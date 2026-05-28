#!/usr/bin/env python3
"""
UTCI post-processing orchestrator.

Stages
------
  0  Write probe_locs and system probes dict
       auto:    run flat cuttingPlane first; if z-spread > 0.5 m or result empty,
                re-run as terrain (patch surface + PED_Z offset) automatically
       flat:    regular grid at constant z (derived from STL bounds)
       terrain: patch surface on ground/street patches, offset PED_Z above terrain
  1  OpenFOAM: calculateqrsw, calcSf, calcWallRadOut, postProcess surfaces + probes
  2  Run umcfUTCIpostprocess binary → Tmrt + UTCI VTK output
  3  Collect all per-timestep VTK files into <output_dir>/results/ with timestep in filename

Usage
-----
  # auto-detect flat vs terrain (default)
  python3 run_utci.py --case /path/to/case [options]

  # force flat
  python3 run_utci.py --case /path/to/case --mode flat

  # force terrain
  python3 run_utci.py --case /path/to/case --mode terrain \\
      [--terrain-patches street ground] [--ped-grid-dx 5] [--ped-grid-dy 5]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.tri as mtri
import numpy as np
import pyvista as pv

try:
    from scipy.interpolate import griddata as _scipy_griddata
    from scipy.spatial import cKDTree as _cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# Force line-buffered stdout so output appears in SLURM logs without delay.
sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

OF_DIR         = os.environ.get('WM_PROJECT_DIR') or (
    os.path.dirname(os.environ['FOAM_ETC']) if 'FOAM_ETC' in os.environ else '/home/strebdom/OpenFOAM-8'
)
UTCI_UTIL      = os.path.join(os.path.dirname(__file__), 'openfoam')
CALC_TMRT_BIN  = os.path.join(os.path.dirname(__file__), 'build', 'umcfUTCIpostprocess')
BREW_BIN       = '/home/linuxbrew/.linuxbrew/bin'

# Default patches — override via CLI if your case differs
WALL_PATCHES   = ('buildings', 'roofs', 'street', 'ground')
SKY_PATCHES    = ('west', 'east', 'north', 'south', 'top')
VEG_PATCH      = 'air_to_vegetation'

PED_Z          = 2.0
PED_GRID_DX    = 3.0
PED_GRID_DY    = 3.0


def _detect_of_major():
    """Detect local OpenFOAM major version.

    Priority:
      1. WM_PROJECT_VERSION / FOAM_VERSION env vars (set when OF is sourced)
      2. META-INFO/api file in OF_DIR (OF-12+ style)
      3. WM_PROJECT_VERSION line in etc/bashrc
      4. Numeric component from OF_DIR path (e.g. OpenFOAM-8, OpenFOAM-12)
    """
    # 1. Environment variables
    for var in ('WM_PROJECT_VERSION', 'FOAM_VERSION'):
        val = os.environ.get(var, '')
        m = re.search(r'\b(\d+)\b', val)
        if m:
            return int(m.group(1))

    # 2. OF-12+ META-INFO/api file
    meta = os.path.join(OF_DIR, 'META-INFO', 'api')
    if os.path.isfile(meta):
        try:
            with open(meta) as f:
                m = re.search(r'\b(\d+)\b', f.read())
            if m:
                return int(m.group(1))
        except OSError:
            pass

    # 3. WM_PROJECT_VERSION in etc/bashrc
    bashrc = os.path.join(OF_DIR, 'etc', 'bashrc')
    if os.path.isfile(bashrc):
        try:
            with open(bashrc) as f:
                for line in f:
                    m = re.match(r'\s*(?:export\s+)?WM_PROJECT_VERSION\s*=\s*["\']?(\d+)', line)
                    if m:
                        return int(m.group(1))
        except OSError:
            pass

    # 4. Parse OF_DIR path components for a version number
    for part in reversed(os.path.normpath(OF_DIR).split(os.sep)):
        m = re.search(r'(?:OpenFOAM|foam)[-_]?(\d+)', part, re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def _detect_case_of_version(case_dir):
    """Detect which OpenFOAM version a case was run with.

    Uses case structure fingerprints from urbanMicroclimateFoam tutorials:
    - OF-12: constant/<region>/thermophysicalTransport exists
    - OF-8:  constant/<region>/momentumTransport has 'object turbulenceProperties'
             (OF-12 has 'object momentumTransport')

    Returns int major version (8 or 12) or None if undetectable.
    """
    for region in ('air', 'vegetation', ''):
        const_dir = (os.path.join(case_dir, 'constant', region)
                     if region else os.path.join(case_dir, 'constant'))

        # OF-12 fingerprint: thermophysicalTransport file exists
        if os.path.isfile(os.path.join(const_dir, 'thermophysicalTransport')):
            return 12

        # momentumTransport object name: turbulenceProperties=OF-8, momentumTransport=OF-12
        mt = os.path.join(const_dir, 'momentumTransport')
        if os.path.isfile(mt):
            try:
                with open(mt, errors='replace') as f:
                    header = f.read(512)
                if re.search(r'\bobject\s+turbulenceProperties\b', header):
                    return 8
                if re.search(r'\bobject\s+momentumTransport\b', header):
                    return 12
            except OSError:
                pass

    return None


def _foam_shell_setup():
    """Shell prefix for OpenFOAM environment and optional Linuxbrew toolchain."""
    cmd = f'source {OF_DIR}/etc/bashrc'
    if os.path.isdir(BREW_BIN):
        cmd += f' && export PATH={BREW_BIN}:$PATH'
    return cmd


OF_MAJOR = _detect_of_major()

# ──────────────────────────────────────────────────────────────────────────────
# PEDESTRIAN SURFACE — OF dict + grid binning (flat and terrain)
# ──────────────────────────────────────────────────────────────────────────────

def _pedestrian_surface_dict(fields=('T',), terrain_patches=None, z=PED_Z):
    """OpenFOAM surfaces dict for pedestrian position generation (vtk output).

    flat mode (terrain_patches=None):
        cuttingPlane at z=PED_Z — points lie in the air mesh only (building
        interiors naturally excluded).
    terrain mode (terrain_patches provided):
        patch surface on ground/street patches — point z follows terrain height;
        caller adds PED_Z offset when binning.
    """
    if terrain_patches is None:
        point_key = 'basePoint' if (OF_MAJOR is not None and OF_MAJOR >= 12) else 'point'
        normal_key = 'normalVector' if (OF_MAJOR is not None and OF_MAJOR >= 12) else 'normal'
        surface_block = """\
    pedestrian
    {{
        type        cuttingPlane;
        planeType   pointAndNormal;
        pointAndNormalDict
        {{
            {point_key}  (0 0 {z});
            {normal_key} (0 0 1);
        }}
        interpolate false;
    }}""".format(z=z, point_key=point_key, normal_key=normal_key)
    else:
        patches_str = ' '.join(terrain_patches)
        surface_block = """\
    pedestrian
    {{
        type        patch;
        patches     ({patches});
        interpolate false;
    }}""".format(patches=patches_str)

    fields_str = '\n    '.join(fields)
    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='surfacesPedestrian')
        + """type            surfaces;
libs            ("libsampling.so");

surfaceFormat   vtk;
interpolationScheme cellPoint;

fields
(
    {fields}
);

surfaces
(
{surface}
);
""".format(fields=fields_str, surface=surface_block)
    )


def _stl_bbox(stl_path, padding=0.0):
    """Return (xmin, xmax, ymin, ymax) of an STL file, expanded by padding."""
    mesh = pv.read(stl_path)
    b = mesh.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
    return (b[0] - padding, b[1] + padding,
            b[2] - padding, b[3] + padding)


def _bin_vtk_to_grid(vtk_path, dx, dy, z_offset=0.0, bbox=None, fill_radius=-1.0,
                     terrain_vtk_path=None):
    """
    Build a regular dx/dy grid of pedestrian positions from a surface VTK.

    flat mode (z_offset=0, terrain_vtk_path=None):
        Snaps cutting-plane cell centres to the dx/dy grid. z is fixed at the
        cutting-plane height (≈ PED_Z for flat terrain).

    terrain mode (terrain_vtk_path provided):
        Same (x,y) positions as flat mode (cutting-plane cell centres), but z
        is looked up per probe from the terrain surface via nearest-neighbour
        and then offset by PED_Z. Gives terrain-following probes at air-mesh
        density — avoids the probe-count explosion of binning the fine ground
        patch mesh directly.

    legacy terrain mode (z_offset != 0, terrain_vtk_path=None):
        Bins terrain-patch cell centres onto the grid; each gets median
        z_terrain + z_offset. Only use for coarse terrain meshes.

    bbox: optional (xmin, xmax, ymin, ymax) to clip the grid extent.
          When None the VTK mesh bounds are used (full domain).
    """
    mesh = pv.read(vtk_path)

    if z_offset == 0.0 or terrain_vtk_path is not None:
        # Flat / terrain-following: snap cutting-plane cell centres to grid.
        # Building interiors are gaps in the cutting-plane mesh → correctly absent.
        mesh = mesh.clean(tolerance=1e-4)
        pts = np.asarray(mesh.points)

        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
        else:
            b = mesh.bounds
            xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]

        xs = np.arange(round(xmin / dx) * dx, xmax + dx, dx)
        ys = np.arange(round(ymin / dy) * dy, ymax + dy, dy)

        centers = np.asarray(mesh.cell_centers().points)
        mask = ((centers[:, 0] >= xmin) & (centers[:, 0] <= xmax) &
                (centers[:, 1] >= ymin) & (centers[:, 1] <= ymax))
        centers = centers[mask]
        gx_c = np.round(centers[:, 0] / dx) * dx
        gy_c = np.round(centers[:, 1] / dy) * dy
        occ_xy = np.unique(np.column_stack([gx_c, gy_c]), axis=0)

        # Terrain z lookup: NN-query terrain cell centres to get z_terrain(x,y).
        if terrain_vtk_path is not None and _HAVE_SCIPY and len(occ_xy) > 0:
            t_mesh = pv.read(terrain_vtk_path)
            t_centers = np.asarray(t_mesh.cell_centers().points)
            t_tree = _cKDTree(t_centers[:, :2])
            _, idx = t_tree.query(occ_xy)
            occ_z = t_centers[idx, 2] + PED_Z
        else:
            z_fixed = float(np.median(pts[:, 2]))
            occ_z = np.full(len(occ_xy), z_fixed)

        # NN-fill: add empty dx/dy cells within the fill radius of any occupied bin.
        #   fill_radius <  0 → adaptive: 2× 95th-pct NN gap among occupied bins
        #   fill_radius == 0 → disabled (return occupied bins only)
        #   fill_radius >  0 → explicit radius [m]
        if fill_radius != 0.0 and _HAVE_SCIPY and len(occ_xy) > 1:
            occ_tree = _cKDTree(occ_xy)
            if fill_radius < 0:
                d_occ, _ = occ_tree.query(occ_xy, k=2)
                r = max(float(np.percentile(d_occ[:, 1], 95)) * 2.0, max(dx, dy) * 2)
                r = min(r, max(dx, dy) * 4)  # cap: large cells in veg meshes inflate NN gap
            else:
                r = fill_radius
            gxg, gyg = np.meshgrid(xs, ys)
            all_xy = np.column_stack([gxg.ravel(), gyg.ravel()])
            d_all, nn_idx = occ_tree.query(all_xy)
            positions = sorted(
                (float(x), float(y), float(occ_z[i]))
                for (x, y), d, i in zip(all_xy, d_all, nn_idx) if d <= r
            )
        else:
            positions = sorted(
                (float(x), float(y), float(z))
                for (x, y), z in zip(occ_xy, occ_z)
            )
        return positions

    else:
        # Terrain: bin face-center (x,y,z_terrain) onto grid, then offset z.
        # After binning, NN-fill empty dx/dy cells so coverage is uniform.
        centers = mesh.cell_centers().points
        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            mask = ((centers[:, 0] >= xmin) & (centers[:, 0] <= xmax) &
                    (centers[:, 1] >= ymin) & (centers[:, 1] <= ymax))
            centers = centers[mask]
        else:
            b = mesh.bounds
            xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]
        gx = np.round(centers[:, 0] / dx) * dx
        gy = np.round(centers[:, 1] / dy) * dy
        bins = defaultdict(list)
        for i in range(len(centers)):
            bins[(float(gx[i]), float(gy[i]))].append(centers[i, 2])

        occ_xy = np.array(list(bins.keys()))
        occ_z  = np.array([float(np.median(zs)) for zs in bins.values()])

        # Build full uniform grid over bbox
        xs = np.arange(round(xmin / dx) * dx, xmax + dx, dx)
        ys = np.arange(round(ymin / dy) * dy, ymax + dy, dy)
        gxg, gyg = np.meshgrid(xs, ys)
        all_xy = np.column_stack([gxg.ravel(), gyg.ravel()])

        # NN-fill: add empty dx/dy cells within the fill radius of any occupied bin.
        #   fill_radius <  0 → adaptive: 2× 95th-pct NN gap among occupied bins
        #   fill_radius == 0 → disabled (return occupied bins only)
        #   fill_radius >  0 → explicit radius [m]
        if fill_radius != 0.0 and _HAVE_SCIPY and len(occ_xy) > 1:
            occ_tree = _cKDTree(occ_xy)
            if fill_radius < 0:
                d_occ, _ = occ_tree.query(occ_xy, k=2)
                r = max(float(np.percentile(d_occ[:, 1], 95)) * 2.0, max(dx, dy) * 2)
                r = min(r, max(dx, dy) * 4)  # cap: large cells in veg meshes inflate NN gap
            else:
                r = fill_radius
            d_all, nn_idx = occ_tree.query(all_xy)
            positions = sorted(
                (float(x), float(y), float(occ_z[i]) + z_offset)
                for (x, y), d, i in zip(all_xy, d_all, nn_idx) if d <= r
            )
        else:
            positions = sorted(
                (x, y, float(np.median(zs)) + z_offset)
                for (x, y), zs in bins.items()
            )
        return positions




# ──────────────────────────────────────────────────────────────────────────────
# OPENFOAM DICT BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

_FOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {obj};
}}
"""

def _surfaces_patch_dict(wall_patches, sky_patches, fields=('Sf', 'qrOut', 'qsOut')):
    """Raw-format surface dict to extract fields at wall + sky boundary patches."""
    wall_str = ' '.join(wall_patches)
    sky_str  = ' '.join(sky_patches)
    fields_str = '\n    '.join(fields)
    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='surfaces')
        + """
type            surfaces;
libs            ("libsampling.so");

surfaceFormat   raw;
interpolationScheme cellPoint;

fields
(
    {fields}
);

surfaces
(
    wallAndTreeSurfaces
    {{
        type            patch;
        patches         ({wall});
        interpolate     false;
        triangulate     false;
    }}

    skySurfaces
    {{
        type            patch;
        patches         ({sky});
        interpolate     false;
        triangulate     false;
    }}
);
""".format(fields=fields_str, wall=wall_str, sky=sky_str)
    )


def _probes_dict(positions):
    """OpenFOAM probes dict to extract T, U, w at pedestrian positions."""
    pts_block = '\n'.join(f'    ({p[0]} {p[1]} {p[2]})' for p in positions)
    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='probes')
        + """
type            probes;
libs            ("libsampling.so");

writeControl    timeStep;
writeInterval   1;

fields
(
    T
    U
    w
);

probeLocations
(
{points}
);
""".format(points=pts_block)
    )


# ──────────────────────────────────────────────────────────────────────────────
# OPENFOAM RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd, case=None, region=None, time_range=None, use_case=True):
    full = f'{_foam_shell_setup()} && {cmd}'
    if use_case and case:
        full += f' -case {case}'
    if region:
        full += f' -region {region}'
    if time_range:
        full += f' -time {time_range}'
    log_path = None
    if case:
        log_dir = os.path.join(case, 'postProcessing', 'run_utci_logs')
        os.makedirs(log_dir, exist_ok=True)
        safe_cmd = ''.join(c if c.isalnum() else '_' for c in cmd).strip('_') or 'command'
        safe_region = region or 'default'
        safe_time = (time_range or 'notime').replace(':', '_')
        log_path = os.path.join(log_dir, f'{safe_cmd}__{safe_region}__{safe_time}.log')
    print(f'  RUN: {cmd}')
    print(f'       shell: {full}')
    if log_path:
        print(f'       log  : {log_path}')
    t0 = time.time()
    result = subprocess.run(['bash', '-c', full], capture_output=True, text=True)
    elapsed = time.time() - t0
    stdout_tail = (result.stdout or '').strip()[-4000:]
    stderr_tail = (result.stderr or '').strip()[-4000:]
    if log_path:
        with open(log_path, 'w') as f:
            f.write(f'command: {cmd}\n')
            f.write(f'shell: {full}\n')
            if case:
                f.write(f'case: {case}\n')
            if region:
                f.write(f'region: {region}\n')
            if time_range:
                f.write(f'time: {time_range}\n')
            f.write(f'returncode: {result.returncode}\n')
            f.write(f'elapsed_seconds: {elapsed:.6f}\n')
            f.write('\n=== STDOUT ===\n')
            f.write(result.stdout or '')
            f.write('\n=== STDERR ===\n')
            f.write(result.stderr or '')
    if result.returncode != 0:
        print('  [ERROR] OpenFOAM command failed')
        print(f'          command : {cmd}')
        if case:
            print(f'          case    : {case}')
        if region:
            print(f'          region  : {region}')
        if time_range:
            print(f'          time    : {time_range}')
        print(f'          code    : {result.returncode}')
        print(f'          elapsed : {elapsed:.2f}s')
        if stdout_tail:
            print('          stdout tail:')
            for line in stdout_tail.splitlines():
                print(f'            {line}')
        if stderr_tail:
            print('          stderr tail:')
            for line in stderr_tail.splitlines():
                print(f'            {line}')
        print('          Stage 1 may continue, but later steps can silently reuse stale outputs.')
    else:
        print(f'  OK: {cmd} ({elapsed:.2f}s)')
        if stdout_tail:
            print('      stdout tail:')
            for line in stdout_tail.splitlines()[-20:]:
                print(f'        {line}')
        if stderr_tail:
            print('      stderr tail:')
            for line in stderr_tail.splitlines()[-20:]:
                print(f'        {line}')
    return result


def _run_required(cmd, case=None, region=None, time_range=None, use_case=True):
    result = _run(cmd, case=case, region=region, time_range=time_range, use_case=use_case)
    if result.returncode != 0:
        print(f'  [FATAL] Aborting because required command failed: {cmd}')
        sys.exit(result.returncode)
    return result


def _run_required_per_timestep(cmd, case, region, timesteps, max_parallel, use_case=True):
    """Run a required OpenFOAM command independently for each timestep.

    Uses a thread pool because the work is external subprocess execution.
    Raises SystemExit on the first failed timestep after all launched jobs finish.
    """
    if not timesteps:
        return

    max_parallel = max(1, min(int(max_parallel), len(timesteps)))
    if max_parallel == 1:
        for t in timesteps:
            _run_required(cmd, case=case, region=region, time_range=str(t), use_case=use_case)
        return

    print(f'  Running {cmd} for {len(timesteps)} timestep(s) with up to {max_parallel} concurrent jobs')
    failures = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        future_map = {
            pool.submit(_run, cmd, case, region, str(t), use_case): t
            for t in timesteps
        }
        for fut in as_completed(future_map):
            t = future_map[fut]
            result = fut.result()
            if result.returncode != 0:
                failures.append((t, result.returncode))

    if failures:
        t, code = sorted(failures)[0]
        print(f'  [FATAL] Aborting because required command failed: {cmd} at timestep {t} (code {code})')
        sys.exit(code)


def _available(binary):
    r = subprocess.run(
        ['bash', '-lc', f'{_foam_shell_setup()} && command -v {binary}'],
        capture_output=True, text=True)
    return r.returncode == 0


def _compile_if_missing(binary, src_dir):
    if not _available(binary):
        print(f'  Compiling {binary} ...')
        _run(f'wmake {src_dir}', use_case=False)
    else:
        print(f'  {binary} found')


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 – Write system files
# ──────────────────────────────────────────────────────────────────────────────

def _projected_mesh_area_xy(mesh):
    """Projected x-y area of a surface mesh."""
    mesh = mesh.triangulate()
    if mesh.n_cells == 0:
        return 0.0
    pts = mesh.points
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    a = pts[tris[:, 0], :2]
    b = pts[tris[:, 1], :2]
    c = pts[tris[:, 2], :2]
    twice_area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    return float(0.5 * np.abs(twice_area).sum())


def _detect_terrain(vtk_path, bbox=None):
    """Return True if vtk_path looks like a terrain domain.

    Criteria (either triggers terrain mode):
      - VTK has no points (cutting plane at z=PED_Z missed the domain entirely)
      - std(z) of mesh points > 0.5 m (sloped terrain)
      - projected x-y coverage of the flat cutting plane is too small
        relative to the STL bbox area (typical when a terrain-following
        pedestrian surface is needed instead of a flat z=PED_Z plane)
    """
    mesh = pv.read(vtk_path)
    if mesh.n_points == 0:
        return True
    z = mesh.points[:, 2]
    if float(np.std(z)) > 0.5:
        return True
    if bbox is not None:
        bbox_area = max(0.0, (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]))
        if bbox_area > 0.0:
            coverage = _projected_mesh_area_xy(mesh) / bbox_area
            if coverage < 0.2:
                return True
    return False


def _find_pedestrian_vtk(case, subdir, t_start, legacy_name):
    """Return first existing pedestrian surface VTK across legacy and OF12 layouts."""
    t = str(t_start)
    candidates = [
        os.path.join(case, 'postProcessing', subdir, t, legacy_name),
        os.path.join(case, 'postProcessing', subdir, t, 'pedestrian.vtk'),
        os.path.join(case, 'postProcessing', 'air', subdir, t, legacy_name),
        os.path.join(case, 'postProcessing', 'air', subdir, t, 'pedestrian.vtk'),
        os.path.join(case, 'postProcessing', 'vegetation', subdir, t, legacy_name),
        os.path.join(case, 'postProcessing', 'vegetation', subdir, t, 'pedestrian.vtk'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _first_existing_path(paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None



def _surface_cell_centres(vtk_path, bbox=None, terrain_vtk_path=None):
    """Return raw cell centres of the cutting-plane mesh as probe positions.
    No grid snapping — each probe corresponds to exactly one surface cell.
    If terrain_vtk_path is provided, z is replaced with z_terrain + PED_Z
    via nearest-neighbour lookup, giving terrain-following probe heights.
    """
    mesh = pv.read(vtk_path)
    mesh = mesh.clean(tolerance=1e-4)
    centres = np.asarray(mesh.cell_centers().points)
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        mask = ((centres[:, 0] >= xmin) & (centres[:, 0] <= xmax) &
                (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax))
        centres = centres[mask]
    if terrain_vtk_path is not None and _HAVE_SCIPY:
        t_mesh = pv.read(terrain_vtk_path)
        t_centres = np.asarray(t_mesh.cell_centers().points)
        t_tree = _cKDTree(t_centres[:, :2])
        _, idx = t_tree.query(centres[:, :2])
        centres[:, 2] = t_centres[idx, 2] + PED_Z
    return sorted((float(c[0]), float(c[1]), float(c[2])) for c in centres)


def _merge_veg_probes(case, t_start, dx, dy, bbox, fill_radius, terrain_vtk_path, positions):
    """Merge vegetation-region cutting-plane probes into existing positions.
    T values are still sampled from the air mesh — this only adds (x,y,z)
    positions inside the vegetation zone for under-canopy coverage.
    """
    sys_veg = os.path.join(case, 'system', 'vegetation')
    if not os.path.isdir(sys_veg):
        return positions
    try:
        with open(os.path.join(sys_veg, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
        r = _run('postProcess -func surfacesPedestrian',
                 case, region='vegetation', time_range=str(t_start))
        vtk_path = _find_pedestrian_vtk(case, 'surfacesPedestrian', t_start, 'T_pedestrian.vtk')
        if r.returncode != 0 or vtk_path is None:
            return positions
        veg_pos = _bin_vtk_to_grid(vtk_path, dx, dy, z_offset=0.0, bbox=bbox,
                                    fill_radius=fill_radius,
                                    terrain_vtk_path=terrain_vtk_path)
        merged = {(p[0], p[1]): p for p in veg_pos}
        merged.update({(p[0], p[1]): p for p in positions})  # air takes priority
        result = sorted(merged.values())
        print(f'  Vegetation probes: {len(veg_pos)} sampled, +{len(result)-len(positions)} new')
        return result
    except Exception as e:
        print(f'  [WARN] Vegetation probe merge failed: {e}')
        return positions


def _stage0_generate_positions(args):
    """Run postProcess to generate T_pedestrian.vtk and return (positions, resolved_mode)."""
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    # Optional STL bounding-box clipping
    bbox = None
    if args.bbox_padding is not None:
        stl_path = _first_existing_path([
            os.path.join(args.case, 'constant', 'triSurface', 'wallAndTreeSurfaces.stl'),
            os.path.join(args.case, 'constant', 'triSurface', 'walls.stl'),
            os.path.join(args.case, 'constant', 'triSurface', 'facades.stl'),
        ])
        if stl_path is not None:
            bbox = _stl_bbox(stl_path, padding=args.bbox_padding)
            print(f'  STL bbox + {args.bbox_padding} m padding: '
                  f'x=[{bbox[0]:.1f}, {bbox[1]:.1f}]  '
                  f'y=[{bbox[2]:.1f}, {bbox[3]:.1f}]')
        else:
            print('  [WARN] STL not found for bbox clipping '
                  '(tried wallAndTreeSurfaces.stl, walls.stl, facades.stl)')

    # ── surface-centres mode: raw cell centres, no grid snapping ─────────────
    if getattr(args, 'surface_centres', False):
        import tempfile, shutil
        # Step 1: cutting-plane VTK for (x,y,z) cell centres
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))
        vtk_path = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
        if r.returncode != 0 or vtk_path is None:
            return None, 'terrain'
        # Step 2: terrain-patch VTK for z_terrain lookup
        terrain_vtk_path = None
        terrain_patches = list(getattr(args, 'terrain_patches', None) or [])
        if terrain_patches and _HAVE_SCIPY:
            tmp_dir = tempfile.mkdtemp()
            try:
                with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
                    f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))
                r2 = _run('postProcess -func surfacesPedestrian',
                          args.case, region='air', time_range=str(args.t_start))
                t_vtk = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
                if r2.returncode == 0 and t_vtk is not None:
                    terrain_vtk_path = os.path.join(tmp_dir, 'terrain.vtk')
                    shutil.copy2(t_vtk, terrain_vtk_path)
            except Exception:
                pass
            # Restore cutting-plane dict
            with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
                f.write(_pedestrian_surface_dict(terrain_patches=None))
        positions = _surface_cell_centres(vtk_path, bbox=bbox,
                                          terrain_vtk_path=terrain_vtk_path)
        z_note = 'terrain z' if terrain_vtk_path else 'cutting-plane z'
        print(f'  Surface-centres mode: {len(positions)} raw cell centres ({z_note}, no grid snapping)')
        MAX_SURFACE_CENTRES = 250000
        if len(positions) > MAX_SURFACE_CENTRES:
            print(f'  [WARN] {len(positions)} > {MAX_SURFACE_CENTRES} limit — subsampling uniformly')
            step = len(positions) // MAX_SURFACE_CENTRES + 1
            positions = positions[::step]
            print(f'  After subsampling: {len(positions)}')
        return positions, 'terrain'

    # ── forced flat ──────────────────────────────────────────────────────────
    if args.mode == 'flat':
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))
        vtk_path = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
        if r.returncode != 0 or vtk_path is None:
            return None, 'flat'
        positions = _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                      z_offset=0.0, bbox=bbox,
                                      fill_radius=args.ped_grid_fill_radius)
        if args.vegetation:
            positions = _merge_veg_probes(args.case, args.t_start, args.ped_grid_dx,
                                          args.ped_grid_dy, bbox, args.ped_grid_fill_radius,
                                          None, positions)
        return positions, 'flat'

    # ── forced terrain ───────────────────────────────────────────────────────
    # Use the cutting-plane VTK (air cells at z≈PED_Z) for probe positions.
    # The ground-patch mesh is typically much finer than the air mesh and
    # produces ~4× more occupied bins, which makes the UTCI solver run
    # out of memory. Air cell centres at z=PED_Z are already "PED_Z above
    # ground" for flat terrain and give the same probe density as flat mode.
    if args.mode == 'terrain':
        terrain_patches = list(args.terrain_patches)
        # Step 1: cutting-plane VTK for (x,y) probe positions (air-mesh density)
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))
        vtk_path = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
        if r.returncode != 0 or vtk_path is None:
            return None, 'terrain'
        # Step 2: terrain-patch VTK for z_terrain lookup (written to a separate file)
        terrain_vtk_path = None
        if terrain_patches and _HAVE_SCIPY:
            import tempfile, shutil
            tmp_dir = tempfile.mkdtemp()
            try:
                with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
                    f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))
                r2 = _run('postProcess -func surfacesPedestrian',
                          args.case, region='air', time_range=str(args.t_start))
                t_vtk = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
                if r2.returncode == 0 and t_vtk is not None:
                    terrain_vtk_path = os.path.join(tmp_dir, 'terrain.vtk')
                    shutil.copy2(t_vtk, terrain_vtk_path)
            except Exception:
                pass
            # Restore cutting-plane dict for T sampling in later stages
            with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
                f.write(_pedestrian_surface_dict(terrain_patches=None))
        positions = _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                      z_offset=0.0, bbox=bbox,
                                      fill_radius=args.ped_grid_fill_radius,
                                      terrain_vtk_path=terrain_vtk_path)
        if args.vegetation:
            positions = _merge_veg_probes(args.case, args.t_start, args.ped_grid_dx,
                                          args.ped_grid_dy, bbox, args.ped_grid_fill_radius,
                                          terrain_vtk_path, positions)
        return positions, 'terrain'

    # ── auto-detect (default) ─────────────────────────────────────────────────
    # Step 1: always try flat (cuttingPlane at PED_Z) first
    with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
        f.write(_pedestrian_surface_dict(terrain_patches=None))
    r = _run('postProcess -func surfacesPedestrian',
             args.case, region='air', time_range=str(args.t_start))
    vtk_path = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')

    if r.returncode != 0 or vtk_path is None:
        return None, 'auto'

    # Step 2: inspect result
    if not _detect_terrain(vtk_path, bbox=bbox):
        positions = _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                      z_offset=0.0, bbox=bbox,
                                      fill_radius=args.ped_grid_fill_radius)
        if args.vegetation:
            positions = _merge_veg_probes(args.case, args.t_start, args.ped_grid_dx,
                                          args.ped_grid_dy, bbox, args.ped_grid_fill_radius,
                                          None, positions)
        return positions, 'flat'

    # Step 3: auto-detected terrain — cutting plane for (x,y), terrain patches for z.
    print('  Auto-detected terrain domain (z-spread > 0.5 m, low flat-plane coverage, '
          'or empty cutting plane) — switching to terrain mode')
    terrain_patches = list(args.terrain_patches)
    terrain_vtk_path = None
    if terrain_patches and _HAVE_SCIPY:
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
                f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))
            r2 = _run('postProcess -func surfacesPedestrian',
                      args.case, region='air', time_range=str(args.t_start))
            t_vtk = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
            if r2.returncode == 0 and t_vtk is not None:
                terrain_vtk_path = os.path.join(tmp_dir, 'terrain.vtk')
                shutil.copy2(t_vtk, terrain_vtk_path)
        except Exception:
            pass
        # Restore cutting-plane dict for T sampling in later stages
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
    positions = _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                  z_offset=0.0, bbox=bbox,
                                  fill_radius=args.ped_grid_fill_radius,
                                  terrain_vtk_path=terrain_vtk_path)
    if args.vegetation:
        positions = _merge_veg_probes(args.case, args.t_start, args.ped_grid_dx,
                                      args.ped_grid_dy, bbox, args.ped_grid_fill_radius,
                                      terrain_vtk_path, positions)
    return positions, 'terrain'


def stage0(args):
    print(f'\n=== Stage 0: Write system files (mode={args.mode}) ===')

    sys_air = os.path.join(args.case, 'system', 'air')
    positions, resolved_mode = _stage0_generate_positions(args)

    if positions is not None:
        label = 'terrain+{:.0f}m'.format(PED_Z) if resolved_mode == 'terrain' else 'flat'
        print(f'  Positions on {args.ped_grid_dx}×{args.ped_grid_dy} m grid '
              f'({label}, from T_pedestrian.vtk): {len(positions)}')
    else:
        print('  [WARN] Pedestrian surface VTK not generated, falling back to probe_locs')
        probe_locs = os.path.join(sys_air, 'probe_locs')
        if not os.path.isfile(probe_locs):
            print('[ERROR] No probe_locs fallback available')
            sys.exit(1)
        positions = []
        with open(probe_locs) as f:
            for line in f:
                line = line.strip().strip('()')
                parts = line.split()
                if len(parts) == 3:
                    positions.append(tuple(float(v) for v in parts))
        print(f'  Loaded {len(positions)} positions from existing probe_locs')

    # probe_locs (used by umcfUTCIpostprocess to know pedestrian positions)
    with open(os.path.join(sys_air, 'probe_locs'), 'w') as f:
        f.write('(\n')
        for p in positions:
            f.write(f'({p[0]} {p[1]} {p[2]})\n')
        f.write(')\n')
    print(f'  Written probe_locs ({len(positions)} positions)')

    # probes dict (used by postProcess to sample T, U, w)
    with open(os.path.join(sys_air, 'probes'), 'w') as f:
        f.write(_probes_dict(positions))
    print(f'  Written system/air/probes')


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 – OpenFOAM postprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _qrsw_cutting_plane_dict():
    point_key = 'basePoint' if (OF_MAJOR is not None and OF_MAJOR >= 12) else 'point'
    normal_key = 'normalVector' if (OF_MAJOR is not None and OF_MAJOR >= 12) else 'normal'
    return f"""\
/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      qrswCuttingPlane;
}}

type            surfaces;
libs            ("libsampling.so");

surfaceFormat   vtk;
interpolationScheme cellPoint;
fields          ( qrsw );

surfaces
(
    pedestrian
    {{
        type            cuttingPlane;
        planeType       pointAndNormal;
        pointAndNormalDict {{ {point_key} (0 0 2); {normal_key} (0 0 1); }}
        interpolate     false;
    }}
);
"""


def _sample_qrsw_at_probes(case, t_start, t_end, t_step):
    from scipy.spatial import cKDTree
    from io import StringIO as _StringIO

    probe_locs = os.path.join(case, 'system', 'air', 'probe_locs')
    if not os.path.isfile(probe_locs):
        print('  [WARN] probe_locs not found – skipping qrsw probe sampling')
        return

    s = open(probe_locs).read().replace('(', '').replace(')', '')
    pos = np.loadtxt(_StringIO(s), ndmin=2)
    px, py = pos[:, 0], pos[:, 1]
    n_probes = len(pos)

    out_dir = os.path.join(case, 'postProcessing', 'probes', 'qrsw')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'qrsw')

    rows = []
    for t in range(t_start, t_end + 1, t_step):
        vtk_path = os.path.join(case, 'postProcessing', 'qrswCuttingPlane', str(t),
                                'qrsw_pedestrian.vtk')
        if not os.path.isfile(vtk_path):
            rows.append((t, np.zeros(n_probes)))
            continue

        ds = pv.read(vtk_path)
        if 'qrsw' in ds.cell_data:
            ds = ds.cell_data_to_point_data()
        q = ds.point_data['qrsw']
        qmag = np.linalg.norm(q, axis=1) if q.ndim == 2 else np.abs(q)
        kd = cKDTree(ds.points[:, :2])
        _, idx = kd.query(np.column_stack([px, py]))
        rows.append((t, qmag[idx]))

    with open(out_path, 'w') as f:
        for i in range(n_probes):
            f.write(f'# Probe {i} ({pos[i,0]:.3f} {pos[i,1]:.3f} {pos[i,2]:.3f})\n')
        for t, vals in rows:
            f.write(f'{t}')
            for v in vals:
                f.write(f'\t{v:.4f}')
            f.write('\n')

    print(f'  qrsw sampled at {n_probes} probe positions -> {out_path}')


def _ensure_symlink(src, dst):
    if os.path.exists(dst) or os.path.islink(dst):
        return
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def _split_of12_surface_xy(src_path, dst_dir, surface_name):
    """Split OpenFOAM-12 raw .xy multi-field surface output into legacy files."""
    if not os.path.isfile(src_path):
        return

    files = {}

    def open_out(field, columns):
        path = os.path.join(dst_dir, f'{field}_{surface_name}.raw')
        f = open(path, 'w')
        f.write('# ' + ' '.join(columns) + '\n')
        f.write('# split from OpenFOAM-12 surfaces .xy output\n')
        files[field] = f

    os.makedirs(dst_dir, exist_ok=True)
    try:
        with open(src_path) as src:
            for line in src:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    if 'Sf' not in files:
                        open_out('Sf', ('face_x', 'face_y', 'face_z', 'Sf_x', 'Sf_y', 'Sf_z'))
                    if 'qrOut' not in files:
                        open_out('qrOut', ('face_x', 'face_y', 'face_z', 'qrOut'))
                    if 'qsOut' not in files:
                        open_out('qsOut', ('face_x', 'face_y', 'face_z', 'qsOut'))
                    files['Sf'].write(' '.join(parts[0:6]) + '\n')
                    files['qrOut'].write(' '.join(parts[0:3] + [parts[6]]) + '\n')
                    files['qsOut'].write(' '.join(parts[0:3] + [parts[7]]) + '\n')
                elif len(parts) >= 5:
                    if 'qrOut' not in files:
                        open_out('qrOut', ('face_x', 'face_y', 'face_z', 'qrOut'))
                    if 'qsOut' not in files:
                        open_out('qsOut', ('face_x', 'face_y', 'face_z', 'qsOut'))
                    files['qrOut'].write(' '.join(parts[0:3] + [parts[3]]) + '\n')
                    files['qsOut'].write(' '.join(parts[0:3] + [parts[4]]) + '\n')
    finally:
        for f in files.values():
            f.close()


def _normalize_of12_postprocessing(case, region, timesteps):
    """Expose OF12 region-scoped outputs at the legacy paths used by Stage 2.

    Safe and idempotent for OF8 legacy layout (no-op when .xy inputs are absent).
    """
    pp = os.path.join(case, 'postProcessing')

    dst_surfaces = os.path.join(pp, 'surfaces')
    src_roots = [
        os.path.join(pp, region, 'surfaces'),
        os.path.join(pp, 'surfaces'),
    ]
    for src_surfaces in src_roots:
        if not os.path.isdir(src_surfaces):
            continue
        for t in timesteps:
            src_t = os.path.join(src_surfaces, str(t))
            if not os.path.isdir(src_t):
                continue
            dst_t = os.path.join(dst_surfaces, str(t))
            _split_of12_surface_xy(
                os.path.join(src_t, 'wallAndTreeSurfaces.xy'),
                dst_t,
                'wallAndTreeSurfaces'
            )
            _split_of12_surface_xy(
                os.path.join(src_t, 'skySurfaces.xy'),
                dst_t,
                'skySurfaces'
            )

    _ensure_symlink(os.path.join(pp, 'air', 'surfacesPedestrian'), os.path.join(pp, 'surfacesPedestrian'))
    _ensure_symlink(os.path.join(pp, 'air', 'surfacesPedestrianAir'), os.path.join(pp, 'surfacesPedestrianAir'))
    _ensure_symlink(os.path.join(pp, region, 'surfacesPedestrianRad'), os.path.join(pp, 'surfacesPedestrianRad'))
    _ensure_symlink(os.path.join(pp, region, 'qrswCuttingPlane'), os.path.join(pp, 'qrswCuttingPlane'))
    _ensure_symlink(os.path.join(pp, 'air', 'probes'), os.path.join(pp, 'probes', 'air'))

def stage1(args):
    print('\n=== Stage 1: OpenFOAM postprocessing ===')
    time_range = f'{args.t_start}:{args.t_end}'
    timesteps = list(range(args.t_start, args.t_end + 1, args.t_step))
    of_parallel = max(1, min(6, args.threads, len(timesteps)))
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    # Compile utilities
    _compile_if_missing('calculateqrsw', os.path.join(UTCI_UTIL, 'calculateqrsw'))
    _compile_if_missing('calcSf',        os.path.join(UTCI_UTIL, 'calcSf'))
    _compile_if_missing('calcWallRadOut', os.path.join(UTCI_UTIL, 'calcWallRadOut'))

    region = 'vegetation' if args.vegetation else 'air'

    # Step 1: Direct solar radiation volume field
    print(f'  calculateqrsw ({region}) ...')
    _run_required_per_timestep('calculateqrsw', args.case, region, timesteps, of_parallel)

    # Step 2: Surface area vectors. Geometry is static, so Sf is only needed
    # for the first timestep used by Stage 2 to build the surface geometry.
    print(f'  calcSf ({region}) ...')
    geometry_time = timesteps[0]
    _run_required('calcSf', args.case, region=region, time_range=str(geometry_time))

    # Step 3: Outgoing wall radiation
    print(f'  calcWallRadOut ({region}) ...')
    _run_required_per_timestep('calcWallRadOut', args.case, region, timesteps, of_parallel)

    # Step 4: Extract Sf, qrOut, qsOut at wall + sky patches → raw files
    wall_patches = list(args.wall_patches)
    if args.vegetation:
        wall_patches.append(VEG_PATCH)
    sys_region = os.path.join(args.case, 'system', region)
    os.makedirs(sys_region, exist_ok=True)
    print(f'  postProcess: Sf, qrOut, qsOut at patches ({region}) ...')
    with open(os.path.join(sys_region, 'surfaces'), 'w') as f:
        f.write(_surfaces_patch_dict(wall_patches, args.sky_patches))
    _run_required('postProcess -func surfaces', args.case, region=region, time_range=str(geometry_time))

    scalar_timesteps = timesteps[1:]
    if scalar_timesteps:
        with open(os.path.join(sys_region, 'surfaces'), 'w') as f:
            f.write(_surfaces_patch_dict(wall_patches, args.sky_patches, fields=('qrOut', 'qsOut')))
        _run_required_per_timestep('postProcess -func surfaces', args.case, region, scalar_timesteps, of_parallel)

    # Step 5: Extract dense pedestrian-surface fields used by C++ Stage 2
    ped_vtk = _find_pedestrian_vtk(args.case, 'surfacesPedestrian', args.t_start, 'T_pedestrian.vtk')
    ped_mode = args.mode
    if ped_mode == 'auto':
        ped_mode = 'terrain' if (ped_vtk and _detect_terrain(ped_vtk)) else 'flat'

    terrain_patches = list(args.terrain_patches) if ped_mode == 'terrain' else None

    with open(os.path.join(sys_air, 'surfacesPedestrianAir'), 'w') as f:
        f.write(_pedestrian_surface_dict(fields=('T', 'U', 'w'),
                                         terrain_patches=terrain_patches))
    print(f'  postProcess: dense pedestrian T, U, w ({ped_mode}, air) ...')
    _run_required('postProcess -func surfacesPedestrianAir',
                  args.case, region='air', time_range=str(args.t_start))
    if len(timesteps) > 1:
        _run_required_per_timestep('postProcess -func surfacesPedestrianAir',
                                   args.case, 'air', timesteps[1:], of_parallel)

    rad_region = 'vegetation' if args.vegetation else 'air'
    sys_rad = os.path.join(args.case, 'system', rad_region)
    os.makedirs(sys_rad, exist_ok=True)
    with open(os.path.join(sys_rad, 'surfacesPedestrianRad'), 'w') as f:
        f.write(_pedestrian_surface_dict(fields=('qrsw',),
                                         terrain_patches=terrain_patches))
    print(f'  postProcess: dense pedestrian qrsw ({ped_mode}, {rad_region}) ...')
    _run_required_per_timestep('postProcess -func surfacesPedestrianRad',
                               args.case, rad_region, timesteps, of_parallel)

    # Step 6: Extract T, U, w at pedestrian positions (air region, probes)
    print('  postProcess: probes T, U, w (air) ...')
    _run_required('postProcess -func probes', args.case, region='air', time_range=time_range)

    # Step 7: Sample qrsw magnitude at probe positions from a z=2 m cutting plane.
    # This provides direct-solar irradiance at pedestrian positions for both
    # vegetation and non-vegetation workflows.
    print(f'  qrsw cutting plane ({rad_region}) ...')
    with open(os.path.join(sys_rad, 'qrswCuttingPlane'), 'w') as f:
        f.write(_qrsw_cutting_plane_dict())
    _run_required_per_timestep('postProcess -func qrswCuttingPlane',
                               args.case, rad_region, timesteps, of_parallel)

    _normalize_of12_postprocessing(args.case, region, timesteps)
    _sample_qrsw_at_probes(args.case, args.t_start, args.t_end, args.t_step)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 – calcTmrt
# ──────────────────────────────────────────────────────────────────────────────

def _interpolate_utci_surface(case, output_dir, t_start, timesteps):
    """Resample C++ UTCI/Tmrt point-cloud results onto the dense CFD pedestrian
    surface mesh using linear interpolation.

    Reads:  postProcessing/surfacesPedestrian/<t_start>/T_pedestrian.vtk  (CFD mesh)
            <output_dir>/<t>/UTCI.vtk                                     (probe point cloud)
    Writes: <output_dir>/<t>/UTCI_surface.vtk                             (dense interpolated mesh)
    """
    if not _HAVE_SCIPY:
        print('  [WARN] Skipping surface interpolation – scipy not available')
        return

    # Load the dense CFD cutting-plane mesh once (building interiors absent)
    mesh_path = _find_pedestrian_vtk(case, 'surfacesPedestrian', t_start, 'T_pedestrian.vtk')
    if not mesh_path:
        print('  [WARN] CFD surface mesh not found (surfacesPedestrian)  – skipping interpolation')
        return

    mesh = pv.read(mesh_path)
    # Ensure point data (cutting plane may write cell data)
    if mesh.n_points == 0:
        mesh = mesh.cell_data_to_point_data()
    mesh_pts = mesh.points
    mx, my = mesh_pts[:, 0], mesh_pts[:, 1]

    # Restrict interpolation to the interior of the probe bounding box (1 m inset
    # on each side) to avoid edge artefacts from interpolation near the boundary.
    # Points outside this box are set to NaN and filled by nearest-neighbour below.
    BBOX_INSET = 1.0
    print(f'  Interpolating onto CFD surface ({len(mesh_pts)} points) ...')

    done = 0
    for t in timesteps:
        utci_path = os.path.join(case, output_dir, str(t), 'UTCI.vtk')
        if not os.path.isfile(utci_path):
            continue

        probe = pv.read(utci_path)
        px, py = probe.points[:, 0], probe.points[:, 1]

        # Mesh points inside the inset probe bounding box
        in_bbox = (
            (mx >= px.min() + BBOX_INSET) & (mx <= px.max() - BBOX_INSET) &
            (my >= py.min() + BBOX_INSET) & (my <= py.max() - BBOX_INSET)
        )

        out = mesh.copy(deep=True)
        out.clear_data()

        for name in probe.point_data.keys():
            vals = probe.point_data[name]
            interp = np.full(len(mx), np.nan)
            interp[in_bbox] = _scipy_griddata(
                (px, py), vals, (mx[in_bbox], my[in_bbox]), method='linear')
            nan_mask = np.isnan(interp)
            if nan_mask.any():
                interp[nan_mask] = _scipy_griddata(
                    (px, py), vals, (mx[nan_mask], my[nan_mask]), method='nearest')
            # Clip interpolation overshoot to the probe data range
            interp = np.clip(interp, vals.min(), vals.max())
            out.point_data[name] = interp.astype('float32')

        out.save(os.path.join(case, output_dir, str(t), 'UTCI_surface.vtk'))
        done += 1

    print(f'  Written UTCI_surface.vtk for {done}/{len(timesteps)} timesteps')


def stage2(args):
    print('\n=== Stage 2: calcTmrt ===')

    # Preflight: verify qrOut files exist for at least one timestep.
    # If missing, Stage 1 (calcWallRadOut) was not run or ran against the wrong
    # region, and the C++ binary would produce near-zero Tmrt for shadowed positions.
    surf_base = os.path.join(args.case, 'postProcessing', 'surfaces')
    qrout_found = False
    if os.path.isdir(surf_base):
        for t_dir in sorted(os.listdir(surf_base)):
            qrout = os.path.join(surf_base, t_dir, 'qrOut_wallAndTreeSurfaces.raw')
            if os.path.isfile(qrout):
                qrout_found = True
                break
    if not qrout_found:
        print('  [ERROR] No qrOut_wallAndTreeSurfaces.raw found under postProcessing/surfaces/.')
        print('  Run Stage 1 first (calcWallRadOut must complete before Stage 2).')
        print('  If using a vegetation region, ensure --vegetation is set (it is the default).')
        sys.exit(1)

    binary = args.calc_tmrt_bin
    if not os.path.isfile(binary):
        print(f'  [ERROR] Binary not found: {binary}')
        print('  Build: mkdir -p build && cd build && cmake .. && make -j$(nproc)')
        sys.exit(1)

    cmd = [
        binary,
        '--case',       args.case,
        '--start',      str(args.t_start),
        '--end',        str(args.t_end),
        '--step',       str(args.t_step),
        '--output-dir', args.output_dir,
        '-j',           str(args.threads),
    ]
    if args.force_recompute:
        cmd.append('--force-recompute')
    if args.skip_utci:
        cmd.append('--skip-utci')
    cmd += ['--utci-method', args.utci_method]
    if args.lut_path:
        cmd += ['--lut-path', args.lut_path]
    cmd += ['--dense-tumrt-interp', args.dense_tumrt_interp]
    cmd += ['--dense-tumrt-smooth-passes', str(args.dense_tumrt_smooth_passes)]
    cmd += ['--dense-interp-clamp', args.dense_interp_clamp]
    cmd += ['--sky-method', args.sky_method]
    if args.sky_method == 'angular':
        cmd += ['--sky-azimuth-samples', str(args.sky_azimuth_samples)]
        cmd += ['--sky-elevation-samples', str(args.sky_elevation_samples)]
        cmd += ['--sky-ray-length', str(args.sky_ray_length)]
    print('  ' + ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print('  calcTmrt finished')

    if not args.skip_utci:
        print('  Dense surface postprocess is handled in the C++ binary')


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 – Collect results
# ──────────────────────────────────────────────────────────────────────────────

def stage3(args):
    print('\n=== Stage 3: Collect VTK results ===')
    base = os.path.join(args.case, args.output_dir)
    results_dir = os.path.join(base, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f'  Results directory: {results_dir}')

    timesteps = range(args.t_start, args.t_end + 1, args.t_step)
    copied = 0
    for t in timesteps:
        t_dir = os.path.join(base, str(t))
        if not os.path.isdir(t_dir):
            print(f'  [WARN] Timestep directory not found: {t_dir}')
            continue
        for src in glob.glob(os.path.join(t_dir, '*.vtk')):
            stem = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(results_dir, f'{stem}_t{t:06d}.vtk')
            shutil.copy2(src, dst)
            copied += 1

    print(f'  Copied {copied} VTK file(s) → {results_dir}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='UTCI orchestrator (4-stage pipeline + C++ calcTmrt)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--case',       required=True,  help='OpenFOAM case directory')
    p.add_argument('--stages',     nargs='*', type=int, default=[0, 1, 2, 3], metavar='N',
                   help='Stages to run: 0=system files  1=OF postProcess  2=calcTmrt  3=collect results')
    p.add_argument('--t-start',    type=int, default=3600,  dest='t_start')
    p.add_argument('--t-end',      type=int, default=86400, dest='t_end')
    p.add_argument('--t-step',     type=int, default=3600,  dest='t_step')
    p.add_argument('--mode',       default='auto', choices=['auto', 'flat', 'terrain'],
                   help='Probe grid mode: auto=detect from cuttingPlane z-spread, '
                        'flat=constant z=PED_Z, terrain=follow ground patches + PED_Z offset')
    p.add_argument('--output-dir', default='UTCI',  dest='output_dir')
    p.add_argument('-j', '--threads', type=int, default=20)
    p.add_argument('--vegetation', action='store_true', default=True,
                   help='Use vegetation region for surface/radiation sampling (default: True)')
    p.add_argument('--no-vegetation', action='store_false', dest='vegetation',
                   help='Use air region instead of vegetation for surface/radiation sampling')
    p.add_argument('--wall-patches', nargs='+', default=list(WALL_PATCHES), dest='wall_patches')
    p.add_argument('--sky-patches',  nargs='+', default=list(SKY_PATCHES),  dest='sky_patches')
    p.add_argument('--sky-method', default='patch', choices=['patch', 'angular'], dest='sky_method',
                   help='Sky treatment in Stage 2: OpenFOAM patch sky or angular hemisphere sampling')
    p.add_argument('--sky-azimuth-samples', default=48, type=int, dest='sky_azimuth_samples',
                   help='Angular sky azimuth bins for Stage 2')
    p.add_argument('--sky-elevation-samples', default=12, type=int, dest='sky_elevation_samples',
                   help='Angular sky elevation bins for Stage 2')
    p.add_argument('--sky-ray-length', default=5000.0, type=float, dest='sky_ray_length',
                   help='Angular sky ray length [m] for Stage 2')
    p.add_argument('--force-recompute', action='store_true', dest='force_recompute')
    p.add_argument('--skip-utci',       action='store_true', dest='skip_utci')
    p.add_argument('--calc-tmrt-bin', default=CALC_TMRT_BIN, dest='calc_tmrt_bin')
    p.add_argument('--utci-method', default='poly', choices=['poly', 'lut'], dest='utci_method',
                   help='UTCI calculation method: poly=165-term polynomial, lut=lookup table')
    p.add_argument('--lut-path', default='', dest='lut_path',
                   help='Path to utci_offset.Dat LUT file (default: auto-resolved near binary)')
    p.add_argument('--dense-interp-clamp', default='local-range', choices=['none', 'local-range'],
                   dest='dense_interp_clamp',
                   help='Clamp dense interpolation output to the local 4x4 stencil range')
    p.add_argument('--dense-tumrt-interp', default='cubic', choices=['cubic', 'idw'],
                   dest='dense_tumrt_interp',
                   help='Dense Tumrt interpolation from sparse raytraced positions')
    p.add_argument('--dense-tumrt-smooth-passes', default=0, type=int,
                   dest='dense_tumrt_smooth_passes',
                   help='Smooth sparse Tumrt before dense interpolation; 0 disables smoothing')
    # pedestrian grid spacing (flat and terrain)
    p.add_argument('--ped-grid-dx', type=float, default=PED_GRID_DX, dest='ped_grid_dx',
                   help='Pedestrian grid x-spacing [m]')
    p.add_argument('--ped-grid-dy', type=float, default=PED_GRID_DY, dest='ped_grid_dy',
                   help='Pedestrian grid y-spacing [m]')
    p.add_argument('--surface-centres', action='store_true', dest='surface_centres',
                   default=True,
                   help='Use raw cutting-plane cell centres as probe positions (no grid '
                        'snapping). Eliminates the sparse→dense interpolation step — '
                        'Tmrt is computed at every surface cell exactly. (default: on)')
    p.add_argument('--no-surface-centres', action='store_false', dest='surface_centres',
                   help='Disable surface-centres mode; use the pedestrian grid instead.')
    p.add_argument('--ped-grid-fill-radius', type=float, default=0.0,
                   dest='ped_grid_fill_radius', metavar='M',
                   help='NN-fill radius for coarse-mesh gaps [m]: '
                        '0=disable/occupied bins only (default), '
                        '-1=adaptive (2× 95th-pct NN gap), '
                        '>0=explicit radius')
    p.add_argument('--bbox-padding', type=float, default=None, dest='bbox_padding',
                   metavar='M',
                   help='Clip probe grid to STL bounding box + M m padding. '
                        'Default: None (use full domain extent).')

    # terrain-following Stage 0 (also used by auto-detect)
    tg = p.add_argument_group('terrain mode (--mode terrain or auto-detected terrain)')
    tg.add_argument('--terrain-patches', nargs='+', default=['street', 'ground'],
                    dest='terrain_patches',
                    help='Ground patches to sample for terrain-following positions')

    return p.parse_args()


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(round(seconds))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h:d}h {m:02d}m {s:02d}s'
    if m:
        return f'{m:d}m {s:02d}s'
    return f'{seconds:.2f}s'


def main():
    args = parse_args()
    args.case = os.path.abspath(args.case)

    print(f'Case:      {args.case}')
    print(f'Time:      {args.t_start}..{args.t_end}  step {args.t_step}')
    print(f'Stages:    {args.stages}')
    print(f'Threads:   {args.threads}')
    print(f'OpenFOAM:  {OF_DIR} (detected major: {OF_MAJOR if OF_MAJOR is not None else "unknown"})')

    case_of = _detect_case_of_version(args.case)
    print(f'Case OF version: {case_of if case_of is not None else "unknown"}')
    if case_of is not None and OF_MAJOR is not None and case_of != OF_MAJOR:
        print(
            f'\n[ERROR] Case was run with OpenFOAM-{case_of} but the active installation'
            f' is OpenFOAM-{OF_MAJOR}. Surface sampling utilities (calcSf, calcWallRadOut,'
            f' calculateqrsw) must match the case version — results will be wrong or the'
            f' run will crash.\n'
            f'        Source the correct OpenFOAM environment before running, e.g.:\n'
            f'          source /home/strebdom/OpenFOAM-{case_of}/etc/bashrc\n'
        )
        sys.exit(1)

    print(f'Vegetation: {args.vegetation}')
    print(f'UTCI method: {args.utci_method}' + (f' ({args.lut_path})' if args.lut_path else ''))

    stage_map = {0: stage0, 1: stage1, 2: stage2, 3: stage3}
    stage_timings = {}
    total_start = time.perf_counter()
    for s in sorted(args.stages):
        if s not in stage_map:
            print(f'[ERROR] Unknown stage {s}')
            sys.exit(1)
        stage_start = time.perf_counter()
        stage_map[s](args)
        elapsed = time.perf_counter() - stage_start
        stage_timings[s] = elapsed
        print(f'=== Stage {s} done in {_format_elapsed(elapsed)} ===')

    total_elapsed = time.perf_counter() - total_start
    print('\n=== Timing Summary ===')
    for s in sorted(stage_timings):
        print(f'  Stage {s}: {_format_elapsed(stage_timings[s])}')
    print(f'  Total:   {_format_elapsed(total_elapsed)}')
    print('\nDone.')


if __name__ == '__main__':
    main()
