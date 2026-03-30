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
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import matplotlib.tri as mtri
import numpy as np
import pyvista as pv

try:
    from scipy.interpolate import griddata as _scipy_griddata
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

# Default patches — override via CLI if your case differs
WALL_PATCHES   = ('buildings', 'roofs', 'street', 'ground')
SKY_PATCHES    = ('west', 'east', 'north', 'south', 'top')
VEG_PATCH      = 'air_to_vegetation'

PED_Z          = 2.0
PED_GRID_DX    = 3.0
PED_GRID_DY    = 3.0

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
        surface_block = """\
    pedestrian
    {{
        type        cuttingPlane;
        planeType   pointAndNormal;
        pointAndNormalDict
        {{
            basePoint    (0 0 {z});
            normalVector (0 0 1);
        }}
        interpolate false;
    }}""".format(z=z)
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
        + """#includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg"

surfaceFormat   vtk;

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


def _bin_vtk_to_grid(vtk_path, dx, dy, z_offset=0.0, bbox=None):
    """
    Build a regular dx/dy grid of pedestrian positions from a surface VTK.

    flat mode (z_offset=0):
        Generates a regular grid and keeps only points inside the surface mesh
        (building interiors excluded via matplotlib TriFinder).

    terrain mode (z_offset=PED_Z):
        Bins face-center points onto the dx/dy grid; each cell gets the median
        terrain z + z_offset.

    bbox: optional (xmin, xmax, ymin, ymax) to clip the grid extent.
          When None the VTK mesh bounds are used (full domain).
    """
    mesh = pv.read(vtk_path)

    if z_offset == 0.0:
        # Flat: regular grid filtered by 2-D triangulation containment test.
        # Build a matplotlib TriFinder on the mesh triangles (exact, O(log N) per point,
        # vectorised over the whole grid in one call).
        mesh = mesh.triangulate()
        pts  = mesh.points
        tris = mesh.faces.reshape(-1, 4)[:, 1:]
        tri    = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        finder = tri.get_trifinder()

        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
        else:
            b = mesh.bounds
            xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]

        xs = np.arange(round(xmin / dx) * dx, xmax + dx, dx)
        ys = np.arange(round(ymin / dy) * dy, ymax + dy, dy)
        gx, gy = np.meshgrid(xs, ys)
        inside = finder(gx.ravel(), gy.ravel()) >= 0

        z = float(np.median(pts[:, 2]))
        vx = gx.ravel()[inside]
        vy = gy.ravel()[inside]
        positions = sorted((float(x), float(y), z) for x, y in zip(vx, vy))
        return positions

    else:
        # Terrain: bin face-center (x,y,z_terrain) onto grid, then offset z
        centers = mesh.cell_centers().points
        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            mask = ((centers[:, 0] >= xmin) & (centers[:, 0] <= xmax) &
                    (centers[:, 1] >= ymin) & (centers[:, 1] <= ymax))
            centers = centers[mask]
        gx = np.round(centers[:, 0] / dx) * dx
        gy = np.round(centers[:, 1] / dy) * dy
        bins = defaultdict(list)
        for i in range(len(centers)):
            bins[(float(gx[i]), float(gy[i]))].append(centers[i, 2])
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
#includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg"

surfaceFormat   raw;

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
#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

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
    full = f'source {OF_DIR}/etc/bashrc && {cmd}'
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


def _available(binary):
    r = subprocess.run(
        ['bash', '-lc', f'source {OF_DIR}/etc/bashrc && command -v {binary}'],
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

def _detect_terrain(vtk_path):
    """Return True if vtk_path looks like a terrain domain.

    Criteria (either triggers terrain mode):
      - VTK has no points (cutting plane at z=PED_Z missed the domain entirely)
      - std(z) of mesh points > 0.5 m (sloped terrain)
    """
    mesh = pv.read(vtk_path)
    if mesh.n_points == 0:
        return True
    z = mesh.points[:, 2]
    return float(np.std(z)) > 0.5


def _stage0_generate_positions(args):
    """Run postProcess to generate T_pedestrian.vtk and return (positions, resolved_mode)."""
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    vtk_path = os.path.join(
        args.case, 'postProcessing', 'surfacesPedestrian',
        str(args.t_start), 'T_pedestrian.vtk')

    # Optional STL bounding-box clipping
    bbox = None
    if args.bbox_padding is not None:
        stl_path = os.path.join(args.case, 'constant', 'triSurface',
                                'wallAndTreeSurfaces.stl')
        if os.path.isfile(stl_path):
            bbox = _stl_bbox(stl_path, padding=args.bbox_padding)
            print(f'  STL bbox + {args.bbox_padding} m padding: '
                  f'x=[{bbox[0]:.1f}, {bbox[1]:.1f}]  '
                  f'y=[{bbox[2]:.1f}, {bbox[3]:.1f}]')
        else:
            print(f'  [WARN] STL not found for bbox clipping: {stl_path}')

    # ── forced flat ──────────────────────────────────────────────────────────
    if args.mode == 'flat':
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=None))
        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))
        if r.returncode != 0 or not os.path.isfile(vtk_path):
            return None, 'flat'
        return _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                 z_offset=0.0, bbox=bbox), 'flat'

    # ── forced terrain ───────────────────────────────────────────────────────
    if args.mode == 'terrain':
        terrain_patches = list(args.terrain_patches)
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))
        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))
        if r.returncode != 0 or not os.path.isfile(vtk_path):
            return None, 'terrain'
        return _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                 z_offset=PED_Z, bbox=bbox), 'terrain'

    # ── auto-detect (default) ─────────────────────────────────────────────────
    # Step 1: always try flat (cuttingPlane at PED_Z) first
    with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
        f.write(_pedestrian_surface_dict(terrain_patches=None))
    r = _run('postProcess -func surfacesPedestrian',
             args.case, region='air', time_range=str(args.t_start))

    if r.returncode != 0 or not os.path.isfile(vtk_path):
        return None, 'auto'

    # Step 2: inspect result
    if not _detect_terrain(vtk_path):
        return _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                                 z_offset=0.0, bbox=bbox), 'flat'

    # Step 3: re-run as terrain
    print('  Auto-detected terrain domain (z-spread > 0.5 m or empty cutting plane) '
          '— switching to terrain mode')
    terrain_patches = list(args.terrain_patches)
    with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
        f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))
    r = _run('postProcess -func surfacesPedestrian',
             args.case, region='air', time_range=str(args.t_start))
    if r.returncode != 0 or not os.path.isfile(vtk_path):
        return None, 'terrain'
    return _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy,
                             z_offset=PED_Z, bbox=bbox), 'terrain'


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
    return """\
/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      qrswCuttingPlane;
}

#includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg"

surfaceFormat   vtk;
fields          ( qrsw );

surfaces
(
    pedestrian
    {
        type            cuttingPlane;
        planeType       pointAndNormal;
        pointAndNormalDict { point (0 0 2); normal (0 0 1); }
        interpolate     false;
    }
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

def stage1(args):
    print('\n=== Stage 1: OpenFOAM postprocessing ===')
    time_range = f'{args.t_start}:{args.t_end}'
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    # Compile utilities
    _compile_if_missing('calculateqrsw', os.path.join(UTCI_UTIL, 'calculateqrsw'))
    _compile_if_missing('calcSf',        os.path.join(UTCI_UTIL, 'calcSf'))
    _compile_if_missing('calcWallRadOut', os.path.join(UTCI_UTIL, 'calcWallRadOut'))

    region = 'vegetation' if args.vegetation else 'air'

    # Step 1: Direct solar radiation volume field
    print(f'  calculateqrsw ({region}) ...')
    _run_required('calculateqrsw', args.case, region=region, time_range=time_range)

    # Step 2: Surface area vectors (first timestep only)
    print(f'  calcSf ({region}) ...')
    _run_required('calcSf', args.case, region=region, time_range=str(args.t_start))

    # Step 3: Outgoing wall radiation
    print(f'  calcWallRadOut ({region}) ...')
    _run_required('calcWallRadOut', args.case, region=region, time_range=time_range)

    # Step 4: Extract Sf, qrOut, qsOut at wall + sky patches → raw files
    wall_patches = list(args.wall_patches)
    if args.vegetation:
        wall_patches.append(VEG_PATCH)
    sys_region = os.path.join(args.case, 'system', region)
    os.makedirs(sys_region, exist_ok=True)
    with open(os.path.join(sys_region, 'surfaces'), 'w') as f:
        f.write(_surfaces_patch_dict(wall_patches, args.sky_patches))
    print(f'  postProcess: Sf, qrOut, qsOut at patches ({region}) ...')
    _run_required('postProcess -func surfaces', args.case, region=region, time_range=time_range)

    # Step 5: Extract dense pedestrian-surface fields used by C++ Stage 2
    ped_vtk = os.path.join(args.case, 'postProcessing', 'surfacesPedestrian',
                           str(args.t_start), 'T_pedestrian.vtk')
    ped_mode = args.mode
    if ped_mode == 'auto':
        ped_mode = 'terrain' if (os.path.isfile(ped_vtk) and _detect_terrain(ped_vtk)) else 'flat'

    terrain_patches = list(args.terrain_patches) if ped_mode == 'terrain' else None

    with open(os.path.join(sys_air, 'surfacesPedestrianAir'), 'w') as f:
        f.write(_pedestrian_surface_dict(fields=('T', 'U', 'w'),
                                         terrain_patches=terrain_patches))
    print(f'  postProcess: dense pedestrian T, U, w ({ped_mode}, air) ...')
    _run_required('postProcess -func surfacesPedestrianAir',
                  args.case, region='air', time_range=time_range)

    rad_region = 'vegetation' if args.vegetation else 'air'
    sys_rad = os.path.join(args.case, 'system', rad_region)
    os.makedirs(sys_rad, exist_ok=True)
    with open(os.path.join(sys_rad, 'surfacesPedestrianRad'), 'w') as f:
        f.write(_pedestrian_surface_dict(fields=('qrsw',),
                                         terrain_patches=terrain_patches))
    print(f'  postProcess: dense pedestrian qrsw ({ped_mode}, {rad_region}) ...')
    _run_required('postProcess -func surfacesPedestrianRad',
                  args.case, region=rad_region, time_range=time_range)

    # Step 6: Extract T, U, w at pedestrian positions (air region, probes)
    print('  postProcess: probes T, U, w (air) ...')
    _run_required('postProcess -func probes', args.case, region='air', time_range=time_range)

    # Step 7: Sample qrsw magnitude at probe positions from a z=2 m cutting plane.
    # This provides direct-solar irradiance including canopy attenuation.
    if args.vegetation:
        print('  qrsw cutting plane (vegetation) ...')
        sys_veg = os.path.join(args.case, 'system', 'vegetation')
        os.makedirs(sys_veg, exist_ok=True)
        with open(os.path.join(sys_veg, 'qrswCuttingPlane'), 'w') as f:
            f.write(_qrsw_cutting_plane_dict())
        _run_required('postProcess -func qrswCuttingPlane',
                      args.case, region='vegetation', time_range=time_range)
        _sample_qrsw_at_probes(args.case, args.t_start, args.t_end, args.t_step)



# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 – calcTmrt
# ──────────────────────────────────────────────────────────────────────────────

def _interpolate_utci_surface(case, output_dir, t_start, timesteps):
    """Resample C++ UTCI/Tmrt point-cloud results onto the dense CFD pedestrian
    surface mesh using cubic interpolation.

    Reads:  postProcessing/surfacesPedestrian/<t_start>/T_pedestrian.vtk  (CFD mesh)
            <output_dir>/<t>/UTCI.vtk                                     (probe point cloud)
    Writes: <output_dir>/<t>/UTCI_surface.vtk                             (dense interpolated mesh)
    """
    if not _HAVE_SCIPY:
        print('  [WARN] Skipping surface interpolation – scipy not available')
        return

    # Load the dense CFD cutting-plane mesh once (building interiors absent)
    mesh_path = os.path.join(case, 'postProcessing', 'surfacesPedestrian',
                             str(t_start), 'T_pedestrian.vtk')
    if not os.path.isfile(mesh_path):
        print(f'  [WARN] CFD surface mesh not found: {mesh_path}  – skipping interpolation')
        return

    mesh = pv.read(mesh_path)
    # Ensure point data (cutting plane may write cell data)
    if mesh.n_points == 0:
        mesh = mesh.cell_data_to_point_data()
    mesh_pts = mesh.points
    mx, my = mesh_pts[:, 0], mesh_pts[:, 1]

    # Restrict interpolation to the interior of the probe bounding box (1 m inset
    # on each side) to avoid edge artefacts from cubic extrapolation.
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
                (px, py), vals, (mx[in_bbox], my[in_bbox]), method='cubic')
            nan_mask = np.isnan(interp)
            if nan_mask.any():
                interp[nan_mask] = _scipy_griddata(
                    (px, py), vals, (mx[nan_mask], my[nan_mask]), method='nearest')
            # Clip cubic overshoot to the probe data range
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
    p.add_argument('--force-recompute', action='store_true', dest='force_recompute')
    p.add_argument('--skip-utci',       action='store_true', dest='skip_utci')
    p.add_argument('--calc-tmrt-bin', default=CALC_TMRT_BIN, dest='calc_tmrt_bin')
    p.add_argument('--utci-method', default='poly', choices=['poly', 'lut'], dest='utci_method',
                   help='UTCI calculation method: poly=165-term polynomial, lut=lookup table')
    p.add_argument('--lut-path', default='', dest='lut_path',
                   help='Path to utci_offset.Dat LUT file (default: auto-resolved near binary)')

    # pedestrian grid spacing (flat and terrain)
    p.add_argument('--ped-grid-dx', type=float, default=PED_GRID_DX, dest='ped_grid_dx',
                   help='Pedestrian grid x-spacing [m]')
    p.add_argument('--ped-grid-dy', type=float, default=PED_GRID_DY, dest='ped_grid_dy',
                   help='Pedestrian grid y-spacing [m]')
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


def main():
    args = parse_args()
    args.case = os.path.abspath(args.case)

    print(f'Case:      {args.case}')
    print(f'Time:      {args.t_start}..{args.t_end}  step {args.t_step}')
    print(f'Stages:    {args.stages}')
    print(f'Threads:   {args.threads}')
    print(f'Vegetation: {args.vegetation}')
    print(f'UTCI method: {args.utci_method}' + (f' ({args.lut_path})' if args.lut_path else ''))

    stage_map = {0: stage0, 1: stage1, 2: stage2, 3: stage3}
    for s in sorted(args.stages):
        if s not in stage_map:
            print(f'[ERROR] Unknown stage {s}')
            sys.exit(1)
        stage_map[s](args)

    print('\nDone.')


if __name__ == '__main__':
    main()
