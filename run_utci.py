#!/usr/bin/env python3
"""
UTCI post-processing orchestrator.

Stages
------
  0  Write probe_locs and system probes dict
       flat:    regular grid at constant z (derived from STL bounds)
       terrain: ray-cast onto a surface mesh VTK, offset 2 m above terrain
  1  OpenFOAM: calculateqrsw, calcSf, calcWallRadOut, postProcess surfaces + probes
  2  Run umcfUTCIpostprocess binary → Tmrt + UTCI VTK output
  3  Collect all per-timestep VTK files into <output_dir>/results/ with timestep in filename

Usage
-----
  # flat domain (default)
  python3 run_utci.py --case /path/to/case [options]

  # terrain-following
  python3 run_utci.py --case /path/to/case --mode terrain \\
      [--terrain-patches street ground] [--ped-grid-dx 5] [--ped-grid-dy 5]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# Force line-buffered stdout so output appears in SLURM logs without delay.
sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

OF_DIR         = '/home/strebdom/OpenFOAM-8'
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

def _pedestrian_surface_dict(terrain_patches=None, z=PED_Z):
    """OpenFOAM surfaces dict for pedestrian position generation (vtk output).

    flat mode (terrain_patches=None):
        cuttingPlane at z=PED_Z — points lie in the air mesh only (building
        interiors naturally excluded).
    terrain mode (terrain_patches provided):
        patch surface on ground/street patches — point z follows terrain height;
        caller adds PED_Z offset when binning.
    """
    if terrain_patches is None:
        surface_block = f"""\
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
    }}"""
    else:
        patches_str = ' '.join(terrain_patches)
        surface_block = f"""\
    pedestrian
    {{
        type        patch;
        patches     ({patches_str});
        interpolate false;
    }}"""

    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='surfacesPedestrian')
        + f"""#includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg"

surfaceFormat   vtk;

fields
(
    T
);

surfaces
(
{surface_block}
);
"""
    )


def _bin_vtk_to_grid(vtk_path, dx, dy, z_offset=0.0):
    """
    Build a regular dx/dy grid of pedestrian positions from a surface VTK.

    flat mode (z_offset=0):
        Generates a regular grid from the VTK bounds and keeps only points
        that fall inside the surface mesh (building interiors excluded).
        Uses pyvista select_interior_points for the inside test.

    terrain mode (z_offset=PED_Z):
        Bins face-center points from the ground-patch VTK onto the dx/dy grid;
        each grid node gets the median terrain z of its bin + z_offset.
    """
    import numpy as np
    import pyvista as pv

    mesh = pv.read(vtk_path)

    if z_offset == 0.0:
        # Flat: regular grid filtered by 2-D triangulation containment test.
        # Build a matplotlib TriFinder on the mesh triangles (exact, O(log N) per point,
        # vectorised over the whole grid in one call).
        import matplotlib.tri as mtri

        mesh = mesh.triangulate()
        pts  = mesh.points
        tris = mesh.faces.reshape(-1, 4)[:, 1:]
        tri    = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        finder = tri.get_trifinder()

        b  = mesh.bounds
        xs = np.arange(round(b[0] / dx) * dx, b[1] + dx, dx)
        ys = np.arange(round(b[2] / dy) * dy, b[3] + dy, dy)
        gx, gy = np.meshgrid(xs, ys)
        inside = finder(gx.ravel(), gy.ravel()) >= 0

        z = float(np.median(pts[:, 2]))
        vx = gx.ravel()[inside]
        vy = gy.ravel()[inside]
        positions = sorted((float(x), float(y), z) for x, y in zip(vx, vy))
        return positions

    else:
        # Terrain: bin face-center (x,y,z_terrain) onto grid, then offset z
        from collections import defaultdict
        centers = mesh.cell_centers().points
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
        + f"""
#includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg"

surfaceFormat   raw;

fields
(
    {fields_str}
);

surfaces
(
    wallAndTreeSurfaces
    {{
        type            patch;
        patches         ({wall_str});
        interpolate     false;
        triangulate     false;
    }}

    skySurfaces
    {{
        type            patch;
        patches         ({sky_str});
        interpolate     false;
        triangulate     false;
    }}
);
"""
    )


def _probes_dict(positions):
    """OpenFOAM probes dict to extract T, U, w at pedestrian positions."""
    pts_block = '\n'.join(f'    ({p[0]} {p[1]} {p[2]})' for p in positions)
    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='probes')
        + f"""
#includeEtc "caseDicts/postProcessing/probes/probes.cfg"

fields
(
    T
    U
    w
);

probeLocations
(
{pts_block}
);
"""
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
    result = subprocess.run(['bash', '-c', full], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [WARN] {cmd!r} returned {result.returncode}:\n{result.stderr[-600:]}')
    else:
        print(f'  OK: {cmd}')
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

def _stage0_generate_positions(args):
    """Run postProcess to generate T_pedestrian.vtk and return binned positions."""
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    terrain_patches = list(args.terrain_patches) if args.mode == 'terrain' else None
    with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
        f.write(_pedestrian_surface_dict(terrain_patches=terrain_patches))

    r = _run('postProcess -func surfacesPedestrian',
             args.case, region='air', time_range=str(args.t_start))

    vtk_path = os.path.join(
        args.case, 'postProcessing', 'surfacesPedestrian',
        str(args.t_start), 'T_pedestrian.vtk')

    if r.returncode != 0 or not os.path.isfile(vtk_path):
        return None

    z_offset = PED_Z if args.mode == 'terrain' else 0.0
    return _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy, z_offset=z_offset)


def stage0(args):
    print(f'\n=== Stage 0: Write system files ({args.mode}) ===')

    sys_air = os.path.join(args.case, 'system', 'air')
    positions = _stage0_generate_positions(args)

    if positions is not None:
        label = 'terrain+{:.0f}m'.format(PED_Z) if args.mode == 'terrain' else 'flat'
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
# STAGE 1 – OpenFOAM postprocessing (utci_clement workflow)
# ──────────────────────────────────────────────────────────────────────────────

def stage1(args):
    print('\n=== Stage 1: OpenFOAM postprocessing ===')
    time_range = f'{args.t_start}:{args.t_end}'

    # Compile utilities
    _compile_if_missing('calculateqrsw', os.path.join(UTCI_UTIL, 'calculateqrsw'))
    _compile_if_missing('calcSf',        os.path.join(UTCI_UTIL, 'calcSf'))
    _compile_if_missing('calcWallRadOut', os.path.join(UTCI_UTIL, 'calcWallRadOut'))

    region = 'vegetation' if args.vegetation else 'air'

    # Step 1: Direct solar radiation volume field
    print(f'  calculateqrsw ({region}) ...')
    _run('calculateqrsw', args.case, region=region, time_range=time_range)

    # Step 2: Surface area vectors (first timestep only)
    print(f'  calcSf ({region}) ...')
    _run('calcSf', args.case, region=region, time_range=str(args.t_start))

    # Step 3: Outgoing wall radiation
    print(f'  calcWallRadOut ({region}) ...')
    _run('calcWallRadOut', args.case, region=region, time_range=time_range)

    # Step 4: Extract Sf, qrOut, qsOut at wall + sky patches → raw files
    wall_patches = list(args.wall_patches)
    if args.vegetation:
        wall_patches.append(VEG_PATCH)
    sys_region = os.path.join(args.case, 'system', region)
    os.makedirs(sys_region, exist_ok=True)
    with open(os.path.join(sys_region, 'surfaces'), 'w') as f:
        f.write(_surfaces_patch_dict(wall_patches, args.sky_patches))
    print(f'  postProcess: Sf, qrOut, qsOut at patches ({region}) ...')
    _run('postProcess -func surfaces', args.case, region=region, time_range=time_range)

    # Step 5: Extract T, U, w at pedestrian positions (air region, probes)
    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)
    print('  postProcess: probes T, U, w (air) ...')
    _run('postProcess -func probes', args.case, region='air', time_range=time_range)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 – calcTmrt
# ──────────────────────────────────────────────────────────────────────────────

def _interpolate_utci_surface(case, output_dir, t_start, timesteps):
    """Resample C++ UTCI/Tmrt point-cloud results onto the dense CFD pedestrian
    surface mesh using cubic interpolation (same approach as utci_clement).

    Reads:  postProcessing/surfacesPedestrian/<t_start>/T_pedestrian.vtk  (CFD mesh)
            <output_dir>/<t>/UTCI.vtk                                     (probe point cloud)
    Writes: <output_dir>/<t>/UTCI_surface.vtk                             (dense interpolated mesh)
    """
    try:
        import numpy as np
        import pyvista as pv
        from scipy.interpolate import griddata
    except ImportError as e:
        print(f'  [WARN] Skipping surface interpolation – missing dependency: {e}')
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
    print(f'  Interpolating onto CFD surface ({len(mesh_pts)} points) ...')

    done = 0
    for t in timesteps:
        utci_path = os.path.join(case, output_dir, str(t), 'UTCI.vtk')
        if not os.path.isfile(utci_path):
            continue

        probe = pv.read(utci_path)
        px, py = probe.points[:, 0], probe.points[:, 1]

        out = mesh.copy(deep=True)
        out.clear_data()

        for name in probe.point_data.keys():
            vals = probe.point_data[name]
            interp = griddata((px, py), vals, (mx, my), method='cubic')
            nan_mask = np.isnan(interp)
            if nan_mask.any():
                interp[nan_mask] = griddata(
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
        timesteps = list(range(args.t_start, args.t_end + 1, args.t_step))
        _interpolate_utci_surface(args.case, args.output_dir, args.t_start, timesteps)


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
        description='UTCI orchestrator (utci_clement workflow + C++ calcTmrt)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--case',       required=True,  help='OpenFOAM case directory')
    p.add_argument('--stages',     nargs='*', type=int, default=[0, 1, 2, 3], metavar='N',
                   help='Stages to run: 0=system files  1=OF postProcess  2=calcTmrt  3=collect results')
    p.add_argument('--t-start',    type=int, default=3600,  dest='t_start')
    p.add_argument('--t-end',      type=int, default=86400, dest='t_end')
    p.add_argument('--t-step',     type=int, default=3600,  dest='t_step')
    p.add_argument('--mode',       default='flat', choices=['flat', 'terrain'])
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

    # terrain-following Stage 0
    tg = p.add_argument_group('terrain mode (--mode terrain)')
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
