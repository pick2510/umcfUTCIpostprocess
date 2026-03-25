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
      --pedestrian-mesh /path/to/T_pedestrian.vtk \\
      --pos-xmin -150 --pos-xmax 500 --pos-ymin -2350 --pos-ymax -1700 \\
      --pos-grid-step 10
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

OF_DIR         = '/home/strebdom/OpenFOAM-8'
UTCI_UTIL      = os.path.join(os.path.dirname(__file__), 'openfoam')
CALC_TMRT_BIN  = os.path.join(os.path.dirname(__file__), 'src', 'build', 'umcfUTCIpostprocess')

# Default patches — override via CLI if your case differs
WALL_PATCHES   = ('buildings', 'roofs', 'street', 'ground')
SKY_PATCHES    = ('west', 'east', 'north', 'south', 'top')
VEG_PATCH      = 'air_to_vegetation'

PED_Z          = 2.0
PED_GRID_DX    = 3.0
PED_GRID_DY    = 3.0

# ──────────────────────────────────────────────────────────────────────────────
# PEDESTRIAN GRID — flat
# ──────────────────────────────────────────────────────────────────────────────

def _pedestrian_plane_dict(z=PED_Z):
    """OpenFOAM surfaces dict: horizontal cutting plane at pedestrian height, vtk output."""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*/\n"
        + _FOAM_HEADER.format(obj='surfacesPedestrian')
        + f"""surfaceFormat   vtk;

fields
(
    T
);

surfaces
(
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
    }}
);
"""
    )


def _bin_vtk_to_grid(vtk_path, dx, dy):
    """
    Read points from a pedestrian-plane VTK and snap them to a regular dx/dy grid.

    Any VTK point within half a cell of a grid node marks that node as valid.
    Returns a sorted list of unique (x, y, z) grid-node positions at the
    z-height of the cutting plane.
    """
    import numpy as np
    pts = np.array(_read_vtk_points(vtk_path))
    if pts.size == 0:
        return []

    # Average z from the cutting plane (should all be ~PED_Z)
    z = float(np.median(pts[:, 2]))

    # Bin each point to the nearest grid node
    gx = np.round(pts[:, 0] / dx) * dx
    gy = np.round(pts[:, 1] / dy) * dy

    unique = set(zip(gx.tolist(), gy.tolist()))
    positions = sorted((float(x), float(y), z) for x, y in unique)
    return positions


def _read_vtk_points(vtk_path):
    """Read point coordinates from a VTK legacy file (ASCII or pyvista)."""
    try:
        import pyvista as pv
        pts = pv.read(vtk_path).points
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    except Exception:
        pass
    # Manual ASCII VTK parser fallback
    with open(vtk_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('POINTS'):
            n = int(line.split()[1])
            data = []
            j = i + 1
            while len(data) < n * 3 and j < len(lines):
                data.extend(map(float, lines[j].split()))
                j += 1
            import numpy as np
            pts = np.array(data).reshape(-1, 3)
            return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    raise RuntimeError(f'Could not parse POINTS from {vtk_path}')


# ──────────────────────────────────────────────────────────────────────────────
# PEDESTRIAN GRID — terrain-following
# ──────────────────────────────────────────────────────────────────────────────

def pedestrian_grid_terrain(mesh_vtk, xmin, xmax, ymin, ymax, grid_step,
                             z_offset=PED_Z, ray_z_start=300.0, ray_z_end=1000.0):
    """
    Cast vertical rays onto a surface mesh VTK and return positions offset
    z_offset metres above each terrain intersection.

    Parameters
    ----------
    mesh_vtk    : path to the pedestrian-level surface VTK (e.g. T_pedestrian.vtk)
    xmin/xmax   : x domain bounds for ray origins
    ymin/ymax   : y domain bounds for ray origins
    grid_step   : horizontal spacing between rays [m]
    z_offset    : height above terrain surface [m] (default 2 m)
    ray_z_start : z of ray origin — below terrain [m]
    ray_z_end   : z of ray terminus — above terrain [m]
    """
    import numpy as np
    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_vtk)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    mesh = reader.GetOutput()

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()

    xs = np.arange(xmin, xmax + 0.5 * grid_step, grid_step)
    ys = np.arange(ymin, ymax + 0.5 * grid_step, grid_step)
    n_rays = len(xs) * len(ys)
    print(f'  Terrain grid: X [{xmin:.0f}..{xmax:.0f}], Y [{ymin:.0f}..{ymax:.0f}] '
          f'→ {n_rays} rays')

    positions = []
    for xi in xs:
        for yi in ys:
            start = [xi, yi, ray_z_start]
            end   = [xi, yi, ray_z_end]
            pts = vtk.vtkPoints()
            obb.IntersectWithLine(start, end, pts, None)
            n = pts.GetData().GetNumberOfTuples()
            if n > 0:
                p = [0.0, 0.0, 0.0]
                pts.GetPoint(0, p)
                positions.append((p[0], p[1], p[2] + z_offset))

    print(f'  Terrain intersections found: {len(positions)} / {n_rays}')
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

def stage0(args):
    print(f'\n=== Stage 0: Write system files ({args.mode}) ===')

    if args.mode == 'terrain':
        if not args.pedestrian_mesh:
            print('[ERROR] --pedestrian-mesh is required for --mode terrain')
            sys.exit(1)
        positions = pedestrian_grid_terrain(
            args.pedestrian_mesh,
            args.pos_xmin, args.pos_xmax,
            args.pos_ymin, args.pos_ymax,
            args.pos_grid_step,
            z_offset=PED_Z,
        )
    else:
        # Flat mode: generate pedestrian cutting plane via OpenFOAM.
        # The cutting plane lies in the air-region mesh only, so building
        # interiors are naturally excluded without any raycasting.
        sys_air = os.path.join(args.case, 'system', 'air')
        os.makedirs(sys_air, exist_ok=True)
        with open(os.path.join(sys_air, 'surfacesPedestrian'), 'w') as f:
            f.write(_pedestrian_plane_dict(PED_Z))

        r = _run('postProcess -func surfacesPedestrian',
                 args.case, region='air', time_range=str(args.t_start))

        vtk_path = os.path.join(
            args.case, 'postProcessing', 'surfacesPedestrian',
            str(args.t_start), 'T_pedestrian.vtk')
        if r.returncode == 0 and os.path.isfile(vtk_path):
            positions = _bin_vtk_to_grid(vtk_path, args.ped_grid_dx, args.ped_grid_dy)
            print(f'  Positions on {args.ped_grid_dx}×{args.ped_grid_dy} m grid '
                  f'(building cutout from T.vtk): {len(positions)}')
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

def stage2(args):
    print('\n=== Stage 2: calcTmrt ===')
    binary = args.calc_tmrt_bin
    if not os.path.isfile(binary):
        print(f'  [ERROR] Binary not found: {binary}')
        print('  Build: cd src && mkdir -p build && cd build && cmake .. && make -j$(nproc)')
        sys.exit(1)

    cmd = [
        binary,
        '--case',       args.case,
        '--mode',       args.mode,
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

    print('  ' + ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print('  calcTmrt finished')


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
    p.add_argument('--vegetation', action='store_true',
                   help='Use vegetation region for surface/radiation sampling')
    p.add_argument('--wall-patches', nargs='+', default=list(WALL_PATCHES), dest='wall_patches')
    p.add_argument('--sky-patches',  nargs='+', default=list(SKY_PATCHES),  dest='sky_patches')
    p.add_argument('--force-recompute', action='store_true', dest='force_recompute')
    p.add_argument('--skip-utci',       action='store_true', dest='skip_utci')
    p.add_argument('--calc-tmrt-bin', default=CALC_TMRT_BIN, dest='calc_tmrt_bin')

    # flat grid spacing
    p.add_argument('--ped-grid-dx', type=float, default=PED_GRID_DX, dest='ped_grid_dx')
    p.add_argument('--ped-grid-dy', type=float, default=PED_GRID_DY, dest='ped_grid_dy')

    # terrain-following Stage 0
    tg = p.add_argument_group('terrain mode (--mode terrain)')
    tg.add_argument('--pedestrian-mesh', default=None, dest='pedestrian_mesh',
                    help='Surface VTK to ray-cast onto (e.g. T_pedestrian.vtk)')
    tg.add_argument('--pos-xmin',      type=float, default=None, dest='pos_xmin')
    tg.add_argument('--pos-xmax',      type=float, default=None, dest='pos_xmax')
    tg.add_argument('--pos-ymin',      type=float, default=None, dest='pos_ymin')
    tg.add_argument('--pos-ymax',      type=float, default=None, dest='pos_ymax')
    tg.add_argument('--pos-grid-step', type=float, default=10.0, dest='pos_grid_step',
                    help='Horizontal grid spacing for terrain ray-casting [m]')

    return p.parse_args()


def main():
    args = parse_args()
    args.case = os.path.abspath(args.case)

    print(f'Case:      {args.case}')
    print(f'Time:      {args.t_start}..{args.t_end}  step {args.t_step}')
    print(f'Stages:    {args.stages}')
    print(f'Threads:   {args.threads}')
    print(f'Vegetation: {args.vegetation}')

    stage_map = {0: stage0, 1: stage1, 2: stage2, 3: stage3}
    for s in sorted(args.stages):
        if s not in stage_map:
            print(f'[ERROR] Unknown stage {s}')
            sys.exit(1)
        stage_map[s](args)

    print('\nDone.')


if __name__ == '__main__':
    main()
