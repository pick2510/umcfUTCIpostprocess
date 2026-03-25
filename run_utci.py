#!/usr/bin/env python3
"""
UTCI post-processing orchestrator — utci_clement workflow + C++ calcTmrt.

Stages
------
  0  Write probe_locs and system probes dict (pedestrian grid at z=2m)
  1  OpenFOAM: calculateqrsw, calcSf, calcWallRadOut, postProcess surfaces, postProcess probes
  2  Run calcTmrt binary → Tmrt + UTCI VTK output

Usage
-----
  python3 run_utci.py --case /path/to/case [options]
  python3 run_utci.py --case /path/to/case --stages 1 2
"""

import argparse
import math
import os
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
# PEDESTRIAN GRID
# ──────────────────────────────────────────────────────────────────────────────

def pedestrian_grid(case, dx=PED_GRID_DX, dy=PED_GRID_DY):
    """Return list of (x, y, z) probe positions derived from wallAndTreeSurfaces.stl bounds."""
    stl = os.path.join(case, 'constant', 'triSurface', 'wallAndTreeSurfaces.stl')
    if not os.path.isfile(stl):
        stl = os.path.join(case, 'constant', 'triSurface', 'walls.stl')

    if os.path.isfile(stl):
        try:
            import vtk
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl)
            reader.Update()
            b = reader.GetOutput().GetBounds()   # (xmin,xmax, ymin,ymax, zmin,zmax)
            xmin = math.floor(b[0] / dx) * dx
            xmax = math.ceil (b[1] / dx) * dx
            ymin = math.floor(b[2] / dy) * dy
            ymax = math.ceil (b[3] / dy) * dy
            import numpy as np
            xs = np.arange(xmin, xmax + 0.5*dx, dx)
            ys = np.arange(ymin, ymax + 0.5*dy, dy)
            print(f'  Grid from STL bounds: X [{xmin:.0f}..{xmax:.0f}], '
                  f'Y [{ymin:.0f}..{ymax:.0f}] → {len(xs)*len(ys)} positions')
            return [(x, y, PED_Z) for x in xs for y in ys]
        except Exception as e:
            print(f'  [WARN] Could not read STL bounds: {e}')

    # Fallback: read existing probe_locs if present
    probe_locs = os.path.join(case, 'system', 'air', 'probe_locs')
    if os.path.isfile(probe_locs):
        positions = []
        with open(probe_locs) as f:
            for line in f:
                line = line.strip().strip('()')
                parts = line.split()
                if len(parts) == 3:
                    positions.append(tuple(float(v) for v in parts))
        if positions:
            print(f'  Loaded {len(positions)} positions from existing probe_locs')
            return positions

    raise RuntimeError('Cannot determine pedestrian grid: no STL bounds and no probe_locs')


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
    print('\n=== Stage 0: Write system files ===')
    positions = pedestrian_grid(args.case)

    sys_air = os.path.join(args.case, 'system', 'air')
    os.makedirs(sys_air, exist_ok=True)

    # probe_locs (used by calcTmrt to know pedestrian positions)
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
        print('  Build: cmake --build /mnt/nvme/UTCI_OF_CPP/UTCI_util/calcTmrt/build -j8')
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='UTCI orchestrator (utci_clement workflow + C++ calcTmrt)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--case',       required=True,  help='OpenFOAM case directory')
    p.add_argument('--stages',     nargs='*', type=int, default=[0, 1, 2], metavar='N',
                   help='Stages to run: 0=system files  1=OF postProcess  2=calcTmrt')
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

    return p.parse_args()


def main():
    args = parse_args()
    args.case = os.path.abspath(args.case)

    print(f'Case:      {args.case}')
    print(f'Time:      {args.t_start}..{args.t_end}  step {args.t_step}')
    print(f'Stages:    {args.stages}')
    print(f'Threads:   {args.threads}')
    print(f'Vegetation: {args.vegetation}')

    stage_map = {0: stage0, 1: stage1, 2: stage2}
    for s in sorted(args.stages):
        if s not in stage_map:
            print(f'[ERROR] Unknown stage {s}')
            sys.exit(1)
        stage_map[s](args)

    print('\nDone.')


if __name__ == '__main__':
    main()
