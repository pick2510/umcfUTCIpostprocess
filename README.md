# umcfUTCIpostprocess

Post-processing tool for computing the **Universal Thermal Climate Index (UTCI)** from [urbanMicroclimateFoam](https://github.com/pick2510/urbanMicroclimateFoam) CFD simulations. Reads OpenFOAM radiation and flow fields, computes mean radiant temperature (Tmrt) at a pedestrian-level grid via a 5-segment body model and view-factor ray casting, then evaluates UTCI. Output is a set of VTK files per timestep.

Key performance features: OpenMP-parallel view factor computation and timestep processing, SoA float32 view-factor cache (v11 plain / v12 gzip), bulk `strtod` file parser, `computeFast()` hot path with per-timestep surface data hoisted, reusable `DenseInterpPlan` for dense surface interpolation, and thread-cap removal for cached runs.

For physics equations and internal dataflow see [PHYSICS_AND_DATAFLOW.md](PHYSICS_AND_DATAFLOW.md).

---

## Requirements

**C++ binary (`umcfUTCIpostprocess`)**
- CMake ≥ 3.16, C++17 compiler, OpenMP
- Eigen3 (bundled as git submodule)
- zlib (optional — enables view-factor cache compression; auto-detected by CMake)
- zstr (bundled as git submodule, used only when zlib is present)

**Python orchestrator (`run_utci.py`)**
- Python ≥ 3.9
- See `requirements.txt` for pinned versions; key packages:
  - numpy, scipy, pyvista, vtk, matplotlib, joblib, tqdm, pandas

**OpenFOAM** (must be sourced in environment)
- Utilities compiled from `openfoam/` on first Stage 1 run: `calculateqrsw`, `calcSf`, `calcWallRadOut`

---

## Build

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The binary is written to `build/umcfUTCIpostprocess`. `run_utci.py` expects it there by default. zlib is detected automatically; install it with `apt install zlib1g-dev` or equivalent to enable cache compression.

---

## Usage

The full pipeline is orchestrated by `run_utci.py`. All four stages run in one command:

```bash
python3 run_utci.py --case /path/to/openfoam/case
```

### Pipeline stages

| Stage | Flag | What it does |
|-------|------|--------------|
| 0 | grid | Generate pedestrian probe positions → `system/air/probe_locs` |
| 1 | OF fields | Run OpenFOAM utilities + postProcess to extract radiation/flow fields |
| 2 | Tmrt+UTCI | Run C++ binary: view factors, Tmrt, UTCI, dense surface outputs |
| 3 | collect | Copy all per-timestep VTK files to `<output-dir>/results/` with `_t<NNNNNN>` suffix |

Use `--stages` to run a subset, e.g. `--stages 2 3`.

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `--case PATH` | *(required)* | OpenFOAM case directory |
| `--stages N …` | `0 1 2 3` | Stages to run |
| `--t-start S` | `3600` | First timestep [s] |
| `--t-end S` | `86400` | Last timestep [s] |
| `--t-step S` | `3600` | Timestep interval [s] |
| `-j N` | `20` | OpenMP threads for Stage 2 |
| `--mode` | `auto` | Probe grid mode: `auto` \| `flat` \| `terrain` |
| `--ped-grid-dx M` | `3.0` | Pedestrian grid x-spacing [m] |
| `--ped-grid-dy M` | `3.0` | Pedestrian grid y-spacing [m] |
| `--output-dir DIR` | `UTCI` | Output directory (relative to `--case`) |
| `--utci-method` | `poly` | UTCI method: `poly` (165-term polynomial) or `lut` (lookup table) |
| `--lut-path FILE` | *(auto)* | Path to `utci_offset.Dat`; default: next to binary |
| `--wall-patches NAME …` | `buildings roofs street ground` | OpenFOAM patch names for wall/roof surfaces |
| `--sky-patches NAME …` | `west east north south top` | OpenFOAM patch names for sky boundaries |
| `--bbox-padding M` | *(none)* | Clip probe grid to STL bounding box + M m padding |
| `--force-recompute` | off | Recompute view factors even if cache exists |
| `--skip-utci` | off | Compute Tmrt only, skip UTCI |
| `--no-vegetation` | off | Use `air` region instead of `vegetation` for radiation fields |
| `--calc-tmrt-bin PATH` | `build/umcfUTCIpostprocess` | Override path to C++ binary |

### Terrain options

| Option | Default | Description |
|--------|---------|-------------|
| `--terrain-patches NAME …` | `street ground` | Ground patch names for terrain height sampling |

### Probe grid mode

`--mode auto` (default) runs a flat cuttingPlane at `z = 2 m` first. If the result is empty or `std(z) > 0.5 m`, the domain is treated as terrain and positions are re-derived from the ground patches with a `+2 m` offset. Use `--mode flat` or `--mode terrain` to override.

### Examples

```bash
# Run all stages, 20 threads, hourly timesteps over a full day
python3 run_utci.py --case /data/case_01 -j 20

# Re-run only Tmrt+UTCI (Stage 2) with existing cached view factors
python3 run_utci.py --case /data/case_01 --stages 2

# Terrain case with explicit patch names
python3 run_utci.py --case /data/hill_case \
    --mode terrain --terrain-patches ground street \
    --ped-grid-dx 5 --ped-grid-dy 5

# Non-standard wall patches
python3 run_utci.py --case /data/case_01 \
    --wall-patches buildings roofs pavement vegetation_wall

# Clip probe grid to geometry bounds with padding
python3 run_utci.py --case /data/case_01 --bbox-padding 5
```

### C++ binary — additional flags

These are only available when calling the binary directly (not via `run_utci.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--compress-cache` | auto | Write gzip-compressed cache files (default when zlib available) |
| `--no-compress-cache` | — | Force uncompressed cache files |
| `--batch-size N` | `500` | View-factor batch size; lower = less peak memory |
| `--max-positions N` | *(none)* | Cap number of positions (for testing) |
| `--filter-radius R` | *(none)* | Keep only positions within radius R of (`--filter-cx`, `--filter-cy`) |
| `--filter-cx X` | — | Filter centre X coordinate |
| `--filter-cy Y` | — | Filter centre Y coordinate |
| `--write-debug-terms` | off | Write per-timestep LW/SW radiation breakdown VTK files |
| `--write-debug-qrsw` | off | Write `qrsw_surface.vtk` per timestep |

---

## Output

Stage 2 writes the following files under `<case>/<output-dir>/<timestep>/`:

### Sparse point-cloud outputs (always written)

| File | Contents |
|------|---------|
| `Tmrt_pedestrian.vtk` | Tmrt [K] at each probe position |
| `RH_pedestrian.vtk` | Relative humidity [%] at each probe position |
| `UTCI.vtk` | Tmrt [°C] and UTCI [°C] at each probe position |

### Dense surface outputs (written when Stage 1 dense VTKs are present)

Interpolated onto the full CFD pedestrian surface mesh (`T_pedestrian.vtk` from Stage 1).

| File | Contents |
|------|---------|
| `Tumrt_surface.vtk` | Pre-solar mean radiant temperature [K] |
| `Tmrt_surface.vtk` | Final mean radiant temperature [K] (with direct solar) |
| `RH_surface.vtk` | Relative humidity [%] |
| `UTCI_surface.vtk` | UTCI [°C] and Tmrt [°C] |

Dense files are silently skipped if `T_pedestrian.vtk`, `qrsw_pedestrian.vtk`, `U_pedestrian.vtk`, or `w_pedestrian.vtk` are absent for a timestep.

### Debug outputs (only with `--write-debug-terms` / `--write-debug-qrsw`)

| File | Contents |
|------|---------|
| `qrsw_surface.vtk` | Solar irradiance magnitude on dense surface mesh |
| `TumrtAvg_terms/` | Per-segment LW/SW radiation breakdown |

Stage 3 copies all per-timestep VTK files to `<output-dir>/results/<stem>_t<NNNNNN>.vtk`.

---

## Repository layout

```
run_utci.py              — pipeline orchestrator (Stages 0–3)
requirements.txt         — pinned Python dependencies
src/
  umcfUTCIpostprocess.cpp  main loop, CLI argument parsing
  tmrtSolver.cpp           Tmrt physics (LW+SW balance, sky temperature)
  viewFactor.cpp           view factor geometry + ray occlusion
  utciSolver.cpp           UTCI 165-term polynomial + LUT
  pedestrian.cpp           5-segment body model
  raycaster.cpp            STL uniform-grid BVH, DDA ray traversal
  denseStage2.cpp          dense surface interpolation and VTK output
  caching.cpp              binary view-factor cache (v11 plain / v12 gzip)
  io.cpp                   raw/VTK/probe file I/O, meteo reader
  constants.h              all physical constants
  logging.h                logging helpers
openfoam/
  calculateqrsw/           OF utility — direct solar vector field (qrsw)
  calcSf/                  OF utility — surface area vectors
  calcWallRadOut/          OF utility — outgoing LW and SW at patches
extern/
  eigen/                   Eigen3 (header-only, git submodule)
  zstr/                    zlib stream wrapper (header-only, git submodule)
PHYSICS_AND_DATAFLOW.md  — physics equations and full dataflow diagram
BENCHMARK.md             — validation results
```
