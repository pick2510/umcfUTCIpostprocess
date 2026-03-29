# umcfUTCIpostprocess

Post-processing tool for computing the **Universal Thermal Climate Index (UTCI)** from [urbanMicroclimateFoam](https://github.com/pick2510/urbanMicroclimateFoam) CFD simulations. Reads OpenFOAM radiation and flow fields, computes mean radiant temperature (Tmrt) at a pedestrian-level grid via a 5-segment body model and view-factor ray casting, then evaluates UTCI using the standard 165-term polynomial. Output is a set of VTK files per timestep.

For a full description of the physics and internal dataflow see [PHYSICS_AND_DATAFLOW.md](PHYSICS_AND_DATAFLOW.md).

---

## Requirements

**C++ binary (`umcfUTCIpostprocess`)**
- CMake ≥ 3.16, C++17 compiler, OpenMP
- Eigen3 (bundled as git submodule)
- zlib (optional — enables view-factor cache compression)
- zstr (bundled as git submodule, used only when zlib is present)

**Python orchestrator (`run_utci.py`)**
- Python ≥ 3.9
- numpy, pyvista, matplotlib, scipy (see `requirements.txt`)

**OpenFOAM utilities** (compiled from `openfoam/` by `run_utci.py` on first run)
- `calculateqrsw` — direct solar volume field
- `calcSf` — surface area vectors
- `calcWallRadOut` — outgoing LW/SW at wall patches

---

## Build

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The binary is written to `build/umcfUTCIpostprocess`. To enable cache compression, zlib must be present (`apt install zlib1g-dev` or equivalent); CMake detects it automatically.

---

## Usage

The full pipeline is orchestrated by `run_utci.py`. All four stages can be run in one command:

```bash
python3 run_utci.py --case /path/to/openfoam/case
```

**Common options**

| Option | Default | Description |
|--------|---------|-------------|
| `--case PATH` | *(required)* | OpenFOAM case directory |
| `--stages N …` | `0 1 2 3` | Stages to run (0=grid, 1=OF fields, 2=Tmrt+UTCI, 3=collect) |
| `--t-start S` | `3600` | First timestep [s] |
| `--t-end S` | `86400` | Last timestep [s] |
| `--t-step S` | `3600` | Timestep interval [s] |
| `-j N` | `20` | OpenMP threads for Stage 2 |
| `--mode` | `auto` | Probe grid mode: `auto` \| `flat` \| `terrain` |
| `--ped-grid-dx M` | `3.0` | Pedestrian grid x-spacing [m] |
| `--ped-grid-dy M` | `3.0` | Pedestrian grid y-spacing [m] |
| `--output-dir DIR` | `UTCI` | Output directory (relative to `--case`) |
| `--utci-method` | `poly` | UTCI method: `poly` (165-term polynomial) or `lut` (lookup table) |
| `--wall-patches NAME …` | `buildings roofs street ground` | OpenFOAM patch names for wall/roof surfaces |
| `--sky-patches NAME …` | `west east north south top` | OpenFOAM patch names for sky boundaries |
| `--bbox-padding M` | *(none)* | Clip probe grid to STL bounding box + padding [m] |
| `--force-recompute` | off | Recompute view factors even if cache exists |
| `--skip-utci` | off | Compute Tmrt only, skip UTCI |
| `--no-vegetation` | off | Use `air` region instead of `vegetation` for radiation fields |

**Terrain options** (used when `--mode terrain` or auto-detected)

| Option | Default | Description |
|--------|---------|-------------|
| `--terrain-patches NAME …` | `street ground` | Ground patch names for terrain height sampling |

### Probe grid mode

`--mode auto` (default) runs the flat cuttingPlane at `z = 2 m` first. If the result is empty or `std(z) > 0.5 m`, the domain is treated as terrain and positions are re-derived from the ground patches with a `+2 m` offset. Use `--mode flat` or `--mode terrain` to override.

### Examples

```bash
# Run all stages, 20 threads, hourly timesteps
python3 run_utci.py --case /data/case_01 -j 20

# Re-run only Stage 2 (Tmrt + UTCI) with existing cached view factors
python3 run_utci.py --case /data/case_01 --stages 2

# Terrain case, explicit patch names
python3 run_utci.py --case /data/hill_case \
    --mode terrain --terrain-patches ground street \
    --ped-grid-dx 5 --ped-grid-dy 5

# Clip probe grid to STL bounding box with 5 m padding
python3 run_utci.py --case /data/case_01 --bbox-padding 5
```

### C++ binary flags (Stage 2 direct invocation)

These flags are only available when calling the binary directly, not via `run_utci.py`:

| Flag | Description |
|------|-------------|
| `--compress-cache` / `--no-compress-cache` | Force gzip-compressed or plain cache files |
| `--write-debug-terms` | Write per-timestep LW/SW radiation breakdown VTK files |
| `--write-debug-qrsw` | Write `qrsw_surface.vtk` debug file per timestep |
| `--max-positions N` | Cap number of positions (for testing) |
| `--batch-size N` | View-factor batch size (default 500) |
| `--filter-radius R` / `--filter-cx X` / `--filter-cy Y` | Spatial filter around a centre point |

---

## Output

Stage 2 writes the following VTK files under `<case>/<output-dir>/<timestep>/`:

| File | Contents |
|------|---------|
| `UTCI.vtk` | Point cloud: Tmrt [°C], UTCI [°C] per pedestrian position |
| `Tmrt_pedestrian.vtk` | Point cloud: Tmrt [K] |
| `RH_pedestrian.vtk` | Point cloud: relative humidity [%] |
| `Tumrt_surface.vtk` | Dense surface mesh: pre-solar Tmrt interpolated onto CFD pedestrian mesh |
| `Tmrt_surface.vtk` | Dense surface mesh: final Tmrt (with solar) |
| `RH_surface.vtk` | Dense surface mesh: relative humidity [%] |
| `UTCI_surface.vtk` | Dense surface mesh: UTCI [°C] and Tmrt [°C] |

The dense surface files require a `T_pedestrian.vtk` mesh from Stage 1 (CFD pedestrian surface). If absent they are skipped silently.

Stage 3 copies all per-timestep VTK files to `<output-dir>/results/` with a `_t<NNNNNN>` suffix.

---

## Repository layout

```
run_utci.py              — pipeline orchestrator (Stages 0–3)
src/                     — C++ sources
  umcfUTCIpostprocess.cpp  main loop, CLI
  tmrtSolver.cpp           Tmrt physics
  viewFactor.cpp           view factor geometry + ray occlusion
  utciSolver.cpp           UTCI 165-term polynomial + LUT
  pedestrian.cpp           5-segment body model
  raycaster.cpp            STL BVH ray intersection
  denseStage2.cpp          dense surface interpolation and output
  caching.cpp              binary view-factor cache
  io.cpp                   field I/O (raw, VTK, probes, meteo)
  constants.h              physical constants
  logging.h                logging helpers
openfoam/                — OpenFOAM utility sources
  calculateqrsw/
  calcSf/
  calcWallRadOut/
extern/                  — git submodules (Eigen, zstr)
PHYSICS_AND_DATAFLOW.md  — physics equations and full dataflow diagram
```
