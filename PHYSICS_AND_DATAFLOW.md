# umcfUTCIpostprocess — Physics & Dataflow

## Application Summary

`umcfUTCIpostprocess` is an urban microclimate thermal comfort post-processor. It reads radiation and flow fields from a multi-region OpenFOAM CFD simulation and computes the **Universal Thermal Climate Index (UTCI)** at a regular pedestrian-level grid across the domain. The pipeline is orchestrated by `run_utci.py` (Python, 4 stages) and performs the core physics in a C++ binary (`umcfUTCIpostprocess`) using OpenMP parallelism. Output is a set of VTK point-cloud and structured-surface files per timestep.

---

## Physics

### 1. Pedestrian Body Model

The human body is approximated as **5 planar segments** placed at a single representative point (center at pedestrian height):

| Segment | Offset from center | Area vector [m²] | Faces toward |
|---------|-------------------|-------------------|--------------|
| Front   | −0.2 m in x       | (−0.68, 0, 0)     | −X           |
| Back    | +0.2 m in x       | (+0.68, 0, 0)     | +X           |
| Left    | −0.2 m in y       | (0, −0.68, 0)     | −Y           |
| Right   | +0.2 m in y       | (0, +0.68, 0)     | +Y           |
| Top     | centre z          | (0, 0, +0.16)     | +Z           |

Total projected area ≈ 2.88 m².  Radiation is computed independently for each segment.

### 2. View Factors

The view factor from segment *i* to surface patch *j* is:

```
Fij = |nᵢ · r̂| × |nⱼ · r̂| × Aⱼ / (π r²)

  nᵢ  = unit normal of body segment i
  nⱼ  = unit normal of surface patch j
  r̂   = unit vector from j → i
  Aⱼ  = area of patch j [m²]
  r   = distance [m]
```

**Implementation details:**
- Only patches within `R_MAG_MAX = 100 m` of the pedestrian are considered
- Pairs where the body segment does not face the surface (`nᵢ · r < 0`) are skipped
- Each candidate pair is tested for ray occlusion against the STL BVH (embree); blocked pairs contribute zero
- Results stored as sparse `(patch_index, float32_value)` pairs per segment
- Cached to `UTCI/pos/<pedestrian_index>.bin` (binary, version 3)

**Sky fraction** (complement approach):
```
Fsky = max(0,  1 − min(Fijsum, 1))

  Fijsum = Σ Fij   (sum over all wall/veg surfaces)
```
This avoids the unphysical normalisation that arises when sky-boundary patches (domain sides) are included in the sum.

### 3. Outgoing Surface Radiation

For each wall/vegetation patch, the outgoing LW flux is pre-computed in the OpenFOAM `vegetation` region by `calcWallRadOut`:

```
QrOut = σ T⁴ + qr (1 − ε) / ε        ε_surf = 0.9

  T    = surface temperature [K]
  qr   = net radiative flux at surface (DO/P1 radiation model) [W/m²]
  σ    = 5.67 × 10⁻⁸ W m⁻² K⁻⁴
```

Outgoing SW (solar reflection) `qsOut` is computed analogously by `calculateqrsw`.

### 4. Sky Temperature

Cloud-cover-adjusted sky temperature:

```
ec   = (1 − 0.84 cc)(0.527 + 0.161 exp(8.45(1 − 273/Ta))) + 0.84 cc
Tsky = ( 9.365574×10⁻⁶ (1−cc) Ta⁶  +  Ta⁴ cc ec )^0.25

  Ta  = ambient air temperature [K]
  cc  = cloud cover fraction [0–1]
```

### 5. Incident Radiation per Segment

```
qin_LW = Σₘ QrOut[m] × Fij[m]  +  σ Tsky⁴ × Fsky
qin_SW = Σₘ qsOut[m] × Fij[m]  +  Idif × Fsky
```

If `Fijsum > 1` (numerical artefact from very close patches), both wall contributions are normalised by `Fijsum`.

### 6. Mean Radiant Temperature (Tmrt)

Per segment:

```
Tmrt_n⁴ = (ε_p × qin_LW + α_sw × qin_SW) / (σ ε_p)

  ε_p   = 0.97   (person longwave emissivity)
  α_sw  = 0.70   (person shortwave absorptivity)
```

Area-weighted average over all segments:

```
Tmrt_avg = Σ(Tmrt_n × |Aₙ|) / Σ|Aₙ|
```

**Direct solar addition** (if position is unshaded — shadow ray test against STL):

```
T⁴_final = Tmrt_avg⁴  +  fp_solar × α_sw × Idn / (σ ε_p)

fp_solar  = 0.308 cos( β (1 − β²/48402) π/180 )    [β = solar elevation in °]
Idn       = direct normal irradiance [W/m²]
```

### 7. UTCI

A **165-term polynomial** in four inputs:

```
UTCI [°C]  =  Ta  +  Δ(Ta, va, D_Tmrt, Pa)

  Ta      = air temperature [°C]
  va      = wind speed at 10 m [m/s]  (clamped to 0.5–17 m/s)
  D_Tmrt  = Tmrt − Ta  [°C]
  Pa      = vapour pressure [hPa]  (from specific humidity + Magnus formula)
```

---

## Dataflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenFOAM CFD Case                               │
│                                                                     │
│  vegetation/<t>/  T, qr, qrOut, qsOut   (radiation fields)         │
│  air/<t>/         T, U, w               (flow fields at positions)  │
│  constant/        triSurface/wallAndTreeSurfaces.stl                │
│                   sunPosVector, Idif, IDN                           │
│  0/air/           Tambient, cloudCover, wambient, U                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 0  (run once — or reuse existing probe_locs)
┌─────────────────────────────────────────────────────────────────────┐
│  Pedestrian position generation                          run_utci.py│
│                                                                     │
│  postProcess cuttingPlane z=2 m  →  T_pedestrian.vtk               │
│  matplotlib TriFinder containment  →  regular dx×dy grid            │
│  (building interiors excluded — air mesh only)                      │
│  → system/air/probe_locs   (N × "(x y z)")                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 1  (re-run whenever CFD fields change)
┌─────────────────────────────────────────────────────────────────────┐
│  OpenFOAM radiation preprocessing        run_utci.py --vegetation   │
│                                                                     │
│  calculateqrsw  →  qrsw  (direct solar volume field)               │
│  calcSf         →  Sf_wallAndTreeSurfaces.raw  (geometry, t_start) │
│  calcWallRadOut →  qrOut, qsOut  per timestep in vegetation/        │
│  postProcess    →  postProcessing/surfaces/<t>/                     │
│                      Sf_wallAndTreeSurfaces.raw  (face area vectors)│
│                      qrOut_wallAndTreeSurfaces.raw                  │
│                      qsOut_wallAndTreeSurfaces.raw                  │
│                      Sf_skySurfaces.raw                             │
│                      qrOut_skySurfaces.raw                          │
│  postProcess    →  postProcessing/probes/air/<t0>/{T, U, w}        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 2  (C++ binary, re-run per new meteorology)
┌─────────────────────────────────────────────────────────────────────┐
│  umcfUTCIpostprocess  -j 20  --mode flat  --start 3600 --end 86400 │
│                                                                     │
│  Load once:                                                         │
│    allGeo     ← Sf_wallAndTreeSurfaces.raw  (329 023 patch faces)   │
│    skyGeo     ← Sf_skySurfaces.raw          (5 boundary patches)    │
│    STL BVH    ← wallAndTreeSurfaces.stl      (ray occlusion)        │
│    meteo[t]   ← Tambient, cc, Idif, Idn, sunDir, va                 │
│    probeT/U/w ← per-position per-timestep rows                      │
│                                                                     │
│  Batch loop (500 pos / batch, OpenMP):                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ For each position p:                                          │  │
│  │   Load UTCI/pos/<p>.bin  OR  compute + cache:                │  │
│  │     Fij[5][sparse]  ←  geometric formula + ray test          │  │
│  │     Fijsum[5],  Fsky[5] = 1 − min(Fijsum, 1)               │  │
│  │                                                              │  │
│  │ For each timestep t:                                         │  │
│  │   qrOut[m], qsOut[m]  ←  .raw files for this t              │  │
│  │   qin_LW, qin_SW  per segment  ← weighted sum + sky         │  │
│  │   Tmrt[5]  ←  (ε_p qin_LW + α_sw qin_SW) / (σ ε_p) ^0.25  │  │
│  │   Tmrt_avg  ←  area-weighted mean                           │  │
│  │   if unshaded:  add fp_solar × Idn  via solar ray test      │  │
│  │   UTCI  ←  165-term polynomial(Ta, va, Tmrt_avg, RH)        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Write per timestep t:                                              │
│    UTCI/<t>/Tmrt_pedestrian.vtk    (point cloud, Kelvin)           │
│    UTCI/<t>/UTCI.vtk               (Tmrt[°C] + UTCI[°C])          │
│    UTCI/<t>/UTCI_surface.vtk       (structured grid)               │
│    UTCI/<t>/RH_pedestrian.vtk                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 3
┌─────────────────────────────────────────────────────────────────────┐
│  Result collection                                       run_utci.py│
│                                                                     │
│  UTCI/<t>/*.vtk  →  UTCI/results/<stem>_t<NNNNNN>.vtk             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Constants (`src/constants.H`)

| Symbol | Value | Meaning |
|--------|-------|---------|
| σ | 5.67 × 10⁻⁸ W m⁻² K⁻⁴ | Stefan-Boltzmann |
| ε_p | 0.97 | Person LW emissivity |
| α_sw | 0.70 | Person SW absorptivity |
| ε_surf | 0.90 | Wall/veg emissivity |
| R_MAG_MAX | 100 m | Max view-factor range |
| PED_Z | 2.0 m | Pedestrian height above ground |

## Key Source Files

| File | Role |
|------|------|
| `run_utci.py` | Orchestration — 4-stage pipeline |
| `src/umcfUTCIpostprocess.C` | Main: batch loop, I/O wiring |
| `src/tmrtSolver.C` | Tmrt physics (LW+SW balance, sky temp) |
| `src/viewFactor.C` | View factor geometry + ray occlusion |
| `src/utciSolver.C` | UTCI 165-term polynomial |
| `src/pedestrian.C` | 5-segment body model |
| `src/caching.C` | Binary VF cache (float32, version 3) |
| `src/io.C` | Raw/probe file readers, VTK writers |
| `src/constants.H` | All physical constants |
| `openfoam/calculateqrsw/` | OF utility — direct solar volume field |
| `openfoam/calcSf/` | OF utility — surface area vectors |
| `openfoam/calcWallRadOut/` | OF utility — outgoing LW at patches |
