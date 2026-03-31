# umcfUTCIpostprocess — Physics & Dataflow

## Application Summary

`umcfUTCIpostprocess` is an urban microclimate thermal comfort post-processor. It reads radiation and flow fields from a multi-region OpenFOAM CFD simulation and computes the **Universal Thermal Climate Index (UTCI)** at a regular pedestrian-level grid across the domain. The pipeline is orchestrated by `run_utci.py` (Python, 4 stages) and performs the core physics in a C++ binary (`umcfUTCIpostprocess`) using OpenMP parallelism. Output is a set of VTK point-cloud and structured-surface files per timestep.

---

## Physics

### 1. Pedestrian Body Model

The human body is approximated as **5 planar segments** placed at a single representative point (center at pedestrian height):

| Segment | Body point (relative to center) | Area vector [m²] | Faces toward |
|---------|--------------------------------|-------------------|--------------|
| Front   | (−0.2, 0, −1.0)                | (−0.68, 0, 0)     | −X           |
| Back    | (+0.2, 0, −1.0)                | (+0.68, 0, 0)     | +X           |
| Left    | (0, −0.2, −1.0)                | (0, −0.68, 0)     | −Y           |
| Right   | (0, +0.2, −1.0)                | (0, +0.68, 0)     | +Y           |
| Top     | (0, 0, 0)                      | (0, 0, +0.16)     | +Z           |

Side segments are placed 1.0 m below the pedestrian center (torso level); the top segment is at the center (head level). The center itself is placed at `PED_Z = 2.0 m` above the ground surface.

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
- Pairs where the body segment does not face the surface (`nᵢ · r ≥ 0`) are skipped
- Each candidate pair is tested for ray occlusion against the STL BVH; blocked pairs contribute zero
- Rays are trimmed to the 3 %–97 % segment of the full body→patch distance to avoid self-intersection artefacts (startOffset = 3 %, tmax = 0.94 × len from the offset origin = 97 % of original length)
- Results stored as SoA sparse arrays (`std::vector<int> indices` + `std::vector<float> fij`) per segment — float32 values are sufficient precision for UTCI and halve cache size vs float64
- Cached to `UTCI/pos/<original_probe_index>.bin`; cache key is the stable file-order index, not the post-filter array position
- Cache format: version 11 (plain binary) or version 12 (gzip-compressed, requires ZLIB at build time); format auto-detected by magic bytes on load

**Sky view factors** (explicit geometry):

Sky patches (`Sf_skySurfaces.raw`) are treated as a separate geometry set. The same differential formula applies, but with `enforceRangeLimit = false` so sky patches can be arbitrarily far away:

```
FijSky = |nᵢ · r̂| × |nⱼ · r̂| × Aⱼ / (π r²)   (no distance cap)
FijsumSky = Σ FijSky
```

Ray occlusion is tested for sky rays with the same BVH but without the `R_MAG_MAX` cutoff.

### 3. Outgoing Surface Radiation

For each wall/vegetation patch, the outgoing LW flux is pre-computed in the OpenFOAM `vegetation` region by `calcWallRadOut`:

```
QrOut = σ T⁴ + qr (1 − ε) / ε        ε_surf = 0.9

  T    = surface temperature [K]
  qr   = net radiative flux at surface (DO/P1 radiation model) [W/m²]
  σ    = 5.67 × 10⁻⁸ W m⁻² K⁻⁴
```

`calcWallRadOut` also computes the outgoing SW (solar reflection) flux `qsOut` for each patch. `calculateqrsw` computes the direct solar volume field `qrsw` used later for the per-position solar addition.

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
qin_LW_surf = Σₘ QrOut[m] × Fij[m]
qin_SW_surf = Σₘ qsOut[m] × Fij[m]

qin_LW_sky  = σ Tsky⁴ × FijsumSky
qin_SW_sky  = Idif × FijsumSky

total_vf = Fijsum + FijsumSky

qin_LW = (qin_LW_surf + qin_LW_sky) / total_vf
qin_SW = (qin_SW_surf + qin_SW_sky) / total_vf
```

Normalising by `total_vf` ensures the weighted-average irradiance is in W/m² regardless of how many patches are visible. This matches the reference implementation.

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

The SW contributions are also categorised by surface orientation (used in debug output `TumrtAvg_terms/`):

| Category | Condition |
|----------|-----------|
| `qswGround` | nz < −0.7 and patch centre z ≤ 2.5 m |
| `qswElevatedDown` | nz < −0.7 and patch centre z > 2.5 m |
| `qswUpward` | nz > +0.7 |
| `qswVertical` | −0.7 ≤ nz ≤ +0.7 |

where nz is the z-component of the normalised surface patch area vector.

**Direct solar addition:**

```
T⁴_final = Tmrt_avg⁴  +  fp_solar × α_sw × qrsw / (σ ε_p)

fp_solar  = 0.308 cos( β (1 − β²/48402) π/180 )    [β = solar elevation in °]
qrsw      = direct solar irradiance reaching the pedestrian [W/m²]
```

`fp_solar` is the projected-area factor for a standing person (Fiala et al. 2012 / Bröde et al. 2012, UTCI standard). `β` is derived from the CFD `sunPosVector` field: `sin β = sunDir.z`, clamped to [0, 1].

`qrsw` is resolved in priority order:
1. **Probe data** — CFD `qrsw` values sampled at pedestrian positions by `calculateqrsw` (Stage 1)
2. **Dense surface VTK** — `qrsw_pedestrian.vtk` interpolated from surface mesh (fallback)
3. **Binary shadow ray** — if both above are absent, fires a ray in the sun direction; if unblocked, uses `Idn` (direct normal irradiance); if blocked, contributes zero

When `Idn = 0` or solar elevation ≤ 0° the solar term is skipped entirely.

### 7. UTCI

Two methods are available, selected at runtime via `--utci-method`:

#### 7a. Polynomial method (`--utci-method poly`, default)

A **165-term polynomial** in four inputs (Fiala/Bröde 2012, UTCI-A):

```
UTCI [°C]  =  Ta  +  Δ(Ta, va, D_Tmrt, Pa)

  Ta      = air temperature [°C]
  va      = wind speed at 10 m reference height [m/s]  (clamped to 0.5–17 m/s)
  D_Tmrt  = Tmrt − Ta  [°C]
  Pa      = vapour pressure [kPa]
```

Terms span Pa⁰ through Pa⁶ combined with Ta⁰⁻⁶, va⁰⁻⁶, and D⁰⁻⁶ cross-products.

#### 7b. Lookup-table method (`--utci-method lut`)

Uses the official tabulated offset file `utci_offset.Dat` (Bröde et al. 2012, supplemental to IJB UTCI special issue), which encodes `UTCI − Ta` as a function of four variables:

| Axis | Range | Values |
|------|-------|--------|
| Ta [°C] | −50 … +50 | 1 °C steps (101 levels) |
| Tr − Ta [°C] | −30 … +70 | 5 °C steps (21 levels) |
| va [m/s] | 0.5 … 30.3 | 10 non-uniform levels |
| RH [%] | 5 … 100 | ~50 non-uniform levels |

The file has ~104 600 rows and 6 columns (Ta, Tr−Ta, va, RH, pa, offset).

**Evaluation:** 4D linear interpolation over the 16 corners of the enclosing hypercube. Values outside the table range are clamped to the nearest boundary.

#### Shared inputs

**Wind speed conversion:** CFD probe values are at pedestrian height (~2 m). Conversion to the 10 m reference height required by both methods:

```
va_ref = v_CFD / 0.667

  0.667 ≈ u(2 m)/u(10 m)  for a log profile with z0 ≈ 0.1 m
```

**Vapour pressure** from CFD specific humidity `w` [kg/kg]:

```
pv   = P_ref × w / (ε_H₂O + w)          [Pa]   (urbanMicroclimateFoam convention)
psat = exp(77.345 + 0.0057 Ta_K − 7235/Ta_K) / Ta_K^8.2   [Pa]   (Alduchov–Eskridge)
RH   = clamp( pv / psat × 100,  0, 100 )   [%]

  P_ref   = 101325 Pa
  ε_H₂O  = 0.621945  (ratio of molar masses M_water/M_dryair)
```

For the **polynomial method**, `utciSolver` internally converts RH → Pa [kPa] using its own 8-term saturation formula (ISO 7933 / Hardy 1998):

```
es [hPa] = 0.01 × exp( 2.7150305 ln(Ta_K) − 2836.5744/Ta_K² − 6028.076559/Ta_K
                       + 19.54263612 − 0.02737830188 Ta_K + 1.6261698×10⁻⁵ Ta_K²
                       + 7.0229056×10⁻¹⁰ Ta_K³ − 1.8680009×10⁻¹³ Ta_K⁴ )
Pa [kPa] = es × RH / 100 / 10
```

For the **LUT method**, RH is used directly as a table axis; no Pa conversion is needed.

`qrsw` is stored as a 3-D vector field in OpenFOAM/VTK (solar irradiance direction × magnitude). The scalar irradiance used in the solar addition is its magnitude: `|qrsw|`.

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
│  auto mode (default):                                               │
│    postProcess cuttingPlane z=2 m  →  T_pedestrian.vtk             │
│    if std(z) > 0.5 m or result empty  →  terrain detected:         │
│      postProcess patch surface (street/ground)  →  T_pedestrian.vtk│
│      bin face centres onto dx×dy grid + 2 m offset                 │
│    else (flat domain):                                              │
│      matplotlib TriFinder containment  →  regular dx×dy grid       │
│      (building interiors excluded — air mesh only)                  │
│  → system/air/probe_locs   (N × "(x y z)")                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 1  (re-run whenever CFD fields change)
┌─────────────────────────────────────────────────────────────────────┐
│  OpenFOAM radiation preprocessing                        run_utci.py│
│                                                                     │
│  Step 1: calculateqrsw  →  qrsw volume field in vegetation/<t>/    │
│  Step 2: calcSf         →  face area vectors (first timestep only) │
│  Step 3: calcWallRadOut →  qrOut, qsOut fields in vegetation/<t>/  │
│  Step 4: postProcess surfaces                                       │
│            →  postProcessing/surfaces/<t>/                          │
│                 Sf_wallAndTreeSurfaces.raw                          │
│                 qrOut_wallAndTreeSurfaces.raw                       │
│                 qsOut_wallAndTreeSurfaces.raw                       │
│                 Sf_skySurfaces.raw                                  │
│                 qrOut_skySurfaces.raw                               │
│  Step 5: postProcess surfacesPedestrianAir (T, U, w on ped mesh)   │
│            →  postProcessing/surfacesPedestrian/<t>/                │
│                 T_pedestrian.vtk  U_pedestrian.vtk  w_pedestrian.vtk│
│  Step 6: postProcess surfacesPedestrianRad (qrsw on ped mesh)      │
│            →  postProcessing/surfaces/<t>/qrsw_pedestrian.vtk      │
│  Step 7: postProcess probes (T, U, w at probe positions)           │
│            →  postProcessing/probes/air/<t0>/{T, U, w}             │
│  Step 8: postProcess qrswCuttingPlane + probe sampling             │
│    (vegetation mode only)                                           │
│            →  postProcessing/probes/qrsw/qrsw  (per-probe qrsw)   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ▼  Stage 2  (C++ binary, re-run per new meteorology)
┌─────────────────────────────────────────────────────────────────────┐
│  umcfUTCIpostprocess  -j 20  --start 3600 --end 86400               │
│                                                                     │
│  Load once:                                                         │
│    allGeo     ← Sf_wallAndTreeSurfaces.raw  (~329 k patch faces)    │
│    skyGeo     ← Sf_skySurfaces.raw          (5 boundary patches)    │
│    STL BVH    ← wallAndTreeSurfaces.stl      (ray occlusion)        │
│    meteo[t]   ← Tambient, cc, Idif, Idn, sunDir, va                 │
│    probeT/U/w/qrsw ← per-position rows; probe points matched to     │
│                 pedestrian positions by xyz coordinate hash (mm      │
│                 precision); falls back to original probe file index  │
│    (files parsed with bulk strtod reader: full file into buffer,     │
│     walked with strtod — no stream or istringstream overhead)        │
│                                                                     │
│  Batch loop (500 pos / batch, OpenMP):                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ For each position p:                                          │  │
│  │   Load UTCI/pos/<originalIndex>.bin  OR  compute + cache:    │  │
│  │     Fij[5][sparse]   ←  geometric formula + ray test         │  │
│  │     FijsumSky[5]     ←  explicit sky geometry, no range cap  │  │
│  │     Fijsum[5] = Σ Fij                                        │  │
│  │                                                              │  │
│  │ For each timestep t:                                         │  │
│  │   SurfaceRadiativeData built once per timestep (hoisted      │  │
│  │     outside inner position loop): qrOut[m], qsOut[m],        │  │
│  │     swClass[m] from .raw files for this t                    │  │
│  │   computeFast() hot path per position (skips breakdown        │  │
│  │     struct; thread cap removed for cached runs):              │  │
│  │   qin_LW, qin_SW  per segment                               │  │
│  │     = (surf contribution + sky contribution) / (Fijsum+FijsumSky)│ │
│  │   Tmrt[5]  ←  (ε_p qin_LW + α_sw qin_SW) / (σ ε_p) ^0.25  │  │
│  │   Tmrt_avg (pre-solar)  ←  area-weighted mean               │  │
│  │   qrsw  ←  probe data  OR  dense VTK  OR  shadow ray → Idn  │  │
│  │   if qrsw > 0:  add fp_solar × qrsw to Tmrt⁴               │  │
│  │   va_ref  ←  v_CFD / 0.667  (2 m → 10 m log profile)       │  │
│  │   pv  ←  P_ref × w / (ε_H₂O + w)                          │  │
│  │   UTCI  ←  165-term polynomial(Ta, va_ref, Tmrt_avg, Pa)    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Write per timestep t:                                              │
│    UTCI/<t>/Tmrt_pedestrian.vtk    (point cloud, Tmrt [K])         │
│    UTCI/<t>/RH_pedestrian.vtk      (point cloud, RH [%])           │
│    UTCI/<t>/UTCI.vtk               (point cloud, Tmrt[°C]+UTCI[°C])│
│    UTCI/<t>/Tumrt_surface.vtk      (dense mesh, pre-solar Tmrt)    │
│    (dense output uses DenseInterpPlan: Catmull-Rom stencil +        │
│     bilinear weights built once and reused across all timesteps)    │
│    UTCI/<t>/Tmrt_surface.vtk       (dense mesh, final Tmrt)        │
│    UTCI/<t>/RH_surface.vtk         (dense mesh, RH [%])            │
│    UTCI/<t>/UTCI_surface.vtk       (dense mesh, UTCI+Tmrt [°C])    │
│  Dense surface files require T_pedestrian.vtk from Stage 1;         │
│  if absent they are silently skipped.                               │
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

## Key Constants (`src/constants.h`)

| Symbol | Value | Meaning |
|--------|-------|---------|
| σ | 5.67 × 10⁻⁸ W m⁻² K⁻⁴ | Stefan-Boltzmann |
| ε_p | 0.97 | Person LW emissivity |
| α_sw | 0.70 | Person SW absorptivity |
| ε_surf | 0.90 | Wall/veg emissivity |
| R_MAG_MAX | 100 m | Max view-factor range |
| PED_Z | 2.0 m | Pedestrian height above ground |
| P_ref | 101325 Pa | Reference atmospheric pressure |
| ε_H₂O | 0.621945 | Molar mass ratio M_water/M_dryair |

## Key Source Files

| File | Role |
|------|------|
| `run_utci.py` | Orchestration — 4-stage pipeline |
| `src/umcfUTCIpostprocess.cpp` | Main: batch loop, CLI, I/O wiring |
| `src/tmrtSolver.cpp` | Tmrt physics (LW+SW balance, sky temp) |
| `src/viewFactor.cpp` | View factor geometry + ray occlusion |
| `src/utciSolver.cpp` | UTCI 165-term polynomial + LUT |
| `src/pedestrian.cpp` | 5-segment body model |
| `src/raycaster.cpp` | STL BVH ray intersection |
| `src/denseStage2.cpp` | Dense surface interpolation and output |
| `src/caching.cpp` | Binary VF cache (v11 plain / v12 gzip) |
| `src/io.cpp` | Raw/probe file readers, VTK writers |
| `src/constants.h` | All physical constants |
| `openfoam/calculateqrsw/` | OF utility — direct solar volume field (qrsw) |
| `openfoam/calcSf/` | OF utility — surface area vectors |
| `openfoam/calcWallRadOut/` | OF utility — outgoing LW and SW at patches |
