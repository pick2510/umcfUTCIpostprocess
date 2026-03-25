#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sys/resource.h>
#include <cerrno>
#include <sys/stat.h>
#include <atomic>
#include <cstring>
#include <iomanip>

#include "constants.H"
#include "types.H"
#include "pedestrian.H"
#include "raycaster.H"
#include "viewFactor.H"
#include "tmrtSolver.H"
#include "utciSolver.H"
#include "io.H"
#include "caching.H"
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace utci;

struct CommandLineArgs {
    std::string casePath;
    WorkflowMode mode = WorkflowMode::Flat;
    int tStart = 3600;
    int tEnd = 86400;
    int tStep = 3600;
    std::string outputDir = "UTCI";
    int nThreads = 1;
    bool computeUtci = true;
    bool forceRecompute = false;
    bool useSkyViewFactors = true;
    bool hasVegetation = true;
    // Optional spatial filter (radius <= 0 means no filter)
    double filterCenterX = -1.0;
    double filterCenterY = -1.0;
    double filterRadius   = -1.0;
    // Limit total positions (≤0 = no limit; useful for quick tests)
    int maxPositions = -1;
    // Batch size for view factor processing (limits peak memory)
    int batchSize = 500;
};

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --case <path>          Case directory (required)\n"
              << "  --mode flat|terrain    Workflow mode (default: flat)\n"
              << "  --start <time>         Start timestep (default: 3600)\n"
              << "  --end <time>           End timestep (default: 86400)\n"
              << "  --step <time>          Timestep interval (default: 3600)\n"
              << "  --output-dir <dir>     Output directory (default: UTCI)\n"
              << "  --skip-utci            Skip UTCI calculation\n"
              << "  --force-recompute      Force recompute view factors\n"
              << "  --filter-radius <r>    Filter positions within radius r of center\n"
              << "  --filter-cx <x>        Filter center X (used with --filter-radius)\n"
              << "  --filter-cy <y>        Filter center Y (used with --filter-radius)\n"
              << "  --max-positions <N>    Cap number of positions (for testing)\n"
              << "  --batch-size <N>       VF batch size (default 500, lower = less memory)\n"
              << "  -j <N>                 Number of threads (default: 1)\n"
              << "  --help                 Show this message\n";
}

CommandLineArgs parseArgs(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--case" && i + 1 < argc) {
            args.casePath = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "terrain") {
                args.mode = WorkflowMode::Terrain;
            } else {
                args.mode = WorkflowMode::Flat;
            }
        } else if (arg == "--start" && i + 1 < argc) {
            args.tStart = std::stoi(argv[++i]);
        } else if (arg == "--end" && i + 1 < argc) {
            args.tEnd = std::stoi(argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            args.tStep = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.outputDir = argv[++i];
        } else if (arg == "--skip-utci") {
            args.computeUtci = false;
        } else if (arg == "--force-recompute") {
            args.forceRecompute = true;
        } else if (arg == "--max-positions" && i + 1 < argc) {
            args.maxPositions = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batchSize = std::stoi(argv[++i]);
        } else if (arg == "--filter-radius" && i + 1 < argc) {
            args.filterRadius = std::stod(argv[++i]);
        } else if (arg == "--filter-cx" && i + 1 < argc) {
            args.filterCenterX = std::stod(argv[++i]);
        } else if (arg == "--filter-cy" && i + 1 < argc) {
            args.filterCenterY = std::stod(argv[++i]);
        } else if (arg == "-j" && i + 1 < argc) {
            args.nThreads = std::stoi(argv[++i]);
        } else if (arg.size() > 2 && arg[0] == '-' && arg[1] == 'j') {
            args.nThreads = std::stoi(arg.substr(2));
        } else if (arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        }
    }

    return args;
}

std::vector<int> getTimesteps(int start, int end, int step) {
    std::vector<int> timesteps;
    for (int t = start; t <= end; t += step) {
        timesteps.push_back(t);
    }
    return timesteps;
}

// Print progress line with ETA
static void printProgress(const char* tag, size_t done, size_t total,
                          std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    double elapsed = duration<double>(steady_clock::now() - t0).count();
    double pct = 100.0 * done / total;
    double eta  = (done > 0) ? elapsed / done * (total - done) : 0.0;
    std::cout << "  " << tag << " " << done << "/" << total
              << " (" << std::fixed << std::setprecision(1) << pct << "%)"
              << "  elapsed=" << (int)elapsed << "s"
              << "  ETA=" << (int)eta << "s\n" << std::flush;
}

// Concatenate two SurfacePatch vectors
static std::vector<SurfacePatch> concat(const std::vector<SurfacePatch>& a,
                                        const std::vector<SurfacePatch>& b) {
    std::vector<SurfacePatch> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

int main(int argc, char* argv[]) {
    auto startTime = std::chrono::high_resolution_clock::now();

    CommandLineArgs args = parseArgs(argc, argv);

    if (args.casePath.empty()) {
        std::cerr << "Error: --case is required\n";
        printUsage(argv[0]);
        return 1;
    }

#ifdef _OPENMP
    if (args.nThreads > 0) {
        omp_set_num_threads(args.nThreads);
    }
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";
#else
    std::cout << "OpenMP not enabled, using single thread\n";
#endif

    std::cout << "=== umcfUTCIpostprocess: C++ Tmrt/UTCI Calculator ===\n";
    std::cout << "Case: " << args.casePath << "\n";
    std::cout << "Mode: " << (args.mode == WorkflowMode::Flat ? "flat" : "terrain") << "\n";

    std::vector<int> timesteps = getTimesteps(args.tStart, args.tEnd, args.tStep);
    std::cout << "Timesteps: " << timesteps.size() << " (";
    for (size_t i = 0; i < std::min(timesteps.size(), size_t(3)); ++i) {
        std::cout << timesteps[i];
        if (i + 1 < std::min(timesteps.size(), size_t(3))) std::cout << ", ";
    }
    if (timesteps.size() > 3) std::cout << ", ...";
    std::cout << ")\n";

    // =========================================================
    // Load pedestrian positions
    // =========================================================
    std::string probeLocsPath = args.casePath + "/system/air/probe_locs";
    std::vector<PedestrianPosition> positions = loadPedestrianPositions(probeLocsPath);

    if (positions.empty()) {
        std::cerr << "Error: No pedestrian positions loaded\n";
        return 1;
    }

    std::cout << "Loaded " << positions.size() << " positions\n";

    // Optional spatial filter
    if (args.filterRadius > 0.0) {
        double cx = args.filterCenterX;
        double cy = args.filterCenterY;
        double r2 = args.filterRadius * args.filterRadius;
        std::cout << "Filtering positions within radius " << args.filterRadius
                  << " of (" << cx << ", " << cy << ")...\n";
        std::vector<PedestrianPosition> filtered;
        for (const auto& pos : positions) {
            double dx = pos.center.x - cx;
            double dy = pos.center.y - cy;
            if (dx*dx + dy*dy <= r2) {
                filtered.push_back(pos);
            }
        }
        if (filtered.empty()) {
            std::cerr << "Warning: filter removed all positions – using all\n";
        } else {
            positions = filtered;
            std::cout << "After filter: " << positions.size() << " positions\n";
        }
    }

    if (args.maxPositions > 0 && (int)positions.size() > args.maxPositions) {
        positions.resize(args.maxPositions);
        std::cout << "Capped to " << positions.size() << " positions (--max-positions)\n";
    }

    // =========================================================
    // Load STL geometry for ray occlusion
    // =========================================================
    std::string wallTreeStl = args.casePath + "/constant/triSurface/wallAndTreeSurfaces.stl";
    Raycaster raycaster;
    raycaster.setNumThreads(args.nThreads);
    std::cout << "Loading STL geometry...\n";
    if (!raycaster.loadGeometry(wallTreeStl)) {
        std::cerr << "Error: could not load " << wallTreeStl << "\n";
        return 1;
    }

    // =========================================================
    // Load surface geometry ONCE (from first timestep)
    // =========================================================
    std::string firstSurfDir = args.casePath + "/postProcessing/surfaces/"
                               + std::to_string(timesteps[0]);

    std::cout << "Loading surface geometry from timestep " << timesteps[0] << "...\n";

    // Try discrete-workflow names first, fall back to utci_clement combined name
    auto wallGeo = loadSurfacePatches(firstSurfDir + "/Sf_wallSurfaces.raw");
    if (wallGeo.empty())
        wallGeo = loadSurfacePatches(firstSurfDir + "/Sf_wallAndTreeSurfaces.raw");
    auto vegGeo  = loadSurfacePatches(firstSurfDir + "/Sf_vegSurfaces.raw");
    auto allGeo  = concat(wallGeo, vegGeo);

    std::cout << "  Wall surfaces: " << wallGeo.size() << "\n";
    std::cout << "  Veg surfaces:  " << vegGeo.size()  << "\n";
    std::cout << "  Total:         " << allGeo.size()  << "\n";

    std::vector<SurfacePatch> skyGeo;
    if (args.useSkyViewFactors) {
        skyGeo = loadSurfacePatches(firstSurfDir + "/Sf_skySurfaces.raw");
        std::cout << "  Sky surfaces:  " << skyGeo.size() << "\n";
    }

    // =========================================================
    // Load meteorological data for all timesteps
    // =========================================================
    std::cout << "\nLoading meteorological data...\n";
    std::vector<MeteoData> meteoData = loadMeteoData(args.casePath, timesteps);
    for (size_t i = 0; i < std::min(meteoData.size(), size_t(3)); ++i) {
        std::cout << "  t=" << timesteps[i]
                  << "  Ta=" << meteoData[i].Ta
                  << "  cc=" << meteoData[i].cc
                  << "  Idif=" << meteoData[i].Idif << "\n";
    }

    // =========================================================
    // Preload all per-timestep scalar fields upfront
    // (wall/veg temperatures + veg qr for each timestep)
    // This avoids repeated disk I/O inside the batch loop.
    // Memory: ~(3 × nSurfaces × nTimesteps × 8 bytes) ≈ small
    // =========================================================
    std::cout << "\nPreloading surface scalars for all timesteps...\n";
    struct TimestepScalars {
        std::vector<double> wallTemps;
        std::vector<double> vegTemps;
        std::vector<double> vegQr;
        std::vector<double> allQrOut;   // Outgoing LW σT⁴+qr*(1-ε)/ε (utci_clement format)
        std::vector<double> allQsOut;   // Outgoing SW
    };
    std::vector<TimestepScalars> tsData(timesteps.size());
    for (size_t tIdx = 0; tIdx < timesteps.size(); ++tIdx) {
        int t = timesteps[tIdx];
        std::string surfDir = args.casePath + "/postProcessing/surfaces/" + std::to_string(t);
        tsData[tIdx].wallTemps = loadScalarField(surfDir + "/T_wallSurfaces.raw");
        tsData[tIdx].vegTemps  = loadScalarField(surfDir + "/T_vegSurfaces.raw");
        tsData[tIdx].vegQr     = loadScalarField(surfDir + "/qr_vegSurfaces.raw");
        // utci_clement: combined outgoing LW radiation (replaces T+qr)
        tsData[tIdx].allQrOut  = loadScalarField(surfDir + "/qrOut_wallAndTreeSurfaces.raw");
        tsData[tIdx].allQsOut  = loadScalarField(surfDir + "/qsOut_wallAndTreeSurfaces.raw");
        if (tsData[tIdx].wallTemps.empty() && tsData[tIdx].allQrOut.empty()) {
            std::cerr << "  Warning: no wall temps or qrOut for t=" << t << "\n";
        }
    }
    std::cout << "Scalars loaded.\n";

    // =========================================================
    // Preload per-position CFD data from probe files (all timestep rows)
    // One row per radiation timestep; we pick the closest row per timestep.
    // =========================================================
    std::cout << "\nLoading per-position probe data...\n";
    std::string probeDir = findProbeDir(args.casePath, "air");
    if (probeDir.empty()) {
        probeDir = args.casePath + "/postProcessing/probes/air/3600";
        std::cout << "  Warning: probe dir not found via scan, falling back to " << probeDir << "\n";
    } else {
        std::cout << "  Probe dir: " << probeDir << "\n";
    }
    auto probeTAll = loadProbeScalarAll(probeDir + "/T");
    auto probeWAll = loadProbeScalarAll(probeDir + "/w");
    auto probeUAll = loadProbeVelocityMagAll(probeDir + "/U");
    std::cout << "  Probe T rows: " << probeTAll.size()
              << "  U rows: " << probeUAll.size() << "\n";
    if (probeTAll.empty())
        std::cout << "  Warning: no probe T – using ambient temperature for UTCI\n";
    if (probeUAll.empty())
        std::cout << "  Warning: no probe U – using reference wind speed for UTCI\n";
    static const std::vector<double> emptyVec;

    // =========================================================
    // Create output directories once (mkdir -p style)
    // =========================================================
    {
        std::string baseDir = args.casePath + "/" + args.outputDir;
        createDirectory(baseDir);
        for (int t : timesteps) {
            std::string outDir = baseDir + "/" + std::to_string(t);
            if (!createDirectory(outDir)) {
                std::cerr << "Warning: could not create " << outDir
                          << ": " << std::strerror(errno) << "\n";
            }
        }
    }

    // =========================================================
    // Batch processing: VF → Tmrt → write, one batch at a time
    // Each batch uses (batchSize × ~20 MB) for ViewFactorResults
    // =========================================================
    ViewFactorCalculator vfCalc(raycaster);
    BinaryCache cache;
    std::string cacheBaseDir = args.casePath + "/" + args.outputDir;
    cache.setBaseDir(cacheBaseDir);
    TmrtSolver tmrtSolver;
    UtciSolver utciSolver;

    size_t nPos      = positions.size();
    size_t batchSz   = static_cast<size_t>(std::max(1, args.batchSize));
    size_t nBatches  = (nPos + batchSz - 1) / batchSz;

    double vfMBperPos = (5.0 * (allGeo.size() + skyGeo.size()) * 8.0) / (1024.0 * 1024.0);
    std::cout << "\n=== Batch processing ===\n"
              << "Positions: " << nPos << "  Batches: " << nBatches
              << "  Batch size: " << batchSz << "\n"
              << "Est. VF memory per batch: "
              << std::fixed << std::setprecision(0) << (batchSz * vfMBperPos) << " MB\n\n";

    // Accumulators: allTmrt/allUtci/allRH[tIdx][pedIdx]
    size_t nT = timesteps.size();
    std::vector<Eigen::VectorXd> allTmrt(nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allUtci(nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allRH  (nT, Eigen::VectorXd::Zero(nPos));

    auto totalT0 = std::chrono::steady_clock::now();

    for (size_t batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
        size_t bStart = batchIdx * batchSz;
        size_t bEnd   = std::min(bStart + batchSz, nPos);
        size_t bN     = bEnd - bStart;

        std::cout << "--- Batch " << (batchIdx + 1) << "/" << nBatches
                  << "  positions [" << bStart << ", " << bEnd << ")\n";

        // --- Compute / load view factors for this batch ---
        std::vector<ViewFactorResult> batchVF(bN);
        std::atomic<size_t> vfDone{0};
        size_t vfInterval = std::max(size_t(1), bN / 10);
        auto vfT0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (size_t bi = 0; bi < bN; ++bi) {
            size_t pedIdx = bStart + bi;
            std::string cachePath = cache.getCachePath(static_cast<int>(pedIdx));
            bool loaded = false;
            if (!args.forceRecompute) {
                loaded = cache.load(cachePath, batchVF[bi]);
            }
            if (!loaded) {
                batchVF[bi] = vfCalc.compute(
                    positions[pedIdx], allGeo, skyGeo, args.useSkyViewFactors
                );
                cache.save(cachePath, batchVF[bi]);
            }
            size_t cur = ++vfDone;
            if (cur % vfInterval == 0 || cur == bN) {
                #pragma omp critical
                { printProgress("  VF", cur, bN, vfT0); }
            }
        }

        // Print Fijsum sample for first batch only
        if (batchIdx == 0) {
            std::cout << "  Sample Fijsum:\n";
            for (size_t i = 0; i < std::min(bN, size_t(2)); ++i) {
                std::cout << "    pos[" << (bStart+i) << "]"
                          << " Fijsum=" << batchVF[i].Fijsum.transpose()
                          << " FijsumSky=" << batchVF[i].FijsumSky.transpose() << "\n";
            }
        }

        // --- Compute Tmrt for each timestep, append to output ---
        for (size_t tIdx = 0; tIdx < timesteps.size(); ++tIdx) {
            int t = timesteps[tIdx];
            const auto& sc    = tsData[tIdx];
            const MeteoData& meteo = meteoData[tIdx];

            // Build allSurfaces with this timestep's scalars
            std::vector<SurfacePatch> allSurfaces = allGeo;
            // utci_clement format: use qrOut directly (σT⁴+qr*(1-ε)/ε already combined)
            if (!sc.allQrOut.empty()) {
                for (size_t i = 0; i < allSurfaces.size() && i < sc.allQrOut.size(); ++i)
                    allSurfaces[i].qrOut = sc.allQrOut[i];
            } else {
                // discrete/legacy format: separate T and qr per region
                for (size_t i = 0; i < wallGeo.size() && i < sc.wallTemps.size(); ++i)
                    allSurfaces[i].temperature = sc.wallTemps[i];
                for (size_t i = 0; i < vegGeo.size(); ++i) {
                    size_t idx = wallGeo.size() + i;
                    if (i < sc.vegTemps.size()) allSurfaces[idx].temperature = sc.vegTemps[i];
                    if (i < sc.vegQr.size())    allSurfaces[idx].qr          = sc.vegQr[i];
                }
            }
            // SW outgoing (qsOut): covers all wall+tree surfaces in order
            for (size_t i = 0; i < allSurfaces.size() && i < sc.allQsOut.size(); ++i)
                allSurfaces[i].qsOut = sc.allQsOut[i];

            Eigen::VectorXd batchTmrt(bN);
            Eigen::VectorXd batchUtci(bN);
            Eigen::VectorXd batchRH  (bN);

            // Pick probe data row for this radiation timestep
            double tDouble = static_cast<double>(t);
            const std::vector<double>& probeT = probeTAll.empty() ? emptyVec
                : [&]() -> const std::vector<double>& {
                    const std::vector<double>* best = &probeTAll.front().second;
                    double bd = std::abs(probeTAll.front().first - tDouble);
                    for (auto& r : probeTAll) { double d=std::abs(r.first-tDouble); if(d<bd){bd=d;best=&r.second;} }
                    return *best;
                }();
            const std::vector<double>& probeW = probeWAll.empty() ? emptyVec
                : [&]() -> const std::vector<double>& {
                    const std::vector<double>* best = &probeWAll.front().second;
                    double bd = std::abs(probeWAll.front().first - tDouble);
                    for (auto& r : probeWAll) { double d=std::abs(r.first-tDouble); if(d<bd){bd=d;best=&r.second;} }
                    return *best;
                }();
            const std::vector<double>& probeU = probeUAll.empty() ? emptyVec
                : [&]() -> const std::vector<double>& {
                    const std::vector<double>* best = &probeUAll.front().second;
                    double bd = std::abs(probeUAll.front().first - tDouble);
                    for (auto& r : probeUAll) { double d=std::abs(r.first-tDouble); if(d<bd){bd=d;best=&r.second;} }
                    return *best;
                }();

            // Sun direction and IDN for this timestep
            const double sinBeta  = meteo.sunDir.z();
            const bool   hasSolar = (meteo.Idn > 0.0 && sinBeta > 0.0);
            double fp_solar = 0.0;
            if (hasSolar) {
                double betaDeg = std::asin(std::min(1.0, sinBeta)) * 180.0 / M_PI;
                fp_solar = 0.308 * std::cos(betaDeg * (1.0 - betaDeg*betaDeg/48402.0) * M_PI/180.0);
            }

            #pragma omp parallel for schedule(dynamic)
            for (size_t bi = 0; bi < bN; ++bi) {
                size_t pedIdx = bStart + bi;

                // --- Tmrt (LW + diffuse SW) ---
                Eigen::VectorXd TmrtBody = tmrtSolver.compute(
                    allSurfaces, skyGeo, batchVF[bi], meteo, args.useSkyViewFactors
                );
                double TumrtAvg = tmrtSolver.computeAreaWeightedAverage(
                    TmrtBody, positions[pedIdx].areaVectors
                );

                // --- Direct solar component (shadow test) ---
                // Ray must be < R_MAG_MAX so the raycaster doesn't skip it.
                if (hasSolar) {
                    Vec3 pedPos = positions[pedIdx].center.toVec3();
                    Vec3 target = pedPos + meteo.sunDir * (R_MAG_MAX * 0.9);
                    if (!raycaster.isBlocked(pedPos, target)) {
                        // fp_solar * Idn gives effective irradiance on person body
                        double T4 = std::pow(TumrtAvg, 4)
                                    + fp_solar * ABS_SW_PERSON * meteo.Idn
                                      / (EPS_LW_PERSON * SIGMA);
                        TumrtAvg = std::pow(std::max(0.0, T4), 0.25);
                    }
                }
                batchTmrt[bi] = TumrtAvg;

                // --- Per-position UTCI inputs ---
                double Ta_K = (pedIdx < probeT.size()) ? probeT[pedIdx] : meteo.Ta;
                if (Ta_K < 200.0) Ta_K = meteo.Ta;  // guard against bad probe values
                double Ta_c = Ta_K - 273.15;
                double va   = (pedIdx < probeU.size()) ? probeU[pedIdx] / 0.667 : meteo.va;
                va = std::max(0.5, std::min(17.0, va));  // UTCI polynomial valid range

                // RH from specific humidity (original Python formula)
                double w   = (pedIdx < probeW.size()) ? probeW[pedIdx] : 0.01;
                double psat = std::exp(77.345 + 0.0057*Ta_K - 7235.0/Ta_K)
                              / std::pow(Ta_K, 8.2);
                double pv   = w * 1e5 / 0.62;
                double RH   = std::min(100.0, std::max(0.0, pv / psat * 100.0));
                batchRH[bi] = RH;

                // --- UTCI ---
                if (args.computeUtci) {
                    double Tmrt_c = batchTmrt[bi] - 273.15;
                    batchUtci[bi] = utciSolver.calculate(Ta_c, va, RH, Tmrt_c);
                }
            }

            // Accumulate into global result vectors
            for (size_t bi = 0; bi < bN; ++bi) {
                allTmrt[tIdx][bStart + bi] = batchTmrt[bi];
                allUtci[tIdx][bStart + bi] = batchUtci[bi];
                allRH  [tIdx][bStart + bi] = batchRH[bi];
            }
        }

        // batchVF goes out of scope here → memory freed

        // Overall progress
        size_t posProcessed = bEnd;
        using namespace std::chrono;
        double elapsed = duration<double>(steady_clock::now() - totalT0).count();
        double eta = (posProcessed > 0) ? elapsed / posProcessed * (nPos - posProcessed) : 0;
        std::cout << "  Batch done. Total progress: " << posProcessed << "/" << nPos
                  << "  elapsed=" << (int)elapsed << "s  ETA=" << (int)eta << "s\n\n";
    }

    // Write VTK output and print stats per timestep
    std::cout << "=== Writing VTK output and per-timestep stats ===\n";
    for (size_t tIdx = 0; tIdx < nT; ++tIdx) {
        int t = timesteps[tIdx];
        std::string outDir = args.casePath + "/" + args.outputDir + "/" + std::to_string(t);

        // Tmrt_pedestrian.vtk  (Kelvin) — point cloud
        writeVtkPolyData(outDir + "/Tmrt_pedestrian.vtk", positions, allTmrt[tIdx], "Tmrt");

        // RH_pedestrian.vtk — point cloud
        writeVtkPolyData(outDir + "/RH_pedestrian.vtk", positions, allRH[tIdx], "RH");

        // UTCI.vtk  (Tmrt in °C + UTCI in °C) — point cloud
        // UTCI_surface.vtk   — interpolated structured grid surface
        if (args.computeUtci) {
            Eigen::VectorXd TmrtC = allTmrt[tIdx].array() - 273.15;
            writeVtkMultiScalar(outDir + "/UTCI.vtk", positions,
                { {"Tmrt", TmrtC}, {"UTCI", allUtci[tIdx]} });
            writeVtkStructuredSurface(outDir + "/UTCI_surface.vtk", positions,
                { {"Tmrt", TmrtC}, {"UTCI", allUtci[tIdx]} });
        }

        // Print stats
        double tMin  = allTmrt[tIdx].minCoeff();
        double tMax  = allTmrt[tIdx].maxCoeff();
        double tMean = allTmrt[tIdx].mean();
        std::cout << "  t=" << t
                  << "  Tmrt[K]: min=" << std::setprecision(1) << tMin
                  << " max=" << tMax << " mean=" << tMean;
        if (args.computeUtci) {
            double uMin  = allUtci[tIdx].minCoeff();
            double uMax  = allUtci[tIdx].maxCoeff();
            double uMean = allUtci[tIdx].mean();
            std::cout << "  UTCI[°C]: min=" << uMin << " max=" << uMax << " mean=" << uMean;
        }
        std::cout << "\n";
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

    std::cout << "\n=== Complete ===\n";
    std::cout << "Total time: " << duration.count() << " seconds\n";

    return 0;
}
