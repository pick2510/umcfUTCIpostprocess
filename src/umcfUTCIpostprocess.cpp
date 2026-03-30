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
#include <limits>
#include <optional>
#include <sstream>
#include <fstream>

#include "constants.h"
#include "types.h"
#include "pedestrian.h"
#include "raycaster.h"
#include "viewFactor.h"
#include "tmrtSolver.h"
#include "utciSolver.h"
#include "denseStage2.h"
#include "logging.h"
#include "io.h"
#include "caching.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace utci;

struct CommandLineArgs {
    std::string casePath;
    int tStart = 3600;
    int tEnd = 86400;
    int tStep = 3600;
    std::string outputDir = "UTCI";
    int nThreads = 1;
    bool computeUtci = true;
    bool forceRecompute = false;
    bool useSkyViewFactors = true;
    // Optional spatial filter (radius <= 0 means no filter)
    double filterCenterX = -1.0;
    double filterCenterY = -1.0;
    double filterRadius   = -1.0;
    // Limit total positions (≤0 = no limit; useful for quick tests)
    int maxPositions = -1;
    // Batch size for view factor processing (limits peak memory)
    int batchSize = 500;
    // UTCI calculation method
    UtciMethod utciMethod = UtciMethod::POLYNOMIAL;
    // LUT file path (empty = look for utci_offset.Dat next to binary)
    std::string lutPath;
    // Write compressed (gzip) cache files; auto-enabled when ZLIB is available
    bool compressCache = BinaryCache::compressionAvailable();
    // Optional debug outputs from the investigation phase
    bool writeDebugTerms = false;
    bool writeDebugQrswSurface = false;
};

struct TimestepScalars {
    std::vector<double> wallTemps;
    std::vector<double> vegTemps;
    std::vector<double> vegQr;
    std::vector<double> allQrOut;   // Outgoing LW σT⁴+qr*(1-ε)/ε (reference format)
    std::vector<double> allQsOut;   // Outgoing SW
    Eigen::VectorXd sparseQrswFromSurface;
};

struct SparseStage2Results {
    std::vector<Eigen::VectorXd> tmrt;
    std::vector<Eigen::VectorXd> tumrtAvg;
    std::vector<Eigen::VectorXd> utci;
    std::vector<Eigen::VectorXd> rh;
    std::vector<Eigen::VectorXd> tumrtNoSolar;
    std::vector<Eigen::VectorXd> qlwSurfaces;
    std::vector<Eigen::VectorXd> qlwSky;
    std::vector<Eigen::VectorXd> qswSurfaces;
    std::vector<Eigen::VectorXd> qswSky;
    std::vector<Eigen::VectorXd> qswDirect;
    std::vector<Eigen::VectorXd> qswGround;
    std::vector<Eigen::VectorXd> qswElevatedDown;
    std::vector<Eigen::VectorXd> qswVertical;
    std::vector<Eigen::VectorXd> qswUpward;
};

struct SparseStage2Context {
    const CommandLineArgs& args;
    const std::vector<int>& timesteps;
    const std::vector<PedestrianPosition>& positions;
    const std::vector<SurfacePatch>& wallGeo;
    const std::vector<SurfacePatch>& vegGeo;
    const std::vector<SurfacePatch>& allGeo;
    const std::vector<SurfacePatch>& skyGeo;
    const std::vector<MeteoData>& meteoData;
    const std::vector<TimestepScalars>& tsData;
    const std::vector<std::pair<double, std::vector<double>>>& probeTAll;
    const std::vector<std::pair<double, std::vector<double>>>& probeWAll;
    const std::vector<std::pair<double, std::vector<double>>>& probeUAll;
    const std::vector<std::pair<double, std::vector<double>>>& probeQrswAll;
    const std::vector<int>& probeIndexForPos;
    const std::vector<double>& emptyVec;
    Raycaster& raycaster;
    ViewFactorCalculator& vfCalc;
    BinaryCache& cache;
    TmrtSolver& tmrtSolver;
    UtciSolver& utciSolver;
};

void utci::logInfo(const std::string& message) {
    std::cout << message << "\n";
}

void utci::logWarn(const std::string& message) {
    std::cerr << "Warning: " << message << "\n";
}

void utci::logError(const std::string& message) {
    std::cerr << "Error: " << message << "\n";
}

void utci::logSection(const std::string& title) {
    logInfo("=== " + title + " ===");
}

void utci::logDetail(const std::string& message) {
    logInfo("  " + message);
}

void utci::logSummary(const std::string& message) {
    logInfo(message);
}

void utci::logProgress(const std::string& tag, size_t done, size_t total,
                       double elapsedSeconds, double etaSeconds) {
    std::ostringstream out;
    out << "  " << tag << " " << done << "/" << total
        << " (" << std::fixed << std::setprecision(1)
        << (total > 0 ? 100.0 * static_cast<double>(done) / static_cast<double>(total) : 0.0) << "%)"
        << "  elapsed=" << static_cast<int>(elapsedSeconds) << "s"
        << "  ETA=" << static_cast<int>(etaSeconds) << "s";
    logInfo(out.str());
}

void printUsage(const char* progName) {
    logInfo("Usage: " + std::string(progName) + " [options]");
    logInfo("Options:");
    logInfo("  --case <path>          Case directory (required)");
    logInfo("  --start <time>         Start timestep (default: 3600)");
    logInfo("  --end <time>           End timestep (default: 86400)");
    logInfo("  --step <time>          Timestep interval (default: 3600)");
    logInfo("  --output-dir <dir>     Output directory (default: UTCI)");
    logInfo("  --skip-utci            Skip UTCI calculation");
    logInfo("  --utci-method <m>      UTCI method: poly (default) or lut");
    logInfo("  --lut-path <file>      Path to utci_offset.Dat (default: next to binary)");
    logInfo("  --force-recompute      Force recompute view factors");
    logInfo("  --filter-radius <r>    Filter positions within radius r of center");
    logInfo("  --filter-cx <x>        Filter center X (used with --filter-radius)");
    logInfo("  --filter-cy <y>        Filter center Y (used with --filter-radius)");
    logInfo("  --max-positions <N>    Cap number of positions (for testing)");
    logInfo("  --batch-size <N>       VF batch size (default 500, lower = less memory)");
    logInfo("  --compress-cache       Write gzip-compressed cache files (default when ZLIB available)");
    logInfo("  --no-compress-cache    Write uncompressed cache files");
    logInfo("  --write-debug-terms    Write TumrtAvg_terms debug output");
    logInfo("  --write-debug-qrsw     Write qrsw_surface.vtk debug output");
    logInfo("  -j <N>                 Number of threads (default: 1)");
    logInfo("  --help                 Show this message");
}

CommandLineArgs parseArgs(int argc, char* argv[]) {
    CommandLineArgs args;
    auto parseIntArg = [&](const std::string& flag, const char* value) -> int {
        try {
            return std::stoi(value);
        } catch (const std::exception&) {
            logError("Invalid integer for " + flag + ": " + value);
            std::exit(1);
        }
    };
    auto parseDoubleArg = [&](const std::string& flag, const char* value) -> double {
        try {
            return std::stod(value);
        } catch (const std::exception&) {
            logError("Invalid number for " + flag + ": " + value);
            std::exit(1);
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--case" && i + 1 < argc) {
            args.casePath = argv[++i];
        } else if (arg == "--start" && i + 1 < argc) {
            args.tStart = parseIntArg(arg, argv[++i]);
        } else if (arg == "--end" && i + 1 < argc) {
            args.tEnd = parseIntArg(arg, argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            args.tStep = parseIntArg(arg, argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.outputDir = argv[++i];
        } else if (arg == "--skip-utci") {
            args.computeUtci = false;
        } else if (arg == "--utci-method" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "lut")  args.utciMethod = UtciMethod::LUT;
            else if (m == "poly") args.utciMethod = UtciMethod::POLYNOMIAL;
            else { logError("Unknown --utci-method: " + m + " (use poly or lut)"); exit(1); }
        } else if (arg == "--lut-path" && i + 1 < argc) {
            args.lutPath = argv[++i];
        } else if (arg == "--force-recompute") {
            args.forceRecompute = true;
        } else if (arg == "--max-positions" && i + 1 < argc) {
            args.maxPositions = parseIntArg(arg, argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batchSize = parseIntArg(arg, argv[++i]);
        } else if (arg == "--filter-radius" && i + 1 < argc) {
            args.filterRadius = parseDoubleArg(arg, argv[++i]);
        } else if (arg == "--filter-cx" && i + 1 < argc) {
            args.filterCenterX = parseDoubleArg(arg, argv[++i]);
        } else if (arg == "--filter-cy" && i + 1 < argc) {
            args.filterCenterY = parseDoubleArg(arg, argv[++i]);
        } else if (arg == "-j" && i + 1 < argc) {
            args.nThreads = parseIntArg(arg, argv[++i]);
        } else if (arg.size() > 2 && arg[0] == '-' && arg[1] == 'j') {
            args.nThreads = parseIntArg("-j", arg.substr(2).c_str());
        } else if (arg == "--compress-cache") {
            args.compressCache = true;
        } else if (arg == "--no-compress-cache") {
            args.compressCache = false;
        } else if (arg == "--write-debug-terms") {
            args.writeDebugTerms = true;
        } else if (arg == "--write-debug-qrsw") {
            args.writeDebugQrswSurface = true;
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
    logProgress(tag, done, total, elapsed, eta);
}

// Concatenate two SurfacePatch vectors
static std::vector<SurfacePatch> concat(const std::vector<SurfacePatch>& a,
                                        const std::vector<SurfacePatch>& b) {
    std::vector<SurfacePatch> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

static double areaWeightedAverage(const Eigen::VectorXd& values,
                                  const std::array<Vec3, 5>& areaVectors) {
    double sum = 0.0;
    double sumArea = 0.0;
    for (int i = 0; i < values.size() && i < static_cast<int>(areaVectors.size()); ++i) {
        const double area = areaVectors[i].norm();
        sum += values[i] * area;
        sumArea += area;
    }
    return sumArea > 0.0 ? sum / sumArea : 0.0;
}

template <typename T>
static const std::vector<T>& selectNearestProbeRow(
        const std::vector<std::pair<double, std::vector<T>>>& rows,
        double timestep,
        const std::vector<T>& fallback) {
    if (rows.empty()) return fallback;
    const std::vector<T>* best = &rows.front().second;
    double bestDt = std::abs(rows.front().first - timestep);
    for (const auto& row : rows) {
        double dt = std::abs(row.first - timestep);
        if (dt < bestDt) {
            bestDt = dt;
            best = &row.second;
        }
    }
    return *best;
}

static bool validateRequiredInputs(const CommandLineArgs& args,
                                   const std::vector<SurfacePatch>& allGeo,
                                   const std::vector<SurfacePatch>& skyGeo,
                                   const std::vector<MeteoData>& meteoData,
                                   const std::vector<int>& timesteps,
                                   const std::vector<std::pair<double, std::vector<double>>>& probeTAll,
                                   const std::vector<std::pair<double, std::vector<double>>>& probeUAll,
                                   const std::vector<std::pair<double, std::vector<double>>>& probeWAll,
                                   const std::vector<TimestepScalars>& tsData) {
    bool ok = true;
    if (allGeo.empty()) {
        logError("no wall/tree surface geometry loaded from postProcessing/surfaces");
        ok = false;
    }
    if (args.useSkyViewFactors && skyGeo.empty()) {
        logError("sky view factors requested but no sky surface geometry was loaded");
        ok = false;
    }
    if (meteoData.empty()) {
        logError("no meteorological data loaded");
        ok = false;
    }
    if (probeTAll.empty()) {
        logError("no probe T data loaded");
        ok = false;
    }
    if (probeUAll.empty()) {
        logError("no probe U data loaded");
        ok = false;
    }
    if (probeWAll.empty()) {
        logError("no probe w data loaded");
        ok = false;
    }
    for (size_t i = 0; i < tsData.size(); ++i) {
        const auto& sc = tsData[i];
        if (sc.allQrOut.empty() && sc.wallTemps.empty()) {
            logError("timestep " + std::to_string(timesteps[i]) +
                     " is missing wall LW inputs (qrOut or wall temperatures)");
            ok = false;
            break;
        }
        if (sc.allQsOut.empty()) {
            logError("timestep " + std::to_string(timesteps[i]) +
                     " is missing qsOut_wallAndTreeSurfaces.raw");
            ok = false;
            break;
        }
    }
    return ok;
}

static SparseStage2Results runSparseStage2(const SparseStage2Context& ctx) {
    const size_t nPos = ctx.positions.size();
    const size_t nT = ctx.timesteps.size();
    const size_t batchSz = static_cast<size_t>(std::max(1, ctx.args.batchSize));
    const size_t nBatches = (nPos + batchSz - 1) / batchSz;

    SparseStage2Results results;
    results.tmrt.assign(nT, Eigen::VectorXd::Zero(nPos));
    results.tumrtAvg.assign(nT, Eigen::VectorXd::Zero(nPos));
    results.utci.assign(nT, Eigen::VectorXd::Zero(nPos));
    results.rh.assign(nT, Eigen::VectorXd::Zero(nPos));
    results.tumrtNoSolar.assign(nT, Eigen::VectorXd::Zero(nPos));
    if (ctx.args.writeDebugTerms) {
        results.qlwSurfaces.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qlwSky.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswSurfaces.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswSky.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswDirect.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswGround.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswElevatedDown.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswVertical.assign(nT, Eigen::VectorXd::Zero(nPos));
        results.qswUpward.assign(nT, Eigen::VectorXd::Zero(nPos));
    }

    double vfMBperPos = (5.0 * (ctx.allGeo.size() + ctx.skyGeo.size()) * 8.0) / (1024.0 * 1024.0);
    logSection("Batch processing");
    {
        std::ostringstream out;
        out << "Positions: " << nPos << "  Batches: " << nBatches
            << "  Batch size: " << batchSz;
        logSummary(out.str());
    }
    {
        std::ostringstream out;
        out << "Est. VF memory per batch: "
            << std::fixed << std::setprecision(0) << (batchSz * vfMBperPos) << " MB";
        logSummary(out.str());
    }

    auto totalT0 = std::chrono::steady_clock::now();

    for (size_t batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
        size_t bStart = batchIdx * batchSz;
        size_t bEnd   = std::min(bStart + batchSz, nPos);
        size_t bN     = bEnd - bStart;

        {
            std::ostringstream out;
            out << "--- Batch " << (batchIdx + 1) << "/" << nBatches
                << "  positions [" << bStart << ", " << bEnd << ")";
            logSummary(out.str());
        }

        std::vector<ViewFactorResult> batchVF(bN);
        std::atomic<size_t> vfDone{0};
        size_t vfInterval = std::max(size_t(1), bN / 10);
        auto vfT0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (size_t bi = 0; bi < bN; ++bi) {
            size_t pedIdx = bStart + bi;
            std::string cachePath = ctx.cache.getCachePath(ctx.positions[pedIdx].originalIndex);
            bool loaded = false;
            if (!ctx.args.forceRecompute) {
                loaded = ctx.cache.load(cachePath, batchVF[bi]);
            }
            if (!loaded) {
                batchVF[bi] = ctx.vfCalc.compute(
                    ctx.positions[pedIdx], ctx.allGeo, ctx.skyGeo, ctx.args.useSkyViewFactors
                );
                ctx.cache.save(cachePath, batchVF[bi]);
            }
            size_t cur = ++vfDone;
            if (cur % vfInterval == 0 || cur == bN) {
                #pragma omp critical
                { printProgress("  VF", cur, bN, vfT0); }
            }
        }

        if (batchIdx == 0) {
            logDetail("Sample Fijsum:");
            for (size_t i = 0; i < std::min(bN, size_t(2)); ++i) {
                std::ostringstream out;
                out << "  pos[" << (bStart + i) << "]"
                    << " Fijsum=" << batchVF[i].Fijsum.transpose()
                    << " FijsumSky=" << batchVF[i].FijsumSky.transpose();
                logDetail(out.str());
            }
        }

        for (size_t tIdx = 0; tIdx < ctx.timesteps.size(); ++tIdx) {
            int t = ctx.timesteps[tIdx];
            const auto& sc = ctx.tsData[tIdx];
            const MeteoData& meteo = ctx.meteoData[tIdx];

            std::vector<SurfacePatch> allSurfaces = ctx.allGeo;
            if (!sc.allQrOut.empty()) {
                for (size_t i = 0; i < allSurfaces.size() && i < sc.allQrOut.size(); ++i) {
                    allSurfaces[i].qrOut = sc.allQrOut[i];
                }
            } else {
                for (size_t i = 0; i < ctx.wallGeo.size() && i < sc.wallTemps.size(); ++i) {
                    allSurfaces[i].temperature = sc.wallTemps[i];
                }
                for (size_t i = 0; i < ctx.vegGeo.size(); ++i) {
                    size_t idx = ctx.wallGeo.size() + i;
                    if (i < sc.vegTemps.size()) allSurfaces[idx].temperature = sc.vegTemps[i];
                    if (i < sc.vegQr.size()) allSurfaces[idx].qr = sc.vegQr[i];
                }
            }
            for (size_t i = 0; i < allSurfaces.size() && i < sc.allQsOut.size(); ++i) {
                allSurfaces[i].qsOut = sc.allQsOut[i];
            }

            Eigen::VectorXd batchTmrt = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchUtci = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchRH = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchTumrtNoSolar = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchQlwSurfaces;
            Eigen::VectorXd batchQlwSky;
            Eigen::VectorXd batchQswSurfaces;
            Eigen::VectorXd batchQswSky;
            Eigen::VectorXd batchQswDirect;
            Eigen::VectorXd batchQswGround;
            Eigen::VectorXd batchQswElevatedDown;
            Eigen::VectorXd batchQswVertical;
            Eigen::VectorXd batchQswUpward;
            if (ctx.args.writeDebugTerms) {
                batchQlwSurfaces = Eigen::VectorXd::Zero(bN);
                batchQlwSky = Eigen::VectorXd::Zero(bN);
                batchQswSurfaces = Eigen::VectorXd::Zero(bN);
                batchQswSky = Eigen::VectorXd::Zero(bN);
                batchQswDirect = Eigen::VectorXd::Zero(bN);
                batchQswGround = Eigen::VectorXd::Zero(bN);
                batchQswElevatedDown = Eigen::VectorXd::Zero(bN);
                batchQswVertical = Eigen::VectorXd::Zero(bN);
                batchQswUpward = Eigen::VectorXd::Zero(bN);
            }

            double tDouble = static_cast<double>(t);
            const std::vector<double>& probeT = selectNearestProbeRow(ctx.probeTAll, tDouble, ctx.emptyVec);
            const std::vector<double>& probeW = selectNearestProbeRow(ctx.probeWAll, tDouble, ctx.emptyVec);
            const std::vector<double>& probeU = selectNearestProbeRow(ctx.probeUAll, tDouble, ctx.emptyVec);
            const std::vector<double>& probeQrsw = selectNearestProbeRow(ctx.probeQrswAll, tDouble, ctx.emptyVec);

            const double sinBeta = meteo.sunDir.z();
            const bool hasSolar = (meteo.Idn > 0.0 && sinBeta > 0.0);
            double fp_solar = 0.0;
            if (hasSolar) {
                double betaDeg = std::asin(std::min(1.0, sinBeta)) * 180.0 / M_PI;
                fp_solar = 0.308 * std::cos(betaDeg * (1.0 - betaDeg * betaDeg / 48402.0) * M_PI / 180.0);
            }

            #pragma omp parallel for schedule(dynamic)
            for (size_t bi = 0; bi < bN; ++bi) {
                size_t pedIdx = bStart + bi;
                TmrtBreakdown tmrtDetail = ctx.tmrtSolver.computeDetailed(
                    allSurfaces, ctx.skyGeo, batchVF[bi], meteo, ctx.args.useSkyViewFactors
                );
                double TumrtNoSolar = ctx.tmrtSolver.computeAreaWeightedAverage(
                    tmrtDetail.Tmrt, ctx.positions[pedIdx].areaVectors
                );
                double TumrtAvg = TumrtNoSolar;
                batchTumrtNoSolar[bi] = TumrtNoSolar;
                if (ctx.args.writeDebugTerms) {
                    batchQlwSurfaces[bi] = areaWeightedAverage(tmrtDetail.qlwSurfaces, ctx.positions[pedIdx].areaVectors);
                    batchQlwSky[bi] = areaWeightedAverage(tmrtDetail.qlwSky, ctx.positions[pedIdx].areaVectors);
                    batchQswSurfaces[bi] = areaWeightedAverage(tmrtDetail.qswSurfaces, ctx.positions[pedIdx].areaVectors);
                    batchQswSky[bi] = areaWeightedAverage(tmrtDetail.qswSky, ctx.positions[pedIdx].areaVectors);
                    batchQswGround[bi] = areaWeightedAverage(tmrtDetail.qswGround, ctx.positions[pedIdx].areaVectors);
                    batchQswElevatedDown[bi] = areaWeightedAverage(tmrtDetail.qswElevatedDown, ctx.positions[pedIdx].areaVectors);
                    batchQswVertical[bi] = areaWeightedAverage(tmrtDetail.qswVertical, ctx.positions[pedIdx].areaVectors);
                    batchQswUpward[bi] = areaWeightedAverage(tmrtDetail.qswUpward, ctx.positions[pedIdx].areaVectors);
                }

                int probeIdx = (pedIdx < ctx.probeIndexForPos.size() && ctx.probeIndexForPos[pedIdx] >= 0)
                    ? ctx.probeIndexForPos[pedIdx]
                    : ctx.positions[pedIdx].originalIndex;
                double localQrsw = (probeIdx >= 0 && probeIdx < static_cast<int>(probeQrsw.size()))
                    ? probeQrsw[probeIdx] : -1.0;
                if (localQrsw < 0.0 && pedIdx < static_cast<size_t>(sc.sparseQrswFromSurface.size())) {
                    localQrsw = sc.sparseQrswFromSurface[pedIdx];
                }
                if (localQrsw >= 0.0) {
                    if (ctx.args.writeDebugTerms) batchQswDirect[bi] = fp_solar * localQrsw;
                    double T4 = std::pow(TumrtAvg, 4)
                              + fp_solar * ABS_SW_PERSON * localQrsw / (EPS_LW_PERSON * SIGMA);
                    TumrtAvg = std::pow(std::max(0.0, T4), 0.25);
                } else if (hasSolar) {
                    Vec3 pedPos = ctx.positions[pedIdx].center.toVec3();
                    Vec3 target = pedPos + meteo.sunDir * (R_MAG_MAX * 0.9);
                    if (!ctx.raycaster.isBlocked(pedPos, target)) {
                        if (ctx.args.writeDebugTerms) batchQswDirect[bi] = fp_solar * meteo.Idn;
                        double T4 = std::pow(TumrtAvg, 4)
                                  + fp_solar * ABS_SW_PERSON * meteo.Idn / (EPS_LW_PERSON * SIGMA);
                        TumrtAvg = std::pow(std::max(0.0, T4), 0.25);
                    }
                }
                results.tumrtAvg[tIdx][pedIdx] = TumrtAvg;
                batchTmrt[bi] = TumrtAvg;

                double Ta_K = (probeIdx < static_cast<int>(probeT.size())) ? probeT[probeIdx] : meteo.Ta;
                if (Ta_K < 200.0) Ta_K = meteo.Ta;
                double Ta_c = Ta_K - 273.15;
                double va = (probeIdx < static_cast<int>(probeU.size())) ? probeU[probeIdx] / 0.667 : meteo.va;
                va = std::max(0.5, std::min(17.0, va));
                double w = (probeIdx < static_cast<int>(probeW.size())) ? probeW[probeIdx] : 0.01;
                double psat = std::exp(77.345 + 0.0057 * Ta_K - 7235.0 / Ta_K) / std::pow(Ta_K, 8.2);
                double pv = P_REF * w / (EPSILON_H2O + w);
                double RH = std::min(100.0, std::max(0.0, pv / psat * 100.0));
                batchRH[bi] = RH;

                if (ctx.args.computeUtci) {
                    double Tmrt_c = batchTmrt[bi] - 273.15;
                    batchUtci[bi] = ctx.utciSolver.calculate(Ta_c, va, RH, Tmrt_c);
                }
            }

            for (size_t bi = 0; bi < bN; ++bi) {
                results.tmrt[tIdx][bStart + bi] = batchTmrt[bi];
                results.utci[tIdx][bStart + bi] = batchUtci[bi];
                results.rh[tIdx][bStart + bi] = batchRH[bi];
                results.tumrtNoSolar[tIdx][bStart + bi] = batchTumrtNoSolar[bi];
                if (ctx.args.writeDebugTerms) {
                    results.qlwSurfaces[tIdx][bStart + bi] = batchQlwSurfaces[bi];
                    results.qlwSky[tIdx][bStart + bi] = batchQlwSky[bi];
                    results.qswSurfaces[tIdx][bStart + bi] = batchQswSurfaces[bi];
                    results.qswSky[tIdx][bStart + bi] = batchQswSky[bi];
                    results.qswDirect[tIdx][bStart + bi] = batchQswDirect[bi];
                    results.qswGround[tIdx][bStart + bi] = batchQswGround[bi];
                    results.qswElevatedDown[tIdx][bStart + bi] = batchQswElevatedDown[bi];
                    results.qswVertical[tIdx][bStart + bi] = batchQswVertical[bi];
                    results.qswUpward[tIdx][bStart + bi] = batchQswUpward[bi];
                }
            }
        }

        size_t posProcessed = bEnd;
        using namespace std::chrono;
        double elapsed = duration<double>(steady_clock::now() - totalT0).count();
        double eta = (posProcessed > 0) ? elapsed / posProcessed * (nPos - posProcessed) : 0;
        std::ostringstream out;
        out << "  Batch done. Total progress: " << posProcessed << "/" << nPos
            << "  elapsed=" << static_cast<int>(elapsed) << "s"
            << "  ETA=" << static_cast<int>(eta) << "s";
        logSummary(out.str());
    }

    return results;
}

static bool writeTumrtTerms(const std::string& path,
                            const std::vector<PedestrianPosition>& positions,
                            int timestep,
                            const Eigen::VectorXd& tumrtNoSolar,
                            const Eigen::VectorXd& tumrtFinal,
                            const Eigen::VectorXd& qlwSurfaces,
                            const Eigen::VectorXd& qlwSky,
                            const Eigen::VectorXd& qswSurfaces,
                            const Eigen::VectorXd& qswSky,
                            const Eigen::VectorXd& qswDirect,
                            const Eigen::VectorXd& qswGround,
                            const Eigen::VectorXd& qswElevatedDown,
                            const Eigen::VectorXd& qswVertical,
                            const Eigen::VectorXd& qswUpward) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < positions.size(); ++i) {
        f << timestep << " "
          << positions[i].center.x << " "
          << positions[i].center.y << " "
          << positions[i].center.z << " "
          << tumrtNoSolar[i] << " "
          << tumrtFinal[i] << " "
          << qlwSurfaces[i] << " "
          << qlwSky[i] << " "
          << qswSurfaces[i] << " "
          << qswSky[i] << " "
          << qswDirect[i] << " "
          << qswGround[i] << " "
          << qswElevatedDown[i] << " "
          << qswVertical[i] << " "
          << qswUpward[i] << "\n";
    }
    return true;
}

int main(int argc, char* argv[]) {
    auto startTime = std::chrono::high_resolution_clock::now();

    CommandLineArgs args = parseArgs(argc, argv);

    if (args.casePath.empty()) {
        logError("--case is required");
        printUsage(argv[0]);
        return 1;
    }

#ifdef _OPENMP
    if (args.nThreads > 0) {
        omp_set_num_threads(args.nThreads);
    }
    logInfo("Using " + std::to_string(omp_get_max_threads()) + " OpenMP threads");
#else
    logInfo("OpenMP not enabled, using single thread");
#endif

    logSection("umcfUTCIpostprocess: C++ Tmrt/UTCI Calculator");
    logInfo("Case: " + args.casePath);

    std::vector<int> timesteps = getTimesteps(args.tStart, args.tEnd, args.tStep);
    {
        std::ostringstream out;
        out << "Timesteps: " << timesteps.size() << " (";
        for (size_t i = 0; i < std::min(timesteps.size(), size_t(3)); ++i) {
            out << timesteps[i];
            if (i + 1 < std::min(timesteps.size(), size_t(3))) out << ", ";
        }
        if (timesteps.size() > 3) out << ", ...";
        out << ")";
        logInfo(out.str());
    }

    // =========================================================
    // Load pedestrian positions
    // =========================================================
    std::string probeLocsPath = args.casePath + "/system/air/probe_locs";
    std::vector<PedestrianPosition> positions = loadPedestrianPositions(probeLocsPath);

    if (positions.empty()) {
        logError("No pedestrian positions loaded");
        return 1;
    }

    logInfo("Loaded " + std::to_string(positions.size()) + " positions");

    // Optional spatial filter
    if (args.filterRadius > 0.0) {
        double cx = args.filterCenterX;
        double cy = args.filterCenterY;
        double r2 = args.filterRadius * args.filterRadius;
        {
            std::ostringstream out;
            out << "Filtering positions within radius " << args.filterRadius
                << " of (" << cx << ", " << cy << ")...";
            logInfo(out.str());
        }
        std::vector<PedestrianPosition> filtered;
        for (const auto& pos : positions) {
            double dx = pos.center.x - cx;
            double dy = pos.center.y - cy;
            if (dx*dx + dy*dy <= r2) {
                filtered.push_back(pos);
            }
        }
        if (filtered.empty()) {
            logWarn("filter removed all positions – using all");
        } else {
            positions = filtered;
            logInfo("After filter: " + std::to_string(positions.size()) + " positions");
        }
    }

    if (args.maxPositions > 0 && (int)positions.size() > args.maxPositions) {
        positions.resize(args.maxPositions);
        logInfo("Capped to " + std::to_string(positions.size()) + " positions (--max-positions)");
    }

    // =========================================================
    // Load STL geometry for ray occlusion
    // =========================================================
    std::string wallTreeStl = args.casePath + "/constant/triSurface/wallAndTreeSurfaces.stl";
    Raycaster raycaster;
    raycaster.setNumThreads(args.nThreads);
    logInfo("Loading STL geometry...");
    if (!raycaster.loadGeometry(wallTreeStl)) {
        logError("could not load " + wallTreeStl);
        return 1;
    }

    // =========================================================
    // Load surface geometry ONCE (from first timestep)
    // =========================================================
    int geometryTimestep = timesteps[0];
    std::string firstSurfDir;
    for (int t : timesteps) {
        std::string candidate = args.casePath + "/postProcessing/surfaces/" + std::to_string(t);
        if (!firstExistingPath({
                candidate + "/Sf_wallSurfaces.raw",
                candidate + "/Sf_wallAndTreeSurfaces.raw",
                candidate + "/Sf_skySurfaces.raw"
            }).empty()) {
            geometryTimestep = t;
            firstSurfDir = candidate;
            break;
        }
    }
    if (firstSurfDir.empty()) {
        const std::string fallback = args.casePath + "/postProcessing/surfaces/3600";
        if (!firstExistingPath({
                fallback + "/Sf_wallSurfaces.raw",
                fallback + "/Sf_wallAndTreeSurfaces.raw",
                fallback + "/Sf_skySurfaces.raw"
            }).empty()) {
            geometryTimestep = 3600;
            firstSurfDir = fallback;
        }
    }
    if (firstSurfDir.empty()) {
        firstSurfDir = args.casePath + "/postProcessing/surfaces/" + std::to_string(timesteps[0]);
    }

    logInfo("Loading surface geometry from timestep " + std::to_string(geometryTimestep) + "...");

    // Try discrete-workflow names first, fall back to combined name
    auto wallGeo = loadSurfacePatches(firstSurfDir + "/Sf_wallSurfaces.raw");
    if (wallGeo.empty())
        wallGeo = loadSurfacePatches(firstSurfDir + "/Sf_wallAndTreeSurfaces.raw");
    auto vegGeo  = loadSurfacePatches(firstSurfDir + "/Sf_vegSurfaces.raw");
    auto allGeo  = concat(wallGeo, vegGeo);

    logDetail("Wall surfaces: " + std::to_string(wallGeo.size()));
    logDetail("Veg surfaces:  " + std::to_string(vegGeo.size()));
    logDetail("Total:         " + std::to_string(allGeo.size()));

    std::vector<SurfacePatch> skyGeo;
    if (args.useSkyViewFactors) {
        skyGeo = loadSurfacePatches(firstSurfDir + "/Sf_skySurfaces.raw");
        logDetail("Sky surfaces:  " + std::to_string(skyGeo.size()));
    }

    // =========================================================
    // Load meteorological data for all timesteps
    // =========================================================
    logSection("Loading meteorological data");
    std::vector<MeteoData> meteoData = loadMeteoData(args.casePath, timesteps);
    for (size_t i = 0; i < std::min(meteoData.size(), size_t(3)); ++i) {
        std::ostringstream out;
        out << "t=" << timesteps[i]
            << "  Ta=" << meteoData[i].Ta
            << "  cc=" << meteoData[i].cc
            << "  Idif=" << meteoData[i].Idif;
        logDetail(out.str());
    }

    // =========================================================
    // Preload all per-timestep scalar fields upfront
    // (wall/veg temperatures + veg qr for each timestep)
    // This avoids repeated disk I/O inside the batch loop.
    // Memory: ~(3 × nSurfaces × nTimesteps × 8 bytes) ≈ small
    // =========================================================
    logSection("Preloading surface scalars for all timesteps");
    std::vector<TimestepScalars> tsData(timesteps.size());
    std::vector<Point3> pedCenters(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) pedCenters[i] = positions[i].center;
    for (size_t tIdx = 0; tIdx < timesteps.size(); ++tIdx) {
        int t = timesteps[tIdx];
        std::string surfDir = args.casePath + "/postProcessing/surfaces/" + std::to_string(t);
        std::string radDir = args.casePath + "/postProcessing/surfacesPedestrianRad/" + std::to_string(t);
        tsData[tIdx].wallTemps = loadScalarField(surfDir + "/T_wallSurfaces.raw");
        tsData[tIdx].vegTemps  = loadScalarField(surfDir + "/T_vegSurfaces.raw");
        tsData[tIdx].vegQr     = loadScalarField(surfDir + "/qr_vegSurfaces.raw");
        // combined outgoing LW radiation (reference format, replaces T+qr)
        tsData[tIdx].allQrOut  = loadScalarField(surfDir + "/qrOut_wallAndTreeSurfaces.raw");
        tsData[tIdx].allQsOut  = loadScalarField(surfDir + "/qsOut_wallAndTreeSurfaces.raw");
        const std::string qrswPath = firstExistingPath({
            surfDir + "/qrsw_pedestrian.vtk",
            radDir + "/qrsw_pedestrian.vtk"
        });
        if (!qrswPath.empty()) {
            VtkMeshData meshQrsw;
            if (readLegacyVtkMesh(qrswPath, meshQrsw)) {
                tsData[tIdx].sparseQrswFromSurface = remapQrswMagnitudesToPoints(meshQrsw, pedCenters);
            }
        }
        if (tsData[tIdx].wallTemps.empty() && tsData[tIdx].allQrOut.empty()) {
            logWarn("no wall temps or qrOut for t=" + std::to_string(t));
        }
    }
    logInfo("Scalars loaded.");

    // =========================================================
    // Preload per-position CFD data from probe files (all timestep rows)
    // One row per radiation timestep; we pick the closest row per timestep.
    // =========================================================
    logSection("Loading per-position probe data");
    std::string probeDir = findProbeDir(args.casePath, "air");
    if (probeDir.empty()) {
        probeDir = args.casePath + "/postProcessing/probes/air/3600";
        logWarn("probe dir not found via scan, falling back to " + probeDir);
    } else {
        logDetail("Probe dir: " + probeDir);
    }
    auto probeTAll = loadProbeScalarAll(probeDir + "/T");
    auto probeWAll = loadProbeScalarAll(probeDir + "/w");
    auto probeUAll = loadProbeVelocityMagAll(probeDir + "/U");
    auto probeQrswAll = loadQrswProbeData(args.casePath);
    auto probePoints = loadProbePoints(probeDir + "/T");
    {
        std::ostringstream out;
        out << "Probe T rows: " << probeTAll.size()
            << "  U rows: " << probeUAll.size();
        logDetail(out.str());
    }
    if (probeTAll.empty())
        logWarn("no probe T – using ambient temperature for UTCI");
    if (probeUAll.empty())
        logWarn("no probe U – using reference wind speed for UTCI");
    if (probeQrswAll.empty())
        logWarn("no probe qrsw data – using dense qrsw surface sampling fallback before binary shadow");
    else
        logDetail("Probe qrsw rows: " + std::to_string(probeQrswAll.size()));
    static const std::vector<double> emptyVec;

    if (!validateRequiredInputs(args, allGeo, skyGeo, meteoData, timesteps,
                                probeTAll, probeUAll, probeWAll, tsData)) {
        return 1;
    }

    std::vector<int> probeIndexForPos(positions.size(), -1);
    if (!probePoints.empty()) {
        std::unordered_map<std::string, int> probePointIndex;
        for (size_t i = 0; i < probePoints.size(); ++i) {
            probePointIndex[probeKey(probePoints[i])] = static_cast<int>(i);
        }
        for (size_t i = 0; i < positions.size(); ++i) {
            auto it = probePointIndex.find(probeKey(positions[i].center));
            if (it != probePointIndex.end()) probeIndexForPos[i] = it->second;
        }
    }

    // =========================================================
    // Create output directories once (mkdir -p style)
    // =========================================================
    {
        std::string baseDir = args.casePath + "/" + args.outputDir;
        createDirectory(baseDir);
        for (int t : timesteps) {
            std::string outDir = baseDir + "/" + std::to_string(t);
            if (!createDirectory(outDir)) {
                logWarn("could not create " + outDir + ": " + std::strerror(errno));
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
    cache.setCompressed(args.compressCache);
    if (args.compressCache && !BinaryCache::compressionAvailable()) {
        logWarn("--compress-cache requested but binary was built without ZLIB support; cache files will be written uncompressed.");
    }
    TmrtSolver tmrtSolver;
    UtciSolver utciSolver;
    utciSolver.setMethod(args.utciMethod);
    if (args.utciMethod == UtciMethod::LUT) {
        // Resolve LUT path: explicit arg → next to binary → next to case
        std::string lutFile = args.lutPath;
        if (lutFile.empty()) {
            // argv[0] gives binary path; look in same directory
            std::string binDir = std::string(getenv("_") ? getenv("_") : "");
            // Fallback: look next to the source build dir
            std::string candidate = std::string(argv[0]);
            auto slash = candidate.rfind('/');
            std::string binDirPath = (slash != std::string::npos)
                ? candidate.substr(0, slash) : ".";
            lutFile = binDirPath + "/../utci_offset.Dat";
            if (!std::ifstream(lutFile).good())
                lutFile = binDirPath + "/utci_offset.Dat";
            if (!std::ifstream(lutFile).good())
                lutFile = args.casePath + "/utci_offset.Dat";
        }
        logInfo("Loading UTCI LUT: " + lutFile);
        if (!utciSolver.loadLUT(lutFile)) {
            logWarn("failed to load LUT — falling back to polynomial");
            utciSolver.setMethod(UtciMethod::POLYNOMIAL);
        }
    }
    logInfo("UTCI method: " + std::string(
        utciSolver.method() == UtciMethod::LUT ? "LUT" : "polynomial"));

    size_t nPos = positions.size();
    size_t nT = timesteps.size();
    SparseStage2Context sparseCtx{
        args, timesteps, positions, wallGeo, vegGeo, allGeo, skyGeo,
        meteoData, tsData, probeTAll, probeWAll, probeUAll, probeQrswAll,
        probeIndexForPos, emptyVec, raycaster, vfCalc, cache, tmrtSolver, utciSolver
    };
    SparseStage2Results sparseResults = runSparseStage2(sparseCtx);

    // Write VTK output and print stats per timestep
    logSection("Writing VTK output and per-timestep stats");
    std::vector<std::string> timestepSummaries(nT);
    #pragma omp parallel for schedule(dynamic)
    for (size_t tIdx = 0; tIdx < nT; ++tIdx) {
        int t = timesteps[tIdx];
        std::string outDir = args.casePath + "/" + args.outputDir + "/" + std::to_string(t);

        // Tmrt_pedestrian.vtk  (Kelvin) — point cloud
        writeVtkPolyData(outDir + "/Tmrt_pedestrian.vtk", positions, sparseResults.tmrt[tIdx], "Tmrt");
        // TumrtAvg is the sparse pre-solar field used for dense interpolation.
        writeTumrtAvg(outDir + "/TumrtAvg", positions, t, sparseResults.tumrtNoSolar[tIdx], false);
        if (args.writeDebugTerms) {
            writeTumrtTerms(outDir + "/TumrtAvg_terms", positions, t,
                            sparseResults.tumrtNoSolar[tIdx], sparseResults.tumrtAvg[tIdx],
                            sparseResults.qlwSurfaces[tIdx], sparseResults.qlwSky[tIdx],
                            sparseResults.qswSurfaces[tIdx], sparseResults.qswSky[tIdx], sparseResults.qswDirect[tIdx],
                            sparseResults.qswGround[tIdx], sparseResults.qswElevatedDown[tIdx],
                            sparseResults.qswVertical[tIdx], sparseResults.qswUpward[tIdx]);
        }

        // RH_pedestrian.vtk — point cloud
        writeVtkPolyData(outDir + "/RH_pedestrian.vtk", positions, sparseResults.rh[tIdx], "RH");

        // UTCI.vtk  (Tmrt in °C + UTCI in °C) — point cloud
        // UTCI_surface.vtk   — interpolated structured grid surface
        if (args.computeUtci) {
            Eigen::VectorXd TmrtC = sparseResults.tmrt[tIdx].array() - 273.15;
            writeVtkMultiScalar(outDir + "/UTCI.vtk", positions,
                { {"Tmrt", TmrtC}, {"UTCI", sparseResults.utci[tIdx]} });
        }
        computeDenseSurfaceOutputs(args.casePath, outDir, t, positions, sparseResults.tumrtNoSolar[tIdx],
                                   utciSolver, args.writeDebugQrswSurface);

        // Print stats
        double tMin  = sparseResults.tmrt[tIdx].minCoeff();
        double tMax  = sparseResults.tmrt[tIdx].maxCoeff();
        double tMean = sparseResults.tmrt[tIdx].mean();
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(1)
                << "  t=" << t
                << "  Tmrt[K]: min=" << tMin
                << " max=" << tMax << " mean=" << tMean;
        if (args.computeUtci) {
            double uMin  = sparseResults.utci[tIdx].minCoeff();
            double uMax  = sparseResults.utci[tIdx].maxCoeff();
            double uMean = sparseResults.utci[tIdx].mean();
            summary << "  UTCI[°C]: min=" << uMin << " max=" << uMax << " mean=" << uMean;
        }
        timestepSummaries[tIdx] = summary.str();
    }
    for (const auto& summary : timestepSummaries) {
        logSummary(summary);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

    logSection("Complete");
    logInfo("Total time: " + std::to_string(duration.count()) + " seconds");

    return 0;
}
