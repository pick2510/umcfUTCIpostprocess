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
#include <unordered_map>

#include "constants.h"
#include "types.h"
#include "pedestrian.h"
#include "raycaster.h"
#include "viewFactor.h"
#include "tmrtSolver.h"
#include "utciSolver.h"
#include "io.h"
#include "caching.h"
#include <fstream>

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

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --case <path>          Case directory (required)\n"
              << "  --start <time>         Start timestep (default: 3600)\n"
              << "  --end <time>           End timestep (default: 86400)\n"
              << "  --step <time>          Timestep interval (default: 3600)\n"
              << "  --output-dir <dir>     Output directory (default: UTCI)\n"
              << "  --skip-utci            Skip UTCI calculation\n"
              << "  --utci-method <m>      UTCI method: poly (default) or lut\n"
              << "  --lut-path <file>      Path to utci_offset.Dat (default: next to binary)\n"
              << "  --force-recompute      Force recompute view factors\n"
              << "  --filter-radius <r>    Filter positions within radius r of center\n"
              << "  --filter-cx <x>        Filter center X (used with --filter-radius)\n"
              << "  --filter-cy <y>        Filter center Y (used with --filter-radius)\n"
              << "  --max-positions <N>    Cap number of positions (for testing)\n"
              << "  --batch-size <N>       VF batch size (default 500, lower = less memory)\n"
              << "  --compress-cache       Write gzip-compressed cache files (default when ZLIB available)\n"
              << "  --no-compress-cache    Write uncompressed cache files\n"
              << "  --write-debug-terms    Write TumrtAvg_terms debug output\n"
              << "  --write-debug-qrsw     Write qrsw_surface.vtk debug output\n"
              << "  -j <N>                 Number of threads (default: 1)\n"
              << "  --help                 Show this message\n";
}

CommandLineArgs parseArgs(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--case" && i + 1 < argc) {
            args.casePath = argv[++i];
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
        } else if (arg == "--utci-method" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "lut")  args.utciMethod = UtciMethod::LUT;
            else if (m == "poly") args.utciMethod = UtciMethod::POLYNOMIAL;
            else { std::cerr << "Unknown --utci-method: " << m << " (use poly or lut)\n"; exit(1); }
        } else if (arg == "--lut-path" && i + 1 < argc) {
            args.lutPath = argv[++i];
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

static std::string probeKey(const Point3& p) {
    auto q = [](double v) {
        return static_cast<long long>(std::llround(v * 1000.0));
    };
    return std::to_string(q(p.x)) + ":" + std::to_string(q(p.y)) + ":" + std::to_string(q(p.z));
}

struct UniformGridField {
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> values;
    std::unordered_map<long long, size_t> indexByCell;
    bool valid = false;
};

static constexpr double DENSE_INTERP_BBOX_INSET = 1.0;

static long long gridKey(int ix, int iy) {
    return (static_cast<long long>(ix) << 32) ^
           static_cast<unsigned int>(iy);
}

static UniformGridField buildUniformGridField(const std::vector<PedestrianPosition>& positions,
                                              const Eigen::VectorXd& values) {
    UniformGridField field;
    field.xs.reserve(positions.size());
    field.ys.reserve(positions.size());
    for (const auto& p : positions) {
        field.xs.push_back(p.center.x);
        field.ys.push_back(p.center.y);
    }
    auto dedupe = [](std::vector<double>& coords) {
        std::sort(coords.begin(), coords.end());
        coords.erase(std::unique(coords.begin(), coords.end(),
            [](double a, double b) { return std::abs(a - b) < 1e-4; }),
            coords.end());
    };
    dedupe(field.xs);
    dedupe(field.ys);
    if (field.xs.size() < 2 || field.ys.size() < 2) return field;

    field.values.assign(field.xs.size() * field.ys.size(),
                        std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < positions.size() && i < static_cast<size_t>(values.size()); ++i) {
        auto xit = std::lower_bound(field.xs.begin(), field.xs.end(), positions[i].center.x - 1e-4);
        auto yit = std::lower_bound(field.ys.begin(), field.ys.end(), positions[i].center.y - 1e-4);
        if (xit == field.xs.end() || yit == field.ys.end()) continue;
        int ix = static_cast<int>(xit - field.xs.begin());
        int iy = static_cast<int>(yit - field.ys.begin());
        if (ix >= 0 && iy >= 0 && ix < static_cast<int>(field.xs.size()) &&
            iy < static_cast<int>(field.ys.size())) {
            field.values[iy * field.xs.size() + ix] = values[i];
            field.indexByCell[gridKey(ix, iy)] = i;
        }
    }
    field.valid = true;
    return field;
}

static double nearestSparseValue(const std::vector<PedestrianPosition>& positions,
                                 const Eigen::VectorXd& values,
                                 double x, double y) {
    double bestD2 = std::numeric_limits<double>::infinity();
    double best = 0.0;
    for (size_t i = 0; i < positions.size() && i < static_cast<size_t>(values.size()); ++i) {
        double dx = positions[i].center.x - x;
        double dy = positions[i].center.y - y;
        double d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
            bestD2 = d2;
            best = values[i];
        }
    }
    return best;
}

static double cubicInterpolate1D(double p0, double p1, double p2, double p3, double t) {
    double a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    double a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    double a2 = -0.5 * p0 + 0.5 * p2;
    double a3 = p1;
    return ((a0 * t + a1) * t + a2) * t + a3;
}

static double interpolateSparseTumrt(const UniformGridField& field,
                                     const std::vector<PedestrianPosition>& positions,
                                     const Eigen::VectorXd& values,
                                     double x, double y) {
    if (!field.valid || field.xs.empty() || field.ys.empty()) {
        return nearestSparseValue(positions, values, x, y);
    }
    if (x < field.xs.front() || x > field.xs.back() || y < field.ys.front() || y > field.ys.back()) {
        return nearestSparseValue(positions, values, x, y);
    }

    // Match the Clement workflow: use cubic interpolation only inside an inset
    // bounding box and fall back to nearest outside that region or when the
    // required 4x4 stencil is incomplete.
    if (x < field.xs.front() + DENSE_INTERP_BBOX_INSET ||
        x > field.xs.back() - DENSE_INTERP_BBOX_INSET ||
        y < field.ys.front() + DENSE_INTERP_BBOX_INSET ||
        y > field.ys.back() - DENSE_INTERP_BBOX_INSET) {
        return nearestSparseValue(positions, values, x, y);
    }

    auto xhi = std::lower_bound(field.xs.begin(), field.xs.end(), x);
    auto yhi = std::lower_bound(field.ys.begin(), field.ys.end(), y);
    if (xhi == field.xs.begin() || yhi == field.ys.begin() ||
        xhi == field.xs.end() || yhi == field.ys.end()) {
        return nearestSparseValue(positions, values, x, y);
    }

    int ix1 = static_cast<int>(xhi - field.xs.begin());
    int iy1 = static_cast<int>(yhi - field.ys.begin());
    int ix0 = ix1 - 1;
    int iy0 = iy1 - 1;
    int ixm1 = ix0 - 1;
    int ix2 = ix1 + 1;
    int iym1 = iy0 - 1;
    int iy2 = iy1 + 1;
    if (ixm1 < 0 || iym1 < 0 ||
        ix2 >= static_cast<int>(field.xs.size()) ||
        iy2 >= static_cast<int>(field.ys.size())) {
        return nearestSparseValue(positions, values, x, y);
    }

    auto getVal = [&](int ix, int iy) -> double {
        auto it = field.indexByCell.find(gridKey(ix, iy));
        if (it == field.indexByCell.end()) return std::numeric_limits<double>::quiet_NaN();
        return values[it->second];
    };

    double x0 = field.xs[ix0];
    double x1 = field.xs[ix1];
    double y0 = field.ys[iy0];
    double y1 = field.ys[iy1];
    double tx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
    double ty = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;
    double rowVals[4];
    const int yIdx[4] = {iym1, iy0, iy1, iy2};
    const int xIdx[4] = {ixm1, ix0, ix1, ix2};
    for (int ry = 0; ry < 4; ++ry) {
        double samples[4];
        for (int rx = 0; rx < 4; ++rx) {
            samples[rx] = getVal(xIdx[rx], yIdx[ry]);
            if (!std::isfinite(samples[rx])) {
                return nearestSparseValue(positions, values, x, y);
            }
        }
        rowVals[ry] = cubicInterpolate1D(samples[0], samples[1], samples[2], samples[3], tx);
    }
    return cubicInterpolate1D(rowVals[0], rowVals[1], rowVals[2], rowVals[3], ty);
}

static Eigen::VectorXd vectorMagnitudes(const std::vector<Vec3>& vecs) {
    Eigen::VectorXd out(vecs.size());
    for (size_t i = 0; i < vecs.size(); ++i) out[i] = vecs[i].norm();
    return out;
}

static std::string firstExistingPath(const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return "";
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

static long long hashGridKey(int ix, int iy) {
    return (static_cast<long long>(ix) << 32)
         ^ static_cast<unsigned int>(iy);
}

struct CellLocator2D {
    double minX = 0.0;
    double minY = 0.0;
    double binSize = 1.0;
    std::vector<Point3> centroids;
    std::vector<std::array<double, 4>> bboxes;
    std::unordered_map<long long, std::vector<int>> bins;
};

static bool pointInPolygonXY(const std::vector<Point3>& points,
                             const std::vector<int>& cell,
                             double x,
                             double y) {
    bool inside = false;
    const size_t n = cell.size();
    if (n < 3) return false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point3& pi = points[cell[i]];
        const Point3& pj = points[cell[j]];
        const bool crosses = ((pi.y > y) != (pj.y > y));
        if (!crosses) continue;
        const double denom = pj.y - pi.y;
        if (std::abs(denom) < 1e-12) continue;
        const double xCross = pi.x + (y - pi.y) * (pj.x - pi.x) / denom;
        if (x < xCross) inside = !inside;
    }
    return inside;
}

static CellLocator2D buildCellLocator2D(const VtkMeshData& mesh) {
    CellLocator2D loc;
    if (mesh.cells.empty() || mesh.points.empty()) return loc;

    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& p : mesh.points) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    loc.minX = minX;
    loc.minY = minY;
    const double domainArea = std::max(1e-6, (maxX - minX) * (maxY - minY));
    loc.binSize = std::max(0.25, std::sqrt(domainArea / std::max<size_t>(1, mesh.cells.size())));
    loc.centroids.resize(mesh.cells.size());
    loc.bboxes.resize(mesh.cells.size());

    for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
        const auto& cell = mesh.cells[ci];
        double cx = 0.0, cy = 0.0, cz = 0.0;
        double bx0 = std::numeric_limits<double>::infinity();
        double by0 = std::numeric_limits<double>::infinity();
        double bx1 = -std::numeric_limits<double>::infinity();
        double by1 = -std::numeric_limits<double>::infinity();
        for (int pid : cell) {
            const auto& p = mesh.points[pid];
            cx += p.x;
            cy += p.y;
            cz += p.z;
            bx0 = std::min(bx0, p.x);
            by0 = std::min(by0, p.y);
            bx1 = std::max(bx1, p.x);
            by1 = std::max(by1, p.y);
        }
        const double invN = cell.empty() ? 0.0 : 1.0 / static_cast<double>(cell.size());
        loc.centroids[ci] = {cx * invN, cy * invN, cz * invN};
        loc.bboxes[ci] = {bx0, by0, bx1, by1};

        const int ix0 = static_cast<int>(std::floor((bx0 - loc.minX) / loc.binSize));
        const int iy0 = static_cast<int>(std::floor((by0 - loc.minY) / loc.binSize));
        const int ix1 = static_cast<int>(std::floor((bx1 - loc.minX) / loc.binSize));
        const int iy1 = static_cast<int>(std::floor((by1 - loc.minY) / loc.binSize));
        for (int ix = ix0; ix <= ix1; ++ix) {
            for (int iy = iy0; iy <= iy1; ++iy) {
                loc.bins[hashGridKey(ix, iy)].push_back(static_cast<int>(ci));
            }
        }
    }
    return loc;
}

static std::vector<Vec3> remapCellVectorsToPoints2D(const VtkMeshData& srcMesh,
                                                    const std::vector<Vec3>& cellVectors,
                                                    const std::vector<Point3>& dstPoints) {
    std::vector<Vec3> out(dstPoints.size(), Vec3::Zero());
    if (srcMesh.cells.empty() || cellVectors.size() != srcMesh.cells.size()) return out;

    const CellLocator2D loc = buildCellLocator2D(srcMesh);
    for (size_t i = 0; i < dstPoints.size(); ++i) {
        const double x = dstPoints[i].x;
        const double y = dstPoints[i].y;
        const int ix = static_cast<int>(std::floor((x - loc.minX) / loc.binSize));
        const int iy = static_cast<int>(std::floor((y - loc.minY) / loc.binSize));

        int bestCell = -1;
        double bestInsideD2 = std::numeric_limits<double>::infinity();
        for (int rx = -1; rx <= 1; ++rx) {
            for (int ry = -1; ry <= 1; ++ry) {
                auto it = loc.bins.find(hashGridKey(ix + rx, iy + ry));
                if (it == loc.bins.end()) continue;
                for (int ci : it->second) {
                    const auto& bb = loc.bboxes[ci];
                    if (x < bb[0] - 1e-9 || x > bb[2] + 1e-9 ||
                        y < bb[1] - 1e-9 || y > bb[3] + 1e-9) {
                        continue;
                    }
                    if (!pointInPolygonXY(srcMesh.points, srcMesh.cells[ci], x, y)) continue;
                    const double dx = loc.centroids[ci].x - x;
                    const double dy = loc.centroids[ci].y - y;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < bestInsideD2) {
                        bestInsideD2 = d2;
                        bestCell = ci;
                    }
                }
            }
        }

        if (bestCell < 0) {
            double bestD2 = std::numeric_limits<double>::infinity();
            for (size_t ci = 0; ci < loc.centroids.size(); ++ci) {
                const double dx = loc.centroids[ci].x - x;
                const double dy = loc.centroids[ci].y - y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < bestD2) {
                    bestD2 = d2;
                    bestCell = static_cast<int>(ci);
                }
            }
        }

        if (bestCell >= 0) out[i] = cellVectors[bestCell];
    }
    return out;
}

static Eigen::VectorXd remapQrswMagnitudesToPoints(const VtkMeshData& meshQrsw,
                                                   const std::vector<Point3>& dstPoints) {
    auto itQCell = meshQrsw.cellVectors.find("qrsw");
    auto itQPoint = meshQrsw.pointVectors.find("qrsw");
    std::vector<Vec3> qrswOnDst;
    if (itQPoint != meshQrsw.pointVectors.end() &&
        meshQrsw.points.size() == dstPoints.size()) {
        qrswOnDst = itQPoint->second;
    } else if (itQCell != meshQrsw.cellVectors.end() && !meshQrsw.cells.empty()) {
        qrswOnDst = remapCellVectorsToPoints2D(meshQrsw, itQCell->second, dstPoints);
    } else if (itQPoint != meshQrsw.pointVectors.end()) {
        qrswOnDst.resize(dstPoints.size(), Vec3::Zero());
        for (size_t i = 0; i < dstPoints.size(); ++i) {
            double bestD2 = std::numeric_limits<double>::infinity();
            size_t best = 0;
            for (size_t j = 0; j < meshQrsw.points.size(); ++j) {
                double dx = meshQrsw.points[j].x - dstPoints[i].x;
                double dy = meshQrsw.points[j].y - dstPoints[i].y;
                double d2 = dx * dx + dy * dy;
                if (d2 < bestD2) {
                    bestD2 = d2;
                    best = j;
                }
            }
            if (best < itQPoint->second.size()) qrswOnDst[i] = itQPoint->second[best];
        }
    }
    return vectorMagnitudes(qrswOnDst);
}

static bool computeDenseSurfaceOutputs(const std::string& casePath,
                                       const std::string& outDir,
                                       int timestep,
                                       const std::vector<PedestrianPosition>& sparsePositions,
                                       const Eigen::VectorXd& sparseTumrtAvg,
                                       UtciSolver& utciSolver,
                                       bool debugWriteQrsw) {
    const std::string surfaceDir = casePath + "/postProcessing/surfaces/" + std::to_string(timestep);
    const std::string airDir = casePath + "/postProcessing/surfacesPedestrianAir/" + std::to_string(timestep);
    const std::string radDir = casePath + "/postProcessing/surfacesPedestrianRad/" + std::to_string(timestep);
    const std::string meshPath = firstExistingPath({
        surfaceDir + "/T_pedestrian.vtk",
        airDir + "/T_pedestrian.vtk"
    });
    const std::string qrswPath = firstExistingPath({
        surfaceDir + "/qrsw_pedestrian.vtk",
        radDir + "/qrsw_pedestrian.vtk"
    });
    const std::string uPath = firstExistingPath({
        surfaceDir + "/U_pedestrian.vtk",
        airDir + "/U_pedestrian.vtk"
    });
    const std::string wPath = firstExistingPath({
        surfaceDir + "/w_pedestrian.vtk",
        airDir + "/w_pedestrian.vtk"
    });

    VtkMeshData meshT, meshQrsw, meshU, meshW;
    if (meshPath.empty() || !readLegacyVtkMesh(meshPath, meshT)) {
        std::cout << "  Dense Stage 2 skipped for t=" << timestep
                  << " (missing dense T_pedestrian.vtk)\n";
        return false;
    }
    if (qrswPath.empty() || uPath.empty() || wPath.empty() ||
        !readLegacyVtkMesh(qrswPath, meshQrsw) ||
        !readLegacyVtkMesh(uPath, meshU) ||
        !readLegacyVtkMesh(wPath, meshW)) {
        std::cout << "  Dense Stage 2 skipped for t=" << timestep
                  << " (missing dense qrsw/U/w fields)\n";
        return false;
    }

    auto itT = meshT.pointScalars.find("T");
    auto itW = meshW.pointScalars.find("w");
    auto itU = meshU.pointVectors.find("U");
    auto itQ = meshQrsw.pointVectors.find("qrsw");
    auto itQCell = meshQrsw.cellVectors.find("qrsw");
    if (itT == meshT.pointScalars.end() || itW == meshW.pointScalars.end() ||
        itU == meshU.pointVectors.end() || itQ == meshQrsw.pointVectors.end()) {
        std::cout << "  Dense Stage 2 skipped for t=" << timestep
                  << " (expected T/U/w/qrsw arrays not found)\n";
        return false;
    }

    const size_t n = meshT.points.size();
    if (itT->second.size() != static_cast<int>(n) ||
        itW->second.size() != static_cast<int>(n) ||
        itU->second.size() != n) {
        std::cout << "  Dense Stage 2 skipped for t=" << timestep
                  << " (dense T/U/w arrays do not align with T mesh)\n";
        return false;
    }

    const auto sparseField = buildUniformGridField(sparsePositions, sparseTumrtAvg);
    Eigen::VectorXd denseTumrtAvg = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < n; ++i) {
        denseTumrtAvg[i] = interpolateSparseTumrt(
            sparseField, sparsePositions, sparseTumrtAvg,
            meshT.points[i].x, meshT.points[i].y
        );
    }

    std::vector<Vec3> qrswOnAirMesh;
    if (meshQrsw.points.size() == n && itQ != meshQrsw.pointVectors.end()) {
        qrswOnAirMesh = itQ->second;
    } else if (itQ != meshQrsw.pointVectors.end()) {
        qrswOnAirMesh.resize(n, Vec3::Zero());
        for (size_t i = 0; i < n; ++i) {
            double bestD2 = std::numeric_limits<double>::infinity();
            size_t best = 0;
            for (size_t j = 0; j < meshQrsw.points.size(); ++j) {
                double dx = meshQrsw.points[j].x - meshT.points[i].x;
                double dy = meshQrsw.points[j].y - meshT.points[i].y;
                double d2 = dx * dx + dy * dy;
                if (d2 < bestD2) {
                    bestD2 = d2;
                    best = j;
                }
            }
            if (best < itQ->second.size()) qrswOnAirMesh[i] = itQ->second[best];
        }
    } else if (itQCell != meshQrsw.cellVectors.end() && !meshQrsw.cells.empty()) {
        qrswOnAirMesh = remapCellVectorsToPoints2D(meshQrsw, itQCell->second, meshT.points);
    }

    Eigen::VectorXd qrswNorm = vectorMagnitudes(qrswOnAirMesh);
    Eigen::VectorXd denseTmrt = denseTumrtAvg;
    for (size_t i = 0; i < n; ++i) {
        double beta = 0.0;
        Vec3 sunPosVector = -qrswOnAirMesh[i];
        if (sunPosVector.norm() > 0.0 && sunPosVector.z() > 0.0) {
            double cosPhi = sunPosVector.z() / sunPosVector.norm();
            cosPhi = std::max(-1.0, std::min(1.0, cosPhi));
            beta = 90.0 - std::acos(cosPhi) * 180.0 / M_PI;
        }
        double fp = 0.308 * std::cos(beta * (1.0 - (beta * beta) / 48402.0) * M_PI / 180.0);
        double T4 = std::pow(denseTumrtAvg[i], 4)
                  + (fp * ABS_SW_PERSON * qrswNorm[i]) / (EPS_LW_PERSON * SIGMA);
        denseTmrt[i] = std::pow(std::max(0.0, T4), 0.25);
        if (denseTumrtAvg[i] < 1.0) denseTmrt[i] = 0.0;
    }

    Eigen::VectorXd denseRH = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd denseUtci = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < n; ++i) {
        double airT = itT->second[i];
        double airw = itW->second[i];
        double psat = std::exp(77.3450 + 0.0057 * airT - 7235.0 / airT) / std::pow(airT, 8.2);
        double pv = P_REF * airw / 0.62;
        denseRH[i] = std::min(100.0, std::max(0.0, pv / psat * 100.0));

        double va = itU->second[i].norm() / 0.667;
        double Ta_c = airT - 273.15;
        double Tmrt_c = denseTmrt[i] - 273.15;
        denseUtci[i] = utciSolver.calculate(Ta_c, va, denseRH[i], Tmrt_c);
    }

    writeLegacyVtkMesh(outDir + "/Tumrt_surface.vtk", meshT, {{"Tumrt", denseTumrtAvg}});
    writeLegacyVtkMesh(outDir + "/Tmrt_surface.vtk", meshT, {{"Tmrt", denseTmrt}});
    writeLegacyVtkMesh(outDir + "/RH_surface.vtk", meshT, {{"RH", denseRH}});
    if (debugWriteQrsw) {
        writeLegacyVtkMesh(outDir + "/qrsw_surface.vtk", meshT, {{"qrsw", qrswNorm}});
    }
    writeLegacyVtkMesh(outDir + "/UTCI_surface.vtk", meshT, {{"UTCI", denseUtci}, {"Tmrt", denseTmrt}});
    return true;
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

    std::cout << "Loading surface geometry from timestep " << geometryTimestep << "...\n";

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
        Eigen::VectorXd sparseQrswFromSurface;
    };
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
        // utci_clement: combined outgoing LW radiation (replaces T+qr)
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
    auto probeQrswAll = loadQrswProbeData(args.casePath);
    auto probePoints = loadProbePoints(probeDir + "/T");
    std::cout << "  Probe T rows: " << probeTAll.size()
              << "  U rows: " << probeUAll.size() << "\n";
    if (probeTAll.empty())
        std::cout << "  Warning: no probe T – using ambient temperature for UTCI\n";
    if (probeUAll.empty())
        std::cout << "  Warning: no probe U – using reference wind speed for UTCI\n";
    if (probeQrswAll.empty())
        std::cout << "  Warning: no probe qrsw data – using dense qrsw surface sampling fallback before binary shadow\n";
    else
        std::cout << "  Probe qrsw rows: " << probeQrswAll.size() << "\n";
    static const std::vector<double> emptyVec;

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
    cache.setCompressed(args.compressCache);
    if (args.compressCache && !BinaryCache::compressionAvailable()) {
        std::cerr << "Warning: --compress-cache requested but binary was built without ZLIB support; "
                     "cache files will be written uncompressed.\n";
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
        std::cout << "Loading UTCI LUT: " << lutFile << "\n";
        if (!utciSolver.loadLUT(lutFile)) {
            std::cerr << "Failed to load LUT — falling back to polynomial.\n";
            utciSolver.setMethod(UtciMethod::POLYNOMIAL);
        }
    }
    std::cout << "UTCI method: "
              << (utciSolver.method() == UtciMethod::LUT ? "LUT" : "polynomial") << "\n";

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
    std::vector<Eigen::VectorXd> allTumrtAvg(nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allUtci(nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allRH  (nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allTumrtNoSolar(nT, Eigen::VectorXd::Zero(nPos));
    std::vector<Eigen::VectorXd> allQlwSurfaces;
    std::vector<Eigen::VectorXd> allQlwSky;
    std::vector<Eigen::VectorXd> allQswSurfaces;
    std::vector<Eigen::VectorXd> allQswSky;
    std::vector<Eigen::VectorXd> allQswDirect;
    std::vector<Eigen::VectorXd> allQswGround;
    std::vector<Eigen::VectorXd> allQswElevatedDown;
    std::vector<Eigen::VectorXd> allQswVertical;
    std::vector<Eigen::VectorXd> allQswUpward;
    if (args.writeDebugTerms) {
        allQlwSurfaces.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQlwSky.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswSurfaces.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswSky.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswDirect.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswGround.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswElevatedDown.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswVertical.assign(nT, Eigen::VectorXd::Zero(nPos));
        allQswUpward.assign(nT, Eigen::VectorXd::Zero(nPos));
    }

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
            std::string cachePath = cache.getCachePath(positions[pedIdx].originalIndex);
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

            Eigen::VectorXd batchTmrt = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchUtci = Eigen::VectorXd::Zero(bN);
            Eigen::VectorXd batchRH   = Eigen::VectorXd::Zero(bN);
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
            if (args.writeDebugTerms) {
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
            const std::vector<double>& probeQrsw = probeQrswAll.empty() ? emptyVec
                : [&]() -> const std::vector<double>& {
                    const std::vector<double>* best = &probeQrswAll.front().second;
                    double bd = std::abs(probeQrswAll.front().first - tDouble);
                    for (auto& r : probeQrswAll) { double d=std::abs(r.first-tDouble); if(d<bd){bd=d;best=&r.second;} }
                    return *best;
                }();

            // Sun direction and IDN for this timestep
            const double sinBeta  = meteo.sunDir.z();
            const bool   hasSolar = (meteo.Idn > 0.0 && sinBeta > 0.0);
            double fp_solar = 0.0;
            if (hasSolar) {
                double betaDeg = std::asin(std::min(1.0, sinBeta)) * 180.0 / M_PI;
                // Projected-area factor for direct solar on a standing person.
                // Source: Fiala et al. (2012) / Bröde et al. (2012), eq. used in UTCI standard.
                // fp_solar = 0.308 * cos(β_rad * (1 − β_deg²/48402))
                fp_solar = 0.308 * std::cos(betaDeg * (1.0 - betaDeg*betaDeg/48402.0) * M_PI/180.0);
            }

            #pragma omp parallel for schedule(dynamic)
            for (size_t bi = 0; bi < bN; ++bi) {
                size_t pedIdx = bStart + bi;

                // --- Tmrt (LW + diffuse SW) ---
                TmrtBreakdown tmrtDetail = tmrtSolver.computeDetailed(
                    allSurfaces, skyGeo, batchVF[bi], meteo, args.useSkyViewFactors
                );
                double TumrtNoSolar = tmrtSolver.computeAreaWeightedAverage(
                    tmrtDetail.Tmrt, positions[pedIdx].areaVectors
                );
                double TumrtAvg = TumrtNoSolar;
                batchTumrtNoSolar[bi] = TumrtNoSolar;
                if (args.writeDebugTerms) {
                    batchQlwSurfaces[bi] = areaWeightedAverage(tmrtDetail.qlwSurfaces, positions[pedIdx].areaVectors);
                    batchQlwSky[bi] = areaWeightedAverage(tmrtDetail.qlwSky, positions[pedIdx].areaVectors);
                    batchQswSurfaces[bi] = areaWeightedAverage(tmrtDetail.qswSurfaces, positions[pedIdx].areaVectors);
                    batchQswSky[bi] = areaWeightedAverage(tmrtDetail.qswSky, positions[pedIdx].areaVectors);
                    batchQswGround[bi] = areaWeightedAverage(tmrtDetail.qswGround, positions[pedIdx].areaVectors);
                    batchQswElevatedDown[bi] = areaWeightedAverage(tmrtDetail.qswElevatedDown, positions[pedIdx].areaVectors);
                    batchQswVertical[bi] = areaWeightedAverage(tmrtDetail.qswVertical, positions[pedIdx].areaVectors);
                    batchQswUpward[bi] = areaWeightedAverage(tmrtDetail.qswUpward, positions[pedIdx].areaVectors);
                }

                // --- Direct solar component ---
                int probeIdx = (pedIdx < probeIndexForPos.size() && probeIndexForPos[pedIdx] >= 0)
                    ? probeIndexForPos[pedIdx]
                    : positions[pedIdx].originalIndex;
                double localQrsw = (probeIdx >= 0 && probeIdx < static_cast<int>(probeQrsw.size()))
                    ? probeQrsw[probeIdx] : -1.0;
                if (localQrsw < 0.0 && pedIdx < static_cast<size_t>(sc.sparseQrswFromSurface.size())) {
                    localQrsw = sc.sparseQrswFromSurface[pedIdx];
                }
                if (localQrsw >= 0.0) {
                    if (args.writeDebugTerms) batchQswDirect[bi] = fp_solar * localQrsw;
                    double T4 = std::pow(TumrtAvg, 4)
                              + fp_solar * ABS_SW_PERSON * localQrsw
                                / (EPS_LW_PERSON * SIGMA);
                    TumrtAvg = std::pow(std::max(0.0, T4), 0.25);
                } else if (hasSolar) {
                    Vec3 pedPos = positions[pedIdx].center.toVec3();
                    Vec3 target = pedPos + meteo.sunDir * (R_MAG_MAX * 0.9);
                    if (!raycaster.isBlocked(pedPos, target)) {
                        if (args.writeDebugTerms) batchQswDirect[bi] = fp_solar * meteo.Idn;
                        double T4 = std::pow(TumrtAvg, 4)
                                    + fp_solar * ABS_SW_PERSON * meteo.Idn
                                      / (EPS_LW_PERSON * SIGMA);
                        TumrtAvg = std::pow(std::max(0.0, T4), 0.25);
                    }
                }
                allTumrtAvg[tIdx][pedIdx] = TumrtAvg;
                batchTmrt[bi] = TumrtAvg;

                // --- Per-position UTCI inputs ---
                double Ta_K = (probeIdx < static_cast<int>(probeT.size())) ? probeT[probeIdx] : meteo.Ta;
                if (Ta_K < 200.0) Ta_K = meteo.Ta;  // guard against bad probe values
                double Ta_c = Ta_K - 273.15;
                // Convert pedestrian-height CFD wind speed to 10 m reference height.
                // Factor 0.667 ≈ u(z_ped)/u(10m) for a log profile with z0≈0.1 m, z_ped=2 m
                // (same convention as urbanMicroclimateFoam / utci_clement).
                double va   = (probeIdx < static_cast<int>(probeU.size())) ? probeU[probeIdx] / 0.667 : meteo.va;
                va = std::max(0.5, std::min(17.0, va));  // UTCI polynomial valid range

                // RH from specific humidity (original Python formula)
                double w   = (probeIdx < static_cast<int>(probeW.size())) ? probeW[probeIdx] : 0.01;
                double psat = std::exp(77.345 + 0.0057*Ta_K - 7235.0/Ta_K)
                              / std::pow(Ta_K, 8.2);
                double pv   = P_REF * w / (EPSILON_H2O + w);
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
                allTumrtNoSolar[tIdx][bStart + bi] = batchTumrtNoSolar[bi];
                if (args.writeDebugTerms) {
                    allQlwSurfaces[tIdx][bStart + bi] = batchQlwSurfaces[bi];
                    allQlwSky[tIdx][bStart + bi] = batchQlwSky[bi];
                    allQswSurfaces[tIdx][bStart + bi] = batchQswSurfaces[bi];
                    allQswSky[tIdx][bStart + bi] = batchQswSky[bi];
                    allQswDirect[tIdx][bStart + bi] = batchQswDirect[bi];
                    allQswGround[tIdx][bStart + bi] = batchQswGround[bi];
                    allQswElevatedDown[tIdx][bStart + bi] = batchQswElevatedDown[bi];
                    allQswVertical[tIdx][bStart + bi] = batchQswVertical[bi];
                    allQswUpward[tIdx][bStart + bi] = batchQswUpward[bi];
                }
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
        // Match Clement: TumrtAvg is the sparse pre-solar field used for dense interpolation.
        writeTumrtAvg(outDir + "/TumrtAvg", positions, t, allTumrtNoSolar[tIdx], false);
        if (args.writeDebugTerms) {
            writeTumrtTerms(outDir + "/TumrtAvg_terms", positions, t,
                            allTumrtNoSolar[tIdx], allTumrtAvg[tIdx],
                            allQlwSurfaces[tIdx], allQlwSky[tIdx],
                            allQswSurfaces[tIdx], allQswSky[tIdx], allQswDirect[tIdx],
                            allQswGround[tIdx], allQswElevatedDown[tIdx],
                            allQswVertical[tIdx], allQswUpward[tIdx]);
        }

        // RH_pedestrian.vtk — point cloud
        writeVtkPolyData(outDir + "/RH_pedestrian.vtk", positions, allRH[tIdx], "RH");

        // UTCI.vtk  (Tmrt in °C + UTCI in °C) — point cloud
        // UTCI_surface.vtk   — interpolated structured grid surface
        if (args.computeUtci) {
            Eigen::VectorXd TmrtC = allTmrt[tIdx].array() - 273.15;
            writeVtkMultiScalar(outDir + "/UTCI.vtk", positions,
                { {"Tmrt", TmrtC}, {"UTCI", allUtci[tIdx]} });
        }
        computeDenseSurfaceOutputs(args.casePath, outDir, t, positions, allTumrtNoSolar[tIdx],
                                   utciSolver, args.writeDebugQrswSurface);

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
