#include "denseStage2.h"

#include "constants.h"
#include "io.h"
#include "logging.h"
#include "utciSolver.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace utci {

namespace {

} // namespace

std::string probeKey(const Point3& p) {
    auto q = [](double v) {
        return static_cast<long long>(std::llround(v * 1000.0));
    };
    return std::to_string(q(p.x)) + ":" + std::to_string(q(p.y)) + ":" + std::to_string(q(p.z));
}

namespace {

struct UniformGridField {
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> values;
    std::unordered_map<long long, size_t> indexByCell;
    bool valid = false;
};

struct DenseInterpPointPlan {
    bool useNearest = true;
    int nearestSparseIndex = -1;
    std::array<int, 16> stencil{};
    std::array<double, 16> weights{};
};

struct DenseInterpPlan {
    std::vector<double> xs;
    std::vector<double> ys;
    std::unordered_map<long long, int> sparseIndexByCell;
    std::vector<DenseInterpPointPlan> pointPlans;
    bool valid = false;
};

enum class QrswRemapMode {
    None,
    DirectPoint,
    NearestPoint,
    CellToPoint
};

struct QrswRemapPlan {
    QrswRemapMode mode = QrswRemapMode::None;
    std::vector<int> indices;
};

static constexpr double DENSE_INTERP_BBOX_INSET = 1.0;

static long long gridKey(int ix, int iy) {
    return (static_cast<long long>(ix) << 32) ^
           static_cast<unsigned int>(iy);
}

static std::string makePlanKey(const std::vector<PedestrianPosition>& sparsePositions,
                               const std::vector<Point3>& densePoints) {
    auto appendPoint = [](std::ostringstream& oss, const Point3& p) {
        oss << std::fixed << std::setprecision(4)
            << p.x << "," << p.y << "," << p.z;
    };
    std::ostringstream oss;
    oss << sparsePositions.size() << "|";
    if (!sparsePositions.empty()) {
        appendPoint(oss, sparsePositions.front().center);
        oss << "|";
        appendPoint(oss, sparsePositions.back().center);
    }
    oss << "|" << densePoints.size() << "|";
    if (!densePoints.empty()) {
        appendPoint(oss, densePoints.front());
        oss << "|";
        appendPoint(oss, densePoints.back());
    }
    return oss.str();
}

static std::string makeQrswPlanKey(const VtkMeshData& meshQrsw,
                                   bool hasPointQrsw,
                                   bool hasCellQrsw,
                                   const std::vector<Point3>& densePoints) {
    auto appendPoint = [](std::ostringstream& oss, const Point3& p) {
        oss << std::fixed << std::setprecision(4)
            << p.x << "," << p.y << "," << p.z;
    };
    std::ostringstream oss;
    oss << static_cast<int>(hasPointQrsw) << "|"
        << static_cast<int>(hasCellQrsw) << "|"
        << meshQrsw.points.size() << "|" << meshQrsw.cells.size();
    if (!meshQrsw.points.empty()) {
        oss << "|";
        appendPoint(oss, meshQrsw.points.front());
        oss << "|";
        appendPoint(oss, meshQrsw.points.back());
    }
    if (!meshQrsw.cells.empty()) {
        oss << "|" << meshQrsw.cells.front().size()
            << "|" << meshQrsw.cells.back().size();
    }
    oss << "|" << densePoints.size();
    if (!densePoints.empty()) {
        oss << "|";
        appendPoint(oss, densePoints.front());
        oss << "|";
        appendPoint(oss, densePoints.back());
    }
    return oss.str();
}

static std::array<double, 4> cubicWeights1D(double t) {
    return {
        -0.5 * t + t * t - 0.5 * t * t * t,
         1.0 - 2.5 * t * t + 1.5 * t * t * t,
         0.5 * t + 2.0 * t * t - 1.5 * t * t * t,
        -0.5 * t * t + 0.5 * t * t * t
    };
}

static DenseInterpPlan buildDenseInterpPlan(const std::vector<PedestrianPosition>& positions,
                                            const std::vector<Point3>& densePoints) {
    DenseInterpPlan plan;
    plan.xs.reserve(positions.size());
    plan.ys.reserve(positions.size());
    for (const auto& p : positions) {
        plan.xs.push_back(p.center.x);
        plan.ys.push_back(p.center.y);
    }
    auto dedupe = [](std::vector<double>& coords) {
        std::sort(coords.begin(), coords.end());
        coords.erase(std::unique(coords.begin(), coords.end(),
            [](double a, double b) { return std::abs(a - b) < 1e-4; }),
            coords.end());
    };
    dedupe(plan.xs);
    dedupe(plan.ys);
    if (plan.xs.size() < 2 || plan.ys.size() < 2) return plan;

    for (size_t i = 0; i < positions.size(); ++i) {
        auto xit = std::lower_bound(plan.xs.begin(), plan.xs.end(), positions[i].center.x - 1e-4);
        auto yit = std::lower_bound(plan.ys.begin(), plan.ys.end(), positions[i].center.y - 1e-4);
        if (xit == plan.xs.end() || yit == plan.ys.end()) continue;
        int ix = static_cast<int>(xit - plan.xs.begin());
        int iy = static_cast<int>(yit - plan.ys.begin());
        plan.sparseIndexByCell[gridKey(ix, iy)] = static_cast<int>(i);
    }

    plan.pointPlans.resize(densePoints.size());
    for (size_t i = 0; i < densePoints.size(); ++i) {
        const double x = densePoints[i].x;
        const double y = densePoints[i].y;
        DenseInterpPointPlan pointPlan;

        double bestD2 = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < positions.size(); ++j) {
            double dx = positions[j].center.x - x;
            double dy = positions[j].center.y - y;
            double d2 = dx * dx + dy * dy;
            if (d2 < bestD2) {
                bestD2 = d2;
                pointPlan.nearestSparseIndex = static_cast<int>(j);
            }
        }

        if (x < plan.xs.front() || x > plan.xs.back() || y < plan.ys.front() || y > plan.ys.back() ||
            x < plan.xs.front() + DENSE_INTERP_BBOX_INSET || x > plan.xs.back() - DENSE_INTERP_BBOX_INSET ||
            y < plan.ys.front() + DENSE_INTERP_BBOX_INSET || y > plan.ys.back() - DENSE_INTERP_BBOX_INSET) {
            plan.pointPlans[i] = pointPlan;
            continue;
        }

        auto xhi = std::lower_bound(plan.xs.begin(), plan.xs.end(), x);
        auto yhi = std::lower_bound(plan.ys.begin(), plan.ys.end(), y);
        if (xhi == plan.xs.begin() || yhi == plan.ys.begin() ||
            xhi == plan.xs.end() || yhi == plan.ys.end()) {
            plan.pointPlans[i] = pointPlan;
            continue;
        }

        int ix1 = static_cast<int>(xhi - plan.xs.begin());
        int iy1 = static_cast<int>(yhi - plan.ys.begin());
        int ix0 = ix1 - 1;
        int iy0 = iy1 - 1;
        int ixm1 = ix0 - 1;
        int ix2 = ix1 + 1;
        int iym1 = iy0 - 1;
        int iy2 = iy1 + 1;
        if (ixm1 < 0 || iym1 < 0 ||
            ix2 >= static_cast<int>(plan.xs.size()) ||
            iy2 >= static_cast<int>(plan.ys.size())) {
            plan.pointPlans[i] = pointPlan;
            continue;
        }

        const int yIdx[4] = {iym1, iy0, iy1, iy2};
        const int xIdx[4] = {ixm1, ix0, ix1, ix2};
        bool stencilOk = true;
        int pos = 0;
        for (int ry = 0; ry < 4 && stencilOk; ++ry) {
            for (int rx = 0; rx < 4; ++rx, ++pos) {
                auto it = plan.sparseIndexByCell.find(gridKey(xIdx[rx], yIdx[ry]));
                if (it == plan.sparseIndexByCell.end()) {
                    stencilOk = false;
                    break;
                }
                pointPlan.stencil[pos] = it->second;
            }
        }
        if (!stencilOk) {
            plan.pointPlans[i] = pointPlan;
            continue;
        }

        double x0 = plan.xs[ix0];
        double x1 = plan.xs[ix1];
        double y0 = plan.ys[iy0];
        double y1 = plan.ys[iy1];
        const double tx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
        const double ty = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;
        const auto wx = cubicWeights1D(tx);
        const auto wy = cubicWeights1D(ty);
        pos = 0;
        for (int ry = 0; ry < 4; ++ry) {
            for (int rx = 0; rx < 4; ++rx, ++pos) {
                pointPlan.weights[pos] = wy[ry] * wx[rx];
            }
        }
        pointPlan.useNearest = false;
        plan.pointPlans[i] = pointPlan;
    }

    plan.valid = true;
    return plan;
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

static Eigen::VectorXd interpolateSparseTumrtWithPlan(const DenseInterpPlan& plan,
                                                      const Eigen::VectorXd& values,
                                                      DenseInterpClampMode clampMode) {
    Eigen::VectorXd out(plan.pointPlans.size());
    for (size_t i = 0; i < plan.pointPlans.size(); ++i) {
        const auto& pp = plan.pointPlans[i];
        if (pp.useNearest) {
            out[i] = (pp.nearestSparseIndex >= 0 && pp.nearestSparseIndex < values.size())
                ? values[pp.nearestSparseIndex] : 0.0;
            continue;
        }
        double interp = 0.0;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < 16; ++k) {
            const double v = values[pp.stencil[k]];
            interp += pp.weights[k] * v;
            if (clampMode == DenseInterpClampMode::LocalRange) {
                localMin = std::min(localMin, v);
                localMax = std::max(localMax, v);
            }
        }
        if (clampMode == DenseInterpClampMode::LocalRange) {
            interp = std::max(localMin, std::min(localMax, interp));
        }
        out[i] = interp;
    }
    return out;
}

static Eigen::VectorXd vectorMagnitudes(const std::vector<Vec3>& vecs) {
    Eigen::VectorXd out(vecs.size());
    for (size_t i = 0; i < vecs.size(); ++i) out[i] = vecs[i].norm();
    return out;
}

std::string firstExistingPathImpl(const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return "";
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
                             double y);

static int findBestCell2D(const CellLocator2D& loc,
                          const VtkMeshData& srcMesh,
                          double x,
                          double y) {
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

    return bestCell;
}

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
        const int bestCell = findBestCell2D(loc, srcMesh, dstPoints[i].x, dstPoints[i].y);
        if (bestCell >= 0) out[i] = cellVectors[bestCell];
    }
    return out;
}

static QrswRemapPlan buildQrswRemapPlan(const VtkMeshData& meshQrsw,
                                        bool hasPointQrsw,
                                        bool hasCellQrsw,
                                        const std::vector<Point3>& dstPoints) {
    QrswRemapPlan plan;
    if (hasPointQrsw && meshQrsw.points.size() == dstPoints.size()) {
        plan.mode = QrswRemapMode::DirectPoint;
        return plan;
    }

    plan.indices.resize(dstPoints.size(), -1);
    if (hasCellQrsw && !meshQrsw.cells.empty()) {
        plan.mode = QrswRemapMode::CellToPoint;
        const CellLocator2D loc = buildCellLocator2D(meshQrsw);
        for (size_t i = 0; i < dstPoints.size(); ++i) {
            plan.indices[i] = findBestCell2D(loc, meshQrsw, dstPoints[i].x, dstPoints[i].y);
        }
        return plan;
    }

    if (hasPointQrsw && !meshQrsw.points.empty()) {
        plan.mode = QrswRemapMode::NearestPoint;
        for (size_t i = 0; i < dstPoints.size(); ++i) {
            double bestD2 = std::numeric_limits<double>::infinity();
            int best = -1;
            for (size_t j = 0; j < meshQrsw.points.size(); ++j) {
                const double dx = meshQrsw.points[j].x - dstPoints[i].x;
                const double dy = meshQrsw.points[j].y - dstPoints[i].y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < bestD2) {
                    bestD2 = d2;
                    best = static_cast<int>(j);
                }
            }
            plan.indices[i] = best;
        }
    }

    return plan;
}

static std::vector<Vec3> remapQrswWithPlan(const QrswRemapPlan& plan,
                                           const std::vector<Vec3>* pointVectors,
                                           const std::vector<Vec3>* cellVectors,
                                           size_t dstSize) {
    if (plan.mode == QrswRemapMode::DirectPoint && pointVectors != nullptr) {
        return *pointVectors;
    }

    std::vector<Vec3> out(dstSize, Vec3::Zero());
    const std::vector<Vec3>* src = nullptr;
    if (plan.mode == QrswRemapMode::NearestPoint) src = pointVectors;
    if (plan.mode == QrswRemapMode::CellToPoint) src = cellVectors;
    if (src == nullptr) return out;

    for (size_t i = 0; i < dstSize && i < plan.indices.size(); ++i) {
        const int idx = plan.indices[i];
        if (idx >= 0 && idx < static_cast<int>(src->size())) out[i] = (*src)[idx];
    }
    return out;
}

} // namespace

Eigen::VectorXd remapQrswMagnitudesToPoints(const VtkMeshData& meshQrsw,
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

std::string firstExistingPath(const std::vector<std::string>& paths) {
    return firstExistingPathImpl(paths);
}

bool computeDenseSurfaceOutputs(const std::string& casePath,
                                const std::string& outDir,
                                int timestep,
                                const std::vector<PedestrianPosition>& sparsePositions,
                                const Eigen::VectorXd& sparseTumrtAvg,
                                UtciSolver& utciSolver,
                                bool debugWriteQrsw,
                                DenseInterpClampMode clampMode) {
    const std::string surfaceDir = casePath + "/postProcessing/surfaces/" + std::to_string(timestep);
    const std::string airDir = casePath + "/postProcessing/surfacesPedestrianAir/" + std::to_string(timestep);
    const std::string radDir = casePath + "/postProcessing/surfacesPedestrianRad/" + std::to_string(timestep);
    const std::string meshPath = firstExistingPathImpl({
        surfaceDir + "/T_pedestrian.vtk",
        airDir + "/T_pedestrian.vtk"
    });
    const std::string qrswPath = firstExistingPathImpl({
        surfaceDir + "/qrsw_pedestrian.vtk",
        radDir + "/qrsw_pedestrian.vtk"
    });
    const std::string uPath = firstExistingPathImpl({
        surfaceDir + "/U_pedestrian.vtk",
        airDir + "/U_pedestrian.vtk"
    });
    const std::string wPath = firstExistingPathImpl({
        surfaceDir + "/w_pedestrian.vtk",
        airDir + "/w_pedestrian.vtk"
    });

    VtkMeshData meshT, meshQrsw, meshU, meshW;
    if (meshPath.empty() || !readLegacyVtkMesh(meshPath, meshT)) {
        logWarn("Dense Stage 2 skipped for t=" + std::to_string(timestep) +
                " (missing dense T_pedestrian.vtk)");
        return false;
    }
    if (qrswPath.empty() || uPath.empty() || wPath.empty() ||
        !readLegacyVtkMesh(qrswPath, meshQrsw) ||
        !readLegacyVtkMesh(uPath, meshU) ||
        !readLegacyVtkMesh(wPath, meshW)) {
        logWarn("Dense Stage 2 skipped for t=" + std::to_string(timestep) +
                " (missing dense qrsw/U/w fields)");
        return false;
    }

    auto itT = meshT.pointScalars.find("T");
    auto itW = meshW.pointScalars.find("w");
    auto itU = meshU.pointVectors.find("U");
    auto itQ = meshQrsw.pointVectors.find("qrsw");
    auto itQCell = meshQrsw.cellVectors.find("qrsw");
    if (itT == meshT.pointScalars.end() || itW == meshW.pointScalars.end() ||
        itU == meshU.pointVectors.end() ||
        (itQ == meshQrsw.pointVectors.end() && itQCell == meshQrsw.cellVectors.end())) {
        logWarn("Dense Stage 2 skipped for t=" + std::to_string(timestep) +
                " (expected T/U/w/qrsw arrays not found)");
        return false;
    }

    const size_t n = meshT.points.size();
    if (itT->second.size() != static_cast<int>(n) ||
        itW->second.size() != static_cast<int>(n) ||
        itU->second.size() != n) {
        logWarn("Dense Stage 2 skipped for t=" + std::to_string(timestep) +
                " (dense T/U/w arrays do not align with T mesh)");
        return false;
    }

    static std::mutex interpPlanMutex;
    static std::unordered_map<std::string, std::shared_ptr<const DenseInterpPlan>> interpPlanCache;
    const std::string interpKey = makePlanKey(sparsePositions, meshT.points);
    std::shared_ptr<const DenseInterpPlan> interpPlan;
    {
        std::lock_guard<std::mutex> lock(interpPlanMutex);
        auto it = interpPlanCache.find(interpKey);
        if (it == interpPlanCache.end()) {
            it = interpPlanCache.emplace(
                interpKey,
                std::make_shared<DenseInterpPlan>(buildDenseInterpPlan(sparsePositions, meshT.points))
            ).first;
        }
        interpPlan = it->second;
    }
    Eigen::VectorXd denseTumrtAvg = interpolateSparseTumrtWithPlan(*interpPlan, sparseTumrtAvg, clampMode);

    static std::mutex qrswPlanMutex;
    static std::unordered_map<std::string, std::shared_ptr<const QrswRemapPlan>> qrswPlanCache;
    const std::string qrswKey = makeQrswPlanKey(
        meshQrsw,
        itQ != meshQrsw.pointVectors.end(),
        itQCell != meshQrsw.cellVectors.end(),
        meshT.points);
    std::shared_ptr<const QrswRemapPlan> qrswPlan;
    {
        std::lock_guard<std::mutex> lock(qrswPlanMutex);
        auto it = qrswPlanCache.find(qrswKey);
        if (it == qrswPlanCache.end()) {
            it = qrswPlanCache.emplace(
                qrswKey,
                std::make_shared<QrswRemapPlan>(buildQrswRemapPlan(
                    meshQrsw,
                    itQ != meshQrsw.pointVectors.end(),
                    itQCell != meshQrsw.cellVectors.end(),
                    meshT.points))
            ).first;
        }
        qrswPlan = it->second;
    }

    std::vector<Vec3> qrswOnAirMesh = remapQrswWithPlan(
        *qrswPlan,
        itQ != meshQrsw.pointVectors.end() ? &itQ->second : nullptr,
        itQCell != meshQrsw.cellVectors.end() ? &itQCell->second : nullptr,
        n);

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
        double pv = P_REF * airw / (EPSILON_H2O + airw);
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

} // namespace utci
