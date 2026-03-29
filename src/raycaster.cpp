#include "raycaster.h"
#include "constants.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace utci {

// --------------------------------------------------------------------------
// Möller–Trumbore ray-triangle intersection
// Returns true if ray [orig, orig+t*dir] (t in (eps, tmax)) hits triangle.
// --------------------------------------------------------------------------
static bool rayTriangle(const double* orig, const double* dir, double tmax,
                         const double* v0, const double* v1, const double* v2)
{
    const double EPS = 1e-9;
    double e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    double e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    double h[3]  = {dir[1]*e2[2]-dir[2]*e2[1],
                    dir[2]*e2[0]-dir[0]*e2[2],
                    dir[0]*e2[1]-dir[1]*e2[0]};
    double a = e1[0]*h[0]+e1[1]*h[1]+e1[2]*h[2];
    if (std::abs(a) < EPS) return false;
    double f = 1.0/a;
    double s[3] = {orig[0]-v0[0], orig[1]-v0[1], orig[2]-v0[2]};
    double u = f*(s[0]*h[0]+s[1]*h[1]+s[2]*h[2]);
    if (u < 0.0 || u > 1.0) return false;
    double q[3] = {s[1]*e1[2]-s[2]*e1[1],
                   s[2]*e1[0]-s[0]*e1[2],
                   s[0]*e1[1]-s[1]*e1[0]};
    double v = f*(dir[0]*q[0]+dir[1]*q[1]+dir[2]*q[2]);
    if (v < 0.0 || u+v > 1.0) return false;
    double t = f*(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2]);
    return (t > EPS && t < tmax);
}

// --------------------------------------------------------------------------
// Uniform 3-D grid spatial index for fast ray-AABB traversal
// --------------------------------------------------------------------------
struct GridMesh {
    struct Tri {
        double v[3][3];   // 3 vertices × 3 coords
        double aabbMin[3], aabbMax[3];
    };

    std::vector<Tri>                tris;
    std::vector<std::vector<int>>   cells;  // per-cell triangle lists

    double origin[3];
    double cellSize;
    int    nx, ny, nz;

    bool empty() const { return tris.empty(); }

    void build(const std::string& stlPath, double cs = 5.0) {
        // Read binary STL
        std::ifstream f(stlPath, std::ios::binary);
        if (!f.is_open()) return;

        char header[80];
        f.read(header, 80);
        uint32_t nTri = 0;
        f.read(reinterpret_cast<char*>(&nTri), 4);

        tris.reserve(nTri);
        double mn[3] = {1e18, 1e18, 1e18};
        double mx[3] = {-1e18,-1e18,-1e18};

        for (uint32_t i = 0; i < nTri; ++i) {
            float buf[12];  // normal(3) + v0(3) + v1(3) + v2(3)
            f.read(reinterpret_cast<char*>(buf), 48);
            uint16_t attr; f.read(reinterpret_cast<char*>(&attr), 2);

            Tri tri;
            for (int k = 0; k < 3; ++k) {
                tri.v[0][k] = buf[3+k];
                tri.v[1][k] = buf[6+k];
                tri.v[2][k] = buf[9+k];
                tri.aabbMin[k] = std::min({tri.v[0][k], tri.v[1][k], tri.v[2][k]});
                tri.aabbMax[k] = std::max({tri.v[0][k], tri.v[1][k], tri.v[2][k]});
                mn[k] = std::min(mn[k], tri.aabbMin[k]);
                mx[k] = std::max(mx[k], tri.aabbMax[k]);
            }
            tris.push_back(tri);
        }

        if (tris.empty()) return;

        cellSize = cs;
        for (int k = 0; k < 3; ++k) origin[k] = mn[k] - cs;
        nx = static_cast<int>((mx[0]-mn[0]+2*cs)/cs) + 1;
        ny = static_cast<int>((mx[1]-mn[1]+2*cs)/cs) + 1;
        nz = static_cast<int>((mx[2]-mn[2]+2*cs)/cs) + 1;
        if ((long long)nx * ny * nz > 50'000'000LL)
            throw std::runtime_error("BVH grid too large (" + std::to_string(nx) + "x"
                + std::to_string(ny) + "x" + std::to_string(nz)
                + ") — reduce domain size or increase cell size");
        cells.resize((size_t)nx * ny * nz);

        for (int i = 0; i < (int)tris.size(); ++i) {
            const Tri& t = tris[i];
            int x0 = std::max(0,(int)((t.aabbMin[0]-origin[0])/cs));
            int x1 = std::min(nx-1,(int)((t.aabbMax[0]-origin[0])/cs));
            int y0 = std::max(0,(int)((t.aabbMin[1]-origin[1])/cs));
            int y1 = std::min(ny-1,(int)((t.aabbMax[1]-origin[1])/cs));
            int z0 = std::max(0,(int)((t.aabbMin[2]-origin[2])/cs));
            int z1 = std::min(nz-1,(int)((t.aabbMax[2]-origin[2])/cs));
            for (int xi=x0;xi<=x1;++xi)
            for (int yi=y0;yi<=y1;++yi)
            for (int zi=z0;zi<=z1;++zi)
                cells[xi*ny*nz + yi*nz + zi].push_back(i);
        }
    }

    // DDA ray traversal: test only triangles in cells the ray passes through
    bool isBlocked(const double* p1, const double* p2, bool enforceRangeLimit) const {
        if (tris.empty()) return false;

        double dir[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
        double len = std::sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
        if (len < 1e-12) return false;
        if (enforceRangeLimit && len >= R_MAG_MAX) return false;  // beyond range → assume unblocked
        double inv = 1.0/len;
        double d[3] = {dir[0]*inv, dir[1]*inv, dir[2]*inv};

        // Match the Clement workflow: trim each ray to the 3%..97% segment.
        double startOffset = len * 0.03;
        double s[3] = {
            p1[0] + d[0] * startOffset,
            p1[1] + d[1] * startOffset,
            p1[2] + d[2] * startOffset
        };
        double tmax = len * 0.94;
        if (tmax <= 0) return false;

        // Starting grid cell
        auto cellIdx = [&](double x, int dim) -> int {
            return std::max(0,std::min(
                (dim==0?nx:(dim==1?ny:nz))-1,
                (int)((x-origin[dim])/cellSize)));
        };
        int cx = cellIdx(s[0],0), cy = cellIdx(s[1],1), cz = cellIdx(s[2],2);

        // DDA step sizes
        double tDelta[3], tNext[3];
        int step[3];
        for (int k = 0; k < 3; ++k) {
            double dimOrigin = (k==0?origin[0]:(k==1?origin[1]:origin[2]));
            int    dimSize   = (k==0?cx:(k==1?cy:cz));
            if (std::abs(d[k]) < 1e-12) {
                tDelta[k] = 1e18; tNext[k] = 1e18; step[k] = 0;
            } else if (d[k] > 0) {
                step[k] = 1;
                tDelta[k] = cellSize/d[k];
                tNext[k]  = ((dimSize+1)*cellSize - (s[k]-dimOrigin)) / d[k];
            } else {
                step[k] = -1;
                tDelta[k] = -cellSize/d[k];
                tNext[k]  = (dimSize*cellSize - (s[k]-dimOrigin)) / d[k];
            }
        }

        // Track which triangles we've already tested
        static thread_local std::vector<bool> tested;
        if ((int)tested.size() < (int)tris.size()) tested.assign(tris.size(), false);
        std::vector<int> toReset;

        bool hit = false;
        while (!hit) {
            // Test triangles in current cell
            int idx = cx*ny*nz + cy*nz + cz;
            if (idx >= 0 && idx < (int)cells.size()) {
                for (int ti : cells[idx]) {
                    if (!tested[ti]) {
                        tested[ti] = true;
                        toReset.push_back(ti);
                        if (rayTriangle(s, d, tmax, tris[ti].v[0], tris[ti].v[1], tris[ti].v[2])) {
                            hit = true; break;
                        }
                    }
                }
            }
            if (hit) break;

            // Advance DDA
            if (tNext[0] < tNext[1] && tNext[0] < tNext[2]) {
                if (tNext[0] > tmax) break;
                cx += step[0]; tNext[0] += tDelta[0];
            } else if (tNext[1] < tNext[2]) {
                if (tNext[1] > tmax) break;
                cy += step[1]; tNext[1] += tDelta[1];
            } else {
                if (tNext[2] > tmax) break;
                cz += step[2]; tNext[2] += tDelta[2];
            }
            if (cx<0||cx>=nx||cy<0||cy>=ny||cz<0||cz>=nz) break;
        }

        // Reset tested flags
        for (int ti : toReset) tested[ti] = false;
        return hit;
    }
};

struct Raycaster::Impl {
    GridMesh wallMesh;
    GridMesh vegMesh;
    bool wallsLoaded = false;
    bool vegLoaded   = false;

    static bool fileExists(const std::string& path) {
        std::ifstream f(path);
        return f.good();
    }
};

Raycaster::Raycaster() : pImpl(std::make_unique<Impl>()) {}
Raycaster::~Raycaster() = default;

void Raycaster::setNumThreads(int) { /* thread-safe; no action needed */ }

bool Raycaster::loadGeometry(const std::string& stlPath) {
    if (stlPath.empty() || !Impl::fileExists(stlPath)) {
        std::cout << "  Warning: Could not load wall geometry: " << stlPath << std::endl;
        return false;
    }
    pImpl->wallMesh.build(stlPath);
    pImpl->wallsLoaded = !pImpl->wallMesh.empty();
    if (pImpl->wallsLoaded) {
        std::cout << "  Loaded wall geometry: " << stlPath
                  << " (" << pImpl->wallMesh.tris.size() << " triangles, grid "
                  << pImpl->wallMesh.nx << "x" << pImpl->wallMesh.ny
                  << "x" << pImpl->wallMesh.nz << ")\n";
    } else {
        std::cout << "  Warning: Could not load wall geometry: " << stlPath << std::endl;
    }
    return pImpl->wallsLoaded;
}

void Raycaster::loadVegetation(const std::string& stlPath) {
    if (stlPath.empty() || !Impl::fileExists(stlPath)) return;
    pImpl->vegMesh.build(stlPath);
    pImpl->vegLoaded = !pImpl->vegMesh.empty();
    if (pImpl->vegLoaded) {
        std::cout << "  Loaded vegetation geometry: " << stlPath << std::endl;
    }
}

bool Raycaster::isBlocked(const Vec3& start, const Vec3& end, bool enforceRangeLimit) const {
    double s[3] = {start.x(), start.y(), start.z()};
    double e[3] = {end.x(),   end.y(),   end.z()};
    if (pImpl->wallMesh.isBlocked(s, e, enforceRangeLimit)) return true;
    if (pImpl->vegLoaded && pImpl->vegMesh.isBlocked(s, e, enforceRangeLimit)) return true;
    return false;
}


bool Raycaster::isValid() const { return pImpl->wallsLoaded; }

}
