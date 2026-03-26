#include "viewFactor.h"
#include "raycaster.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

namespace utci {

// ---------------------------------------------------------------------------
// Patch quadrature helper
// ---------------------------------------------------------------------------
// For large patches (side > r/2) the single-centre differential approximation
// introduces spatial oscillations in Fijsum because cosTI varies strongly
// across the patch face.  We replace it with an N×N quadrature that samples
// the patch uniformly, each sample having area A/N².  The cache still stores
// ONE Fij value per original patch — the quadrature sum — so cache size and
// the tmrtSolver interface are unchanged.
//
// N is chosen so sub-cell side ≤ r/2:
//   N = clamp( ceil(2·√A / r), 1, N_MAX )
// N is chosen so that sub-cell side = √A/N ≤ r/2, i.e. A_sub/(πr²) ≤ 1/π < 0.32.
// For the differential formula to be accurate we need A_sub/(πr²) < 0.1, which
// is achieved when sub-cell side ≤ 0.56·r.  With r_min = pedestrian height ≈ 2 m
// and the largest observed patch side ≈ 27 m, this requires N ≈ 24.  Cap at 25
// (625 evaluations — still fast thanks to parallelism and the prefilter).
static constexpr int QUAD_N_MAX = 25;  // at most 25×25 = 625 sample points

static double computePatchFij(
    const Vec3& i_n,       // probe body point
    const Vec3& Ai_n,      // body area vector (for orientation check, not normalised)
    const Vec3& ni,        // unit body normal
    const Vec3& j_center,  // patch centre
    const Vec3& nj,        // unit patch normal (already loaded/normalised by caller)
    double AjMag)          // patch area
{
    Vec3 r0 = i_n - j_center;
    double r0Mag = r0.norm();
    if (r0Mag <= 0.0 || r0Mag >= R_MAG_MAX) return 0.0;
    if (Ai_n.dot(r0) >= 0.0) return 0.0;   // body faces away from patch centre

    // Determine quadrature order
    double patchSide = std::sqrt(AjMag);
    int N = (patchSide > r0Mag * 0.5)
            ? std::min(QUAD_N_MAX, (int)std::ceil(2.0 * patchSide / r0Mag))
            : 1;

    if (N == 1) {
        double cosTI = std::abs(ni.dot(r0)) / r0Mag;
        double cosTJ = std::abs(nj.dot(r0)) / r0Mag;
        double r2 = r0Mag * r0Mag;
        return cosTI * cosTJ * AjMag / (r2 * M_PI + AjMag);
    }

    // Build two unit tangent vectors in the patch plane
    Vec3 ref = (std::abs(nj.x()) < 0.9) ? Vec3(1,0,0) : Vec3(0,1,0);
    Vec3 t1 = (ref - nj * nj.dot(ref)).normalized();
    Vec3 t2 = nj.cross(t1).normalized();

    const double subArea   = AjMag / (N * N);
    const double cellSide  = patchSide / N;
    const double halfSpan  = patchSide * 0.5 - cellSide * 0.5;  // = cellSide*(N-1)/2

    // Use the DIFFERENTIAL formula (dA / π r²) at each sub-cell — not the
    // finite-area form.  The finite-area formula A/(πr²+A) is exact only for
    // a single coaxial disk; summing N² copies each with area A/N² gives a
    // total that exceeds 1 when A/N² >> πr².  The differential integral
    // ∫ cosTI·cosTJ·dA/(πr²) converges correctly because cosTI→0 for
    // horizontal body segments looking straight down, eliminating the 1/r²
    // singularity.
    double total = 0.0;
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            double di = ii * cellSide - halfSpan;
            double dj = jj * cellSide - halfSpan;
            Vec3 sub = j_center + t1 * di + t2 * dj;
            Vec3 r   = i_n - sub;
            double rMag = r.norm();
            if (rMag < 1e-4) continue;
            if (Ai_n.dot(r) >= 0.0) continue;
            double cosTI = std::abs(ni.dot(r)) / rMag;
            double cosTJ = std::abs(nj.dot(r)) / rMag;
            double r2    = rMag * rMag;
            total += cosTI * cosTJ * subArea / (r2 * M_PI);  // differential
        }
    }
    return total;
}

struct ViewFactorCalculator::Impl {
    const Raycaster& raycaster;

    Impl(const Raycaster& rc) : raycaster(rc) {}

    bool isRayBlocked(const Vec3& start, const Vec3& end) const {
        return raycaster.isBlocked(start, end);
    }
};

ViewFactorCalculator::ViewFactorCalculator(const Raycaster& raycaster)
    : pImpl(std::make_unique<Impl>(raycaster)) {}

ViewFactorCalculator::~ViewFactorCalculator() = default;

ViewFactorResult ViewFactorCalculator::compute(
    const PedestrianPosition& ped,
    const std::vector<SurfacePatch>& surfaces,
    const std::vector<SurfacePatch>& sky,
    bool useSky)
{
    ViewFactorResult result;

    const int nBody = 5;
    const int nSurf = static_cast<int>(surfaces.size());
    const int nSky  = useSky ? static_cast<int>(sky.size()) : 0;

    result.Fijsum    = Eigen::VectorXd::Zero(nBody);
    result.FijsumSky = Eigen::VectorXd::Zero(nBody);

    // Prefilter: collect surface indices within R_MAG_MAX of pedestrian center
    // Uses squared distance to avoid sqrt; reduces inner loop from ~400k to ~few thousand
    const double R2 = R_MAG_MAX * R_MAG_MAX;
    Vec3 pedCenter = ped.center.toVec3();

    std::vector<int> nearSurf;
    nearSurf.reserve(4096);
    for (int m = 0; m < nSurf; ++m) {
        if (surfaces[m].area < 1e-10) continue;
        double dx = pedCenter.x() - surfaces[m].center.x;
        double dy = pedCenter.y() - surfaces[m].center.y;
        double dz = pedCenter.z() - surfaces[m].center.z;
        if (dx*dx + dy*dy + dz*dz < R2)
            nearSurf.push_back(m);
    }

    // Sky patches have no range limit (sky dome can be far away) — include all valid ones
    std::vector<int> nearSky;
    if (useSky && nSky > 0) {
        nearSky.reserve(nSky);
        for (int m = 0; m < nSky; ++m) {
            if (sky[m].area >= 1e-10)
                nearSky.push_back(m);
        }
    }

    for (int n = 0; n < nBody; ++n) {
        Vec3 i_n = ped.bodyPoints[n].toVec3();
        Vec3 Ai_n = ped.areaVectors[n];
        double AiMag = Ai_n.norm();

        if (AiMag < 1e-10) continue;

        Vec3 ni = Ai_n / AiMag;

        for (int mi = 0; mi < (int)nearSurf.size(); ++mi) {
            int m = nearSurf[mi];
            const auto& surf = surfaces[m];
            Vec3 j_m = surf.center.toVec3();
            double AjMag = surf.area;

            // Quick distance / orientation pre-check using patch centre
            Vec3 r0 = i_n - j_m;
            double r0Mag2 = r0.squaredNorm();
            if (r0Mag2 <= 0.0 || r0Mag2 >= R_MAG_MAX * R_MAG_MAX) continue;
            if (Ai_n.dot(r0) >= 0) continue;

            // Ray occlusion: test at patch centre only
            if (pImpl->isRayBlocked(i_n, j_m)) continue;

            Vec3 nj = surf.areaVector / AjMag;

            // For large nearby patches, use quadrature sampling to avoid
            // Fijsum oscillations caused by the single-centre approximation.
            double Fij_val = computePatchFij(i_n, Ai_n, ni, j_m, nj, AjMag);
            if (Fij_val <= 0.0) continue;
            result.Fij[n].emplace_back(m, Fij_val);
            result.Fijsum[n] += Fij_val;
        }

        if (useSky && !nearSky.empty()) {
            for (int mi = 0; mi < (int)nearSky.size(); ++mi) {
                int m = nearSky[mi];
                const auto& surfSky = sky[m];
                Vec3 j_sky = surfSky.center.toVec3();
                double AjSkyMag = surfSky.area;

                Vec3 r0 = i_n - j_sky;
                if (r0.norm() <= 0.0) continue;
                if (Ai_n.dot(r0) >= 0) continue;

                // Ray occlusion: skip if blocked by geometry
                if (pImpl->isRayBlocked(i_n, j_sky)) continue;

                Vec3 nj = surfSky.areaVector / AjSkyMag;
                double FijSky_val = computePatchFij(i_n, Ai_n, ni, j_sky, nj, AjSkyMag);
                if (FijSky_val <= 0.0) continue;
                result.FijSky[n].emplace_back(m, FijSky_val);
                result.FijsumSky[n] += FijSky_val;
            }
        }
    }
    
    return result;
}



}
