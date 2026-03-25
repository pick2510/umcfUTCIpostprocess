#include "viewFactor.h"
#include "raycaster.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

namespace utci {

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
            Vec3 Aj_m = surf.areaVector;
            double AjMag = surf.area;

            Vec3 r = i_n - j_m;
            double rMag = r.norm();

            if (!(rMag > 0.0 && rMag < R_MAG_MAX)) continue;

            // Check pedestrian surface faces toward wall surface (single-sided, matching Python reference)
            if (Ai_n.dot(r) >= 0) continue;

            // Ray occlusion: skip if blocked by geometry
            if (pImpl->isRayBlocked(i_n, j_m)) continue;

            Vec3 nj = Aj_m / AjMag;
            double cosTI = std::abs(ni.dot(r)) / rMag;
            double cosTJ = std::abs(nj.dot(r)) / rMag;

            double Fij_val = cosTI * cosTJ * AjMag / (rMag * rMag * M_PI);
            result.Fij[n].emplace_back(m, Fij_val);
            result.Fijsum[n] += Fij_val;
        }

        if (useSky && !nearSky.empty()) {
            for (int mi = 0; mi < (int)nearSky.size(); ++mi) {
                int m = nearSky[mi];
                const auto& surfSky = sky[m];
                Vec3 j_sky = surfSky.center.toVec3();
                Vec3 Aj_sky = surfSky.areaVector;
                double AjSkyMag = surfSky.area;

                Vec3 r = i_n - j_sky;
                double rMag = r.norm();

                if (rMag <= 0.0) continue;

                // Check pedestrian surface faces toward sky patch
                if (Ai_n.dot(r) >= 0) continue;

                // Ray occlusion: skip if blocked by geometry
                if (pImpl->isRayBlocked(i_n, j_sky)) continue;

                Vec3 nj = Aj_sky / AjSkyMag;
                double cosTI = std::abs(ni.dot(r)) / rMag;
                double cosTJ = std::abs(nj.dot(r)) / rMag;

                double FijSky_val = cosTI * cosTJ * AjSkyMag / (rMag * rMag * M_PI);
                result.FijSky[n].emplace_back(m, FijSky_val);
                result.FijsumSky[n] += FijSky_val;
            }
        }
    }
    
    return result;
}



}
