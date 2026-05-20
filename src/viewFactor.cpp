#include "viewFactor.h"
#include "raycaster.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace utci {

static double computePatchFij(
    const Vec3& i_n,       // probe body point
    const Vec3& Ai_n,      // body area vector (for orientation check, not normalised)
    const Vec3& ni,        // unit body normal
    const Vec3& j_center,  // patch centre
    const Vec3& nj,        // unit patch normal (already loaded/normalised by caller)
    double AjMag,          // patch area
    bool enforceRangeLimit = true)
{
    Vec3 r0 = i_n - j_center;
    double r0Mag = r0.norm();
    if (r0Mag <= 0.0) return 0.0;
    if (enforceRangeLimit && r0Mag >= R_MAG_MAX) return 0.0;
    if (Ai_n.dot(r0) >= 0.0) return 0.0;   // body faces away from patch centre
    double cosTI = std::abs(ni.dot(r0)) / r0Mag;
    double cosTJ = std::abs(nj.dot(r0)) / r0Mag;
    double r2 = r0Mag * r0Mag;
    return cosTI * cosTJ * AjMag / (r2 * M_PI);
}

struct ViewFactorCalculator::Impl {
    const Raycaster& raycaster;
    bool angularSky = false;
    double skyRayLength = 5000.0;
    std::vector<Vec3> skyDirs;
    std::vector<double> skyWeights;

    Impl(const Raycaster& rc,
         bool angularSky_,
         int skyAzimuthSamples,
         int skyElevationSamples,
         double skyRayLength_)
        : raycaster(rc), angularSky(angularSky_), skyRayLength(skyRayLength_) {
        if (angularSky) {
            const int nPhi = std::max(1, skyAzimuthSamples);
            const int nMu = std::max(1, skyElevationSamples);
            skyDirs.reserve(static_cast<size_t>(nPhi * nMu));
            skyWeights.reserve(static_cast<size_t>(nPhi * nMu));
            const double dPhi = 2.0 * M_PI / static_cast<double>(nPhi);
            const double dMu = 1.0 / static_cast<double>(nMu);
            for (int iMu = 0; iMu < nMu; ++iMu) {
                const double mu = (static_cast<double>(iMu) + 0.5) * dMu;
                const double sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
                    const double phi = (static_cast<double>(iPhi) + 0.5) * dPhi;
                    skyDirs.emplace_back(
                        sinTheta * std::cos(phi),
                        sinTheta * std::sin(phi),
                        mu
                    );
                    skyWeights.push_back(dPhi * dMu);
                }
            }
        }
    }

    bool isRayBlocked(const Vec3& start, const Vec3& end) const {
        return raycaster.isBlocked(start, end);
    }

    bool isSkyRayBlocked(const Vec3& start, const Vec3& end) const {
        return raycaster.isBlocked(start, end, false);
    }
};

ViewFactorCalculator::ViewFactorCalculator(const Raycaster& raycaster,
                                           bool angularSky,
                                           int skyAzimuthSamples,
                                           int skyElevationSamples,
                                           double skyRayLength)
    : pImpl(std::make_unique<Impl>(
        raycaster, angularSky, skyAzimuthSamples, skyElevationSamples, skyRayLength
      )) {}

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
    const int nSky  = (useSky && !pImpl->angularSky) ? static_cast<int>(sky.size()) : 0;

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
        const Vec3 i_n = ped.bodyPoints[n].toVec3();
        const Vec3 Ai_n = ped.areaVectors[n];
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

            double Fij_val = computePatchFij(i_n, Ai_n, ni, j_m, nj, AjMag, true);
            if (Fij_val <= 0.0) continue;
            result.Fij[n].indices.push_back(m);
            result.Fij[n].fij.push_back(static_cast<float>(Fij_val));
            result.Fijsum[n] += Fij_val;
        }

        if (useSky && pImpl->angularSky) {
            const size_t nSamples = pImpl->skyDirs.size();
            for (size_t s = 0; s < nSamples; ++s) {
                const Vec3& dir = pImpl->skyDirs[s];
                const double cosThetaI = ni.dot(dir);
                if (cosThetaI <= 0.0) continue;
                const Vec3 end = i_n + dir * pImpl->skyRayLength;
                if (pImpl->isSkyRayBlocked(i_n, end)) continue;
                result.FijsumSky[n] += (cosThetaI * pImpl->skyWeights[s]) / M_PI;
            }
        } else if (useSky && !nearSky.empty()) {
            for (int mi = 0; mi < (int)nearSky.size(); ++mi) {
                int m = nearSky[mi];
                const auto& surfSky = sky[m];
                Vec3 j_sky = surfSky.center.toVec3();
                double AjSkyMag = surfSky.area;

                Vec3 r0 = i_n - j_sky;
                if (r0.norm() <= 0.0) continue;
                if (Ai_n.dot(r0) >= 0) continue;

                // Ray occlusion: skip if blocked by geometry
                if (pImpl->isSkyRayBlocked(i_n, j_sky)) continue;

                Vec3 nj = surfSky.areaVector / AjSkyMag;
                double FijSky_val = computePatchFij(i_n, Ai_n, ni, j_sky, nj, AjSkyMag, false);
                if (FijSky_val <= 0.0) continue;
                result.FijSky[n].indices.push_back(m);
                result.FijSky[n].fij.push_back(static_cast<float>(FijSky_val));
                result.FijsumSky[n] += FijSky_val;
            }
        }
    }
    
    return result;
}



}
