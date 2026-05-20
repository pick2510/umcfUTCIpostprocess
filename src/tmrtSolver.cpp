#include "tmrtSolver.h"
#include "constants.h"
#include <cmath>
#include <iostream>

namespace utci {

struct TmrtSolver::Impl {};

TmrtSolver::TmrtSolver() : pImpl(std::make_unique<Impl>()) {}
TmrtSolver::~TmrtSolver() = default;

double computeSkyTemperature(double Ta, double cc) {
    double ec = (1.0 - 0.84 * cc) * (0.527 + 0.161 * std::exp(8.45 * (1.0 - 273.0 / Ta))) + 0.84 * cc;
    double Tsky_pow6 = 9.365574e-6 * (1.0 - cc) * std::pow(Ta, 6);
    double Tsky_pow4 = Ta * Ta * Ta * Ta;
    return std::pow(Tsky_pow6 + Tsky_pow4 * cc * ec, 0.25);
}


std::array<double, 5> TmrtSolver::computeFast(
    const SurfaceRadiativeData& surfaceData,
    const ViewFactorResult& vf,
    const MeteoData& meteo)
{
    std::array<double, 5> out{};

    if (surfaceData.qrOut.empty()) {
        return out;
    }

    const double Tsky = computeSkyTemperature(meteo.Ta, meteo.cc);
    const double sigmaTsky4 = SIGMA * std::pow(Tsky, 4);

    for (int n = 0; n < 5; ++n) {
        double qin_lw_surfaces = 0.0;
        double qin_sw_surfaces = 0.0;

        const auto& seg = vf.Fij[n];
        const size_t segSize = seg.indices.size();
        for (size_t k = 0; k < segSize; ++k) {
            const int m = seg.indices[k];
            const double fij_val = static_cast<double>(seg.fij[k]);
            qin_lw_surfaces += surfaceData.qrOut[m] * fij_val;
            qin_sw_surfaces += surfaceData.qsOut[m] * fij_val;
        }

        double qin_lw_sky = sigmaTsky4 * vf.FijsumSky[n];
        double qin_sw_sky = meteo.Idif * vf.FijsumSky[n];
        double total_vf = vf.Fijsum[n] + vf.FijsumSky[n];
        if (total_vf > 0.0) {
            qin_lw_surfaces /= total_vf;
            qin_sw_surfaces /= total_vf;
            qin_lw_sky /= total_vf;
            qin_sw_sky /= total_vf;
        }

        const double qin_lw = qin_lw_surfaces + qin_lw_sky;
        const double qin_sw = qin_sw_surfaces + qin_sw_sky;
        const double Tmrt4 = (EPS_LW_PERSON * qin_lw + ABS_SW_PERSON * qin_sw)
                           / (SIGMA * EPS_LW_PERSON);
        out[n] = (Tmrt4 > 0.0) ? std::pow(Tmrt4, 0.25) : meteo.Ta;
    }

    return out;
}

TmrtBreakdown TmrtSolver::computeDetailed(
    const SurfaceRadiativeData& surfaceData,
    const ViewFactorResult& vf,
    const MeteoData& meteo)
{
    const int nBody = 5;
    TmrtBreakdown out;
    
    if (surfaceData.qrOut.empty()) {
        return out;
    }
    
    // Compute sky temperature once
    const double Tsky = computeSkyTemperature(meteo.Ta, meteo.cc);
    const double sigmaTsky4 = SIGMA * std::pow(Tsky, 4);

    for (int n = 0; n < nBody; ++n) {
        // Split into long-wave (LW) and short-wave (SW) incident radiation
        double qin_lw_surfaces = 0.0;
        double qin_sw_surfaces = 0.0;

        // LW and SW from wall/veg surfaces
        const auto& seg = vf.Fij[n];
        const size_t segSize = seg.indices.size();
        for (size_t k = 0; k < segSize; ++k) {
            const int m = seg.indices[k];
            const double fij_val = static_cast<double>(seg.fij[k]);
            qin_lw_surfaces += surfaceData.qrOut[m] * fij_val;
            double qsw = surfaceData.qsOut[m] * fij_val;
            qin_sw_surfaces += qsw;
            if (surfaceData.swClass[m] == SurfaceSwClass::Ground) {
                out.qswGround[n] += qsw;
            } else if (surfaceData.swClass[m] == SurfaceSwClass::ElevatedDown) {
                out.qswElevatedDown[n] += qsw;
            } else if (surfaceData.swClass[m] == SurfaceSwClass::Upward) {
                out.qswUpward[n] += qsw;
            } else {
                out.qswVertical[n] += qsw;
            }
        }

        // Normalization: add sky radiation weighted by explicit FijsumSky,
        // then divide both LW and SW by (Fijsum + FijsumSky) so the total sums to 1.
        double qin_lw_sky = sigmaTsky4 * vf.FijsumSky[n];
        double qin_sw_sky = meteo.Idif * vf.FijsumSky[n];
        double total_vf = vf.Fijsum[n] + vf.FijsumSky[n];
        if (total_vf > 0.0) {
            qin_lw_surfaces /= total_vf;
            qin_sw_surfaces /= total_vf;
            qin_lw_sky /= total_vf;
            qin_sw_sky /= total_vf;
            out.qswGround[n]       /= total_vf;
            out.qswElevatedDown[n] /= total_vf;
            out.qswUpward[n]       /= total_vf;
            out.qswVertical[n]     /= total_vf;
        }
        out.qlwSurfaces[n] = qin_lw_surfaces;
        out.qlwSky[n] = qin_lw_sky;
        out.qswSurfaces[n] = qin_sw_surfaces;
        out.qswSky[n] = qin_sw_sky;

        // Apply person's absorption coefficients (matching original Python):
        // Tmrt^4 = (eps_pers * qin_lw + abs_sw * qin_sw) / (sigma * eps_pers)
        double qin_lw = qin_lw_surfaces + qin_lw_sky;
        double qin_sw = qin_sw_surfaces + qin_sw_sky;
        double Tmrt4 = (EPS_LW_PERSON * qin_lw + ABS_SW_PERSON * qin_sw)
                       / (SIGMA * EPS_LW_PERSON);
        out.Tmrt[n] = (Tmrt4 > 0.0) ? std::pow(Tmrt4, 0.25) : meteo.Ta;
    }
    
    return out;
}

double TmrtSolver::computeAreaWeightedAverage(
    const std::array<double, 5>& Tmrt,
    const std::array<Vec3, 5>& areaVectors) const
{
    double sumTmrtArea = 0.0;
    double sumArea = 0.0;
    
    for (int i = 0; i < 5; ++i) {
        double area = areaVectors[i].norm();
        sumTmrtArea += Tmrt[i] * area;
        sumArea += area;
    }
    
    return (sumArea > 0) ? sumTmrtArea / sumArea : 0.0;
}

}
