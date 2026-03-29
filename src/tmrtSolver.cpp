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

Eigen::VectorXd TmrtSolver::compute(
    const std::vector<SurfacePatch>& surfaces,
    const std::vector<SurfacePatch>& sky,
    const ViewFactorResult& vf,
    const MeteoData& meteo,
    bool useSkyViewFactors)
{
    return computeDetailed(surfaces, sky, vf, meteo, useSkyViewFactors).Tmrt;
}

TmrtBreakdown TmrtSolver::computeDetailed(
    const std::vector<SurfacePatch>& surfaces,
    const std::vector<SurfacePatch>& sky,
    const ViewFactorResult& vf,
    const MeteoData& meteo,
    bool useSkyViewFactors)
{
    const int nBody = 5;
    TmrtBreakdown out;
    out.Tmrt = Eigen::VectorXd::Zero(nBody);
    out.qlwSurfaces = Eigen::VectorXd::Zero(nBody);
    out.qlwSky = Eigen::VectorXd::Zero(nBody);
    out.qswSurfaces = Eigen::VectorXd::Zero(nBody);
    out.qswSky = Eigen::VectorXd::Zero(nBody);
    out.qswGround = Eigen::VectorXd::Zero(nBody);
    out.qswElevatedDown = Eigen::VectorXd::Zero(nBody);
    out.qswVertical = Eigen::VectorXd::Zero(nBody);
    out.qswUpward = Eigen::VectorXd::Zero(nBody);
    
    if (surfaces.empty()) {
        return out;
    }
    
    // Precompute outgoing LW per surface.
    // utci_clement format: qrOut is already σT⁴ + qr*(1-ε)/ε (use directly).
    // Legacy format: qrOut==0, compute from temperature and qr separately.
    std::vector<double> QrOut(surfaces.size());
    for (size_t i = 0; i < surfaces.size(); ++i) {
        if (surfaces[i].qrOut > 0.0) {
            QrOut[i] = surfaces[i].qrOut;
        } else {
            QrOut[i] = SIGMA * std::pow(surfaces[i].temperature, 4)
                       + surfaces[i].qr * (1.0 - EPS_SURF) / EPS_SURF;
        }
    }
    
    // Compute sky temperature once
    const double Tsky = computeSkyTemperature(meteo.Ta, meteo.cc);
    const double sigmaTsky4 = SIGMA * std::pow(Tsky, 4);

    for (int n = 0; n < nBody; ++n) {
        // Split into long-wave (LW) and short-wave (SW) incident radiation
        double qin_lw_surfaces = 0.0;
        double qin_sw_surfaces = 0.0;

        // LW and SW from wall/veg surfaces
        for (const auto& [m, fij_val] : vf.Fij[n]) {
            qin_lw_surfaces += QrOut[m] * fij_val;
            double qsw = surfaces[m].qsOut * fij_val;
            qin_sw_surfaces += qsw;
            double areaMag = surfaces[m].areaVector.norm();
            double nz = (areaMag > 0.0) ? surfaces[m].areaVector.z() / areaMag : 0.0;
            if (nz < -0.7) {
                if (surfaces[m].center.z <= 2.5) out.qswGround[n] += qsw;
                else out.qswElevatedDown[n] += qsw;
            } else if (nz > 0.7) {
                out.qswUpward[n] += qsw;
            } else {
                out.qswVertical[n] += qsw;
            }
        }

        // Clement normalization: add sky radiation weighted by explicit FijsumSky,
        // then divide both LW and SW by (Fijsum + FijsumSky) so the total sums to 1.
        double qin_lw_sky = sigmaTsky4 * vf.FijsumSky[n];
        double qin_sw_sky = meteo.Idif * vf.FijsumSky[n];
        double total_vf = vf.Fijsum[n] + vf.FijsumSky[n];
        if (total_vf > 0.0) {
            qin_lw_surfaces /= total_vf;
            qin_sw_surfaces /= total_vf;
            qin_lw_sky /= total_vf;
            qin_sw_sky /= total_vf;
        }
        (void)useSkyViewFactors;

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
    const Eigen::VectorXd& Tmrt,
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
