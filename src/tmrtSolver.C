#include "tmrtSolver.H"
#include "constants.H"
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
    const int nBody = 5;
    Eigen::VectorXd Tmrt(nBody);
    Tmrt.setZero();
    
    if (surfaces.empty()) {
        return Tmrt;
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
    double Tsky = computeSkyTemperature(meteo.Ta, meteo.cc);
    double sigmaTsky4 = SIGMA * std::pow(Tsky, 4);

    for (int n = 0; n < nBody; ++n) {
        // Split into long-wave (LW) and short-wave (SW) incident radiation
        double qin_lw = 0.0;
        double qin_sw = 0.0;

        // LW and SW from wall/veg surfaces
        for (const auto& [m, fij_val] : vf.Fij[n]) {
            qin_lw += QrOut[m] * fij_val;
            qin_sw += surfaces[m].qsOut * fij_val;
        }

        if (useSkyViewFactors && vf.FijsumSky.size() == nBody) {
            double totalF = vf.Fijsum[n] + vf.FijsumSky[n];
            if (totalF > 1e-12) {
                double skyF = vf.FijsumSky[n] / totalF;
                qin_lw /= totalF;
                qin_lw += sigmaTsky4  * skyF;   // sky LW
                qin_sw /= totalF;               // normalize wall SW
                qin_sw += meteo.Idif  * skyF;   // add sky SW diffuse
            }
        } else {
            // Fallback: sky fraction = 1 - Fijsum
            if (vf.Fijsum[n] > 1.0) qin_lw /= vf.Fijsum[n];
            double Fsky = std::max(0.0, 1.0 - vf.Fijsum[n]);
            qin_lw += sigmaTsky4 * Fsky;
            qin_sw /= std::max(1.0, vf.Fijsum[n]);
            qin_sw += meteo.Idif * Fsky;
        }

        // Apply person's absorption coefficients (matching original Python):
        // Tmrt^4 = (eps_pers * qin_lw + abs_sw * qin_sw) / (sigma * eps_pers)
        double Tmrt4 = (EPS_LW_PERSON * qin_lw + ABS_SW_PERSON * qin_sw)
                       / (SIGMA * EPS_LW_PERSON);
        Tmrt[n] = (Tmrt4 > 0.0) ? std::pow(Tmrt4, 0.25) : meteo.Ta;
    }
    
    return Tmrt;
}

double TmrtSolver::computeAreaWeightedAverage(
    const Eigen::VectorXd& Tmrt,
    const std::array<Vec3, 5>& areaVectors)
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
