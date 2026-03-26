#ifndef UTCISOLVER_H
#define UTCISOLVER_H

#include "types.h"
#include <string>
#include <vector>

namespace utci {

enum class UtciMethod { POLYNOMIAL, LUT };

class UtciSolver {
public:
    UtciSolver() = default;
    ~UtciSolver() = default;

    void setMethod(UtciMethod m) { method_ = m; }
    UtciMethod method() const { return method_; }

    // Load LUT from utci_offset.Dat (required before using LUT method).
    // Returns true on success.
    bool loadLUT(const std::string& path);
    bool lutLoaded() const { return !lut_off_.empty(); }

    double calculate(double Ta_c, double va, double RH, double Tmrt_c);

private:
    double calcVaporPressure(double Ta_K, double RH);
    double calculatePoly(double Ta_c, double va, double RH, double Tmrt_c);
    double calculateLUT (double Ta_c, double va, double RH, double Tmrt_c);

    UtciMethod method_ = UtciMethod::POLYNOMIAL;

    // LUT data — 4D array indexed [iTa][iva][iTrTa][irH]
    std::vector<double> lut_Ta_, lut_TrTa_, lut_va_, lut_rH_;
    std::vector<double> lut_off_;
    int lut_nTa_=0, lut_nva_=0, lut_nTrTa_=0, lut_nrH_=0;

    double lutGet(int iT, int iV, int iM, int iR) const {
        return lut_off_[((iT * lut_nva_ + iV) * lut_nTrTa_ + iM) * lut_nrH_ + iR];
    }
    void lutSet(int iT, int iV, int iM, int iR, double v) {
        lut_off_[((iT * lut_nva_ + iV) * lut_nTrTa_ + iM) * lut_nrH_ + iR] = v;
    }
    static int lowerIdx(const std::vector<double>& g, double x, double& frac);
};

}

#endif
