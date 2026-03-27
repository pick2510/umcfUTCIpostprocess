#include "utciSolver.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace utci {

double UtciSolver::calcVaporPressure(double Ta_K, double RH) {
    double es = 0.01 * std::exp(
        2.7150305 * std::log(Ta_K)
        - 2836.5744 / (Ta_K * Ta_K)
        - 6028.076559 / Ta_K
        + 19.54263612
        - 0.02737830188 * Ta_K
        + 0.000016261698 * Ta_K * Ta_K
        + 7.0229056e-10 * Ta_K * Ta_K * Ta_K
        - 1.8680009e-13 * Ta_K * Ta_K * Ta_K * Ta_K
    );
    return es * RH / 100.0;
}

double UtciSolver::calculate(double Ta_c, double va, double RH, double Tmrt_c) {
    if (method_ == UtciMethod::LUT)
        return calculateLUT(Ta_c, va, RH, Tmrt_c);
    return calculatePoly(Ta_c, va, RH, Tmrt_c);
}

double UtciSolver::calculatePoly(double Ta_c, double va, double f, double Tmrt_c) {
    double Ta = Ta_c;                               // polynomial expects °C
    double D  = Tmrt_c - Ta_c;                     // delta Tmrt [K or °C, same]
    double Pa = calcVaporPressure(Ta_c + 273.15, f) / 10.0;  // hPa → kPa (polynomial expects kPa)
    
    double UTCI = Ta
        + 6.07562052e-01
        - 2.27712343e-02 * Ta
        + 8.06470249e-04 * Ta * Ta
        - 1.54271372e-04 * Ta * Ta * Ta
        - 3.24651735e-06 * Ta * Ta * Ta * Ta
        + 7.32602852e-08 * Ta * Ta * Ta * Ta * Ta
        + 1.35959073e-09 * Ta * Ta * Ta * Ta * Ta * Ta
        - 2.25836520 * va
        + 8.80326035e-02 * Ta * va
        + 2.16844454e-03 * Ta * Ta * va
        - 1.53347087e-05 * Ta * Ta * Ta * va
        - 5.72983704e-07 * Ta * Ta * Ta * Ta * va
        - 2.55090145e-09 * Ta * Ta * Ta * Ta * Ta * va
        - 7.51269505e-01 * va * va
        - 4.08350271e-03 * Ta * va * va
        - 5.21670675e-05 * Ta * Ta * va * va
        + 1.94544667e-06 * Ta * Ta * Ta * va * va
        + 1.14099531e-08 * Ta * Ta * Ta * Ta * va * va
        + 1.58137256e-01 * va * va * va
        - 6.57263143e-05 * Ta * va * va * va
        + 2.22697524e-07 * Ta * Ta * va * va * va
        - 4.16117031e-08 * Ta * Ta * Ta * va * va * va
        - 1.27762753e-02 * va * va * va * va
        + 9.66891875e-06 * Ta * va * va * va * va
        + 2.52785852e-09 * Ta * Ta * va * va * va * va
        + 4.56306672e-04 * va * va * va * va * va
        - 1.74202546e-07 * Ta * va * va * va * va * va
        - 5.91491269e-06 * va * va * va * va * va * va
        + 3.98374029e-01 * D
        + 1.83945314e-04 * Ta * D
        - 1.73754510e-04 * Ta * Ta * D
        - 7.60781159e-07 * Ta * Ta * Ta * D
        + 3.77830287e-08 * Ta * Ta * Ta * Ta * D
        + 5.43079673e-10 * Ta * Ta * Ta * Ta * Ta * D
        - 2.00518269e-02 * va * D
        + 8.92859837e-04 * Ta * va * D
        + 3.45433048e-06 * Ta * Ta * va * D
        - 3.77925774e-07 * Ta * Ta * Ta * va * D
        - 1.69699377e-09 * Ta * Ta * Ta * Ta * va * D
        + 1.69992415e-04 * va * va * D
        - 4.99204314e-05 * Ta * va * va * D
        + 2.47417178e-07 * Ta * Ta * va * va * D
        + 1.07596466e-08 * Ta * Ta * Ta * va * va * D
        + 8.49242932e-05 * va * va * va * D
        + 1.35191328e-06 * Ta * va * va * va * D
        - 6.21531254e-09 * Ta * Ta * va * va * va * D
        - 4.99410301e-06 * va * va * va * va * D
        - 1.89489258e-08 * Ta * va * va * va * va * D
        + 8.15300114e-08 * va * va * va * va * va * D
        + 7.55043090e-04 * D * D
        - 5.65095215e-05 * Ta * D * D
        - 4.52166564e-07 * Ta * Ta * D * D
        + 2.46688878e-08 * Ta * Ta * Ta * D * D
        + 2.42674348e-10 * Ta * Ta * Ta * Ta * D * D
        + 1.54547250e-04 * va * D * D
        + 5.24110970e-06 * Ta * va * D * D
        - 8.75874982e-08 * Ta * Ta * va * D * D
        - 1.50743064e-09 * Ta * Ta * Ta * va * D * D
        - 1.56236307e-05 * va * va * D * D
        - 1.33895614e-07 * Ta * va * va * D * D
        + 2.49709824e-09 * Ta * Ta * va * va * D * D
        + 6.51711721e-07 * va * va * va * D * D
        + 1.94960053e-09 * Ta * va * va * va * D * D
        - 1.00361113e-08 * va * va * va * va * D * D
        - 1.21206673e-05 * D * D * D
        - 2.18203660e-07 * Ta * D * D * D
        + 7.51269482e-09 * Ta * Ta * D * D * D
        + 9.79063848e-11 * Ta * Ta * Ta * D * D * D
        + 1.25006734e-06 * va * D * D * D
        - 1.81584736e-09 * Ta * va * D * D * D
        - 3.52197671e-10 * Ta * Ta * va * D * D * D
        - 3.36514630e-08 * va * va * D * D * D
        + 1.35908359e-10 * Ta * va * va * D * D * D
        + 4.17032620e-10 * va * va * va * D * D * D
        - 1.30369025e-09 * D * D * D * D
        + 4.13908461e-10 * Ta * D * D * D * D
        + 9.22652254e-12 * Ta * Ta * D * D * D * D
        - 5.08220384e-09 * va * D * D * D * D
        - 2.24730961e-11 * Ta * va * D * D * D * D
        + 1.17139133e-10 * va * va * D * D * D * D
        + 6.62154879e-10 * D * D * D * D * D
        + 4.03863260e-13 * Ta * D * D * D * D * D
        + 1.95087203e-12 * va * D * D * D * D * D
        - 4.73602469e-12 * D * D * D * D * D * D
        + 5.12733497 * Pa
        - 3.12788561e-01 * Ta * Pa
        - 1.96701861e-02 * Ta * Ta * Pa
        + 9.99690870e-04 * Ta * Ta * Ta * Pa
        + 9.51738512e-06 * Ta * Ta * Ta * Ta * Pa
        - 4.66426341e-07 * Ta * Ta * Ta * Ta * Ta * Pa
        + 5.48050612e-01 * va * Pa
        - 3.30552823e-03 * Ta * va * Pa
        - 1.64119440e-03 * Ta * Ta * va * Pa
        - 5.16670694e-06 * Ta * Ta * Ta * va * Pa
        + 9.52692432e-07 * Ta * Ta * Ta * Ta * va * Pa
        - 4.29223622e-02 * va * va * Pa
        + 5.00845667e-03 * Ta * va * va * Pa
        + 1.00601257e-06 * Ta * Ta * va * va * Pa
        - 1.81748644e-06 * Ta * Ta * Ta * va * va * Pa
        - 1.25813502e-03 * va * va * va * Pa
        - 1.79330391e-04 * Ta * va * va * va * Pa
        + 2.34994441e-06 * Ta * Ta * va * va * va * Pa
        + 1.29735808e-04 * va * va * va * va * Pa
        + 1.29064870e-06 * Ta * va * va * va * va * Pa
        - 2.28558686e-06 * va * va * va * va * va * Pa
        - 3.69476348e-02 * D * Pa
        + 1.62325322e-03 * Ta * D * Pa
        - 3.14279680e-05 * Ta * Ta * D * Pa
        + 2.59835559e-06 * Ta * Ta * Ta * D * Pa
        - 4.77136523e-08 * Ta * Ta * Ta * Ta * D * Pa
        + 8.64203390e-03 * va * D * Pa
        - 6.87405181e-04 * Ta * va * D * Pa
        - 9.13863872e-06 * Ta * Ta * va * D * Pa
        + 5.15916806e-07 * Ta * Ta * Ta * va * D * Pa
        - 3.59217476e-05 * va * va * D * Pa
        + 3.28696511e-05 * Ta * va * va * D * Pa
        - 7.10542454e-07 * Ta * Ta * va * va * D * Pa
        - 1.24382300e-05 * va * va * va * D * Pa
        - 7.38584400e-09 * Ta * va * va * va * D * Pa
        + 2.20609296e-07 * va * va * va * va * D * Pa
        - 7.32469180e-04 * D * D * Pa
        - 1.87381964e-05 * Ta * D * D * Pa
        + 4.80925239e-06 * Ta * Ta * D * D * Pa
        - 8.75492040e-08 * Ta * Ta * Ta * D * D * Pa
        + 2.77862930e-05 * va * D * D * Pa
        - 5.06004592e-06 * Ta * va * D * D * Pa
        + 1.14325367e-07 * Ta * Ta * va * D * D * Pa
        + 2.53016723e-06 * va * va * D * D * Pa
        - 1.72857035e-08 * Ta * va * va * D * D * Pa
        - 3.95079398e-08 * va * va * va * D * D * Pa
        - 3.59413173e-07 * D * D * D * Pa
        + 7.04388046e-07 * Ta * D * D * D * Pa
        - 1.89309167e-08 * Ta * Ta * D * D * D * Pa
        - 4.79768731e-07 * va * D * D * D * Pa
        + 7.96079978e-09 * Ta * va * D * D * D * Pa
        + 1.62897058e-09 * va * va * D * D * D * Pa
        + 3.94367674e-08 * D * D * D * D * Pa
        - 1.18566247e-09 * Ta * D * D * D * D * Pa
        + 3.34678041e-10 * va * D * D * D * D * Pa;

    return UTCI;
}

// ── LUT loading ────────────────────────────────────────────────────────────

bool UtciSolver::loadLUT(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "  [LUT] Cannot open: " << path << "\n";
        return false;
    }

    struct Row { double Ta, TrTa, va, rH, offset; };
    std::vector<Row> rows;

    std::string line;
    bool header_seen = false;
    while (std::getline(f, line)) {
        // Strip UTF-8 BOM if present
        if (!line.empty() && (unsigned char)line[0] == 0xEF) {
            line = line.substr(line.find_first_not_of("\xEF\xBB\xBF"));
        }
        if (line.empty() || line[0] == '*') continue;
        // Header row
        if (!header_seen) { header_seen = true; continue; }
        std::istringstream ss(line);
        Row r;
        double pa;
        if (ss >> r.Ta >> r.TrTa >> r.va >> r.rH >> pa >> r.offset)
            rows.push_back(r);
    }

    if (rows.empty()) {
        std::cerr << "  [LUT] No data rows parsed from " << path << "\n";
        return false;
    }

    // Build sorted unique axis grids
    auto uniq = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    };
    std::vector<double> all_Ta, all_TrTa, all_va, all_rH;
    for (auto& r : rows) {
        all_Ta.push_back(r.Ta); all_TrTa.push_back(r.TrTa);
        all_va.push_back(r.va); all_rH.push_back(r.rH);
    }
    lut_Ta_    = uniq(all_Ta);
    lut_TrTa_  = uniq(all_TrTa);
    lut_va_    = uniq(all_va);
    lut_rH_    = uniq(all_rH);
    lut_nTa_   = (int)lut_Ta_.size();
    lut_nTrTa_ = (int)lut_TrTa_.size();
    lut_nva_   = (int)lut_va_.size();
    lut_nrH_   = (int)lut_rH_.size();

    lut_off_.assign((size_t)lut_nTa_ * lut_nva_ * lut_nTrTa_ * lut_nrH_, 0.0);

    // The raw data is sparse in the rH axis: each (Ta, TrTa, va) group has
    // only a subset of rH values. Interpolate onto the full rH grid per group,
    // matching the Python np.interp approach, so no LUT cell is left at 0.
    auto idx = [](const std::vector<double>& g, double v) {
        return (int)(std::lower_bound(g.begin(), g.end(), v) - g.begin());
    };

    // Group rows by (Ta, TrTa, va)
    using Key = std::tuple<double,double,double>;
    std::map<Key, std::vector<std::pair<double,double>>> groups; // key → [(rH, offset)]
    for (auto& r : rows)
        groups[{r.Ta, r.TrTa, r.va}].emplace_back(r.rH, r.offset);

    for (auto& [key, pts] : groups) {
        auto [Ta, TrTa, va] = key;
        int iT = idx(lut_Ta_,   Ta);
        int iM = idx(lut_TrTa_, TrTa);
        int iV = idx(lut_va_,   va);

        // Sort by rH
        std::sort(pts.begin(), pts.end());

        // Interpolate onto the full lut_rH_ grid (matches Python np.interp)
        for (int iR = 0; iR < lut_nrH_; ++iR) {
            double rh = lut_rH_[iR];
            // Linear interpolation / extrapolation clamped to endpoints
            double val;
            if (rh <= pts.front().first) {
                val = pts.front().second;
            } else if (rh >= pts.back().first) {
                val = pts.back().second;
            } else {
                // find bracket
                auto it = std::lower_bound(pts.begin(), pts.end(),
                                           std::make_pair(rh, -1e30));
                auto hi = it; auto lo = std::prev(it);
                const double f = (rh - lo->first) / (hi->first - lo->first);
                val = lo->second + f * (hi->second - lo->second);
            }
            lutSet(iT, iV, iM, iR, val);
        }
    }

    std::cout << "  [LUT] Loaded " << rows.size() << " rows  "
              << "Ta[" << lut_Ta_.front() << ".." << lut_Ta_.back() << "]  "
              << "TrTa[" << lut_TrTa_.front() << ".." << lut_TrTa_.back() << "]  "
              << "va[" << lut_va_.front() << ".." << lut_va_.back() << "]  "
              << "rH[" << lut_rH_.front() << ".." << lut_rH_.back() << "]\n";
    return true;
}

// ── LUT interpolation ──────────────────────────────────────────────────────

int UtciSolver::lowerIdx(const std::vector<double>& g, double x, double& frac) {
    double xc = std::clamp(x, g.front(), g.back());
    int i = (int)(std::lower_bound(g.begin(), g.end(), xc) - g.begin());
    i = std::clamp(i - 1, 0, (int)g.size() - 2);
    frac = (xc - g[i]) / (g[i+1] - g[i]);
    return i;
}

double UtciSolver::calculateLUT(double Ta_c, double va, double RH, double Tmrt_c) {
    double TrTa = Tmrt_c - Ta_c;
    double fT, fV, fM, fR;
    int iT = lowerIdx(lut_Ta_,   Ta_c, fT);
    int iV = lowerIdx(lut_va_,   va,   fV);
    int iM = lowerIdx(lut_TrTa_, TrTa, fM);
    int iR = lowerIdx(lut_rH_,   RH,   fR);

    // 4D linear interpolation over 16 corners
    double offset = 0.0;
    for (int dT = 0; dT < 2; ++dT)
    for (int dV = 0; dV < 2; ++dV)
    for (int dM = 0; dM < 2; ++dM)
    for (int dR = 0; dR < 2; ++dR) {
        double w = (dT ? fT : 1-fT) * (dV ? fV : 1-fV)
                 * (dM ? fM : 1-fM) * (dR ? fR : 1-fR);
        offset += w * lutGet(iT+dT, iV+dV, iM+dM, iR+dR);
    }
    return Ta_c + offset;
}

}
