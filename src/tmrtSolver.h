#ifndef TMRTSOLVER_H
#define TMRTSOLVER_H

#include "types.h"
#include <string>
#include <vector>
#include <memory>

namespace utci {

enum class SurfaceSwClass {
    Ground,
    ElevatedDown,
    Vertical,
    Upward
};

struct SurfaceRadiativeData {
    std::vector<double> qrOut;
    std::vector<double> qsOut;
    std::vector<SurfaceSwClass> swClass;
};

struct TmrtBreakdown {
    std::array<double, 5> Tmrt{};
    std::array<double, 5> qlwSurfaces{};
    std::array<double, 5> qlwSky{};
    std::array<double, 5> qswSurfaces{};
    std::array<double, 5> qswSky{};
    std::array<double, 5> qswGround{};
    std::array<double, 5> qswElevatedDown{};
    std::array<double, 5> qswVertical{};
    std::array<double, 5> qswUpward{};
};

class TmrtSolver {
public:
    TmrtSolver();
    ~TmrtSolver();
    
    Eigen::VectorXd compute(
        const std::vector<SurfacePatch>& surfaces,
        const std::vector<SurfacePatch>& sky,
        const ViewFactorResult& vf,
        const MeteoData& meteo,
        bool useSkyViewFactors
    );

    TmrtBreakdown computeDetailed(
        const SurfaceRadiativeData& surfaceData,
        const ViewFactorResult& vf,
        const MeteoData& meteo
    );
    
    double computeAreaWeightedAverage(
        const std::array<double, 5>& Tmrt,
        const std::array<Vec3, 5>& areaVectors
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}

#endif
