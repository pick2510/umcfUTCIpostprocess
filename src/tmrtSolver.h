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
    Eigen::VectorXd Tmrt;
    Eigen::VectorXd qlwSurfaces;
    Eigen::VectorXd qlwSky;
    Eigen::VectorXd qswSurfaces;
    Eigen::VectorXd qswSky;
    Eigen::VectorXd qswGround;
    Eigen::VectorXd qswElevatedDown;
    Eigen::VectorXd qswVertical;
    Eigen::VectorXd qswUpward;
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
        const std::vector<SurfacePatch>& surfaces,
        const SurfaceRadiativeData& surfaceData,
        const std::vector<SurfacePatch>& sky,
        const ViewFactorResult& vf,
        const MeteoData& meteo,
        bool useSkyViewFactors
    );
    
    double computeAreaWeightedAverage(
        const Eigen::VectorXd& Tmrt,
        const std::array<Vec3, 5>& areaVectors
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}

#endif
