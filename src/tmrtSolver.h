#ifndef TMRTSOLVER_H
#define TMRTSOLVER_H

#include "types.h"
#include <string>
#include <vector>
#include <memory>

namespace utci {

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
