#ifndef DENSESTAGE2_H
#define DENSESTAGE2_H

#include "types.h"
#include <string>
#include <vector>

namespace utci {

class UtciSolver;

enum class DenseInterpClampMode {
    None,
    LocalRange
};

enum class DenseTumrtInterpMode {
    Cubic,
    Idw
};

std::string probeKey(const Point3& p);

std::string firstExistingPath(const std::vector<std::string>& paths);

Eigen::VectorXd remapQrswMagnitudesToPoints(const VtkMeshData& meshQrsw,
                                           const std::vector<Point3>& dstPoints);

bool computeDenseSurfaceOutputs(const std::string& casePath,
                                const std::string& outDir,
                                int timestep,
                                const std::vector<PedestrianPosition>& sparsePositions,
                                const Eigen::VectorXd& sparseTumrtAvg,
                                UtciSolver& utciSolver,
                                bool debugWriteQrsw,
                                DenseTumrtInterpMode interpMode,
                                int smoothPasses,
                                DenseInterpClampMode clampMode);

}

#endif
