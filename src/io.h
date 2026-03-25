#ifndef IO_H
#define IO_H

#include "types.h"
#include <string>
#include <vector>

namespace utci {

std::vector<PedestrianPosition> loadPedestrianPositions(const std::string& probeLocsPath);

std::vector<SurfacePatch> loadSurfacePatches(const std::string& rawPath);

std::vector<double> loadQrData(const std::string& qrPath);

// Load a scalar field from a 4-column raw file (x y z value) → returns the value column
std::vector<double> loadScalarField(const std::string& rawPath);

std::vector<MeteoData> loadMeteoData(const std::string& casePath,
                                      const std::vector<int>& timesteps);

// append=false truncates the file; append=true appends to an existing file
bool writeTumrtAvg(const std::string& path,
                   const std::vector<PedestrianPosition>& positions,
                   int timestep,
                   const Eigen::VectorXd& TumrtAvg,
                   bool append = false);

// Load all rows from a probe scalar file: vector of (time, values).
std::vector<std::pair<double,std::vector<double>>>
    loadProbeScalarAll(const std::string& path);

// Load all rows from a probe velocity file: vector of (time, magnitudes).
std::vector<std::pair<double,std::vector<double>>>
    loadProbeVelocityMagAll(const std::string& path);

// Find the probe directory under postProcessing/probes/<region>/
// Returns the first timestep subdirectory found, or empty string.
std::string findProbeDir(const std::string& casePath, const std::string& region = "air");

// Write pedestrian positions + one scalar array as legacy ASCII VTK PolyData (point cloud)
bool writeVtkPolyData(const std::string& path,
                       const std::vector<PedestrianPosition>& positions,
                       const Eigen::VectorXd& values,
                       const std::string& arrayName);

// Write pedestrian positions + multiple scalar arrays as legacy ASCII VTK PolyData (point cloud)
bool writeVtkMultiScalar(const std::string& path,
                          const std::vector<PedestrianPosition>& positions,
                          const std::vector<std::pair<std::string, Eigen::VectorXd>>& arrays);

// Write pedestrian positions + multiple scalar arrays as a VTK STRUCTURED_GRID surface.
// Detects the regular x/y grid automatically from position coordinates.
// Fills 0 for any missing grid points.
bool writeVtkStructuredSurface(const std::string& path,
                                const std::vector<PedestrianPosition>& positions,
                                const std::vector<std::pair<std::string, Eigen::VectorXd>>& arrays);

}

#endif
