#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <unordered_map>

namespace utci {

using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using VecX = Eigen::VectorXd;

using Mat3 = Eigen::Matrix3d;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using MatXxX = Eigen::MatrixXd;

struct Point3 {
    double x, y, z;
    
    Vec3 toVec3() const { return Vec3(x, y, z); }
    Point3 operator+(const Vec3& v) const { return {x + v.x(), y + v.y(), z + v.z()}; }
    Vec3 operator-(const Point3& other) const { 
        return Vec3(x - other.x, y - other.y, z - other.z); 
    }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

struct SurfacePatch {
    Point3 center;
    Vec3 areaVector;
    double temperature;
    double qr;          // Net LW radiation flux [W/m²]
    double qrOut;       // Outgoing LW radiation flux σT⁴+qr*(1-ε)/ε [W/m²] (utci_clement format)
    double qsOut;       // Outgoing SW radiation flux [W/m²]
    double area;        // |areaVector|
};

struct PedestrianPosition {
    int originalIndex = 0;              // Index in probe_locs file; stable cache key across filtered runs
    Point3 center;
    std::array<Point3, 5> bodyPoints;   // 5 body segments
    std::array<Vec3, 5> areaVectors;    // Area vectors for each segment
};

struct ViewFactorResult {
    // Sparse per body segment: only non-zero entries stored as (surface_index, fij_value)
    // Dense dense dense is avoided: only surfaces within R_MAG_MAX with line-of-sight contribute
    std::array<std::vector<std::pair<int,double>>, 5> Fij;
    Eigen::VectorXd Fijsum;
    std::array<std::vector<std::pair<int,double>>, 5> FijSky;
    Eigen::VectorXd FijsumSky;
};

struct TmrtResult {
    Eigen::VectorXd Tmrt;        // 5 directional values
    double TumrtAvg;              // Area-weighted average
};

struct MeteoData {
    double Ta;           // Ambient temperature [K]
    double Tsky;         // Sky temperature [K]
    double cc;           // Cloud cover [0-1]
    double Idif;         // Diffuse irradiance [W/m²]
    double Idn;          // Direct normal irradiance [W/m²]
    Vec3 sunDir;         // Sun direction vector
    double va;           // Wind speed at 10m [m/s]
    double RH;           // Relative humidity [%]
};

struct UtciResult {
    double UTCI;         // UTCI value [°C]
    double Tmrt;         // Full Tmrt [K]
    double RH;           // Relative humidity [%]
    double va;           // Wind speed at 10m [m/s]
};

enum class VtkDatasetType {
    POLYDATA,
    STRUCTURED_GRID
};

struct VtkMeshData {
    VtkDatasetType datasetType = VtkDatasetType::POLYDATA;
    std::string title = "VTK mesh";
    int dimX = 0;
    int dimY = 0;
    int dimZ = 0;
    std::vector<Point3> points;
    std::vector<std::vector<int>> cells;
    std::unordered_map<std::string, Eigen::VectorXd> cellScalars;
    std::unordered_map<std::string, std::vector<Vec3>> cellVectors;
    std::unordered_map<std::string, Eigen::VectorXd> pointScalars;
    std::unordered_map<std::string, std::vector<Vec3>> pointVectors;
};

}

#endif
