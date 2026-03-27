#include "io.h"
#include "constants.h"
#include "pedestrian.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <dirent.h>

namespace utci {

static std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    if (start == s.end()) return {};
    auto end = s.end();
    while (end != start && std::isspace(*std::prev(end))) --end;
    return std::string(start, end);
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

static double parseOpenFOAMNumber(const std::string& s) {
    std::string cleaned;
    for (char c : s) {
        if (c != '(' && c != ')' && c != ';') {
            cleaned += c;
        }
    }
    try {
        return std::stod(cleaned);
    } catch (...) {
        return 0.0;
    }
}

std::vector<PedestrianPosition> loadPedestrianPositions(const std::string& probeLocsPath) {
    std::vector<PedestrianPosition> positions;
    
    std::ifstream file(probeLocsPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open probe_locs: " << probeLocsPath << std::endl;
        return positions;
    }
    
    std::string line;
    bool inList = false;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        if (line == "(") {
            inList = true;
            continue;
        }
        if (line == ")") {
            break;
        }
        
        if (inList && line.find("(") != std::string::npos) {
            std::string coords = line;
            coords.erase(std::remove(coords.begin(), coords.end(), '('), coords.end());
            coords.erase(std::remove(coords.begin(), coords.end(), ')'), coords.end());
            
            auto parts = split(coords, ' ');
            if (parts.size() >= 3) {
                try {
                    Point3 center{
                        std::stod(parts[0]),
                        std::stod(parts[1]),
                        std::stod(parts[2])
                    };
                    positions.push_back(createPedestrianPosition(center));
                } catch (...) {
                    // Skip invalid lines
                }
            }
        }
    }
    
    std::cout << "  Loaded " << positions.size() << " pedestrian positions" << std::endl;
    return positions;
}

std::vector<SurfacePatch> loadSurfacePatches(const std::string& rawPath) {
    std::vector<SurfacePatch> patches;

    std::ifstream file(rawPath);
    if (!file.is_open()) {
        return patches;
    }

    std::string line;
    int headerLines = 0;

    while (std::getline(file, line)) {
        if (headerLines < 2) {
            headerLines++;
            continue;
        }

        line = trim(line);
        if (line.empty()) continue;

        auto parts = split(line, ' ');
        if (parts.size() >= 6) {
            try {
                SurfacePatch patch;
                patch.center.x = std::stod(parts[0]);
                patch.center.y = std::stod(parts[1]);
                patch.center.z = std::stod(parts[2]);
                // Negate area vectors: OpenFOAM Sf points outward from owner cell,
                // we need inward-facing normals for the view factor calculation.
                patch.areaVector = Vec3(
                    -std::stod(parts[3]),
                    -std::stod(parts[4]),
                    -std::stod(parts[5])
                );
                patch.area = patch.areaVector.norm();
                patch.temperature = 0.0;
                patch.qr    = 0.0;
                patch.qrOut = 0.0;
                patch.qsOut = 0.0;
                patches.push_back(patch);
            } catch (...) {
                // Skip invalid lines
            }
        }
    }

    return patches;
}

// Generic: load scalar column (index 3) from a 4-column raw file
std::vector<double> loadScalarField(const std::string& rawPath) {
    std::vector<double> vals;

    std::ifstream file(rawPath);
    if (!file.is_open()) {
        return vals;
    }

    std::string line;
    int headerLines = 0;

    while (std::getline(file, line)) {
        if (headerLines < 2) {
            headerLines++;
            continue;
        }
        line = trim(line);
        if (line.empty()) continue;

        auto parts = split(line, ' ');
        // Format: x y z value  (column 3 is the scalar)
        if (parts.size() >= 4) {
            try {
                vals.push_back(std::stod(parts[3]));
            } catch (...) {
                vals.push_back(0.0);
            }
        }
    }

    return vals;
}

std::vector<double> loadQrData(const std::string& qrPath) {
    return loadScalarField(qrPath);
}

// Parse OpenFOAM time-vector3 table: "( (t (sx sy sz)) ... )"
static std::vector<std::pair<double,Vec3>> parseSunPosVectorFile(const std::string& path) {
    std::vector<std::pair<double,Vec3>> data;
    std::ifstream file(path);
    if (!file.is_open()) return data;
    std::string line;
    while (std::getline(file, line)) {
        for (char& c : line) if (c=='(' || c==')' || c=='\t' || c==';') c = ' ';
        std::istringstream iss(line);
        double t, sx, sy, sz;
        if (iss >> t >> sx >> sy >> sz)
            data.push_back({t, Vec3(sx, sy, sz)});
    }
    return data;
}

static Vec3 interpVec3(const std::vector<std::pair<double,Vec3>>& data, double t) {
    if (data.empty()) return Vec3(0, 0, 1);
    if (t <= data.front().first) return data.front().second;
    if (t >= data.back().first)  return data.back().second;
    for (size_t i = 0; i + 1 < data.size(); ++i) {
        if (data[i].first <= t && t <= data[i+1].first) {
            double frac = (t - data[i].first) / (data[i+1].first - data[i].first);
            return data[i].second + frac * (data[i+1].second - data[i].second);
        }
    }
    return data.back().second;
}

// Parse OpenFOAM time-value table: "(  \n( t v )\n( t v )\n)"
static std::vector<std::pair<double,double>> parseTimeValueFile(const std::string& path) {
    std::vector<std::pair<double,double>> data;
    std::ifstream file(path);
    if (!file.is_open()) return data;

    std::string line;
    while (std::getline(file, line)) {
        // Replace delimiters with spaces
        for (char& c : line) {
            if (c == '(' || c == ')' || c == ';' || c == '\t') c = ' ';
        }
        std::istringstream iss(line);
        double t, v;
        if (iss >> t >> v) {
            data.push_back({t, v});
        }
    }
    return data;
}

static double interpTV(const std::vector<std::pair<double,double>>& data, double t) {
    if (data.empty()) return 0.0;
    if (t <= data.front().first) return data.front().second;
    if (t >= data.back().first)  return data.back().second;
    for (size_t i = 0; i + 1 < data.size(); ++i) {
        if (data[i].first <= t && t <= data[i+1].first) {
            double frac = (t - data[i].first) / (data[i+1].first - data[i].first);
            return data[i].second + frac * (data[i+1].second - data[i].second);
        }
    }
    return data.back().second;
}

// Specific humidity [kg/kg] + temperature [K] → relative humidity [%]
static double specificHumidityToRH(double q, double T_K) {
    // Saturation vapor pressure via Bolton (1980)
    double T_c = T_K - 273.15;
    double e_sat = 611.2 * std::exp(17.67 * T_c / (T_c + 243.5));  // [Pa]
    double w_sat = 0.622 * e_sat / (101325.0 - e_sat);              // [kg/kg]
    if (w_sat <= 0.0) return 100.0;
    return std::min(100.0, q / w_sat * 100.0);
}

std::vector<MeteoData> loadMeteoData(const std::string& casePath,
                                      const std::vector<int>& timesteps) {
    auto tambient   = parseTimeValueFile(casePath + "/0/air/Tambient");
    auto cloudCover = parseTimeValueFile(casePath + "/0/air/cloudCover");
    auto idif       = parseTimeValueFile(casePath + "/constant/Idif");
    auto idn        = parseTimeValueFile(casePath + "/constant/IDN");
    auto wambient   = parseTimeValueFile(casePath + "/0/air/wambient");
    auto sunVecs    = parseSunPosVectorFile(casePath + "/constant/sunPosVector");

    if (tambient.empty()) {
        std::cerr << "Warning: Could not read Tambient from "
                  << casePath << "/0/air/Tambient\n";
    }
    if (cloudCover.empty()) {
        std::cout << "  No cloudCover file found – assuming clear sky\n";
    }
    if (idif.empty()) {
        std::cout << "  No Idif file found – diffuse irradiance set to 0\n";
    }
    if (idn.empty()) {
        std::cout << "  No IDN file found – direct normal irradiance set to 0\n";
    }
    if (sunVecs.empty()) {
        std::cout << "  No sunPosVector file found – direct solar disabled\n";
    }
    if (wambient.empty()) {
        std::cout << "  No wambient file found – assuming RH=50%\n";
    }

    // Try to read reference wind speed from 0/air/U internalField
    double va_ref = 1.5;  // default [m/s]
    {
        std::ifstream uFile(casePath + "/0/air/U");
        std::string line;
        while (std::getline(uFile, line)) {
            if (line.find("internalField") != std::string::npos
                && line.find("uniform") != std::string::npos) {
                // format: internalField   uniform (ux uy uz);
                double ux = 0, uy = 0, uz = 0;
                auto pos = line.find('(');
                if (pos != std::string::npos) {
                    std::istringstream iss(line.substr(pos + 1));
                    iss >> ux >> uy >> uz;
                    va_ref = std::sqrt(ux*ux + uy*uy + uz*uz);
                }
                break;
            }
        }
        if (va_ref > 0.01)
            std::cout << "  Reference wind speed from 0/air/U: " << va_ref << " m/s\n";
    }

    std::vector<MeteoData> result(timesteps.size());
    for (size_t i = 0; i < timesteps.size(); ++i) {
        double t = static_cast<double>(timesteps[i]);
        result[i].Ta      = interpTV(tambient,   t);
        result[i].cc      = interpTV(cloudCover, t);
        result[i].Idif    = interpTV(idif,       t);
        result[i].Idn     = interpTV(idn,        t);
        result[i].Tsky    = 0.0;   // computed in tmrtSolver
        result[i].sunDir  = sunVecs.empty() ? Vec3(0,0,-1) : interpVec3(sunVecs, t);
        result[i].va      = va_ref;
        if (!wambient.empty()) {
            double q = interpTV(wambient, t);
            result[i].RH = specificHumidityToRH(q, result[i].Ta);
        } else {
            result[i].RH = 50.0;
        }
    }
    return result;
}


bool writeTumrtAvg(const std::string& path,
                   const std::vector<PedestrianPosition>& positions,
                   int timestep,
                   const Eigen::VectorXd& TumrtAvg,
                   bool append) {
    auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
    std::ofstream file(path, mode);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write TumrtAvg: " << path << std::endl;
        return false;
    }
    
    file << std::fixed << std::setprecision(4);
    
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i].center;
        file << timestep << " " << pos.x << " " << pos.y << " " << pos.z 
             << " " << TumrtAvg[i] << "\n";
    }
    
    return true;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> loadIdnSunvec(const std::string&) {
    return {Eigen::VectorXd(0), Eigen::VectorXd(0)};
}

// --------------------------------------------------------------------------
// Load all rows from a probe scalar file.
// Format: t  v0  v1  v2  ...  (one row per output time)
// --------------------------------------------------------------------------
std::vector<std::pair<double,std::vector<double>>>
loadProbeScalarAll(const std::string& path) {
    std::vector<std::pair<double,std::vector<double>>> rows;
    std::ifstream file(path);
    if (!file.is_open()) return rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double t; iss >> t;
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (!vals.empty()) rows.push_back({t, std::move(vals)});
    }
    return rows;
}

// --------------------------------------------------------------------------
// Load all rows from a probe velocity file.
// Format: t  (ux0 uy0 uz0)  (ux1 uy1 uz1)  ...
// --------------------------------------------------------------------------
std::vector<std::pair<double,std::vector<double>>>
loadProbeVelocityMagAll(const std::string& path) {
    std::vector<std::pair<double,std::vector<double>>> rows;
    std::ifstream file(path);
    if (!file.is_open()) return rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == '(' || c == ')') c = ' ';
        std::istringstream iss(line);
        double t; iss >> t;
        std::vector<double> mags;
        double ux, uy, uz;
        while (iss >> ux >> uy >> uz)
            mags.push_back(std::sqrt(ux*ux + uy*uy + uz*uz));
        if (!mags.empty()) rows.push_back({t, std::move(mags)});
    }
    return rows;
}

// --------------------------------------------------------------------------
// Find first timestep directory under postProcessing/probes/<region>/
// --------------------------------------------------------------------------
std::string findProbeDir(const std::string& casePath, const std::string& region) {
    std::string base = casePath + "/postProcessing/probes/" + region;
    DIR* dir = opendir(base.c_str());
    if (!dir) return "";
    struct dirent* entry;
    std::string found;
    double bestTime = 1e18;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        try {
            double t = std::stod(name);
            if (t < bestTime) { bestTime = t; found = base + "/" + name; }
        } catch (...) {}
    }
    closedir(dir);
    return found;
}

// --------------------------------------------------------------------------
// Write legacy ASCII VTK PolyData with one scalar array
// --------------------------------------------------------------------------
bool writeVtkPolyData(const std::string& path,
                       const std::vector<PedestrianPosition>& positions,
                       const Eigen::VectorXd& values,
                       const std::string& arrayName) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write VTK: " << path << "\n";
        return false;
    }
    size_t N = positions.size();
    f << "# vtk DataFile Version 2.0\n"
      << arrayName << "\n"
      << "ASCII\n"
      << "DATASET POLYDATA\n"
      << "POINTS " << N << " float\n";
    f << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < N; ++i)
        f << positions[i].center.x << " " << positions[i].center.y << " " << positions[i].center.z << "\n";
    f << "\nVERTICES " << N << " " << (2*N) << "\n";
    for (size_t i = 0; i < N; ++i)
        f << "1 " << i << "\n";
    f << "\nPOINT_DATA " << N << "\n"
      << "SCALARS " << arrayName << " float 1\n"
      << "LOOKUP_TABLE default\n";
    f << std::setprecision(4);
    for (size_t i = 0; i < N; ++i)
        f << values[i] << "\n";
    return true;
}

// --------------------------------------------------------------------------
// Write legacy ASCII VTK PolyData with multiple scalar arrays
// --------------------------------------------------------------------------
bool writeVtkMultiScalar(const std::string& path,
                          const std::vector<PedestrianPosition>& positions,
                          const std::vector<std::pair<std::string, Eigen::VectorXd>>& arrays) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write VTK: " << path << "\n";
        return false;
    }
    size_t N = positions.size();
    f << "# vtk DataFile Version 2.0\n"
      << "UTCI results\n"
      << "ASCII\n"
      << "DATASET POLYDATA\n"
      << "POINTS " << N << " float\n";
    f << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < N; ++i)
        f << positions[i].center.x << " " << positions[i].center.y << " " << positions[i].center.z << "\n";
    f << "\nVERTICES " << N << " " << (2*N) << "\n";
    for (size_t i = 0; i < N; ++i)
        f << "1 " << i << "\n";
    f << "\nPOINT_DATA " << N << "\n";
    f << std::setprecision(4);
    for (const auto& [name, vals] : arrays) {
        f << "SCALARS " << name << " float 1\n"
          << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            f << vals[i] << "\n";
    }
    return true;
}

// --------------------------------------------------------------------------
// Write VTK STRUCTURED_GRID surface (one grid point per pedestrian position).
// Positions are assumed to lie on a regular x/y grid at constant z.
// --------------------------------------------------------------------------
bool writeVtkStructuredSurface(const std::string& path,
                                const std::vector<PedestrianPosition>& positions,
                                const std::vector<std::pair<std::string, Eigen::VectorXd>>& arrays) {
    if (positions.empty()) return false;

    // Collect unique sorted X and Y coordinates
    std::vector<double> xs, ys;
    xs.reserve(positions.size());
    ys.reserve(positions.size());
    for (const auto& p : positions) {
        xs.push_back(p.center.x);
        ys.push_back(p.center.y);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end(),
        [](double a, double b){ return std::abs(a-b) < 0.01; }), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end(),
        [](double a, double b){ return std::abs(a-b) < 0.01; }), ys.end());

    int NX = static_cast<int>(xs.size());
    int NY = static_cast<int>(ys.size());
    double zVal = positions[0].center.z;

    // Build (ix,iy) → position index map
    double dx = (NX > 1) ? (xs[1] - xs[0]) : 1.0;
    double dy = (NY > 1) ? (ys[1] - ys[0]) : 1.0;
    double xmin = xs.front(), ymin = ys.front();

    std::vector<int> grid(NX * NY, -1);
    for (size_t i = 0; i < positions.size(); ++i) {
        int ix = static_cast<int>(std::round((positions[i].center.x - xmin) / dx));
        int iy = static_cast<int>(std::round((positions[i].center.y - ymin) / dy));
        if (ix >= 0 && ix < NX && iy >= 0 && iy < NY)
            grid[iy * NX + ix] = static_cast<int>(i);
    }

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write VTK surface: " << path << "\n";
        return false;
    }

    f << "# vtk DataFile Version 2.0\nUTCI surface\nASCII\n"
      << "DATASET STRUCTURED_GRID\n"
      << "DIMENSIONS " << NX << " " << NY << " 1\n"
      << "POINTS " << (NX * NY) << " float\n";
    f << std::fixed << std::setprecision(3);
    for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix)
            f << xs[ix] << " " << ys[iy] << " " << zVal << "\n";

    f << "\nPOINT_DATA " << (NX * NY) << "\n";
    f << std::setprecision(4);
    for (const auto& [name, vals] : arrays) {
        f << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        for (int iy = 0; iy < NY; ++iy)
            for (int ix = 0; ix < NX; ++ix) {
                int idx = grid[iy * NX + ix];
                f << (idx >= 0 ? vals[idx] : 0.0) << "\n";
            }
    }
    return true;
}

}
