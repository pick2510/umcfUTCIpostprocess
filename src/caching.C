#include "caching.H"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <errno.h>

namespace utci {

bool createDirectory(const std::string& path) {
    // Create all intermediate directories (like mkdir -p)
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] == '/') {
            std::string sub = path.substr(0, i);
            if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// Helper: write a sparse segment vector
static void writeSparseSegments(std::ofstream& f,
    const std::array<std::vector<std::pair<int,double>>, 5>& segs)
{
    for (int n = 0; n < 5; ++n) {
        int sz = static_cast<int>(segs[n].size());
        f.write(reinterpret_cast<const char*>(&sz), sizeof(int));
        for (const auto& [idx, val] : segs[n]) {
            f.write(reinterpret_cast<const char*>(&idx), sizeof(int));
            f.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
}

// Helper: read a sparse segment vector
static bool readSparseSegments(std::ifstream& f,
    std::array<std::vector<std::pair<int,double>>, 5>& segs)
{
    for (int n = 0; n < 5; ++n) {
        int sz = 0;
        if (!f.read(reinterpret_cast<char*>(&sz), sizeof(int))) return false;
        segs[n].resize(sz);
        for (auto& [idx, val] : segs[n]) {
            if (!f.read(reinterpret_cast<char*>(&idx), sizeof(int))) return false;
            if (!f.read(reinterpret_cast<char*>(&val), sizeof(double))) return false;
        }
    }
    return true;
}

bool BinaryCache::load(const std::string& path, ViewFactorResult& result) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // Version tag
    int version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != 2) return false;   // old dense cache → force recompute

    result.Fijsum.resize(5);
    file.read(reinterpret_cast<char*>(result.Fijsum.data()), 5 * sizeof(double));

    result.FijsumSky.resize(5);
    file.read(reinterpret_cast<char*>(result.FijsumSky.data()), 5 * sizeof(double));

    if (!readSparseSegments(file, result.Fij))    return false;
    if (!readSparseSegments(file, result.FijSky)) return false;

    return file.good();
}

bool BinaryCache::save(const std::string& path, const ViewFactorResult& result) {
    std::string dir = path.substr(0, path.find_last_of('/'));
    if (!createDirectory(dir)) {
        std::cerr << "Error: Cannot create cache directory: " << dir << std::endl;
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write cache: " << path << std::endl;
        return false;
    }

    int version = 2;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    file.write(reinterpret_cast<const char*>(result.Fijsum.data()),    5 * sizeof(double));
    file.write(reinterpret_cast<const char*>(result.FijsumSky.data()), 5 * sizeof(double));
    writeSparseSegments(file, result.Fij);
    writeSparseSegments(file, result.FijSky);

    return true;
}

std::string BinaryCache::getCachePath(int pedIndex) const {
    std::ostringstream oss;
    oss << baseDir_ << "/pos/" << pedIndex << "/F.bin";
    return oss.str();
}

}
