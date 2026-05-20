#include "caching.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <errno.h>
#include <cstdio>

#ifdef UTCI_HAVE_ZLIB
#include "zstr.hpp"
#endif

namespace utci {

// Cache version tags
static constexpr int VERSION_PLAIN      = 11;
static constexpr int VERSION_COMPRESSED = 12;

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

bool BinaryCache::compressionAvailable() {
#ifdef UTCI_HAVE_ZLIB
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Helper: detect gzip magic bytes without consuming the stream
// ---------------------------------------------------------------------------
static bool fileIsGzip(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    unsigned char b[2] = {0, 0};
    f.read(reinterpret_cast<char*>(b), 2);
    return b[0] == 0x1f && b[1] == 0x8b;
}

// ---------------------------------------------------------------------------
// Helpers: write/read sparse segment vectors (values stored as float32)
// ---------------------------------------------------------------------------

template<typename Stream>
static void writeSparseSegments(Stream& f,
    const std::array<ViewFactorResult::SparseSegment, 5>& segs)
{
    for (int n = 0; n < 5; ++n) {
        if (segs[n].indices.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return;
        }
        int sz = static_cast<int>(segs[n].indices.size());
        f.write(reinterpret_cast<const char*>(&sz), sizeof(int));
        for (int i = 0; i < sz; ++i) {
            const int idx = segs[n].indices[i];
            const float fval = segs[n].fij[i];
            f.write(reinterpret_cast<const char*>(&idx),  sizeof(int));
            f.write(reinterpret_cast<const char*>(&fval), sizeof(float));
        }
    }
}

template<typename Stream>
static bool readSparseSegments(Stream& f,
    std::array<ViewFactorResult::SparseSegment, 5>& segs)
{
    for (int n = 0; n < 5; ++n) {
        int sz = 0;
        if (!f.read(reinterpret_cast<char*>(&sz), sizeof(int))) return false;
        if (sz < 0) return false;
        segs[n].indices.resize(sz);
        segs[n].fij.resize(sz);
        for (int i = 0; i < sz; ++i) {
            int idx = 0;
            float fval = 0.0f;
            if (!f.read(reinterpret_cast<char*>(&idx), sizeof(int)))   return false;
            if (!f.read(reinterpret_cast<char*>(&fval), sizeof(float))) return false;
            segs[n].indices[i] = idx;
            segs[n].fij[i] = fval;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared body for load, templated on stream type
// ---------------------------------------------------------------------------

template<typename Stream>
static bool loadFromStream(Stream& file, ViewFactorResult& result) {
    result.Fijsum.resize(5);
    file.read(reinterpret_cast<char*>(result.Fijsum.data()), 5 * sizeof(double));

    result.FijsumSky.resize(5);
    file.read(reinterpret_cast<char*>(result.FijsumSky.data()), 5 * sizeof(double));

    if (!readSparseSegments(file, result.Fij))    return false;
    if (!readSparseSegments(file, result.FijSky)) return false;

    return file.good();
}

// ---------------------------------------------------------------------------
// BinaryCache::load
// ---------------------------------------------------------------------------

bool BinaryCache::load(const std::string& path, ViewFactorResult& result) {
    auto removeCorrupt = [&]() { std::remove(path.c_str()); };

    if (fileIsGzip(path)) {
#ifdef UTCI_HAVE_ZLIB
        zstr::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        int version = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(int));
        if (version != VERSION_COMPRESSED) { removeCorrupt(); return false; }

        if (!loadFromStream(file, result)) { removeCorrupt(); return false; }
        return true;
#else
        std::cerr << "Warning: compressed cache found but ZLIB support not compiled in; recomputing.\n";
        return false;
#endif
    } else {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        int version = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(int));
        if (version != VERSION_PLAIN) { removeCorrupt(); return false; }

        if (!loadFromStream(file, result)) { removeCorrupt(); return false; }
        return true;
    }
}

// ---------------------------------------------------------------------------
// BinaryCache::save
// ---------------------------------------------------------------------------

bool BinaryCache::save(const std::string& path, const ViewFactorResult& result) {
    std::string dir = path.substr(0, path.find_last_of('/'));
    if (!createDirectory(dir)) {
        std::cerr << "Error: Cannot create cache directory: " << dir << std::endl;
        return false;
    }

#ifdef UTCI_HAVE_ZLIB
    if (compressed_) {
        zstr::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot write cache: " << path << std::endl;
            return false;
        }
        int version = VERSION_COMPRESSED;
        file.write(reinterpret_cast<const char*>(&version),               sizeof(int));
        file.write(reinterpret_cast<const char*>(result.Fijsum.data()),    5 * sizeof(double));
        file.write(reinterpret_cast<const char*>(result.FijsumSky.data()), 5 * sizeof(double));
        writeSparseSegments(file, result.Fij);
        writeSparseSegments(file, result.FijSky);
        return true;
    }
#endif

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write cache: " << path << std::endl;
        return false;
    }
    int version = VERSION_PLAIN;
    file.write(reinterpret_cast<const char*>(&version),               sizeof(int));
    file.write(reinterpret_cast<const char*>(result.Fijsum.data()),    5 * sizeof(double));
    file.write(reinterpret_cast<const char*>(result.FijsumSky.data()), 5 * sizeof(double));
    writeSparseSegments(file, result.Fij);
    writeSparseSegments(file, result.FijSky);
    return true;
}

// ---------------------------------------------------------------------------

std::string BinaryCache::getCachePath(int pedIndex, const Point3& center) const {
    auto q = [](double v) -> long long {
        return static_cast<long long>(std::llround(v * 1000.0));
    };
    std::ostringstream oss;
    oss << baseDir_ << "/pos/"
        << pedIndex
        << variantTag_
        << "_x" << q(center.x)
        << "_y" << q(center.y)
        << "_z" << q(center.z)
        << ".bin";
    return oss.str();
}

}
