#ifndef CACHING_H
#define CACHING_H

#include "types.h"
#include <string>

namespace utci {

bool createDirectory(const std::string& path);

class BinaryCache {
public:
    BinaryCache() = default;
    ~BinaryCache() = default;
    
    void setBaseDir(const std::string& dir) { baseDir_ = dir; }
    const std::string& getBaseDir() const { return baseDir_; }
    
    bool load(const std::string& path, ViewFactorResult& result);
    bool save(const std::string& path, const ViewFactorResult& result);
    
    std::string getCachePath(int pedIndex) const;

private:
    std::string baseDir_;
};

}

#endif
