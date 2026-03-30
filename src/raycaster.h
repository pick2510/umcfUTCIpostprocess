#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "types.h"
#include <string>
#include <memory>

namespace utci {

class Raycaster {
public:
    Raycaster();
    ~Raycaster();
    
    void setNumThreads(int n);
    bool loadGeometry(const std::string& stlPath);
    void loadVegetation(const std::string& stlPath);
    
    bool isBlocked(const Vec3& start, const Vec3& end, bool enforceRangeLimit = true) const;

    bool isValid() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}

#endif
