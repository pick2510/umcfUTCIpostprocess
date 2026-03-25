#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "types.H"
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
    
    bool isBlocked(const Vec3& start, const Vec3& end) const;
    bool isBlockedBatch(const std::vector<Vec3>& starts, const std::vector<Vec3>& ends) const;
    
    bool isValid() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}

#endif
