#ifndef VIEWFACTOR_H
#define VIEWFACTOR_H

#include "types.H"
#include <string>
#include <memory>

namespace utci {

class Raycaster;

class ViewFactorCalculator {
public:
    ViewFactorCalculator(const Raycaster& raycaster);
    ~ViewFactorCalculator();
    
    ViewFactorResult compute(
        const PedestrianPosition& ped,
        const std::vector<SurfacePatch>& surfaces,
        const std::vector<SurfacePatch>& sky,
        bool useSky = true
    );
    
    bool loadCache(const std::string& cachePath);
    void saveCache(const std::string& cachePath) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}

#endif
