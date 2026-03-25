#ifndef PEDESTRIAN_H
#define PEDESTRIAN_H

#include "types.H"
#include <array>

namespace utci {

PedestrianPosition createPedestrianPosition(const Point3& center);

std::array<Vec3, 5> getPedestrianAreaVectors(const PedestrianPosition& ped);

}

#endif
