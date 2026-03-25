#include "pedestrian.h"
#include "constants.h"
#include <cmath>

namespace utci {

PedestrianPosition createPedestrianPosition(const Point3& center) {
    PedestrianPosition ped;
    ped.center = center;
    
    const double dx = 0.2;
    const double dy = 0.2;
    const double dz = 1.0;
    
    // Matching Python reference:
    // Body points: front at x-0.2, back at x+0.2, left at y-0.2, right at y+0.2, top at center
    // Area vectors: front in -X, back in +X, left in -Y, right in +Y, top in +Z
    ped.bodyPoints[0] = {center.x - dx, center.y, center.z - dz};  // front
    ped.bodyPoints[1] = {center.x + dx, center.y, center.z - dz};  // back
    ped.bodyPoints[2] = {center.x, center.y - dy, center.z - dz};  // left
    ped.bodyPoints[3] = {center.x, center.y + dy, center.z - dz};  // right
    ped.bodyPoints[4] = {center.x, center.y, center.z};            // top
    
    // Area vectors: front/back in ±X, left/right in ±Y, top in +Z
    ped.areaVectors[0] = Vec3(-0.68, 0.0, 0.0);  // front (facing -X)
    ped.areaVectors[1] = Vec3(0.68, 0.0, 0.0);   // back (facing +X)
    ped.areaVectors[2] = Vec3(0.0, -0.68, 0.0);  // left (facing -Y)
    ped.areaVectors[3] = Vec3(0.0, 0.68, 0.0);   // right (facing +Y)
    ped.areaVectors[4] = Vec3(0.0, 0.0, 0.16);    // top (facing +Z)
    
    return ped;
}

}
