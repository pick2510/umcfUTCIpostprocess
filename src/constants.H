#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>

namespace utci {

constexpr double SIGMA = 5.67e-8;           // Stefan-Boltzmann constant [W/m²/K⁴]
constexpr double EPS_LW_PERSON = 0.97;      // Long-wave emissivity of person
constexpr double ABS_SW_PERSON = 0.7;       // Short-wave absorptivity of person
constexpr double EPS_SURF = 0.9;            // Default surface emissivity
constexpr double R_MAG_MAX = 100.0;         // Maximum ray distance for view factors [m]
constexpr double PED_Z = 2.0;               // Pedestrian height [m]

constexpr double PI = M_PI;
constexpr double PI_INV = 1.0 / M_PI;

constexpr double AIR_GAS_CONSTANT = 461.5;  // Gas constant for water vapor [J/(kg·K)]

constexpr double INVALID_SENTINEL = -1.79769e+307;

}

#endif
