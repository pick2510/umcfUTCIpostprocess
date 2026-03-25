#ifndef UTCISOLVER_H
#define UTCISOLVER_H

#include "types.H"

namespace utci {

class UtciSolver {
public:
    UtciSolver() = default;
    ~UtciSolver() = default;
    
    double calculate(double Ta_c, double va, double RH, double Tmrt_c);
    
private:
    double calcVaporPressure(double Ta_K, double RH);
};

}

#endif
