#ifndef ION_TRAP_3D_LIB
#define ION_TRAP_3D_LIB

#include <cstdlib>
#include <vector>

std::vector<double> acceleration(
        const int n,
        const std::vector<double>& r,
        const int k
    );

std::vector<double> acceleration(
    const int n,
    const std::vector<double>& r,
    const std::vector<double>& v,
    const int k
    );

std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > sim_leapfrog(
    const int n,
    const double T,
    const double dt,
    std::vector<double>& r_0,
    std::vector<double>& v_0
    );

std::vector<std::vector<std::vector<double> > > sim_er(
    const int n,
    const int n_tsteps,
    double dt,
    double etol,
    std::vector<double>& r_0,
    std::vector<double>& v_0
    );  

#endif