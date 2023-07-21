#ifndef ION_TRAP_RING_LIB
#define ION_TRAP_RING_LIB

#include <cstdlib>
#include <vector>

double pot_energy(
        const int n,
        const std::vector<double>& r
    );

std::vector<double> total_energy(
    const int n,
    const std::vector<std::vector<double> >& r,
    const std::vector<std::vector<double> >& v
    );

std::vector<double> acceleration(
        const int n,
        const std::vector<double>& r,
        const int k
    );

std::vector<std::vector<std::vector<double> > > sim_leapfrog(
    const int n,
    const double T,
    const double dt,
    std::vector<double>& r_0,
    std::vector<double>& v_0
    );

#endif