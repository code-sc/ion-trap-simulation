#ifndef ION_TRAP_1D_LIB
#define ION_TRAP_1D_LIB

#include <cstdlib>
#include <vector>

double total_energy(
    const int n,
    std::vector<double>& x
    );

double neg_grad_energy(
    const int n,
    std::vector<double>& x,
    const int k,
    const double dx=1e-9
    );

double force(
    const int n,
    std::vector<double>& x,
    const int k
    );

std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > sim_leapfrog(
    const int n,
    const double T,
    const double dt,
    const double M,
    std::vector<double>& x_0,
    std::vector<double>& v_0,
    double dx=1e-9
);

double a_dless(
    const int n,
    std::vector<double>& x, 
    const int k
    );

std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > sim_leapfrog_dless(
        const int n,
        double T,
        double dt, 
        std::vector<double>& x_0,
        std::vector<double>& v_0
    );

#endif