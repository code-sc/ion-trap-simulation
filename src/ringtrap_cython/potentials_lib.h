#ifndef ION_TRAP_POTENTIALS
#define ION_TRAP_POTENTIALS

#include <cstdlib>
#include <vector>

double harmonic_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
    );

std::vector<double> harmonic_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
    );

double mutual_coulomb_potential(
        const int n,
        const double charge,
        const std::vector<double> &r
    );

std::vector<double> mutual_coulomb_jac(
        const int n,
        const double charge,
        const std::vector<double> &r
    );

double point_charge_potential(
        const int n,
        const double point_charge,
        const std::vector<double> charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
    );

std::vector<double> point_charge_jac(
        const int n,
        const double point_charge,
        const std::vector<double> charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
    );

#endif