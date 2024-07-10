#ifndef ION_TRAP_POTENTIALS
#define ION_TRAP_POTENTIALS

#include <cstdlib>
#include <vector>

double ring_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
    );

std::vector<double> ring_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
    );

double ring_trap_harmonic_potential(
        const int n,
        const double mass, 
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    );

std::vector<double> ring_trap_harmonic_jac(
        const int n,
        const double mass, 
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
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

double mutual_coulomb_harmonic_potential(
        const int n,
        const double charge,
        const std::vector<std::vector<double> > &d,
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    );

std::vector<double> mutual_coulomb_harmonic_jac(
        const int n,
        const double charge,
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    );

double point_charge_potential(
        const int n,
        const double point_charge,
        const std::vector<double> &charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
    );

std::vector<double> point_charge_jac(
        const int n,
        const double point_charge,
        const std::vector<double> &charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
    );

double local_harmonic_potential_1d(
    const int n,
    const double mass,
    const std::vector<double> &w,
    const std::vector<double> &r
    );

std::vector<double> local_harmonic_jac_1d(
    const int n,
    const double mass,
    const std::vector<double> &w,
    const std::vector<double> &r
    );

double inverse_square_potential_1d(
    const int n,
    const std::vector<double> &d,
    const double ensemble_charge,
    const std::vector<double> &r
    );

std::vector<double> inverse_square_jac_1d(
    const int n,
    const std::vector<double> &d,
    const double ensemble_charge,
    const std::vector<double> &r
    );

#endif