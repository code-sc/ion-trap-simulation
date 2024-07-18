from libcpp.vector cimport vector

cdef extern from "potentials_lib.h":
    double ring_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector[double]& r
    );
    vector[double] ring_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector[double]& r
    );
    double ring_trap_harmonic_potential(
        const int n,
        const double mass, 
        const vector[vector[double]] &hess,
        const vector[double] &r0,
        const vector[double] &r
    );
    vector[double] ring_trap_harmonic_jac(
        const int n,
        const double mass, 
        const vector[vector[double]] &hess,
        const vector[double] &r0,
        const vector[double] &r
    );
    double mutual_coulomb_potential(
        const int n,
        const double charge,
        const vector[double]& r
    );
    vector[double] mutual_coulomb_jac(
        const int n,
        const double charge,
        const vector[double]& r
    );
    double mutual_coulomb_harmonic_potential(
        const int n,
        const double charge,
        const vector[vector[double]] &d,
        const vector[vector[double]] &hess,
        const vector[double] &r0,
        const vector[double] &r
    );
    vector[double] mutual_coulomb_harmonic_jac(
        const int n,
        const double charge,
        const vector[vector[double]] &hess,
        const vector[double] &r0,
        const vector[double] &r
    );
    double point_charge_potential(
        const int n,
        const double point_charge,
        const vector[double]& charge_r,
        const double ensemble_charge,
        const vector[double]& r
    );
    vector[double] point_charge_jac(
        const int n,
        const double point_charge,
        const vector[double]& charge_r,
        const double ensemble_charge,
        const vector[double]& r
    );
    double local_harmonic_potential_1d(
        const int n,
        const vector[double]& mass,
        const vector[double]& w,
        const vector[double]& r
    );
    vector[double] local_harmonic_jac_1d(
        const int n,
        const vector[double]& mass,
        const vector[double]& w,
        const vector[double]& r
    );
    double inverse_square_potential_1d(
        const int n,
        const vector[double]& d,
        const vector[double]& ensemble_charge,
        const vector[double]& r
    );
    vector[double] inverse_square_jac_1d(
        const int n,
        const vector[double]& d,
        const vector[double]& ensemble_charge,
        const vector[double]& r
    );
