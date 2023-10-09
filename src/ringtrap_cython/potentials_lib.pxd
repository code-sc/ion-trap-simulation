from libcpp.vector cimport vector

cdef extern from "potentials_lib.h":
    double harmonic_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector[double]& r
    );
    vector[double] harmonic_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector[double]& r
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