from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "ion_trap_ring_lib.h":
    double pot_energy(
        const int n,
        const vector[double]& r
    );
    vector[double] total_energy(
        const int n,
        const vector[vector[double]]& r,
        const vector[vector[double]]& v
    );
    vector[double] acceleration(
        const int n,
        const vector[double]& r,
        const int k
    );
    vector[vector[vector[double]]] sim_leapfrog(
        const int n,
        const double T,
        const double dt,
        vector[double]& r_0,
        vector[double]& v_0
    );
