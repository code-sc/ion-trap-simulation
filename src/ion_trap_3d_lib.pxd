from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "ion_trap_3d_lib.h":
    vector[double] acceleration(
        const int n,
        const vector[double]& r,
        const int k
    );
    vector[double] acceleration(
        const int n,
        const vector[double]& r,
        const vector[double]& v,
        const int k
    );
    pair[vector[vector[double]], vector[vector[double]]] sim_leapfrog(
        const int n,
        const double T,
        const double dt,
        vector[double]& r_0,
        vector[double]& v_0
    );
    vector[vector[vector[double]]] sim_er(
        const int n,
        const int n_tsteps,
        double dt,
        double etol,
        vector[double]& r_0,
        vector[double]& v_0
    );
