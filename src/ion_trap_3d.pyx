import cython
cimport ion_trap_3d_lib
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
def acceleration(vector[double] r, int k):
    cdef n = r.size() // 3
    return ion_trap_3d_lib.acceleration(n, r, k)

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_leapfrog(int n, double T, double dt, vector[double] r_0, vector[double] v_0):
    return ion_trap_3d_lib.sim_leapfrog(n, T, dt, r_0, v_0)
