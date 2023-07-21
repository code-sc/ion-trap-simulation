import cython
cimport ion_trap_ring_lib
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
def pot_energy(vector[double] r):
    cdef n = r.size() // 3
    return ion_trap_ring_lib.pot_energy(n, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def total_energy(vector[vector[double]] r, vector[vector[double]] v):
    cdef n = r[0].size() // 3
    return ion_trap_ring_lib.total_energy(n, r, v)

@cython.boundscheck(False)
@cython.wraparound(False)
def acceleration(vector[double] r, int k):
    cdef n = r.size() // 3
    return ion_trap_ring_lib.acceleration(n, r, k)

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_leapfrog(int n, double T, double dt, vector[double] r_0, vector[double] v_0):
    return ion_trap_ring_lib.sim_leapfrog(n, T, dt, r_0, v_0)
