import cython
cimport ion_trap_1d_lib
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
def total_energy(vector[double] x):
    cdef n = x.size()
    return ion_trap_1d_lib.total_energy(n, x)


@cython.boundscheck(False)
@cython.wraparound(False)
def neg_grad_energy(vector[double] x,
                    int k,
                    double dx=1e-9):
    cdef n = x.size()
    return ion_trap_1d_lib.neg_grad_energy(n, x, k, dx)

@cython.boundscheck(False)
@cython.wraparound(False)
def force(vector[double] x,
          int k):
    cdef n = x.size()
    return ion_trap_1d_lib.force(n, x, k);

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_leapfrog(int n, double T, double dt, double M, vector[double] x_0, vector[double] v_0, double dx=1e-9):
    return ion_trap_1d_lib.sim_leapfrog(n, T, dt, M, x_0, v_0, dx)

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_er(int n, int n_tsteps, double dt, double etol, double M, double b, vector[double] x_0, vector[double] v_0):
    return ion_trap_1d_lib.sim_er(n, n_tsteps, dt, etol, M, b, x_0, v_0)

@cython.boundscheck(False)
@cython.wraparound(False)
def a_dless(vector[double] x, const int k):
    cdef n = x.size();
    return ion_trap_1d_lib.a_dless(n, x, k);

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_leapfrog_dless(int n, double T, double dt, vector[double] x_0, vector[double] v_0):
    return ion_trap_1d_lib.sim_leapfrog_dless(n, T, dt, x_0, v_0);
