import cython
cimport potentials_lib
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
def harmonic_trap_potential(double mass, double trap_radius, double wr, double wz, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.harmonic_trap_potential(n, mass, trap_radius, wr, wz, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def harmonic_trap_jac(double mass, double trap_radius, double wr, double wz, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.harmonic_trap_jac(n, mass, trap_radius, wr, wz, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_potential(double charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_potential(n, charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_jac(double charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_jac(n, charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def point_charge_potential(double point_charge, vector[double] charge_r, double ensemble_charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.point_charge_potential(n, point_charge, charge_r, ensemble_charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def point_charge_jac(double point_charge, vector[double] charge_r, double ensemble_charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.point_charge_jac(n, point_charge, charge_r, ensemble_charge, r)