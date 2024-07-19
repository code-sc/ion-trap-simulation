import cython
cimport potentials_lib
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
def ring_trap_potential(double mass, double trap_radius, double wr, double wz, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.ring_trap_potential(n, mass, trap_radius, wr, wz, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def ring_trap_jac(double mass, double trap_radius, double wr, double wz, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.ring_trap_jac(n, mass, trap_radius, wr, wz, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def ring_trap_harmonic_potential(double mass, vector[vector[double]] hess, vector[double] r0, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.ring_trap_harmonic_potential(n, mass, hess, r0, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def ring_trap_harmonic_jac(double mass, vector[vector[double]] hess, vector[double] r0, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.ring_trap_harmonic_jac(n, mass, hess, r0, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_potential(vector[double] charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_potential(n, charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_jac(vector[double] charge, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_jac(n, charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_harmonic_potential(double charge, vector[vector[double]] d, vector[vector[double]] hess, vector[double] r0, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_harmonic_potential(n, charge, d, hess, r0, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_coulomb_harmonic_jac(double charge, vector[vector[double]] hess, vector[double] r0, vector[double] r):
    cdef n = r.size() // 3
    return potentials_lib.mutual_coulomb_harmonic_jac(n, charge, hess, r0, r)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def local_harmonic_potential_1d(vector[double] mass, vector[double] w, vector[double] r):
    cdef n = r.size()
    return potentials_lib.local_harmonic_potential_1d(n, mass, w, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def local_harmonic_jac_1d(vector[double] mass, vector[double] w, vector[double] r):
    cdef n = r.size()
    return potentials_lib.local_harmonic_jac_1d(n, mass, w, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def inverse_square_potential_1d(vector[double] d, vector[double] ensemble_charge, vector[double] r):
    cdef n = r.size()
    return potentials_lib.inverse_square_potential_1d(n, d, ensemble_charge, r)

@cython.boundscheck(False)
@cython.wraparound(False)
def inverse_square_jac_1d(vector[double] d, vector[double] ensemble_charge, vector[double] r):
    cdef n = r.size()
    return potentials_lib.inverse_square_jac_1d(n, d, ensemble_charge, r)
