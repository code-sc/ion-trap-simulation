import numpy as np
import scipy.optimize as optimize

def _pot_energy(positions, ensemble_properties, potentials=[], dims=3):
    energy = 0
    for potential in potentials:
        energy += potential.potential(positions, ensemble_properties)
    return energy

def _jac(positions, ensemble_properties, potentials=[], dims=3):
    grad = np.zeros(dims*ensemble_properties["n"])
    for potential in potentials:
        grad += potential.jac(positions, ensemble_properties)
    return grad

def get_ring_eq_pos(ensemble_properties, offset=0, potentials=[], initial_radius=1e-4, method="BFGS"):
    n = ensemble_properties["n"]
    r_0 = np.zeros(3*n)
    for i in range(n):
        r_0[i] = initial_radius * np.cos(2*np.pi*(i/n))
        r_0[i+n] = initial_radius * np.sin(2*np.pi*(i/n))
        r_0[i+2*n] = 0
    
    if method == "BFGS":
        bfgs_tolerance = 1e-34
        opt = optimize.minimize(_pot_energy,
                                r_0,
                                args=(ensemble_properties, potentials, 3),
                                jac=_jac,
                                options={"gtol": bfgs_tolerance, "disp": False})
    return opt.x

def get_linear_eq_pos(ensemble_properties, initial_dist, potentials=[], method="BFGS"):
    n = ensemble_properties["n"]
    r_0 = np.zeros(n)
    for i in range(n):
        r_0[i] = i*initial_dist - (n // 2) * initial_dist
    
    if method == "BFGS":
        bfgs_tolerance = 1e-34
        opt = optimize.minimize(_pot_energy,
                                r_0,
                                args=(ensemble_properties, potentials, 1),
                                jac=_jac,
                                options={"gtol": bfgs_tolerance, "disp": False})
    return opt.x
