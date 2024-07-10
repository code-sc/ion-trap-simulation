import numpy as np
import scipy.optimize as optimize

def _pot_energy(positions, ensemble_properties, potentials=[], dims=3):
    """Helper function to compute the total potential energy given a list of potentials and the ion positions."""
    energy = 0
    for potential in potentials:
        energy += potential.potential(positions, ensemble_properties)
    return energy

def _jac(positions, ensemble_properties, potentials=[], dims=3):
    """Helper function to compute the gradient of the total potential energy given a list of potentials and the ion positions."""
    grad = np.zeros(dims*ensemble_properties["n"])
    for potential in potentials:
        grad += potential.jac(positions, ensemble_properties)
    return grad

def get_ring_eq_pos(ensemble_properties, offset=0, potentials=[], initial_radius=1e-4, method="BFGS"):
    """
    Compute the equilibrium positions of the ions in a ring trap
    offset (units: rad) is the angular offset of the first ion in the ring
    initial_radius (units: m) is the initial radius of the ion chain before minimizing the potential
    """
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
    """
    Compute the equilibrium positions of the ions in a 1D linear trap
    initial_dist (units: m) is the initial distance between ions before minimizing the potential.
    """
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
