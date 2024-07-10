import numpy as np

def sim_leapfrog(T, dt, r_0, v_0, ensemble_properties, potentials=[], language="python", dims=3):
    """
    Use the leapfrog algorithm to simulate the positions and velocity of the ions over time.
    T (units: s) is the total duration of the simulation
    dt (units: s) is the timestep of the simulation
    r_0 (units: m) is the initial positions of the ions.
    v_0 (units: m/s) is the initial velocities of the ions.
    ensemble_properties is an object of properties that are passed to each potential in the simulation.
    potentials is a list of potentials to include in the simulation
    language can be either "python" or "c++"
    dims is the number of dimensions in the system of interest (ex. can design potentials for simulation of a 1D ion chain)
    """
    n = ensemble_properties["n"]
    n_tsteps = int(T/dt)
    r_sim = np.zeros((n_tsteps+1, dims*n))
    v_sim = np.zeros((n_tsteps+1, dims*n))
    a_sim = np.zeros((n_tsteps+1, dims*n))

    r_sim[0] = r_0
    v_sim[0] = v_0
    for potential in potentials:
        a_sim[0] += potential.acceleration(r_sim[0], ensemble_properties, language=language)

    for i in range(n_tsteps):
        r_sim[i+1] = r_sim[i] + dt*v_sim[i] + 0.5*dt*dt*a_sim[i]
        for potential in potentials:
            a_sim[i+1] += potential.acceleration(r_sim[i+1], ensemble_properties, language=language)
        v_sim[i+1] = v_sim[i] + 0.5*dt*(a_sim[i] + a_sim[i+1])

    return r_sim, v_sim, a_sim
