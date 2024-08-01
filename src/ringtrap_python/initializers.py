import numpy as np

def fibonacciSphere(samples=1000,r=1):
    """Distribute sample number of points on a sphere using the fibonacci spiral
    samples : The number of particles on the sphere
    r       : Radius of the sphere
    """
    
    points = np.zeros(3*samples)
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        points[i+samples] = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - points[i+samples] * points[i+samples])  # radius at y

        theta = phi * i  # golden angle increment

        points[i] = np.cos(theta) * radius
        points[i+2*samples] = np.sin(theta) * radius

    return points*r

def genMaxwellVelocities(T,mass):
    """For the given temperature and mass sample velocity(m/s) from a Maxwell-Boltzmann distribution
    T       : Temperature (Kelvin)
    mass    : Array of masses of each particle (kg)
    """
    kb = 1.38e-23
    scale = np.sqrt(kb*T/mass)
    vel = np.random.normal(loc=0.0,scale=1.0,size=len(mass)*3)
    for idx,s in enumerate(scale):
        vel[idx::len(mass)] *= s
    return vel

def steepestDescent(positions,ensembleProperties,potentials,stepsize,threshold,maxSteps=10000,debug=False):
    """Returns the energy minimiseed structure for the given initial positions and array of potentials
    """
    n = ensembleProperties["n"]
    forceList = np.zeros(3*n)
    maxForce = 0
    for i in range(maxSteps):
        if debug:
            if i%1000 ==0:
                print("step : ",i)
        forceList[::] = 0
        for pot in potentials:
            forceList += pot.force(positions,ensembleProperties)
        maxForce = max(forceList.max() , forceList.min(), key=abs)
        if maxForce < threshold:
            print("[",i,"]","Structure energy minimised with max Force : ",maxForce)
            return positions
        positions += forceList*stepsize
    print("Structure not minimised")
    return positions
    

