import numpy as np
from ..ringtrap_cython import potentials as cpotentials

eps0 = 8.854187817e-12

class HarmonicTrapPotential:
    def __init__(self, trap_radius, wr, wz):
        self.trap_radius = trap_radius
        self.wr = wr
        self.wz = wz

    def potential(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            r_coords = np.sqrt(positions[:n]**2 + positions[n:2*n]**2)
            energy = np.sum(0.5 * ensemble_properties["mass"] * ( (self.wr ** 2) * ((r_coords - self.trap_radius) ** 2) + (self.wz **2) * (positions[2*n:3*n] ** 2) ))
            return energy
        else:
            return cpotentials.harmonic_trap_potential(ensemble_properties["mass"], 
                                                       self.trap_radius,
                                                       self.wr, 
                                                       self.wz,
                                                       positions)


    def jac(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            x = positions[:n]
            y = positions[n:2*n]
            z = positions[2*n:3*n]
            r_coords = np.sqrt(x**2 + y**2)
            grad[:n] = ((self.wr)**2) * (r_coords - self.trap_radius) * (x / r_coords)
            grad[n:2*n] = ((self.wr)**2) * (r_coords - self.trap_radius) * (y / r_coords)
            grad[2*n:3*n] = ((self.wz)**2) * z
            return ensemble_properties["mass"] * grad
        else:
            return np.array(cpotentials.harmonic_trap_jac(ensemble_properties["mass"],
                                                 self.trap_radius,
                                                 self.wr, 
                                                 self.wz,
                                                 positions))

    def force(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class MutualCoulombPotential:
    def __init__(self, cutoff_distance):
        self.cutoff_distance = cutoff_distance

    def potential(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            energy = 0
            for i in range(n):
                for j in range(i+1, n):
                    ri = positions[i::n]
                    rj = positions[j::n]
                    energy += ((ensemble_properties["charge"] ** 2) / (4*np.pi*eps0)) * (1 / np.linalg.norm(ri - rj))
            return energy
        else:
            return cpotentials.mutual_coulomb_potential(ensemble_properties["charge"], positions)


    def jac(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            for i in range(n):
                for j in range(i+1,n):
                    ri = positions[i::n]
                    rj = positions[j::n]
                    coulomb = ((ensemble_properties["charge"] ** 2) / (4*np.pi*eps0)) * (1 / (np.linalg.norm(ri - rj) ** 2))
                    dr = (ri - rj) / np.linalg.norm(ri - rj)
                    grad[i::n] -= coulomb * dr 
                    grad[j::n] += coulomb * dr
            return grad
        else:
            return np.array(cpotentials.mutual_coulomb_jac(ensemble_properties["charge"], positions))

    def force(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class PointChargePotential:
    def __init__(self, position, charge):
        self.position = position
        self.charge = charge
    
    def potential(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            energy = 0
            for i in range(n):
                ri = positions[i::n]
                energy += ((ensemble_properties["charge"] * self.charge) / (4*np.pi*eps0)) * (1 / np.linalg.norm(ri - self.position))
            return energy
        else:
            return cpotentials.point_charge_potential(self.charge,
                                                      self.position,
                                                      ensemble_properties["charge"],
                                                      positions)

    def jac(self, positions, ensemble_properties, language="python"):
        if language == "python":
            n = ensemble_properties["n"]
            grad = np.zeros(3*n)
            for i in range(n):
                ri = positions[i::n]
                coulomb = ((ensemble_properties["charge"] * self.charge) / (4*np.pi*eps0)) * (1 / (np.linalg.norm(ri - self.position) ** 2))
                dr = (ri - self.position) / np.linalg.norm(ri - self.position)
                grad[i::n] -= coulomb * dr
            return grad
        else:
            return np.array(cpotentials.point_charge_jac(self.charge,
                                                      self.position,
                                                      ensemble_properties["charge"],
                                                      positions))

    def force(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language)

    def acceleration(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class LocalHarmonicPotential1D:
    def __init__(self, w):
        self.w = w

    def potential(self, positions, ensemble_properties, language="python"):
        return 0.5 * ensemble_properties["mass"] * np.sum(self.w * self.w * positions * positions)
    
    def jac(self, positions, ensemble_properties, language="python"):
        return ensemble_properties["mass"] * self.w * self.w * positions
    
    def force(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language)
    
    def acceleration(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]

class InverseSquarePotential1D:
    def __init__(self, d):
        self.d = d

    def potential(self, positions, ensemble_properties, language="python"):
        n = ensemble_properties["n"]
        energy = 0
        for i in range(n):
            for j in range(i+1, n):
                energy += (ensemble_properties["charge"] ** 2) / np.abs( (positions[i] + self.d[i]) - (positions[j] + self.d[j]) )
        return energy
    
    def jac(self, positions, ensemble_properties, language="python"):
        n = ensemble_properties["n"]
        grad = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                mag = (ensemble_properties["charge"] ** 2) / (( (positions[i] + self.d[i]) - (positions[j] + self.d[j]) ) ** 2)
                grad[i] += (-1 if (positions[i] + self.d[i] > positions[j] + self.d[j]) else 1) * mag
                grad[j] += (1 if (positions[i] + self.d[i] > positions[j] + self.d[j]) else -1) * mag
        return grad
    
    def force(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language)
    
    def acceleration(self, positions, ensemble_properties, language="python"):
        return -self.jac(positions, ensemble_properties, language=language) / ensemble_properties["mass"]
