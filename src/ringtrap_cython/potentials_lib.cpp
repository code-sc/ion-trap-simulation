#define _USE_MATH_DEFINES

#include "potentials_lib.h"
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

// physical constants
const double eps0 = 8.854187817e-12;

double harmonic_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector<double> &r
    )
{
    double energy = 0;
    double r_coord;
    for (int i = 0; i < n; i++) {
        r_coord = sqrt(r[i]*r[i] + r[n+i]*r[n+i]);
        energy += 0.5 * mass * (wr*wr * (r_coord-trapRadius)*(r_coord-trapRadius) + wz*wz * r[2*n+i]*r[2*n+i]);
    }
    return energy;
}

vector<double> harmonic_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const vector<double> &r
    )
{
    vector<double> grad(3*n, 0);
    double r_coord;
    for (int i = 0; i < n; i++) {
        r_coord = sqrt(r[i]*r[i] + r[n+i]*r[n+i]);
        grad[i] = mass * wr*wr * (r_coord - trapRadius) * (r[i] / r_coord);
        grad[n+i] = mass * wr*wr * (r_coord - trapRadius) * (r[n+i] / r_coord);
        grad[2*n+i] = mass * wz*wz * r[2*n+i];
    }
    return grad;
}

double mutual_coulomb_potential(
        const int n,
        const double charge,
        const std::vector<double> &r
    )
{
    double energy = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double dist = sqrt(
                            (r[i]-r[j])*(r[i]-r[j])
                            + (r[n+i]-r[n+j])*(r[n+i]-r[n+j])
                            + (r[2*n+i]-r[2*n+j])*(r[2*n+i]-r[2*n+j]) );
            energy += ( (charge*charge) / (4*M_PI*eps0) ) * (1 / dist);
        }
    }
    return energy;
}

vector<double> mutual_coulomb_jac(
        const int n,
        const double charge,
        const std::vector<double> &r
    )
{
    vector<double> grad(3*n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double dist = sqrt(
                            (r[i]-r[j])*(r[i]-r[j])
                            + (r[n+i]-r[n+j])*(r[n+i]-r[n+j])
                            + (r[2*n+i]-r[2*n+j])*(r[2*n+i]-r[2*n+j]) );
            double coulomb = ( (charge*charge) / (4*M_PI*eps0) ) * (1 / (dist * dist));
            grad[i] -= coulomb * ((r[i] - r[j]) / dist);
            grad[n+i] -= coulomb * ((r[n+i] - r[n+j]) / dist);
            grad[2*n+i] -= coulomb * ((r[2*n+i] - r[2*n+j]) / dist);
            grad[j] += coulomb * ((r[i] - r[j]) / dist);
            grad[n+j] += coulomb * ((r[n+i] - r[n+j]) / dist);
            grad[2*n+j] += coulomb * ((r[2*n+i] - r[2*n+j]) / dist);
        }
    }
    return grad;
}

double point_charge_potential(
        const int n,
        const double point_charge,
        const vector<double> charge_r,
        const double ensemble_charge,
        const vector<double> &r
    )
{
    double energy = 0;
    for (int i = 0; i < n; i++) {
        double dist = sqrt(
                        (r[i]-charge_r[0])*(r[i]-charge_r[0])
                        + (r[n+i]-charge_r[1])*(r[n+i]-charge_r[1])
                        + (r[2*n+i]-charge_r[2])*(r[2*n+i]-charge_r[2]) );
        energy += ( (point_charge*ensemble_charge) / (4*M_PI*eps0) ) * (1 / dist);
    }
    return energy;
}

vector<double> point_charge_jac(
        const int n,
        const double point_charge,
        const vector<double> charge_r,
        const double ensemble_charge,
        const vector<double> &r
    )
{
    vector<double> grad(3*n, 0);
    for (int i = 0; i < n; i++) {
        double dist = sqrt(
                        (r[i]-charge_r[0])*(r[i]-charge_r[0])
                        + (r[n+i]-charge_r[1])*(r[n+i]-charge_r[1])
                        + (r[2*n+i]-charge_r[2])*(r[2*n+i]-charge_r[2]) );
        double coulomb = ( (point_charge*ensemble_charge) / (4*M_PI*eps0) ) * (1 / (dist * dist));
        grad[i] -= coulomb * ((r[i] - charge_r[0]) / dist);
        grad[n+i] -= coulomb * ((r[n+i] - charge_r[1]) / dist);
        grad[2*n+i] -= coulomb * ((r[2*n+i] - charge_r[2]) / dist);
    }
    return grad;
}