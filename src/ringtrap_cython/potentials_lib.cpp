#define _USE_MATH_DEFINES

#include "potentials_lib.h"
#include <cstdlib>
#include <cmath>
#include <vector>

// physical constants
const double eps0 = 8.854187817e-12;
const double k = 1 / (4 * M_PI * eps0);

double ring_trap_potential(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
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

std::vector<double> ring_trap_jac(
        const int n,
        const double mass,
        const double trapRadius,
        const double wr,
        const double wz,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(3*n, 0);
    double r_coord;
    for (int i = 0; i < n; i++) {
        r_coord = sqrt(r[i]*r[i] + r[n+i]*r[n+i]);
        grad[i] = mass * wr*wr * (r_coord - trapRadius) * (r[i] / r_coord);
        grad[n+i] = mass * wr*wr * (r_coord - trapRadius) * (r[n+i] / r_coord);
        grad[2*n+i] = mass * wz*wz * r[2*n+i];
    }
    return grad;
}

double ring_trap_harmonic_potential(
        const int n,
        const double mass, 
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    )
{
    double energy = 0;
    for (int i = 0; i < n; i++) {
        energy += 0.5 * mass * hess[i][i] * (r[i] - r0[i]) * (r[i] - r0[i]);
        energy += 0.5 * mass * hess[n+i][n+i] * (r[n+i] - r0[n+i]) * (r[n+i] - r0[n+i]);
        energy += 0.5 * mass * hess[2*n+i][2*n+i] * (r[2*n+i] - r0[2*n+i]) * (r[2*n+i] - r0[2*n+i]);
        energy += mass * hess[i][n+i] * (r[i] - r0[i]) * (r[n+i] - r0[n+i]);
    }
    return energy;
}

std::vector<double> ring_trap_harmonic_jac(
        const int n,
        const double mass, 
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(3*n, 0);
    for (int i = 0; i < n; i++) {
        grad[i] = mass * hess[i][i] * (r[i] - r0[i]) + mass * hess[i][i+n] * (r[i+n] - r0[i+n]);
        grad[n+i] = mass * hess[n+i][i] * (r[i] - r0[i]) + mass * hess[n+i][i+n] * (r[i+n] - r0[i+n]);
        grad[2*n+i] = mass * hess[2*n+i][2*n+i] * (r[2*i+n] - r0[2*i+n]);
    }
    return grad;
}

double mutual_coulomb_potential(
        const int n,
        const std::vector<double> &charge,
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
            energy += ( (charge[i]*charge[j]) / (4*M_PI*eps0) ) * (1 / dist);
        }
    }
    return energy;
}

std::vector<double> mutual_coulomb_jac(
        const int n,
        const std::vector<double> &charge,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(3*n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double dist = sqrt(
                            (r[i]-r[j])*(r[i]-r[j])
                            + (r[n+i]-r[n+j])*(r[n+i]-r[n+j])
                            + (r[2*n+i]-r[2*n+j])*(r[2*n+i]-r[2*n+j]) );
            double coulomb = ( (charge[i]*charge[j]) / (4*M_PI*eps0) ) * (1 / (dist * dist));
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

double mutual_coulomb_harmonic_potential(
        const int n,
        const double charge,
        const std::vector<std::vector<double> > &d,
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    )
{
    double halfCharge2 = 0.5 * charge * charge;
    double energy = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) energy += (k*charge*charge) / d[i][j];
            energy += halfCharge2 * hess[i][j] * (r[i] - r0[i]) * (r[j] - r0[j]);
            energy += halfCharge2 * hess[n+i][n+j] * (r[n+i] - r0[n+i]) * (r[n+j] - r0[n+j]);
            energy += halfCharge2 * hess[2*n+i][2*n+j] * (r[2*n+i] - r0[2*n+i]) * (r[2*n+j] - r0[2*n+j]);
            energy += halfCharge2 * hess[i][n+j] * (r[i] - r0[i]) * (r[n+j] - r0[n+j]);
            energy += halfCharge2 * hess[n+i][2*n+j] * (r[n+i] - r0[n+i]) * (r[2*n+j] - r0[2*n+j]);
            energy += halfCharge2 * hess[2*n+i][j] * (r[2*n+i] - r0[2*n+i]) * (r[j] - r0[j]);
        }
    }
    return energy;
}

std::vector<double> mutual_coulomb_harmonic_jac(
        const int n,
        const double charge,
        const std::vector<std::vector<double> > &hess,
        const std::vector<double> &r0,
        const std::vector<double> &r
    )
{
    int charge2 = charge * charge;
    std::vector<double> grad(3*n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grad[i] += charge2 * ( hess[i][j]*(r[j] - r0[j]) + hess[i][n+j]*(r[n+j] - r0[n+j]) + hess[i][2*n+j]*(r[2*n+j] - r0[2*n+j]) );
            grad[n+i] += charge2 * ( hess[n+i][j]*(r[j] - r0[j]) + hess[n+i][n+j]*(r[n+j] - r0[n+j]) + hess[n+i][2*n+j]*(r[2*n+j] - r0[2*n+j]) );
            grad[2*n+i] += charge2 * ( hess[2*n+i][j]*(r[j] - r0[j]) + hess[2*n+i][n+j]*(r[n+j] - r0[n+j]) + hess[2*n+i][2*n+j]*(r[2*n+j] - r0[2*n+j]) );
        }
    }
    return grad;
}

double point_charge_potential(
        const int n,
        const double point_charge,
        const std::vector<double> &charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
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

std::vector<double> point_charge_jac(
        const int n,
        const double point_charge,
        const std::vector<double> &charge_r,
        const double ensemble_charge,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(3*n, 0);
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

double local_harmonic_potential_1d(
        const int n,
        const std::vector<double> &mass,
        const std::vector<double> &w,
        const std::vector<double> &r
    )
{
    double energy = 0;
    for (int i = 0; i < n; i++)
        energy += 0.5 * mass[i] * w[i] * w[i] * r[i] * r[i];
    return energy;
}

std::vector<double> local_harmonic_jac_1d(
        const int n,
        const std::vector<double> &mass,
        const std::vector<double> &w,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(n, 0);
    for (int i = 0; i < n; i++)
        grad[i] = mass[i] * w[i] * w[i] * r[i];
    return grad;
}

double inverse_square_potential_1d(
        const int n,
        const std::vector<double> &d,
        const std::vector<double> &ensemble_charge,
        const std::vector<double> &r
    )
{
    double energy = 0;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            energy += (ensemble_charge[i] * ensemble_charge[j]) / abs( (r[i] + d[i]) - (r[j] + d[j]) );
    return energy;
}

std::vector<double> inverse_square_jac_1d(
        const int n,
        const std::vector<double> &d,
        const std::vector<double> &ensemble_charge,
        const std::vector<double> &r
    )
{
    std::vector<double> grad(n, 0);
    double mag;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            mag = (ensemble_charge[i] * ensemble_charge[j]) / pow( (r[i] + d[i]) - (r[j] + d[j]), 2);
            grad[i] += ( (r[i] + d[i] > r[j] + d[j]) ? -1 : 1) * mag;
            grad[j] += ( (r[i] + d[i] > r[j] + d[j]) ? 1 : -1) * mag;
        }
    }
    return grad;
}

