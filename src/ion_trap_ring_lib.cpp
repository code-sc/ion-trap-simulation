#define _USE_MATH_DEFINES

#include "ion_trap_ring_lib.h"
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

// physical constants
const double Z = 1;
const double e = 1.60217883e-19;
const double eps0 = 8.854187817e-12;
const double M_Yb = 2.8733965e-25;
const double hbar = 1.05457182e-34;

// trap parameters
const double trap_radius = (40 * 1e-6) / (2*M_PI); // 1000 * 1e-6; // m
// const double trap_radius = 1e-3; // m
const double wr = 9 * 2*M_PI*1e6;
const double wz = 10 * 2*M_PI*1e6;

double pot_energy(
        const int n,
        const vector<double>& r
    )
{
    double energy = 0;

    // harmonic potential
    for (int i = 0; i < n; i++) {
        double r_coord = sqrt(r[3*i]*r[3*i] + r[3*i+1]*r[3*i+1]);
        energy += 0.5 * M_Yb * ( pow(wr*(r_coord-trap_radius), 2) + pow(wz*r[3*i+2], 2) );
    }

    // coulomb potential
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double dist = sqrt( pow(r[3*i]-r[3*j], 2) + pow(r[3*i+1]-r[3*j+1], 2) + pow(r[3*i+2]-r[3*j+2], 2) );
            energy += ((Z*Z*e*e) / (4*M_PI*eps0)) * (1/dist);
        }
    }

    return energy;
}

vector<double> total_energy(
    const int n,
    const vector<vector<double> >& r,
    const vector<vector<double> >& v
    )
{
    int n_tsteps = r.size();
    vector<double> tot_energy(n_tsteps);
    for (int i = 0; i < n_tsteps; i++) {
        tot_energy[i] = pot_energy(n, r[i]);
        for (int j = 0; j < 3*n; j++) tot_energy[i] += (v[i][j] * v[i][j]);
    }
    return tot_energy;
}

vector<double> acceleration(
        const int n,
        const vector<double>& r,
        const int k
    )
{
    vector<double> a(3);

    // trap forces
    const double r_coord = sqrt(r[3*k]*r[3*k] + r[3*k+1]*r[3*k+1]);
    a[0] = -(wr*wr) * (r_coord - trap_radius) * (r[3*k] / r_coord);
    a[1] = -(wr*wr) * (r_coord - trap_radius) * (r[3*k+1] / r_coord);
    a[2] = -(wz*wz) * r[3*k+2];

    // coulomb forces
    for (int i = 0; i < n; i++) {
        if (k != i) {
            double dri_mag = sqrt(pow(r[3*k]-r[3*i], 2) + pow(r[3*k+1]-r[3*i+1], 2) + pow(r[3*k+2]-r[3*i+2], 2));
            vector<double> dri(3);
            dri[0] = (r[3*k]-r[3*i]) / dri_mag; dri[1] = (r[3*k+1]-r[3*i+1]) / dri_mag; dri[2] = (r[3*k+2]-r[3*i+2]) / dri_mag;
            double ai_mag = ((Z * Z * e * e) / (4 * M_PI * eps0 * M_Yb)) * (1 / pow(dri_mag, 2));

            a[0] += ai_mag * dri[0];
            a[1] += ai_mag * dri[1];
            a[2] += ai_mag * dri[2];
        }
    }
    return a;
}

vector<vector<vector<double> > > sim_leapfrog(
        const int n,
        const double T,
        const double dt,
        std::vector<double>& r_0,
        std::vector<double>& v_0
    )
{
    const int n_tsteps = T / dt;
    vector<vector<double> > r(n_tsteps + 1, vector<double>(3*n));
    vector<vector<double> > v(n_tsteps + 1, vector<double>(3*n));
    vector<vector<double> > a(n_tsteps + 1, vector<double>(3*n));

    for (int k = 0; k < n; k++) {
        vector<double> accel = acceleration(n, r_0, k);
        for (int j = 0; j < 3; j++) {
            r[0][3*k+j] = r_0[3*k+j];
            v[0][3*k+j] = v_0[3*k+j];
            a[0][3*k+j] = accel[j];
        }
    }

    for (int i = 0; i < n_tsteps; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < 3; j++) {
                r[i+1][3*k+j] = r[i][3*k+j] + dt*v[i][3*k+j] + 0.5*dt*dt*a[i][3*k+j];
            }
        }
        for (int k = 0; k < n; k++) {
            vector<double> accel = acceleration(n, r[i+1], k);
            for (int j = 0; j < 3; j++) {
                a[i+1][3*k+j] = accel[j];
                v[i+1][3*k+j] = v[i][3*k+j] + 0.5*dt*(a[i][3*k+j] + a[i+1][3*k+j]);
            }
        }
    }

    vector<vector<vector<double> > > ret(3);
    ret[0] = r;
    ret[1] = v;
    ret[2] = a;
    return ret;
}