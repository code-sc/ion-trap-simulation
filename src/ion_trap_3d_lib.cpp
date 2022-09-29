#define _USE_MATH_DEFINES

#include "ion_trap_3d_lib.h"
#include <cstdlib>
#include <cmath>
#include <vector>

// physical constants
static const double Z = 1;
static const double e = 1.60217883e-19;
static const double eps0 = 8.854187817e-12;
static const double M_Yb = 2.8733965e-25;

// trap parameters
static const double wx = 5.7 * 2*M_PI*1e6;
static const double wy = 5.7 * 2*M_PI*1e6;
static const double wz = 1 * 2*M_PI*1e6;

using namespace std;

vector<double> acceleration(
        const int n,
        std::vector<double> &r,
        const int k
    )
{
    vector<double> a(3);
    a[0] = -(wx*wx)*r[3*k]; a[1] = -(wy*wy)*r[3*k+1]; a[2] = -(wz*wz)*r[3*k+2];
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

pair<vector<vector<double> >, vector<vector<double> > > sim_leapfrog(
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
    vector<vector<double> > vhalf(n_tsteps + 1, vector<double>(3*n));
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
                vhalf[i][3*k+j] = v[i][3*k+j] + 0.5*dt*a[i][3*k+j];
                r[i+1][3*k+j] = r[i][3*k+j] + dt*vhalf[i][3*k+j];
            }
        }
        for (int k = 0; k < n; k++) {
            vector<double> accel = acceleration(n, r[i+1], k);
            for (int j = 0; j < 3; j++) {
                a[i+1][3*k+j] = accel[j];
                v[i+1][3*k+j] = vhalf[i][3*k+j] + 0.5*dt*a[i+1][3*k+j];
            }
        }
    }

    pair<vector<vector<double> >, vector<vector<double> > > ret = make_pair(r, v);
    return ret;
}
