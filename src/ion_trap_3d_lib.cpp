#define _USE_MATH_DEFINES

#include "ion_trap_3d_lib.h"
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
const double wx = 5.7 * 2*M_PI*1e6;
const double wy = 5.7 * 2*M_PI*1e6;
const double wz = 1.5 * 2*M_PI*1e6;

// laser cooling parameters
const double laserWavelength = 369 * 1e-9;
const double k = (2*M_PI) / laserWavelength;
const double laserOrigin[] = {0, 0, 0};
const double kvector[] = { k/sqrt(2), k/sqrt(2), 0 };
const double laserWidth = 1e-5;

const double decayRate = 20 * 1e6 * 2*M_PI;
const double detuning = -decayRate / 2;
const double s = 1;

vector<double> acceleration(
        const int n,
        const vector<double> &r,
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

vector<double> acceleration(
        const int n,
        const vector<double> &r,
        const vector<double> &v,
        const int k
    )
{
    vector<double> a(3);
    // Harmonic Potential Force
    a[0] = -(wx*wx)*r[3*k];
    a[1] = -(wy*wy)*r[3*k+1];
    a[2] = -(wz*wz)*r[3*k+2];

    // Laser Cooling
    double kdotv = kvector[0]*v[3*k] + kvector[1]*v[3*k+1] + kvector[2]*v[3*k+2];
    double rhoee = (s/2) / (1 + s + pow( (2*(detuning - kdotv)) / decayRate, 2) );
    a[0] += kvector[0] * (hbar * decayRate * rhoee) / M_Yb;
    a[1] += kvector[1] * (hbar * decayRate * rhoee) / M_Yb;
    a[2] += kvector[2] * (hbar * decayRate * rhoee) / M_Yb;

    // Coulomb Force
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

double dist_to_laser(const vector<double> &r, const int k) 
{
    vector<double> rminuso(3);
    for (int i = 0; i < 3; i++)
        rminuso[i] = r[3*k+i] - laserOrigin[i];
    double kvectorLength = sqrt(kvector[0]*kvector[0] + kvector[1]*kvector[1] + kvector[2]*kvector[2]);
    double crossProductLength = sqrt(pow(rminuso[1]*kvector[2]-rminuso[2]*kvector[1], 2) 
                                        + pow(rminuso[2]*kvector[0]-rminuso[0]*kvector[2], 2)
                                        + pow(rminuso[0]*kvector[1]-rminuso[1]*kvector[0], 2));
    return crossProductLength / kvectorLength;
}

vector<double> acceleration(
    const int n,
    const vector<double> &r,
    const vector<double> &v,
    const int k,
    const double laser_width
    )
{
    vector<double> a(3);
    // Harmonic Potential Force
    a[0] = -(wx*wx)*r[3*k];
    a[1] = -(wy*wy)*r[3*k+1];
    a[2] = -(wz*wz)*r[3*k+2];

    // Laser Cooling
    double kdotv = kvector[0]*v[3*k] + kvector[1]*v[3*k+1] + kvector[2]*v[3*k+2];
    double satParam = s * exp( -2 * pow(dist_to_laser(r, k), 2) / (laser_width * laser_width));
    double scattering_rate = (decayRate * satParam) / (1 + 2*satParam + pow( (2*(detuning - kdotv)) / decayRate, 2) );
    a[0] += kvector[0] * (hbar * scattering_rate) / M_Yb;
    a[1] += kvector[1] * (hbar * scattering_rate) / M_Yb;
    a[2] += kvector[2] * (hbar * scattering_rate) / M_Yb;

    // Coulomb Force
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

vector<vector<vector<double> > > sim_er(
    const int n,
    const int n_tsteps,
    double dt,
    double etol,
    vector<double>& r_0,
    vector<double>& v_0
    )
{
    vector<vector<double> > r(n_tsteps + 1, vector<double>(3*n, 0));
    vector<vector<double> > v(n_tsteps + 1, vector<double>(3*n, 0));
    vector<vector<double> > a(n_tsteps + 1, vector<double>(3*n, 0));
    vector<vector<double> > t(n_tsteps + 1, vector<double>(3*n, 0));
    vector<vector<double> > err(n_tsteps + 1, vector<double>(3*n, 0));
    double ahalf, rerr, verr, maxerr;

    for (int k = 0; k < n; k++) {
        vector<double> accel = acceleration(n, r_0, v_0, k, laserWidth);
        for (int j = 0; j < 3; j++) {
            r[0][3*k+j] = r_0[3*k+j];
            v[0][3*k+j] = v_0[3*k+j];
            a[0][3*k+j] = accel[j];
        }
    }

    for (int i = 0; i < n_tsteps; i++) {
        vector<double> rhalf(3*n, 0);
        vector<double> vhalf(3*n, 0);
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < 3; j++) {
                rhalf[3*k+j] = r[i][3*k+j] + v[i][3*k+j]*(dt/2);
                vhalf[3*k+j] = v[i][3*k+j] + a[i][3*k+j]*(dt/2);
            }
        }
        for (int k = 0; k < n; k++) {
            vector<double> ahalf = acceleration(n, rhalf, vhalf, k, laserWidth);
            for (int j = 0; j < 3; j++) {
                r[i+1][3*k+j] = r[i][3*k+j] + vhalf[3*k+j]*dt;
                v[i+1][3*k+j] = v[i][3*k+j] + ahalf[j]*dt;
                // rerr = abs( ( (v[i][3*k+j] - vhalf[3*k+j])*dt ) / 2);
                // verr = abs( ( (a[i][3*k+j] - ahalf[j])*dt ) / 2);
                // err[i][3*k+j] = max(rerr, verr);
            }
        }
        for (int k = 0; k < n; k++) {
            vector<double> accel = acceleration(n, r[i+1], v[i+1], k, laserWidth);
            for (int j = 0; j < 3; j++)
                a[i+1][3*k+j] = accel[j];
        }
        t[i+1][0] = t[i][0] + dt;
    }

    vector<vector<vector<double> > > ret(5);
    ret[0] = r;
    ret[1] = v;
    ret[2] = a;
    ret[3] = t;
    ret[4] = err;
    return ret;
}
