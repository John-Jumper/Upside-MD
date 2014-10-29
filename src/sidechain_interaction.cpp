#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

inline double
nonbonded_kernel(double r_mag2)
{
    // V(r) = 1/(1+exp(s*(r**2-d**2)))
    // V(d+width) is approximately V(d) + V'(d)*width
    // this equals 1/2 - (1/2)^2 * 2*s*r * width = (1/2) * (1 - s*(r*width))
    // Based on this, to get a characteristic scale of width,
    //   s should be 1/(wall_radius * width)

    // V'(r) = -2*s*r * z/(1+z)^2 where z = exp(s*(r**2-d**2))
    // V'(r)/r = -2*s*z / (1+z)^2

    const double wall = 3.2f;  // corresponds to vdW *diameter*
    const double wall_squared = wall*wall;  
    const double width = 0.15f;
    const double scale_factor = 1.f/(wall*width);  // ensure character

    // overflow protection prevents NaN
    double z = exp(scale_factor * (r_mag2-wall_squared));
    double w = 1.f/(1.f + z);  // include protection from 0

    return w;
}


inline double
mayer_f_function(double energy_scale, double r_mag2)
{
    // f(r) = exp(-V(r)) - 1
    return exp(-energy_scale*nonbonded_kernel(r_mag2)) - 1.;
}

extern "C" {
    void interaction_function(double* f_potential, int n_dists, const double* restrict dists, 
            double energy_scale, int n_pairs, double sigma1, double sigma2, unsigned seed);
}

void interaction_function(double* f_potential, int n_dists, const double* restrict dists, 
        double energy_scale, int n_pairs, double sigma1, double sigma2, unsigned seed)
{

    for(int nd=0; nd<n_dists; ++nd) f_potential[nd] = 0.;

    auto gen = mt19937{seed};
    auto normal_dist = normal_distribution<>(0.,1.);

    auto weight = 1. / n_pairs;

    for(int np=0; np<n_pairs; ++np) {
        // first compute displacement at center distance = 0
        double disp[3] = {
            sigma1*normal_dist(gen) - sigma2*normal_dist(gen),
            sigma1*normal_dist(gen) - sigma2*normal_dist(gen),
            sigma1*normal_dist(gen) - sigma2*normal_dist(gen)};

        for(int nd=0; nd<n_dists; ++nd) {
            auto d2 = disp[0]*disp[0] + disp[1]*disp[1] + (disp[2]+dists[nd])*(disp[2]+dists[nd]);
            f_potential[nd] += weight * mayer_f_function(energy_scale, d2);
        }
    }
}


inline void lprob(double * restrict out, const double* pt, const double* center, double scale)
{
    // derivative is with respect to the *center*
    double disp[3] = {scale*(pt[0]-center[0]), scale*(pt[1]-center[1]), scale*(pt[2]-center[2])};
    out[0] = scale*disp[0];
    out[1] = scale*disp[1];
    out[2] = scale*disp[2];
    out[3] = -0.5*(disp[0]*disp[0] + disp[1]*disp[1] + disp[2]*disp[2]);
}

extern "C" {
    double mixture_loglikelihood_3d(
            double* restrict d_component_centers,
            int n_pts,  const double* points, 
            int n_comp, const double* component_centers,
            double sigma);
}


double mixture_loglikelihood_3d(
        double* restrict d_component_centers,
        int n_pts,  const double* points, 
        int n_comp, const double* component_centers,
        double sigma)
{
    double scale = 1./sigma;
    vector<double> tmp(n_comp*4);
    double prefactor = -3. * log(sigma * sqrt(2*M_PI)) - log(double(n_comp));

    double retval = 0.;

    for(int np=0; np<n_pts; ++np) {
        double total_prob = 0.;
        for(int nc=0; nc<n_comp; ++nc) {
            lprob(&tmp[nc*4], points+np*3, component_centers+nc*3, scale);
            total_prob += exp(tmp[nc*4+3]);
        }
        double total_lprob = log(total_prob+1e-100);

        for(int nc=0; nc<n_comp; ++nc) {
            double weight = exp(tmp[nc*4+3]-total_lprob) * (1./n_pts);
            d_component_centers[nc*3+0] += weight * tmp[nc*4+0];
            d_component_centers[nc*3+1] += weight * tmp[nc*4+1];
            d_component_centers[nc*3+2] += weight * tmp[nc*4+2];
        }
        retval += total_lprob;
    }

    return prefactor + retval/n_pts;
}
