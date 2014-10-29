#ifndef HBOND_H
#define HBOND_H

#include <math.h>
#include "md.h"
#include "coord.h"
#include <algorithm>

#include "random.h"
using namespace std;


template <typename CoordT, typename FuncT>
void finite_difference(FuncT& f, CoordT& x, float* expected, float eps = 1e-2) 
{
    int ndim_output = decltype(f(x))::n_dim;
    auto y = x;
    int ndim_input  = decltype(y)::n_dim;

    vector<float> ret(ndim_output*ndim_input);
    for(int d=0; d<ndim_input; ++d) {
        CoordT x_prime1 = x; x_prime1.v[d] += eps;
        CoordT x_prime2 = x; x_prime2.v[d] -= eps;

        auto val1 = f(x_prime1);
        auto val2 = f(x_prime2);
        for(int no=0; no<ndim_output; ++no) ret[no*ndim_input+d] = (val1.v[no]-val2.v[no]) / (2*eps);
    }
    float z = 0.f;
    for(int no=0; no<ndim_output; ++no) {
        printf("exp:");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", expected[no*ndim_input+ni]);
        printf("\n");

        printf("fd: ");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", ret     [no*ndim_input+ni]);
        printf("\n\n");
        for(int ni=0; ni<ndim_input; ++ni) {
            float t = expected[no*ndim_input+ni]-ret[no*ndim_input+ni];
            z += t*t;
        }
    }
    printf("rmsd % f\n\n\n", sqrt(z/ndim_output/ndim_input));

}


void infer_HN_OC_pos_and_dir(
        float* HN_OC,
        const float* pos,
        float* pos_deriv,
        const VirtualParams* params,
        int n_term);





float count_hbond(
        const float * restrict virtual_pos,
        float       * restrict virtual_pos_deriv,
        int n_donor,    const VirtualHBondParams * restrict donor_params,
        int n_acceptor, const VirtualHBondParams * restrict acceptor_params,
        const float hbond_energy);


void helical_probabilities(
        int n_residue, float * restrict helicity,  // size (n_residue,), corresponds to donor_params residue_id's
        const float * restrict virtual_pos,
        int n_donor,    const VirtualHBondParams * restrict donor_params,
        int n_acceptor, const VirtualHBondParams * restrict acceptor_params);

#endif
