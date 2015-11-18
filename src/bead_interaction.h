#ifndef BEAD_INTERACTION_H
#define BEAD_INTERACTION_H

#include "vector_math.h"
#include "spline.h"

namespace {
    constexpr static const int n_bit_rotamer = 4; // max number of rotamers is 2**n_bit_rotamer
    bool exclude_by_id_base(unsigned id1, unsigned id2) {return id1>>n_bit_rotamer == id2>>n_bit_rotamer;}

    struct PosDistSplineInteraction {
        // spline-based distance interaction
        // n_param is the number of basis splines (including those required to get zero
        //   derivative in the clamped spline)
        // spline is constant over [0,dx] to avoid funniness at origin
        // spline should be clamped at zero at the large end for cutoffs

        constexpr static float inv_dx = 1.f/0.5f;  // half-angstrom bins
        constexpr static bool  symmetric = true;
        constexpr static int   n_param=18, n_dim1=3, n_dim2=3, simd_width=1;  // 8 angstrom cutoff

        static float cutoff(const float* p) {
            return (n_param-2-1e-6)/inv_dx;  // 1e-6 just insulates us from round-off error
        }

        static bool is_compatible(const float* p1, const float* p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return exclude_by_id_base(id1,id2);
        }

        static float compute_edge(Vec<n_dim1> &d1, Vec<n_dim2> &d2, const float* p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
            auto disp       = x1-x2;
            auto dist2      = mag2(disp);
            auto inv_dist   = rsqrt(dist2+1e-7f);  // 1e-7 is divergence protection
            auto dist_coord = dist2*(inv_dist*inv_dx);

            auto en = clamped_deBoor_value_and_deriv(p, dist_coord, n_param);
            d1 = disp*(inv_dist*inv_dx*en.y());
            d2 = -d1;
            return en.x();
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
            auto dist_coord = inv_dx*mag(x1-x2);

            int starting_bin;
            float result[4];
            clamped_deBoor_coeff_deriv(&starting_bin, result, p, dist_coord, n_param);
            for(int i: range(4)) d_param[starting_bin+i] = result[i];
        }
    };


}

typedef PosDistSplineInteraction preferred_bead_type;

#endif
