#ifndef BEAD_INTERACTION_H
#define BEAD_INTERACTION_H

#include "vector_math.h"

namespace {
    constexpr static const int n_bit_rotamer = 4; // max number of rotamers is 2**n_bit_rotamer
    bool exclude_by_id_base(unsigned id1, unsigned id2) {return id1>>n_bit_rotamer == id2>>n_bit_rotamer;}

    struct PosDistInteraction {
        // energy_inner, radius_inner, scale_inner,  energy_outer, radius_outer, scale_outer
        constexpr static const int n_param=6, n_dim=3, n_deriv=3;

        static float cutoff(const Vec<n_param> &p) {
            float radius_i=p[1], scale_i=p[2]; float cutoff_i = radius_i + compact_sigmoid_cutoff(scale_i);
            float radius_o=p[4], scale_o=p[5]; float cutoff_o = radius_o + compact_sigmoid_cutoff(scale_o);
            return cutoff_i<cutoff_o? cutoff_o: cutoff_i;
        }

        static bool is_compatible(const Vec<n_param> &p1, const Vec<n_param> &p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return exclude_by_id_base(id1,id2);
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto disp      = x1-x2;
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;

            auto en = p[0]*compact_sigmoid(dist-p[1], p[2])
                    + p[3]*compact_sigmoid(dist-p[4], p[5]);

            d_base = en.y() * (disp*inv_dist);
            return en.x();
        }

        static void expand_deriv(Vec<n_dim> &d1, Vec<n_dim> &d2, const Vec<n_deriv> &d_base) {
            d1 =  d_base;
            d2 = -d_base;
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto dist = mag(x1-x2);
            for(int i: range(2)) {
                int off = 3*i;
                auto sig = compact_sigmoid(dist-p[off+1], p[off+2]);

                // I need the derivative with respect to the scale, but
                // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
                float2 sig_s = compact_sigmoid((dist-p[off+1]) * p[off+2], 1.f);

                d_param[off+0] =  sig  .x();  // energy
                d_param[off+1] = -sig  .y() * p[off+0];  // radius
                d_param[off+2] =  sig_s.y() * (dist-p[off+1]) * p[off+0];  // scale
            }
        }
    };


    struct PosDistInteraction6 {
        // energy_inner, radius_inner, scale_inner,  energy_outer, radius_outer, scale_outer
        constexpr static const int n_param=6, n_dim=6, n_deriv=3;

        static float cutoff(const Vec<n_param> &p) {
            float radius_i=p[1], scale_i=p[2]; float cutoff_i = radius_i + compact_sigmoid_cutoff(scale_i);
            float radius_o=p[4], scale_o=p[5]; float cutoff_o = radius_o + compact_sigmoid_cutoff(scale_o);
            return cutoff_i<cutoff_o? cutoff_o: cutoff_i;
        }

        static bool is_compatible(const Vec<n_param> &p1, const Vec<n_param> &p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return exclude_by_id_base(id1,id2);
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto disp      = extract<0,3>(x1)-extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;

            auto en = p[0]*compact_sigmoid(dist-p[1], p[2])
                    + p[3]*compact_sigmoid(dist-p[4], p[5]);

            d_base = en.y() * (disp*inv_dist);
            return en.x();
        }

        static void expand_deriv(Vec<n_dim> &d1, Vec<n_dim> &d2, const Vec<n_deriv> &d_base) {
            store<0,3>(d1, d_base); store<3,6>(d1, make_zero<3>());
            store<0,3>(d2,-d_base); store<3,6>(d2, make_zero<3>());
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto dist = mag(extract<0,3>(x1)-extract<0,3>(x2));
            for(int i: range(2)) {
                int off = 3*i;
                auto sig = compact_sigmoid(dist-p[off+1], p[off+2]);

                // I need the derivative with respect to the scale, but
                // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
                float2 sig_s = compact_sigmoid((dist-p[off+1]) * p[off+2], 1.f);

                d_param[off+0] =  sig  .x();  // energy
                d_param[off+1] = -sig  .y() * p[off+0];  // radius
                d_param[off+2] =  sig_s.y() * (dist-p[off+1]) * p[off+0];  // scale
            }
        }
    };


    struct PosDistDirInteraction {
        constexpr static const int n_param=6+6, n_dim=6, n_deriv=9;

        struct ParamInterpret {
            float energy_steric, dist_loc_steric, dist_scale_steric,
                  energy_gauss,  dist_loc_gauss,  dist_scale_gauss,
                  dp1_shift_steric, dp2_shift_steric,
                  dp1_loc_gauss,    dp2_loc_gauss,
                  dp1_scale_gauss,  dp2_scale_gauss;
            ParamInterpret(const Vec<n_param> &p):
                energy_steric(p[0]), dist_loc_steric(p[1]), dist_scale_steric(p[2]),
                energy_gauss(p[3]),  dist_loc_gauss(p[4]),  dist_scale_gauss(p[5]),
                dp1_shift_steric(p[6]),  dp2_shift_steric(p[7]),
                dp1_loc_gauss(p[8]),     dp2_loc_gauss(p[9]),
                dp1_scale_gauss(p[10]),  dp2_scale_gauss(p[11]) {}
        };

        static float cutoff(const Vec<n_param> &param) {
            auto p = ParamInterpret(param);
            auto maxf = [](float a, float b) {return a>b ? a : b;};
            float gauss_cutoff  = p.dist_loc_gauss + 3.f*sqrtf(0.5f/p.dist_scale_gauss);  // 3 sigma cutoff
            float steric_cutoff = p.dist_loc_steric + compact_sigmoid_cutoff(p.dist_scale_steric) +
                                  maxf(0.f,p.dp1_shift_steric) + maxf(0.f,p.dp2_shift_steric);
            return maxf(steric_cutoff, gauss_cutoff);
        }

        static bool is_compatible(const Vec<n_param> &p1, const Vec<n_param> &p2) {
            for(int i: range(6))      if(p1[i]!=p2[i]) return false;
            for(int i: range(6,12,2)) if((p1[i]!=p2[i+1]) | (p1[i+1]!=p2[i])) return false;
            return true;
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return exclude_by_id_base(id1,id2);
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &param, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;
            auto disp_dir  = disp*inv_dist;

            auto dp1 = dot(extract<3,6>(x1),  disp_dir);
            auto dp2 = dot(extract<3,6>(x2), -disp_dir);

            auto p = ParamInterpret(param);
            auto eff_dist_loc_steric = p.dist_loc_steric + dp1*p.dp1_shift_steric + dp2*p.dp2_shift_steric;

            auto steric_en = p.energy_steric*compact_sigmoid(dist-eff_dist_loc_steric, p.dist_scale_steric);

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss) + 
                                     p. dp1_scale_gauss * sqr( dp1 - p. dp1_loc_gauss) + 
                                     p. dp2_scale_gauss * sqr( dp2 - p. dp2_loc_gauss)));

            auto gauss_en = p.energy_gauss * gauss_base;

            // FIXME calculate d_base
            return steric_en.x() + gauss_en;
        }

        static void expand_deriv(Vec<n_dim> &d1, Vec<n_dim> &d2, const Vec<n_deriv> &d_base) {
            throw "broken";
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &param, 
                const Vec<n_dim> &x1, const Vec<n_dim> &x2) {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;
            auto disp_dir  = disp*inv_dist;

            auto dp1 = (dot(extract<3,6>(x1),  disp_dir));
            auto dp2 = (dot(extract<3,6>(x2), -disp_dir));

            auto p = ParamInterpret(param);
            auto eff_dist_loc_steric = p.dist_loc_steric + dp1*p.dp1_shift_steric + dp2*p.dp2_shift_steric;

            // I need the derivative with respect to the scale, but
            // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
            auto steric_sig   = compact_sigmoid( dist-eff_dist_loc_steric, p.dist_scale_steric);
            auto steric_sig_s = compact_sigmoid((dist-eff_dist_loc_steric)*p.dist_scale_steric, 1.f);

            d_param[0] =  steric_sig  .x();  // energy
            d_param[1] = -steric_sig  .y() * p.energy_steric; // radius
            d_param[2] =  steric_sig_s.y() * (dist-eff_dist_loc_steric) * p.energy_steric;  // scale

            d_param[6] = dp1*d_param[1];  // dp1_shift_steric
            d_param[7] = dp2*d_param[1];  // dp2_shift_steric

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss) + 
                                     p. dp1_scale_gauss * sqr( dp1 - p. dp1_loc_gauss) + 
                                     p. dp2_scale_gauss * sqr( dp2 - p. dp2_loc_gauss)));

            auto c = p.energy_gauss * gauss_base;

            d_param[ 3] =  gauss_base;  // energy_gauss
            d_param[ 4] =  c * 2.f*p.dist_scale_gauss*(dist - p.dist_loc_gauss); // dist_loc_gauss
            d_param[ 5] = -c * sqr(dist - p.dist_loc_gauss); // dist_scale_gauss
            d_param[ 8] =  c * 2.f*p.dp1_scale_gauss* ( dp1 - p. dp1_loc_gauss); //  dp1_loc_gauss
            d_param[ 9] =  c * 2.f*p.dp2_scale_gauss* ( dp2 - p. dp2_loc_gauss); //  dp2_loc_gauss
            d_param[10] = -c * sqr( dp1 - p. dp1_loc_gauss); //  dp1_scale_gauss
            d_param[11] = -c * sqr( dp2 - p. dp2_loc_gauss); //  dp2_scale_gauss
        }
    };
}

typedef PosDistInteraction6 preferred_bead_type;

#endif
