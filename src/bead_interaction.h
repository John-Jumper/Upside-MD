#ifndef BEAD_INTERACTION_H
#define BEAD_INTERACTION_H

#include "vector_math.h"
#include "spline.h"

// Due to a cascade of requirements because I made n_param essentially a template parameter
// of InteractionGraph (as a constexpr member of IType), I have to switch out the number of 
// knots for the various splines using the preprocessor defines.  This is poor practice
// FIXME Make n_param and the various n_knots proper runtime flags, probably by making sure
// that InteractionGraph receives an IType object rather than just using it as a namespace.
#if defined(PARAM_OLD)
    #define N_KNOT_SC_SC   16
    #define N_KNOT_SC_BB   12
    #define N_KNOT_ANGULAR 15
    #define KNOT_SPACING   0.5f
#elif defined(PARAM_10A_CUTOFF)
    #define N_KNOT_SC_SC   12
    #define N_KNOT_SC_BB   12
    #define N_KNOT_ANGULAR 8
    #define KNOT_SPACING   1.f
#else
    #define N_KNOT_SC_SC   9
    #define N_KNOT_SC_BB   7
    #define N_KNOT_ANGULAR 8
    #define KNOT_SPACING   1.f
#endif

namespace {
    template<int n_knot_angular, int n_knot, int n_dim1, int n_dim2>
        inline Float4 quadspline(
                Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2,
                const float inv_dtheta, const float inv_dx, const float* p[4],
                const Vec<n_dim1,Float4> &x1, const Vec<n_dim2,Float4> &x2)
        {
            Float4 one(1.f);
            auto displace = extract<0,3>(x2)-extract<0,3>(x1);
            auto rvec1 = extract<3,6>(x1);
            auto rvec2 = extract<3,6>(x2);

            auto dist2 = mag2(displace);
            auto inv_dist = rsqrt(dist2);
            auto dist_coord = dist2*(inv_dist*Float4(inv_dx));
            auto displace_unitvec = inv_dist*displace;

            auto cos_cov_angle1 = dot(rvec1, displace_unitvec);
            auto cos_cov_angle2 = dot(rvec2,-displace_unitvec);

            // Spline evaluation
            auto angular_sigmoid1 = deBoor_value_and_deriv(p,  (cos_cov_angle1+one)*Float4(inv_dtheta)+one);
            int o = n_knot_angular; const float* pp[4] = {p[0]+o, p[1]+o, p[2]+o, p[3]+o};

            auto angular_sigmoid2 = deBoor_value_and_deriv(pp, (cos_cov_angle2+one)*Float4(inv_dtheta)+one);
            o=n_knot_angular; pp[0]+=o; pp[1]+=o; pp[2]+=o; pp[3]+=o;

            auto wide_cover   = clamped_deBoor_value_and_deriv(pp, dist_coord, n_knot);
            o=n_knot; pp[0]+=o; pp[1]+=o; pp[2]+=o; pp[3]+=o;

            auto narrow_cover = clamped_deBoor_value_and_deriv(pp, dist_coord, n_knot);

            // Partition derivatives
            auto angular_weight = angular_sigmoid1.x() * angular_sigmoid2.x();

            auto radial_deriv   = Float4(inv_dx    ) * (wide_cover.y() + angular_weight*narrow_cover.y());
            auto angular_deriv1 = Float4(inv_dtheta) * angular_sigmoid1.y()*angular_sigmoid2.x()*narrow_cover.x();
            auto angular_deriv2 = Float4(inv_dtheta) * angular_sigmoid1.x()*angular_sigmoid2.y()*narrow_cover.x();

            auto rXX = angular_deriv1*rvec1 - angular_deriv2*rvec2;
            auto deriv_dir = inv_dist * (rXX - dot(displace_unitvec,rXX)*displace_unitvec);

            auto d_displace =    radial_deriv * displace_unitvec + deriv_dir;
            auto d_rvec1    =  angular_deriv1 * displace_unitvec;
            auto d_rvec2    = -angular_deriv2 * displace_unitvec;

            auto coverage = wide_cover.x() + angular_weight*narrow_cover.x();

            store<0,3>(d1, -d_displace);
            store<3,6>(d1,  d_rvec1);

            store<0,3>(d2, d_displace);
            store<3,6>(d2, d_rvec2);

            return coverage;
        }

    template<int n_knot_angular, int n_knot, int n_param, int n_dim1, int n_dim2>
        inline void quadspline_param_deriv(
                Vec<n_param> &d_param,
                const float inv_dtheta, const float inv_dx, const float* p,
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2)
        {
            d_param = make_zero<n_param>();

            float3 displace = extract<0,3>(x2)-extract<0,3>(x1);
            float3 rvec1 = extract<3,6>(x1);
            float3 rvec2 = extract<3,6>(x2);

            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist_coord = dist2*(inv_dist*inv_dx);
            float3 displace_unitvec = inv_dist*displace;

            float  cos_cov_angle1 = dot(rvec1, displace_unitvec);
            float  cos_cov_angle2 = dot(rvec2,-displace_unitvec);

            float2 angular_sigmoid1 = deBoor_value_and_deriv(p,                (cos_cov_angle1+1.f)*inv_dtheta+1.f);
            float2 angular_sigmoid2 = deBoor_value_and_deriv(p+n_knot_angular, (cos_cov_angle2+1.f)*inv_dtheta+1.f);

            // wide_cover derivative
            int starting_bin;
            float result[4];
            clamped_deBoor_coeff_deriv(&starting_bin, result, dist_coord, n_knot);
            for(int i: range(4)) d_param[2*n_knot_angular+starting_bin+i] = result[i];

            // narrow_cover derivative
            clamped_deBoor_coeff_deriv(&starting_bin, result, dist_coord, n_knot);
            for(int i: range(4))
                d_param[2*n_knot_angular+n_knot+starting_bin+i] = angular_sigmoid1.x()*angular_sigmoid2.x()*result[i];

            // angular_sigmoid derivatives
            float2 narrow_cover = clamped_deBoor_value_and_deriv(p+2*n_knot_angular+n_knot, dist_coord, n_knot);

            deBoor_coeff_deriv(&starting_bin, result, (cos_cov_angle1+1.f)*inv_dtheta+1.f);
            for(int i: range(4)) d_param[starting_bin+i] = angular_sigmoid2.x()*narrow_cover.x()*result[i];

            deBoor_coeff_deriv(&starting_bin, result,                 (cos_cov_angle2+1.f)*inv_dtheta+1.f);
            for(int i: range(4))
                d_param[n_knot_angular+starting_bin+i] = angular_sigmoid1.x()*narrow_cover.x()*result[i];
        }

    constexpr static const int n_bit_rotamer = 4; // max number of rotamers is 2**n_bit_rotamer

    struct PosDistSplineInteraction {
        // spline-based distance interaction
        // n_param is the number of basis splines (including those required to get zero
        //   derivative in the clamped spline)
        // spline is constant over [0,dx] to avoid funniness at origin
        // spline should be clamped at zero at the large end for cutoffs

        constexpr static float inv_dx = 1.f/0.50f;  // half-angstrom bins
        constexpr static bool  symmetric = true;
        constexpr static int   n_param=N_KNOT_SC_SC, n_dim1=3, n_dim2=3, simd_width=1;  // 8 angstrom cutoff

        static float cutoff(const float* p) {
            return (n_param-2-1e-6)/inv_dx;  // 1e-6 just insulates us from round-off error
        }

        static bool is_compatible(const float* p1, const float* p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            return id1.srl(n_bit_rotamer) != id2.srl(n_bit_rotamer);
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &x1, const Vec<n_dim2,Float4> &x2) {
            auto disp       = x1-x2;
            auto dist2      = mag2(disp);
            auto inv_dist   = rsqrt(dist2+Float4(1e-7f));  // 1e-7 is divergence protection
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
            clamped_deBoor_coeff_deriv(&starting_bin, result, dist_coord, n_param);
            for(int i: range(4)) d_param[starting_bin+i] = result[i];
        }
    };
}

    struct PosQuadSplineInteraction {
        // radius scale angular_width angular_scale
        // first group is donors; second group is acceptors

        constexpr static bool  symmetric = true;
        constexpr static int   n_knot = N_KNOT_SC_SC, n_knot_angular=N_KNOT_ANGULAR;
        constexpr static int   n_param=2*n_knot_angular+2*n_knot, n_dim1=6, n_dim2=6, simd_width=1;
        constexpr static float inv_dx = 1.f/KNOT_SPACING, inv_dtheta = (n_knot_angular-3)/2.f;

        static float cutoff(const float* p) {
            return (n_knot-2-1e-6)/inv_dx;  // 1e-6 insulates from roundoff
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            return id1.srl(n_bit_rotamer) != id2.srl(n_bit_rotamer);
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &sc_pos1, const Vec<n_dim2,Float4> &sc_pos2) {
            return quadspline<n_knot_angular, n_knot>(d1,d2, inv_dtheta,inv_dx,p, sc_pos1,sc_pos2);
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p,
                const Vec<n_dim1> &sc_pos1, const Vec<n_dim2> &sc_pos2) {
            quadspline_param_deriv<n_knot_angular, n_knot>(d_param, inv_dtheta,inv_dx,p, sc_pos1,sc_pos2);
        }

        static bool is_compatible(const float* p1, const float* p2) {
            for(int nka: range(n_knot_angular))
                if(p1[nka]!=p2[nka+n_knot_angular] || p1[nka+n_knot_angular]!=p2[nka])
                    throw std::string("bad angular match");
            for(int nk: range(2*n_knot))
                if(p1[2*n_knot_angular + nk] != p2[2*n_knot_angular + nk])
                    return false;

            return true;
        }
};

typedef PosQuadSplineInteraction preferred_bead_type;

#endif
