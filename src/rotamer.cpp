#include "rotamer.h"
#include "vector_math.h"
#include <vector>
#include <string>
#include "spline.h"
#include <tuple>
#include "h5_support.h"
#include "affine.h"
#include <algorithm>
#include <map>
#include "deriv_engine.h"
#include "timing.h"
#include <memory>
#include "state_logger.h"
#include <tuple>

using namespace std;
using namespace h5;


struct PosBead {
    constexpr static const int n_pos_point  = 1;
    constexpr static const int n_pos_vector = 0;
    constexpr static const int n_pos_scalar = 1; // rotamer energy
    constexpr static const int n_pos_dim    = 3*n_pos_point + 3*n_pos_vector + n_pos_scalar;
    constexpr static const int n_param = 6;

    struct InteractionDeriv {
        Vec<3> bead1_deriv;

        Vec<n_pos_dim-1> d1() const {return  bead1_deriv;}
        Vec<n_pos_dim-1> d2() const {return -bead1_deriv;}
    };

    struct SidechainInteraction {
        float cutoff2;
        Vec<n_param> params;  // energy_inner, radius_inner, scale_inner,  energy_outer, radius_outer, scale_outer

        void update_cutoff2() {
            float radius_i=params[1], scale_i=params[2]; float cutoff2_i = sqr(radius_i + 1.f/scale_i);
            float radius_o=params[4], scale_o=params[5]; float cutoff2_o = sqr(radius_o + 1.f/scale_o);
            cutoff2 = cutoff2_i<cutoff2_o? cutoff2_o: cutoff2_i;
        }
    
        bool compatible(const SidechainInteraction& other) const {
            bool comp = cutoff2 == other.cutoff2;
            for(int i: range(6)) comp &= params[i] == other.params[i];
            return comp;
        }
    
        float evaluate(InteractionDeriv& deriv, const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto disp      = x1-x2;
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;

            auto en = params[0]*compact_sigmoid(dist-params[1], params[2])
                    + params[3]*compact_sigmoid(dist-params[4], params[5]);
            // fprintf(stderr, "dist %6.2f % .3f\n", dist, en.x());

            deriv.bead1_deriv = en.y() * (disp*inv_dist);
            return en.x();
        }
    
        Vec<n_param> parameter_deriv(const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto dist = mag(x1-x2);
            Vec<n_param> result;
            for(int i: range(2)) {
                int off = 3*i;
                auto sig = compact_sigmoid(dist-params[off+1], params[off+2]);

                // I need the derivative with respect to the scale, but
                // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
                float2 sig_s = compact_sigmoid((dist-params[off+1]) * params[off+2], 1.f);

                result[off+0] =  sig  .x();  // energy
                result[off+1] = -sig  .y() * params[off+0];  // radius
                result[off+2] =  sig_s.y() * (dist-params[off+1]) * params[off+0];  // scale
            }
            return result;
        }
    };

    static Vec<n_param> parameter_deriv_swap_restype(const Vec<n_param>& deriv) {return deriv;}
};


struct PosExprBead {
    constexpr static const int n_pos_point  = 1;
    constexpr static const int n_pos_vector = 0;
    constexpr static const int n_pos_scalar = 1; // rotamer energy
    constexpr static const int n_pos_dim    = 3*n_pos_point + 3*n_pos_vector + n_pos_scalar;
    constexpr static const int n_param = 9;

    struct InteractionDeriv {
        Vec<n_pos_dim-1> d1() const {return make_zero<3>();}  // FIXME dummy until real calculation
        Vec<n_pos_dim-1> d2() const {return make_zero<3>();}
    };

    struct SidechainInteraction {
        float cutoff2;
        Vec<n_param> params;

        struct ParamInterpret {
            float energy_steric, dist_loc_steric, dist_scale_steric,
                  energy_gauss,  dist_loc_gauss,  dist_scale_gauss,
                  energy_gauss2,  dist_loc_gauss2,  dist_scale_gauss2;
            ParamInterpret(const Vec<n_param> &p):
                energy_steric(p[0]), dist_loc_steric(p[1]), dist_scale_steric(p[2]),
                energy_gauss(p[3]),  dist_loc_gauss(p[4]),  dist_scale_gauss(p[5]),
                energy_gauss2(p[6]),  dist_loc_gauss2(p[7]),  dist_scale_gauss2(p[8]) {}
        };
        
        void update_cutoff2() {
            auto p = ParamInterpret(params);

            float steric_cutoff = p.dist_loc_steric + 1.f/p.dist_scale_steric;

            float gauss_cutoff  = p.dist_loc_gauss + 3.f*sqrtf(0.5f/p.dist_scale_gauss);   // 3 sigma cutoff
            float gauss_cutoff2 = p.dist_loc_gauss2+ 3.f*sqrtf(0.5f/p.dist_scale_gauss2);  // 3 sigma cutoff

            cutoff2 = sqr(max(max(steric_cutoff, gauss_cutoff), gauss_cutoff2));
        }
    
        bool compatible(const SidechainInteraction& other) const {
            bool comp = cutoff2 == other.cutoff2;
            for(int i: range(9)) comp &= params[i] == other.params[i];
            return comp;
        }
    
        float evaluate(InteractionDeriv& deriv, const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;

            auto p = ParamInterpret(params);

            auto steric_en = p.energy_steric*compact_sigmoid(dist-p.dist_loc_steric, p.dist_scale_steric);

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss)));
            auto gauss_en = p.energy_gauss * gauss_base;

            auto gauss_base2 = expf(-(p.dist_scale_gauss2 * sqr(dist - p.dist_loc_gauss2)));
            auto gauss_en2 = p.energy_gauss2 * gauss_base2;

            // deriv.bead1_deriv = en.y() * disp_dir;
            return steric_en.x() + gauss_en + gauss_en2;
        }
    
        Vec<n_param> parameter_deriv(const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;

            auto p = ParamInterpret(params);

            auto eff_dist_loc_steric = p.dist_loc_steric;

            // I need the derivative with respect to the scale, but
            // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
            auto steric_sig   = compact_sigmoid( dist-eff_dist_loc_steric, p.dist_scale_steric);
            auto steric_sig_s = compact_sigmoid((dist-eff_dist_loc_steric)*p.dist_scale_steric, 1.f);

            Vec<n_param> result;
            // fprintf(stderr,"param"); for(int i: range(n_param)) fprintf(stderr, " %.2f", param[i]);
            // fprintf(stderr,"\n");
            result[0] =  steric_sig  .x();  // energy
            result[1] = -steric_sig  .y() * p.energy_steric; // radius
            result[2] =  steric_sig_s.y() * (dist-eff_dist_loc_steric) * p.energy_steric;  // scale

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss)));
            auto c = p.energy_gauss * gauss_base;

            result[ 3] =  gauss_base;  // energy_gauss
            result[ 4] =  c * 2.f*p.dist_scale_gauss*(dist - p.dist_loc_gauss); // dist_loc_gauss
            result[ 5] = -c * sqr(dist - p.dist_loc_gauss); // dist_scale_gauss

            auto gauss_base2 = expf(-(p.dist_scale_gauss2 * sqr(dist - p.dist_loc_gauss2)));
            auto c2 = p.energy_gauss2 * gauss_base2;

            result[ 6] =  gauss_base2;  // energy_gauss
            result[ 7] =  c2 * 2.f*p.dist_scale_gauss2*(dist - p.dist_loc_gauss2); // dist_loc_gauss
            result[ 8] = -c2 * sqr(dist - p.dist_loc_gauss2); // dist_scale_gauss

            return result;
        }
    };

    static Vec<n_param> parameter_deriv_swap_restype(const Vec<n_param>& deriv) {
        auto ret = deriv;
        return ret;
    }
};
struct PosDirBead {
    constexpr static const int n_pos_point  = 1;
    constexpr static const int n_pos_vector = 1; // CB->CG bond direction
    constexpr static const int n_pos_scalar = 1; // rotamer energy
    constexpr static const int n_pos_dim    = 3*n_pos_point + 3*n_pos_vector + n_pos_scalar;
    constexpr static const int n_param = 6+6;

    struct InteractionDeriv {
        Vec<n_pos_dim-1> d1() const {return make_zero<6>();}  // FIXME dummy until real calculation
        Vec<n_pos_dim-1> d2() const {return make_zero<6>();}
    };

    struct SidechainInteraction {
        float cutoff2;
        Vec<n_param> params;

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
        
        void update_cutoff2() {
            auto p = ParamInterpret(params);

            float steric_cutoff = p.dist_loc_steric + 1.f/p.dist_scale_steric +
                                  max(0.f,p.dp1_shift_steric) + max(0.f,p.dp2_shift_steric);

            float gauss_cutoff  = p.dist_loc_gauss + 3.f*sqrtf(0.5f/p.dist_scale_gauss);  // 3 sigma cutoff

            cutoff2 = sqr(max(steric_cutoff, gauss_cutoff));
        }
    
        bool compatible(const SidechainInteraction& other) const {
            bool comp = cutoff2 == other.cutoff2;
            for(int i: range(6)) comp &= params[i] == other.params[i];
            for(int i=7; i<12; i+=2) {
                comp &= params[i]   == other.params[i+1];  // switch of restype1 and restype2
                comp &= params[i+1] == other.params[i];    // switch of restype1 and restype2
            }
            return comp;
        }
    
        float evaluate(InteractionDeriv& deriv, const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;
            auto disp_dir  = disp*inv_dist;

            auto dp1 = (dot(extract<3,6>(x1),  disp_dir));
            auto dp2 = (dot(extract<3,6>(x2), -disp_dir));

            auto p = ParamInterpret(params);
            // fprintf(stderr, "interacting with");
            // for(int i:range(12)) fprintf(stderr, " %.4f", params[i]);
            // fprintf(stderr,"\n");

            auto eff_dist_loc_steric = p.dist_loc_steric + dp1*p.dp1_shift_steric + dp2*p.dp2_shift_steric;

            auto steric_en = p.energy_steric*compact_sigmoid(dist-eff_dist_loc_steric, p.dist_scale_steric);

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss) + 
                                     p. dp1_scale_gauss * sqr( dp1 - p. dp1_loc_gauss) + 
                                     p. dp2_scale_gauss * sqr( dp2 - p. dp2_loc_gauss)));

            auto gauss_en = p.energy_gauss * gauss_base;

            // deriv.bead1_deriv = en.y() * disp_dir;
            return steric_en.x() + gauss_en;
        }
    
        Vec<n_param> parameter_deriv(const Vec<n_pos_dim-1>& x1, const Vec<n_pos_dim-1>& x2) const {
            auto disp      = extract<0,3>(x1) - extract<0,3>(x2);
            auto dist2     = mag2(disp);
            auto inv_dist  = rsqrt(dist2);
            auto dist      = dist2*inv_dist;
            auto disp_dir  = disp*inv_dist;

            auto dp1 = (dot(extract<3,6>(x1),  disp_dir));
            auto dp2 = (dot(extract<3,6>(x2), -disp_dir));

            auto p = ParamInterpret(params);

            // fprintf(stderr, "interacting with");
            // for(int i:range(12)) fprintf(stderr, " %.2f", params[i]);
            // fprintf(stderr,"\n");

            auto eff_dist_loc_steric = p.dist_loc_steric + dp1*p.dp1_shift_steric + dp2*p.dp2_shift_steric;

            // I need the derivative with respect to the scale, but
            // compact_sigmoid(x,s) == compact_sigmoid(x*s,1.), so I can cheat
            auto steric_sig   = compact_sigmoid( dist-eff_dist_loc_steric, p.dist_scale_steric);
            auto steric_sig_s = compact_sigmoid((dist-eff_dist_loc_steric)*p.dist_scale_steric, 1.f);

            Vec<n_param> result;
            result[0] =  steric_sig  .x();  // energy
            result[1] = -steric_sig  .y() * p.energy_steric; // radius
            result[2] =  steric_sig_s.y() * (dist-eff_dist_loc_steric) * p.energy_steric;  // scale

            result[6] = dp1*result[1];  // dp1_shift_steric
            result[7] = dp2*result[1];  // dp2_shift_steric

            auto gauss_base = expf(-(p.dist_scale_gauss * sqr(dist - p.dist_loc_gauss) + 
                                     p. dp1_scale_gauss * sqr( dp1 - p. dp1_loc_gauss) + 
                                     p. dp2_scale_gauss * sqr( dp2 - p. dp2_loc_gauss)));

            auto c = p.energy_gauss * gauss_base;

            result[ 3] =  gauss_base;  // energy_gauss
            result[ 4] =  c * 2.f*p.dist_scale_gauss*(dist - p.dist_loc_gauss); // dist_loc_gauss
            result[ 5] = -c * sqr(dist - p.dist_loc_gauss); // dist_scale_gauss
            result[ 8] =  c * 2.f*p.dp1_scale_gauss* ( dp1 - p. dp1_loc_gauss); //  dp1_loc_gauss
            result[ 9] =  c * 2.f*p.dp2_scale_gauss* ( dp2 - p. dp2_loc_gauss); //  dp2_loc_gauss
            result[10] = -c * sqr( dp1 - p. dp1_loc_gauss); //  dp1_scale_gauss
            result[11] = -c * sqr( dp2 - p. dp2_loc_gauss); //  dp2_scale_gauss

            return result;
        }
    };

    static Vec<n_param> parameter_deriv_swap_restype(const Vec<n_param>& deriv) {
        auto ret = deriv;
        swap(ret[ 6],ret[ 7]);
        swap(ret[ 8],ret[ 9]);
        swap(ret[10],ret[11]);
        return ret;
    }
};

typedef PosBead preferred_bead_type;


struct ResidueLoc {
    int restype;
    CoordPair affine_idx;
    CoordPair rama_idx;
};

template <int n_rot, typename BT>  // BT is a bead-type
struct RotamerPlacement {
    int n_res;

    vector<ResidueLoc> loc;
    vector<int> global_restype;

    LayeredPeriodicSpline2D<n_rot*BT::n_pos_dim> spline; // include both location and potential
    SysArrayStorage pos;
    SysArrayStorage pos_deriv;
    SysArrayStorage phi_deriv;  // d_pos/d_phi vector
    SysArrayStorage psi_deriv;  // d_pos/d_psi vector
    int n_system;

    RotamerPlacement(): spline(10,10,10) {}

    RotamerPlacement(
            const vector<ResidueLoc> &loc_, const vector<int> global_restype_,
            int n_restype_, int nx_, int ny_, double* spline_data,
            int n_system_):
        n_res(loc_.size()),
        loc(loc_), global_restype(global_restype_),
        spline(n_restype_, nx_, ny_),
        pos      (n_system_, n_rot*BT::n_pos_dim, loc.size()), // order points, vectors, scalars
        pos_deriv(n_system_, n_rot*BT::n_pos_dim, loc.size()), // to be filled by compute_free_energy_and_derivative
        phi_deriv(n_system_, n_rot*BT::n_pos_dim, loc.size()),
        psi_deriv(n_system_, n_rot*BT::n_pos_dim, loc.size()),
        n_system(n_system_)
    {
        if(int(loc.size()) != n_res || int(global_restype.size()) != n_res) throw string("Internal error");
        spline.fit_spline(spline_data);
    }


    void place_rotamers(const SysArray& affine_pos, const SysArray& rama_pos, int ns) {
        const float scale_x = spline.nx * (0.5f/M_PI_F - 1e-7f);
        const float scale_y = spline.ny * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        VecArray affine = affine_pos[ns];
        VecArray rama   = rama_pos  [ns];
        VecArray pos_s  = pos       [ns];
        VecArray phi_d  = phi_deriv [ns];
        VecArray psi_d  = psi_deriv [ns];

        for(int nr: range(n_res)) {
            auto aff = load_vec<7>(affine, loc[nr].affine_idx.index);
            auto r   = load_vec<2>(rama,   loc[nr].rama_idx.index);
            auto t   = make_vec3(aff[0], aff[1], aff[2]);
            float U[9]; quat_to_rot(U, aff.v+3);

            float germ[n_rot*BT::n_pos_dim*3];  // 3 here is deriv_x, deriv_y, value
            spline.evaluate_value_and_deriv(germ, loc[nr].restype, 
                    (r[0]+shift)*scale_x, (r[1]+shift)*scale_y);

            for(int no: range(n_rot)) {
                float* val = germ+no*BT::n_pos_dim*3; // 3 here is deriv_x, deriv_y, value

                int j = 0; // index of dimension that we are on

                for(int nvec: range(BT::n_pos_point + BT::n_pos_vector)) {
                    store_vec(phi_d.shifted(no*BT::n_pos_dim+j), nr, 
                            apply_rotation(U,   make_vec3(val[(j+0)*3+0], val[(j+1)*3+0], val[(j+2)*3+0])) * scale_x);
                    store_vec(psi_d.shifted(no*BT::n_pos_dim+j), nr, 
                            apply_rotation(U,   make_vec3(val[(j+0)*3+1], val[(j+1)*3+1], val[(j+2)*3+1])) * scale_y);
                    store_vec(pos_s.shifted(no*BT::n_pos_dim+j), nr, (nvec<BT::n_pos_point 
                                ? apply_affine  (U,t, make_vec3(val[(j+0)*3+2], val[(j+1)*3+2], val[(j+2)*3+2]))
                                : apply_rotation(U,   make_vec3(val[(j+0)*3+2], val[(j+1)*3+2], val[(j+2)*3+2]))));
                    j += 3;
                }

                for(int i=0; i<BT::n_pos_scalar; ++i) {
                    phi_d(no*BT::n_pos_dim+j,nr) = val[j*3+0] * scale_x;
                    psi_d(no*BT::n_pos_dim+j,nr) = val[j*3+1] * scale_y;
                    pos_s(no*BT::n_pos_dim+j,nr) = val[j*3+2];
                    j += 1;
                }
            }
        }
    }


    void push_derivatives(VecArray affine_pos, VecArray affine_deriv, VecArray rama_deriv, 
            int ns, float scale_final_energy) {
        VecArray v_pos = pos[ns];
        VecArray v_pos_deriv = pos_deriv[ns];

        for(int nr: range(n_res)) {
            auto d = scale_final_energy*load_vec<n_rot*BT::n_pos_dim>(v_pos_deriv, nr);
            store_vec(rama_deriv,loc[nr].rama_idx.slot, make_vec2(
                        dot(d,load_vec<n_rot*BT::n_pos_dim>(phi_deriv[ns],nr)),
                        dot(d,load_vec<n_rot*BT::n_pos_dim>(psi_deriv[ns],nr))));

            Vec<6> z; z[0]=z[1]=z[2]=z[3]=z[4]=z[5]=0.f;
            for(int no: range(n_rot)) {
                // only difference between points and vectors is whether to subtract off the translation
                int j=0;
                for(int nvec: range(BT::n_pos_point + BT::n_pos_vector)) {
                    auto dx = make_vec3(d[no*BT::n_pos_dim+3*j+0], d[no*BT::n_pos_dim+3*j+1], d[no*BT::n_pos_dim+3*j+2]);
                    auto t  = load_vec<3>(affine_pos,loc[nr].affine_idx.index);
                    auto x  = load_vec<3>(v_pos.shifted(no*BT::n_pos_dim+3*j),nr);
                    auto tq = cross((nvec<BT::n_pos_point?x-t:x),dx);  // torque relative to the residue center
                    // only points, not vectors, contribute to the CoM derivative
                    if(nvec<BT::n_pos_point) {z[0] += dx[0]; z[1] += dx[1]; z[2] += dx[2];}
                    z[3] += tq[0]; z[4] += tq[1]; z[5] += tq[2];
                    j += 3;
                }
            }
            store_vec(affine_deriv,loc[nr].affine_idx.slot, z);
        }
    }
};


template <int n_rot1, int n_rot2, typename BT>
struct PairInteraction {
    int nr1,nr2;
    float potential[n_rot1][n_rot2];
    typename BT::InteractionDeriv deriv[n_rot1][n_rot2];
};



template <int n_rot1, int n_rot2, typename BT>  // must have n_dim1 <= n_dim2
void compute_graph_elements(
        int &n_edge,  
        PairInteraction<n_rot1,n_rot2,BT>* edges,  // must be large enough to fit all generated edges
        int n_res1, const int* restype1, VecArray pos1,  // dimensionality n_rot1*BT::n_pos_dim
        int n_res2, const int* restype2, VecArray pos2,  // dimensionality n_rot2*BT::n_pos_dim
        int n_restype, const typename BT::SidechainInteraction* interactions,
        bool is_self_interaction) {
    n_edge = 0;

    for(int nr1: range(n_res1)) {
        int nt1 = restype1[nr1];
        float3 rot1[n_rot1]; 
        for(int no1: range(n_rot1)) 
            rot1[no1] = load_vec<3>(pos1.shifted(BT::n_pos_dim*no1), nr1);

        for(int nr2: range((is_self_interaction ? nr1+1 : 0), n_res2)) {
            int nt2 = restype2[nr2];

            auto &p = interactions[nt1*n_restype+nt2];
            float cutoff2 = p.cutoff2;

            // FIXME introduce some sort of early cutoff to reduce the cost of 
            //   checking every rotamer when there cannot be a hit.
            bool within_cutoff = 0;
            for(int no2: range(n_rot2)) {
                float3 rot2 = load_vec<3>(pos2.shifted(BT::n_pos_dim*no2), nr2);
                for(int no1: range(n_rot1))
                    within_cutoff |= mag2(rot1[no1]-rot2) < cutoff2;
            }

            if(within_cutoff) {
                // grab the next edge location available and increment the count of required slots
                auto& edge = edges[n_edge++];
                edge.nr1 = nr1;
                edge.nr2 = nr2;

                for(int no2: range(n_rot2)) {
                    auto rot2 = load_vec<BT::n_pos_dim-1>(pos2.shifted(BT::n_pos_dim*no2), nr2);
                    for(int no1: range(n_rot1)) {
                        // printf("evaluate %i %i\n", nr1,nr2);
                        edge.potential[no1][no2] = p.evaluate(
                                edge.deriv[no1][no2], 
                                load_vec<BT::n_pos_dim-1>(pos1.shifted(no1*BT::n_pos_dim), nr1),
                                rot2);
                    }
                }
            }
        }
    }
}


template <typename BT>
void compute_all_graph_elements(
        int &n_edge11, PairInteraction<1,1,BT>* edges11,  // must be large enough to fit all generated edges
        int &n_edge13, PairInteraction<1,3,BT>* edges13,  // must be large enough to fit all generated edges
        int &n_edge33, PairInteraction<3,3,BT>* edges33,  // must be large enough to fit all generated edges
        int n_res1, const int* restype1, VecArray pos1,  // dimensionality 1*BT::n_pos_dim
        int n_res3, const int* restype3, VecArray pos3,  // dimensionality 3*BT::n_pos_dim
        int n_restype, const typename BT::SidechainInteraction* interactions) {

            compute_graph_elements<1,1,BT>(
                    n_edge11, edges11,
                    n_res1, restype1, pos1,
                    n_res1, restype1, pos1,
                    n_restype, interactions, true);
            compute_graph_elements<1,3,BT>(
                    n_edge13, edges13,
                    n_res1, restype1, pos1,
                    n_res3, restype3, pos3,
                    n_restype, interactions, false);
            compute_graph_elements<3,3,BT>(
                    n_edge33, edges33,
                    n_res3, restype3, pos3,
                    n_res3, restype3, pos3,
                    n_restype, interactions, true);
}


void calculate_new_beliefs(
        VecArray new_node_belief, VecArray new_edge_belief, // output
        VecArray old_node_belief, VecArray old_edge_belief, // input (edge beliefs go in both directions, so has length 6)
        int n_node, const VecArray node_prob, 
        int n_edge, const VecArray edge_prob, int* edge_indices,
        float damping)  // in range [0.,1.).  0 indicates no damping
{
    const int n_rot = 3;
    for(int d: range(n_rot)) copy_n(&node_prob(d,0), n_node, &new_node_belief(d,0));

    for(int ne: range(n_edge)) {
        int node1 = edge_indices[2*ne+0];
        int node2 = edge_indices[2*ne+1];

        auto old_node_belief1 = load_vec<n_rot>(old_node_belief, node1);
        auto old_node_belief2 = load_vec<n_rot>(old_node_belief, node2);

        auto ep = load_vec<n_rot*n_rot>(edge_prob, ne);

        auto old_edge_belief1 = load_vec<n_rot>(old_edge_belief               ,ne);
        auto old_edge_belief2 = load_vec<n_rot>(old_edge_belief.shifted(n_rot),ne);

        auto new_edge_belief1 =  left_multiply_matrix(ep, old_node_belief2 * vec_rcp(old_edge_belief2));
        auto new_edge_belief2 = right_multiply_matrix(    old_node_belief1 * vec_rcp(old_edge_belief1), ep);
        new_edge_belief1 *= rcp(max(new_edge_belief1)); // rescale to avoid underflow in the future
        new_edge_belief2 *= rcp(max(new_edge_belief2));

        // store edge beliefs
        Vec<2*n_rot> neb;
        for(int i: range(n_rot)) neb[i]       = new_edge_belief1[i];
        for(int i: range(n_rot)) neb[i+n_rot] = new_edge_belief2[i];
        store_vec(new_edge_belief,ne, neb);

        // update our beliefs about nodes (normalization is L2, but this still keeps us near 1)
        store_vec(new_node_belief, node1, normalized(new_edge_belief1 * load_vec<n_rot>(new_node_belief, node1)));
        store_vec(new_node_belief, node2, normalized(new_edge_belief2 * load_vec<n_rot>(new_node_belief, node2)));
    }

    // normalize node beliefs to avoid underflow in the future
    for(int nn: range(n_node)) {
        auto b = load_vec<n_rot>(new_node_belief, nn);
        b *= rcp(max(b));
        store_vec(new_node_belief, nn, b);
    }

    if(damping) {
        for(int d: range(  n_rot)) for(int nn: range(n_node)) new_node_belief(d,nn) = new_node_belief(d,nn)*(1.f-damping) + old_node_belief(d,nn)*damping;
        for(int d: range(2*n_rot)) for(int ne: range(n_edge)) new_edge_belief(d,ne) = new_edge_belief(d,ne)*(1.f-damping) + old_edge_belief(d,ne)*damping;
    }
}


// return value is number of iterations completed
pair<int,float> solve_for_beliefs(
        VecArray node_belief,      VecArray edge_belief, 
        VecArray temp_node_belief, VecArray temp_edge_belief,
        int n_node, VecArray node_prob,
        int n_edge, VecArray edge_prob, int* edge_indices_this_system,
        float damping, // 0.f indicates no damping
        int max_iter, float tol, bool re_initialize_node_beliefs) {
    const int n_rot = 3;

    if(re_initialize_node_beliefs) {
        for(int d: range(  n_rot)) for(int nn: range(n_node)) node_belief(d,nn) = node_prob(d,nn);
    }
    for(int d: range(2*n_rot)) for(int ne: range(n_edge)) temp_edge_belief(d,ne) = 1.f;

    // now let's construct the edge beliefs that are correctly related to the node beliefs
    // since old_node_belief sets new_edge_belief and old_edge_belief sets new_node_belief, 
    //   we will do a weird mix to get node_belief sets edge_belief
    calculate_new_beliefs(
            temp_node_belief, edge_belief,
            node_belief,      temp_edge_belief,
            n_node, node_prob,
            n_edge, edge_prob, edge_indices_this_system,
            min(damping,0.1f));

    float max_deviation = 1e10f;
    int iter = 0;
    for(; max_deviation>tol && iter<max_iter; iter+=2) {
        calculate_new_beliefs(
                temp_node_belief, temp_edge_belief,
                node_belief,      edge_belief,
                n_node, node_prob,
                n_edge, edge_prob, edge_indices_this_system,
                damping);

        calculate_new_beliefs(
                node_belief,      edge_belief,
                temp_node_belief, temp_edge_belief,
                n_node, node_prob,
                n_edge, edge_prob, edge_indices_this_system,
                damping);

        // compute max deviation
        float node_dev = 0.f;
        for(int d: range(  n_rot)) 
            for(int nn: range(n_node)) 
                node_dev = max(node_belief(d,nn)-temp_node_belief(d,nn), node_dev);

        float edge_dev = 0.f;
        for(int d: range(2*n_rot)) 
            for(int ne: range(n_edge)) 
                edge_dev = max(edge_belief(d,ne)-temp_edge_belief(d,ne), edge_dev);

        max_deviation = max(node_dev, edge_dev);
    }

    return make_pair(iter, max_deviation);
}


template <typename BT>
void convert_potential_graph_to_probability_graph(
        VecArray node_prob, VecArray edge_prob, int* edge_indices,
        int n_res3, VecArray pos3, // last dimension is 1-residue potential
        int n_edge13, PairInteraction<1,3,BT>* edges13,
        int n_edge33, PairInteraction<3,3,BT>* edges33) {

    // float potential_shift = 0.f;
    for(int ne33: range(n_edge33)) {
        auto &e = edges33[ne33];
        edge_indices[ne33*2 + 0] = e.nr1;
        edge_indices[ne33*2 + 1] = e.nr2;

        float min_pot = 1e10f;
        for(int no1: range(3))
            for(int no2: range(3))
                min_pot = min(e.potential[no1][no2], min_pot);
        // potential_shift += min_pot;

        for(int no1: range(3))
            for(int no2: range(3))
                edge_prob(no1*3+no2,ne33) = expf(-(e.potential[no1][no2] - min_pot));  // shift to avoid underflow later
    }

    for(int no: range(3))
        for(int nr: range(n_res3)) 
            node_prob(no,nr) = pos3(no*BT::n_pos_dim+BT::n_pos_dim-1,nr);

    for(int ne13: range(n_edge13)) {
        auto &e = edges13[ne13];
        int nr = e.nr2;

        for(int no: range(3)) node_prob(no,nr) += e.potential[0][no];
    }

    for(int nr: range(n_res3)) {
        float min_pot = 1e10f;
        for(int no: range(3))
            min_pot = min(node_prob(no,nr), min_pot);
        // potential_shift += min_pot;

        for(int no: range(3)) 
            node_prob(no,nr) = expf(-(node_prob(no,nr)-min_pot));
    }
    // printf("potential_shift %.4f\n", potential_shift);
}


template <typename BT>
void compute_free_energy_and_derivative(
        float* potential, VecArray node_marginal_prob, VecArray edge_marginal_prob,
        VecArray pos,
        VecArray deriv1, VecArray deriv3,
        int n_res1, int n_res3,
        VecArray node_belief, VecArray edge_belief,
        VecArray edge_prob,
        int n_edge11, const PairInteraction<1,1,BT>* edges11,
        int n_edge13, const PairInteraction<1,3,BT>* edges13,
        int n_edge33, const PairInteraction<3,3,BT>* edges33) {

    // start with zero derivative
    fill(deriv1, 1*BT::n_pos_dim, n_res1, 0.f);
    fill(deriv3, 3*BT::n_pos_dim, n_res3, 0.f);

    double free_energy = 0.f;

    // node beliefs couple directly to the 1-body potentials
    for(int nr: range(n_res3)) {
        auto b = load_vec<3>(node_belief, nr);
        b *= rcp(sum(b)); // normalize probability

        for(int no: range(3)) {
            deriv3(no*BT::n_pos_dim+3,nr) = b[no];
            // potential is given by the 3th element of position (note that -S is p*log p with no minus)
            if(potential) {
                float v =  b[no]*pos(no*BT::n_pos_dim+BT::n_pos_dim-1,nr);
                float s = -b[no]*logf(1e-10f+b[no]); // 1-body entropies
                free_energy += v-s;
                node_marginal_prob(no,nr) = b[no];
            }
        }
    }

    // edge beliefs couple to the positions
    for(int ne11: range(n_edge11)) {
        auto &e = edges11[ne11];
        update_vec(deriv1, e.nr1, e.deriv[0][0].d1());
        update_vec(deriv1, e.nr2, e.deriv[0][0].d2());
        if(potential) {
            float v = e.potential[0][0];
            free_energy += v; // no entropy since only 1 state for each
        }
    }

    for(int ne13: range(n_edge13)) {
        auto &e = edges13[ne13];
        Vec<3> b = load_vec<3>(node_belief, e.nr2);
        b *= rcp(sum(b)); // normalize probability

        if(potential) {
            float v = b[0]*e.potential[0][0]+b[1]*e.potential[0][1]+b[2]*e.potential[0][2];
            free_energy += v;  // no mutual information since one of the residues has only a single state
        }

        auto d1 = make_zero<BT::n_pos_dim-1>();
        for(int no2: range(3)) {
            d1 += b[no2]*e.deriv[0][no2].d1();
            update_vec(deriv3.shifted(BT::n_pos_dim*no2),e.nr2, b[no2]*e.deriv[0][no2].d2());
        }
        update_vec(deriv1,e.nr1, d1);
    }

    // The edge marginal distributions are given by p(x1,x2) *
    // node_belief_1(x1) * node_belief_2(x2) / (edge_belief_12(x1) *
    // edge_belief_21(x2)) up to normalization.
    for(int ne33: range(n_edge33)) {
        auto &e = edges33[ne33];
        float3 b1 = load_vec<3>(node_belief, e.nr1);
        float3 b2 = load_vec<3>(node_belief, e.nr2);

        // correct for self interaction
        float3 bc1 = b1 * vec_rcp(1e-10f + load_vec<3>(edge_belief,            ne33));
        float3 bc2 = b2 * vec_rcp(1e-10f + load_vec<3>(edge_belief.shifted(3), ne33));

        Vec<9> pair_distrib = load_vec<9>(edge_prob, ne33);
        for(int no1: range(3))
            for(int no2: range(3))
                pair_distrib[no1*3+no2] *= bc1[no1]*bc2[no2];
        pair_distrib *= rcp(sum(pair_distrib));

        // normalize beliefs to obtain node marginals again
        b1 *= rcp(sum(b1));
        b2 *= rcp(sum(b2));

        if(potential) {
            float v = 0.f;
            float s = 0.f;  // mutual information
            for(int no1: range(3)) {
                for(int no2: range(3)) {
                    auto p = pair_distrib[no1*3+no2];
                    v += p*e.potential[no1][no2];
                    s -= p*logf((1e-10f+p)*rcp((1e-10f+b1[no1]*b2[no2])));
                }
            }
            free_energy += v-s;
            store_vec(edge_marginal_prob, ne33, pair_distrib);
        }

        #define p(no1,no2) (pair_distrib[(no1)*3+(no2)])
        for(int no1: range(3)) 
            update_vec(deriv3.shifted(BT::n_pos_dim*no1),e.nr1,  
                p(no1,0)*e.deriv[no1][0].d1()+p(no1,1)*e.deriv[no1][1].d1()+p(no1,2)*e.deriv[no1][2].d1());
        for(int no2: range(3)) 
            update_vec(deriv3.shifted(BT::n_pos_dim*no2),e.nr2, 
                p(0,no2)*e.deriv[0][no2].d2()+p(1,no2)*e.deriv[1][no2].d2()+p(2,no2)*e.deriv[2][no2].d2());
        #undef p
    }

    if(potential) *potential = free_energy;
} 

template <typename BT>
void compute_parameter_derivatives(
        float* buffer, 
        VecArray node_marginal_prob, VecArray edge_marginal_prob,
        int n_edge11, PairInteraction<1,1,BT>* edges11,
        int n_edge13, PairInteraction<1,3,BT>* edges13,
        int n_edge33, PairInteraction<3,3,BT>* edges33,
        int n_res1, const int* restype1, VecArray pos1,  // dimensionality 1*BT::n_pos_dim
        int n_res3, const int* restype3, VecArray pos3,  // dimensionality 3*BT::n_pos_dim
        int n_restype, const typename BT::SidechainInteraction* interactions) {

    auto inter_deriv = [&](VecArray pos1, VecArray pos2, int rt1, int nr1, int no1, int rt2, int nr2, int no2) {
        return interactions[rt1*n_restype + rt2].parameter_deriv(
                load_vec<BT::n_pos_dim-1>(pos1.shifted(BT::n_pos_dim*no1),nr1),
                load_vec<BT::n_pos_dim-1>(pos2.shifted(BT::n_pos_dim*no2),nr2));
    };

    SysArrayStorage s_deriv(1, BT::n_param, n_restype*n_restype);
    VecArray deriv = s_deriv[0];
    fill(deriv, BT::n_param, n_restype*n_restype, 0.f);

    for(int ne11: range(n_edge11)) {
        auto &e = edges11[ne11];
        int rt1 = restype1[e.nr1];
        int rt2 = restype1[e.nr2];
        update_vec(deriv, rt1*n_restype+rt2, inter_deriv(pos1,pos1, rt1,e.nr1,0, rt2,e.nr2,0));
    }

    for(int ne13: range(n_edge13)) {
        auto &e = edges13[ne13];
        Vec<3> b = load_vec<3>(node_marginal_prob, e.nr2);

        int rt1 = restype1[e.nr1];
        int rt2 = restype3[e.nr2];

        auto dval = make_zero<BT::n_param>();
        for(int no2: range(3))
            dval += b[no2]*inter_deriv(pos1,pos3, rt1,e.nr1,0, rt2,e.nr2,no2); 

        update_vec(deriv, rt1*n_restype+rt2, dval);
    }

    for(int ne33: range(n_edge33)) {
        auto &e = edges33[ne33];
        Vec<9> bp = load_vec<9>(edge_marginal_prob, ne33);

        int rt1 = restype3[e.nr1];
        int rt2 = restype3[e.nr2];

        auto dval = make_zero<BT::n_param>();
        for(int no1: range(3)) for(int no2: range(3))
            dval += bp[no1*3+no2]*inter_deriv(pos3,pos3, rt1,e.nr1,no1, rt2,e.nr2,no2); 
        update_vec(deriv, rt1*n_restype+rt2, dval);
    }

    // now re-order, handle symmetry, and handle scale = 1/width
    for(int rt1: range(n_restype)) {
        for(int rt2: range(n_restype)) {
            auto d1 =                                  load_vec<BT::n_param>(deriv, rt1*n_restype+rt2);
            auto d2 = BT::parameter_deriv_swap_restype(load_vec<BT::n_param>(deriv, rt2*n_restype+rt1));

            // impose symmetry
            auto d = (rt1==rt2 ? 0.5f : 1.0f) * (d1+d2);

            float* base_loc = buffer + (rt1*n_restype+rt2)*BT::n_param;
            for(int i: range(BT::n_param)) base_loc[i] = d[i];
        }
    }
}


template <typename BT>
struct RotamerSidechain: public PotentialNode {
    struct RotamerIndices {
        int start;
        int stop;
    };

    int n_restype;
    CoordNode& rama;
    CoordNode& alignment;
    vector<typename BT::SidechainInteraction> interactions;
    map<string,int> index_from_restype;

    vector<RotamerIndices> rotamer_indices;  // start and stop

    vector<string> sequence;
    vector<int>    restype;

    unique_ptr<RotamerPlacement<1,BT>> placement1;
    unique_ptr<RotamerPlacement<3,BT>> placement3;

    int             max_edges11, max_edges13, max_edges33;
    vector<int>     n_edge11, n_edge13, n_edge33;
    vector<PairInteraction<1,1,BT>> edges11;
    vector<PairInteraction<1,3,BT>> edges13;
    vector<PairInteraction<3,3,BT>> edges33;

    vector<int> edge_indices;
    SysArrayStorage node_prob, edge_prob;
    SysArrayStorage node_belief, edge_belief, temp_node_belief, temp_edge_belief;

    SysArrayStorage s_residue_energy1;
    SysArrayStorage s_residue_energy3;

    SysArrayStorage node_marginal_prob, edge_marginal_prob;
    vector<int>     fixed_rotamers3;

    float damping;
    int   max_iter;
    float tol;

    float scale_final_energy;

    bool energy_fresh_relative_to_derivative;
    int n_res_all;
    map<int, vector<ResidueLoc>> local_loc;

    RotamerSidechain(hid_t grp, CoordNode& rama_, CoordNode& alignment_):
        PotentialNode(alignment_.n_system),
        n_restype(get_dset_size(1, grp, "restype_order")[0]), 
        rama(rama_),
        alignment(alignment_),
        interactions(n_restype*n_restype),
        rotamer_indices(n_restype),

        n_edge11(n_system), n_edge13(n_system), n_edge33(n_system),

        damping (read_attribute<float>(grp, ".", "damping")),
        max_iter(read_attribute<int  >(grp, ".", "max_iter")),
        tol     (read_attribute<float>(grp, ".", "tol")),
        scale_final_energy(read_attribute<float>(grp, ".", "scale_final_energy")),

        energy_fresh_relative_to_derivative(false)

    {
        check_size(grp, "interaction_params", n_restype, n_restype, BT::n_param);

        traverse_dset<3,float>(grp, "interaction_params", [&](size_t rt1, size_t rt2, int par, float x) {
                interactions[rt1*n_restype+rt2].params[par] = x;});

        for(auto& p: interactions) 
            p.update_cutoff2();

        for(int rt1: range(n_restype))
            for(int rt2: range(n_restype))
                if(!interactions[rt1*n_restype + rt2].compatible(interactions[rt2*n_restype + rt1]))
                    throw string("interaction matrix must be symmetric");

        traverse_string_dset<1>(grp, "restype_order", [&](size_t idx, string nm) {index_from_restype[nm] = idx;});

        check_size(grp, "rotamer_start_stop", n_restype, 2);
        traverse_dset<2,int>(grp, "rotamer_start_stop", [&](size_t rt, size_t is_stop, int x) {
                (is_stop ? rotamer_indices[rt].stop : rotamer_indices[rt].start) = x;});

        traverse_string_dset<1>(grp, "restype", [&](size_t nr, string resname) {
                sequence.push_back(resname);
                restype .push_back(index_from_restype[resname]);
                });

        if(h5_exists(grp, "fixed_rotamers")) {
            check_size(grp, "fixed_rotamers", sequence.size());
            traverse_dset<1,int>(grp, "fixed_rotamers", [&](size_t nr, int no) {
                    int rt = restype[nr];
                    int n_rot = rotamer_indices[rt].stop - rotamer_indices[rt].start;
                    if(!(0<=no && no<n_rot)) throw string("Invalid fixed_rotamers");
                    if(n_rot==3) fixed_rotamers3.push_back(no);});
        }

        // Parse the rotamer data and place it in an array specialized for the number of rotamers in this array
        // This code is a mess.
        map<int, vector<int>> local_to_global;
        for(int i: {1,3}) local_to_global[i] = vector<int>();

        for(int rt: range(n_restype)) {
            int n_rot = rotamer_indices[rt].stop - rotamer_indices[rt].start;
            if(local_to_global.find(n_rot) == end(local_to_global)) 
                throw "Invalid number of rotamers " + to_string(n_rot);
            local_to_global[n_rot].push_back(rt);
        }

        int n_bin       = get_dset_size(3,grp, "rotamer_prob")[0];
        int n_total_rot = get_dset_size(3,grp, "rotamer_prob")[2];

        check_size(grp, "rotamer_center", n_bin, n_bin, n_total_rot, BT::n_pos_dim-1);
        check_size(grp, "rotamer_prob",   n_bin, n_bin, n_total_rot);

        vector<double> all_data_to_fit(n_total_rot*n_bin*n_bin*BT::n_pos_dim);
        traverse_dset<4,double>(grp, "rotamer_center", [&](size_t ix, size_t iy, size_t i_pt, size_t d, double x) {
                all_data_to_fit.at(((i_pt*n_bin + ix)*n_bin + iy)*BT::n_pos_dim + d) = x;});
        traverse_dset<3,double>(grp, "rotamer_prob",   [&](size_t ix, size_t iy, size_t i_pt,           double x) {
                all_data_to_fit.at(((i_pt*n_bin + ix)*n_bin + iy)*BT::n_pos_dim + BT::n_pos_dim-1) = -log(x);}); // last entry is potential, not prob

        // Copy the data into local arrays
        map<int, vector<double>> data_to_fit;
        for(auto &kv: local_to_global) {
            int n_rot = kv.first;
            data_to_fit[n_rot] = vector<double>(kv.second.size()*n_bin*n_bin*n_rot*BT::n_pos_dim);
            auto &v = data_to_fit[n_rot];

            for(int local_rt: range(kv.second.size())) {
                int rt = kv.second[local_rt];
                int start = rotamer_indices[rt].start;
                int stop  = rotamer_indices[rt].stop;

                for(int no: range(stop-start))
                    for(int ix: range(n_bin))
                        for(int iy: range(n_bin))
                            for(int d: range(BT::n_pos_dim))
                                v[(((local_rt*n_bin + ix)*n_bin + iy)*n_rot + no)*BT::n_pos_dim + d] = 
                                    all_data_to_fit[(((start+no)*n_bin + ix)*n_bin + iy)*BT::n_pos_dim + d];
            }
        }

        if(int(sequence.size()) != alignment.n_elem || int(sequence.size()) != rama.n_elem) 
            throw string("Excluded residues not allowed for Rama potential");

        if(alignment.n_system != rama.n_system) throw string("Internal error");

        map<int, vector<int>>        global_restype;
        for(auto &kv: local_to_global) {
            int n_rot = kv.first;
            local_loc[n_rot] = vector<ResidueLoc>();

            map<int,int> global_to_local; 
            for(int i: range(kv.second.size())) global_to_local[kv.second[i]] = i;

            for(int i: range(restype.size())) {
                int rt = restype[i];
                if(global_to_local.count(rt)) {
                    global_restype[n_rot].push_back(rt);
                    local_loc[n_rot].emplace_back();
                    local_loc[n_rot].back().restype = global_to_local[rt];
                    local_loc[n_rot].back().affine_idx.index = i;
                    local_loc[n_rot].back().rama_idx.index   = i;
                    alignment.slot_machine.add_request(1,local_loc[n_rot].back().affine_idx);
                    rama     .slot_machine.add_request(1,local_loc[n_rot].back().rama_idx);
                }
            }
        }
        placement1 = unique_ptr<RotamerPlacement<1,BT>>(new RotamerPlacement<1,BT>(
                    local_loc[1], global_restype[1], local_to_global[1].size(), n_bin, n_bin, data_to_fit[1].data(), n_system));
        placement3 = unique_ptr<RotamerPlacement<3,BT>>(new RotamerPlacement<3,BT>(
                    local_loc[3], global_restype[3], local_to_global[3].size(), n_bin, n_bin, data_to_fit[3].data(), n_system));


        // initialize edge storages to maximum possible sizes
        max_edges11 = placement1->n_res * (placement1->n_res-1) / 2;
        max_edges13 = placement1->n_res *  placement3->n_res;
        max_edges33 = placement3->n_res * (placement3->n_res-1) / 2;

        edges11.resize(max_edges11 * n_system);
        edges13.resize(max_edges13 * n_system);
        edges33.resize(max_edges33 * n_system);

        edge_indices.resize(2*max_edges33*n_system);

        node_prob       .reset(n_system, 3, placement3->n_res);
        node_belief     .reset(n_system, 3, placement3->n_res);
        temp_node_belief.reset(n_system, 3, placement3->n_res);

        edge_prob       .reset(n_system, 3*3, max_edges33);
        edge_belief     .reset(n_system, 2*3, max_edges33);  // there are two edge beliefs for each edge, each being beliefs about a node
        temp_edge_belief.reset(n_system, 2*3, max_edges33);

        node_marginal_prob.reset(n_system, 3,   placement3->n_res);
        edge_marginal_prob.reset(n_system, 3*3, max_edges33);

        s_residue_energy1.reset(n_system, 1, placement1->n_res);
        s_residue_energy3.reset(n_system, 2, placement3->n_res);

        n_res_all = placement1->n_res + placement3->n_res;

        auto &p1 = *placement1;
        auto &p3 = *placement3;

        if(logging(LOG_DETAILED))
            default_logger->add_logger<float>("rotamer_potential_entropy", {n_system, n_res_all, 2}, [&](float* buffer) {
                    this->ensure_fresh_energy();
                    this->calculate_per_residue_energies();

                    for(int ns: range(n_system)) {
                        VecArray residue_energy1 = s_residue_energy1[ns];
                        VecArray residue_energy3 = s_residue_energy3[ns];
                        
                        // copy into buffer
                        for(int nr1: range(p1.n_res)) {
                            buffer[ns*n_res_all*2 + local_loc[1][nr1].affine_idx.index*2 + 0] = residue_energy1(0,nr1);
                            buffer[ns*n_res_all*2 + local_loc[1][nr1].affine_idx.index*2 + 1] = 0.f; // no 1-state entropy
                        }
                        for(int nr3: range(p3.n_res)) {
                            buffer[ns*n_res_all*2 + local_loc[3][nr3].affine_idx.index*2 + 0] = residue_energy3(0,nr3);
                            buffer[ns*n_res_all*2 + local_loc[3][nr3].affine_idx.index*2 + 1] = residue_energy3(1,nr3);
                        }
                    }});


        if(logging(LOG_EXTENSIVE)) {
            default_logger->log_once<int>("rotamer_restype1", {placement1->n_res}, [&](int* buffer) {
                    for(int nr: range(placement1->n_res)) buffer[nr]=placement1->global_restype[nr];});
            default_logger->log_once<int>("rotamer_restype3", {placement3->n_res}, [&](int* buffer) {
                    for(int nr: range(placement3->n_res)) buffer[nr]=placement3->global_restype[nr];});

            default_logger->add_logger<float>("rotamer_pos1", {n_system, placement1->n_res,BT::n_pos_dim}, [&](float* buffer) {
                    auto &p1 = *placement1;
                    for(int ns: range(n_system))
                        for(int nr: range(p1.n_res))
                            for(int d: range(BT::n_pos_dim))
                                buffer[ns*p1.n_res*BT::n_pos_dim + nr*BT::n_pos_dim + d] = p1.pos[ns](d,nr);});
            default_logger->add_logger<float>("rotamer_pos3", {n_system, placement3->n_res,3*BT::n_pos_dim}, [&](float* buffer) {
                    auto &p3 = *placement3;
                    for(int ns: range(n_system))
                        for(int nr: range(p3.n_res))
                            for(int d: range(3*BT::n_pos_dim))
                                buffer[ns*p3.n_res*3*BT::n_pos_dim + nr*3*BT::n_pos_dim + d] = p3.pos[ns](d,nr);});



            // let's log the derivative of the free energy with respect to the interaction parameters
            default_logger->add_logger<float>("rotamer_interaction_parameter_gradient", {n_system, n_restype, n_restype, BT::n_param}, [&](float* buffer) {
                    this->ensure_fresh_energy();
                    auto &p1 = *placement1;
                    auto &p3 = *placement3;

                    for(int ns=0; ns<n_system; ++ns) {
                        compute_parameter_derivatives(
                            buffer + ns*n_restype*n_restype*BT::n_param, 
                            node_marginal_prob[ns], edge_marginal_prob[ns],
                            n_edge11[ns], edges11.data() + ns*max_edges11,
                            n_edge13[ns], edges13.data() + ns*max_edges13,
                            n_edge33[ns], edges33.data() + ns*max_edges33,
                            p1.n_res, p1.global_restype.data(), p1.pos[ns],  // dimensionality 1*BT::n_pos_dim
                            p3.n_res, p3.global_restype.data(), p3.pos[ns],  // dimensionality 3*BT::n_pos_dim
                            n_restype, interactions.data());
                    }});

                        
            default_logger->add_logger<float>("node_marginal_prob", {n_system,p3.n_res,3}, [&](float* buffer) {
                    for(int ns:range(n_system))
                        for(int nn: range(p3.n_res)) 
                            for(int no: range(3))
                                buffer[(ns*p3.n_res + nn)*3 + no] = node_marginal_prob[ns](no,nn);});

            default_logger->add_logger<float>("edge_marginal_prob", {n_system,max_edges33,3,3}, [&](float* buffer) {
                    for(int ns:range(n_system))
                        for(int ne: range(max_edges33))
                            for(int no1: range(3))
                                for(int no2: range(3))
                                    buffer[((ns*max_edges33 + ne)*3 + no1)*3 + no2] = (ne<n_edge33[ns])
                                        ? edge_marginal_prob[ns](no1*3+no2,ne)
                                        : 0.f;});
        }
    }

    void ensure_fresh_energy() {
        if(!energy_fresh_relative_to_derivative) compute_value(PotentialAndDerivMode);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer("rotamer");  // Timer code is not thread-safe, so cannot be used within parallel for
        energy_fresh_relative_to_derivative = mode==PotentialAndDerivMode;

        auto &p1 = *placement1;
        auto &p3 = *placement3;

        #pragma omp parallel for schedule(dynamic)
        for(int ns=0; ns<n_system; ++ns) {
            // Timer t1("rotamer_place");
            p1.place_rotamers(alignment.coords().value, rama.coords().value, ns);
            p3.place_rotamers(alignment.coords().value, rama.coords().value, ns);

            // t1.stop(); Timer t2("rotamer_compute");
            compute_all_graph_elements(
                    n_edge11[ns], edges11.data() + ns*max_edges11,
                    n_edge13[ns], edges13.data() + ns*max_edges13,
                    n_edge33[ns], edges33.data() + ns*max_edges33,
                    p1.n_res, p1.global_restype.data(), p1.pos[ns],
                    p3.n_res, p3.global_restype.data(), p3.pos[ns],
                    n_restype, interactions.data());

            // t2.stop(); Timer t3("rotamer_convert");
            convert_potential_graph_to_probability_graph(
                    node_prob[ns], edge_prob[ns], edge_indices.data() + ns*2*max_edges33,
                    p3.n_res, p3.pos[ns],  // last dimension is 1-residue potential
                    n_edge13[ns], edges13.data()+ns*max_edges13,
                    n_edge33[ns], edges33.data()+ns*max_edges33);

            // t3.stop(); Timer t4("rotamer_solve");
            if(!fixed_rotamers3.size()) {
                auto result = solve_for_beliefs(
                        node_belief[ns], edge_belief[ns], 
                        temp_node_belief[ns], temp_edge_belief[ns],
                        p3.n_res, node_prob[ns],
                        n_edge33[ns], edge_prob[ns], edge_indices.data() + ns*2*max_edges33,
                        damping, max_iter, tol, true); // do re-initialize beliefs

                if(result.first >= max_iter-1) 
                    printf("%2i solved in %i iterations with error of %f\n", ns, result.first, result.second);
            } else {
                // 0,1 beliefs are equivalent to fixed rotamers
                // just populate the beliefs with certainties
                fill(node_belief[ns], 3, p3.n_res, 0.f);
                for(int nn: range(p3.n_res))
                    node_belief[ns](fixed_rotamers3[nn], nn) = 1.f;

                fill(edge_belief[ns], 2*3, n_edge33[ns], 0.f);
                for(int ne: range(n_edge33[ns])) {
                    int nr1 = edge_indices[ns*2*max_edges33+ne*2+0];
                    int nr2 = edge_indices[ns*2*max_edges33+ne*2+1];
                    edge_belief[ns](fixed_rotamers3[nr1]  ,ne) = 1.f;
                    edge_belief[ns](fixed_rotamers3[nr2]+3,ne) = 1.f;
                }
            }

            // t4.stop(); Timer t5("rotamer_diff");
            compute_free_energy_and_derivative(
                    (mode==PotentialAndDerivMode ? potential.data()+ns : nullptr), 
                    node_marginal_prob[ns], edge_marginal_prob[ns],
                    p3.pos[ns],
                    p1.pos_deriv[ns], p3.pos_deriv[ns],
                    p1.n_res,         p3.n_res,
                    node_belief[ns], edge_belief[ns],
                    edge_prob[ns],
                    n_edge11[ns], edges11.data() + ns*max_edges11,
                    n_edge13[ns], edges13.data() + ns*max_edges13,
                    n_edge33[ns], edges33.data() + ns*max_edges33);

            if(mode==PotentialAndDerivMode) potential[ns] *= scale_final_energy;

            p1.push_derivatives(alignment.coords().value[ns],alignment.coords().deriv[ns],rama.coords().deriv[ns],ns,scale_final_energy);
            p3.push_derivatives(alignment.coords().value[ns],alignment.coords().deriv[ns],rama.coords().deriv[ns],ns,scale_final_energy);
        }
    }

    void calculate_per_residue_energies() {
        auto &p1 = *placement1;
        auto &p3 = *placement3;

        #pragma omp parallel for schedule(static,1)
        for(int ns=0; ns<n_system; ++ns) {
            VecArray residue_energy1 = s_residue_energy1[ns];
            VecArray residue_energy3 = s_residue_energy3[ns];
            fill(residue_energy1, 1, p1.n_res, 0.f);

            for(int nr: range(p3.n_res)) {
                auto b = load_vec<3>(node_marginal_prob[ns], nr);
                auto vs = make_vec2(0.f,0.f);
                for(int no: range(3)) {vs[0] += b[no]*p3.pos[ns](no*BT::n_pos_dim+3,nr); vs[1] += -b[no]*logf(1e-10f+b[no]);}
                store_vec(residue_energy3, nr, vs);
            }

            for(int ne11: range(n_edge11[ns])) {
                auto &e = edges11[ns*max_edges11+ne11];
                update_vec(residue_energy1, e.nr1, 0.5f*make_vec1(e.potential[0][0]));
                update_vec(residue_energy1, e.nr2, 0.5f*make_vec1(e.potential[0][0]));
            }

            for(int ne13: range(n_edge13[ns])) {
                auto &e = edges13[ns*max_edges13+ne13];
                Vec<3> b = load_vec<3>(node_marginal_prob[ns], e.nr2);
                auto v = make_vec1(0.f);  // 1-body entropies were already handled
                for(int no2: range(3)) v[0] += b[no2]*e.potential[0][no2];
                update_vec(residue_energy1, e.nr1, 0.5f*v);
                update_vec(residue_energy3, e.nr2, 0.5f*v);
            }

            for(int ne33: range(n_edge33[ns])) {
                auto &e = edges33[ns*max_edges33+ne33];
                Vec<9> bp = load_vec<9>(edge_marginal_prob[ns], ne33);
                Vec<3> b1 = load_vec<3>(node_marginal_prob[ns], e.nr1);
                Vec<3> b2 = load_vec<3>(node_marginal_prob[ns], e.nr2);
                auto vs = make_vec2(0.f,0.f);
                for(int no1: range(3)) for(int no2: range(3)) {
                    int i = no1*3+no2;
                    vs[0] += bp[i]*e.potential[no1][no2];
                    vs[1] +=-bp[i]*(logf((1e-10f+bp[i])*rcp((1e-10f+b1[no1]*b2[no2]))));
                }
                update_vec(residue_energy3, e.nr1, 0.5f*vs);
                update_vec(residue_energy3, e.nr2, 0.5f*vs);
            }
        }
    }


    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs_affine(1);
        vector<vector<CoordPair>> coord_pairs_rama  (1);

        for(auto &p: placement1->loc) {
            coord_pairs_affine.back().push_back(p.affine_idx);
            coord_pairs_rama  .back().push_back(p.rama_idx);
        }
        for(auto &p: placement3->loc) {
            coord_pairs_affine.back().push_back(p.affine_idx);
            coord_pairs_rama  .back().push_back(p.rama_idx);
        }

        // double affine_dev = compute_relative_deviation_for_node<7,RotamerSidechain,BODY_VALUE>(
        //         *this, alignment, coord_pairs_affine);
        double rama_dev   = compute_relative_deviation_for_node<2>(
                *this, rama,      coord_pairs_rama);

        return rama_dev;
    }
};
static RegisterNodeType<RotamerSidechain<preferred_bead_type>,2> rotamer_node ("rotamer");


struct RotamerConstructAndSolve {
    typedef preferred_bead_type BT;

    int n_restype;
    vector<BT::SidechainInteraction> interactions;

    vector<int>     restype1,   restype3;
    int             n_res1,     n_res3;
    SysArrayStorage pos1,       pos3;
    SysArrayStorage pos_deriv1, pos_deriv3;

    int n_edge11, n_edge13, n_edge33;
    vector<PairInteraction<1,1,BT>> edges11;
    vector<PairInteraction<1,3,BT>> edges13;
    vector<PairInteraction<3,3,BT>> edges33;

    vector<int> edge_indices;

    SysArrayStorage node_prob, edge_prob;
    SysArrayStorage node_belief, edge_belief, temp_node_belief, temp_edge_belief;

    SysArrayStorage node_marginal_prob, edge_marginal_prob;
    vector<int>     fixed_rotamers3;

    float damping;
    int   max_iter;
    float tol;

    float free_energy_and_parameter_deriv(float* parameter_deriv, const float* interactions_) {
        for(int i: range(n_restype*n_restype)) {
            for(int ip: range(BT::n_param))
                interactions[i].params[ip] = interactions_[BT::n_param*i+ip];
            interactions[i].update_cutoff2();
        }

        compute_all_graph_elements(
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data(),
                n_res1, restype1.data(), pos1[0],
                n_res3, restype3.data(), pos3[0],
                n_restype, interactions.data());

        convert_potential_graph_to_probability_graph(
                node_prob[0], edge_prob[0], edge_indices.data(),
                n_res3, pos3[0],  // last dimension is 1-residue potential
                n_edge13, edges13.data(),
                n_edge33, edges33.data());

        if(!fixed_rotamers3.size()) {
            auto result = solve_for_beliefs(
                    node_belief[0], edge_belief[0], 
                    temp_node_belief[0], temp_edge_belief[0],
                    n_res3, node_prob[0],
                    n_edge33, edge_prob[0], edge_indices.data(),
                    damping, max_iter, tol, true); // do re-initialize beliefs

            if(result.first >= max_iter-1) 
                fprintf(stderr,"solved in %i iterations with error of %f\n", result.first, result.second);
        } else {
            // 0,1 beliefs are equivalent to fixed rotamers
            // just populate the beliefs with certainties
            fill(node_belief[0], 3, n_res3, 0.f);
            for(int nn: range(n_res3))
                node_belief[0](fixed_rotamers3[nn], nn) = 1.f;

            fill(edge_belief[0], 2*3, n_edge33, 0.f);
            for(int ne: range(n_edge33)) {
                int nr1 = edge_indices[ne*2+0];
                int nr2 = edge_indices[ne*2+1];
                edge_belief[0](fixed_rotamers3[nr1]  ,ne) = 1.f;
                edge_belief[0](fixed_rotamers3[nr2]+3,ne) = 1.f;
            }
        }

        float free_energy = 0.f;
        compute_free_energy_and_derivative(
                &free_energy,
                node_marginal_prob[0], edge_marginal_prob[0],
                pos3[0],
                pos_deriv1[0], pos_deriv3[0],
                n_res1,        n_res3,
                node_belief[0], edge_belief[0],
                edge_prob[0],
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data());

        compute_parameter_derivatives(
                parameter_deriv, 
                node_marginal_prob[0], edge_marginal_prob[0],
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data(),
                n_res1, restype1.data(), pos1[0],  // dimensionality 1*BT::n_pos_dim
                n_res3, restype3.data(), pos3[0],  // dimensionality 3*BT::n_pos_dim
                n_restype, interactions.data());

        return free_energy;
    }
};

// C-friendly interface, so we can connect to python
RotamerConstructAndSolve* new_rotamer_construct_and_solve(
             int n_restype_, 
             int n_res1_, int* restype1_, float* pos1_,
             int n_res3_, int* restype3_, float* pos3_,
             float damping_, int max_iter_, float tol_,
             int* fixed_rotamers3_) {
    typedef preferred_bead_type BT;

    auto rcas = new RotamerConstructAndSolve;
    auto &z = *rcas;

    z.n_restype = n_restype_;
    z.interactions.resize(z.n_restype*z.n_restype);

    z.n_res1 = n_res1_; z.restype1 = vector<int>(restype1_, restype1_+z.n_res1);
    z.n_res3 = n_res3_; z.restype3 = vector<int>(restype3_, restype3_+z.n_res3);

    z.pos1.reset(1,1*BT::n_pos_dim,z.n_res1); z.pos_deriv1.reset(1,1*BT::n_pos_dim,z.n_res1);
    z.pos3.reset(1,3*BT::n_pos_dim,z.n_res3); z.pos_deriv3.reset(1,3*BT::n_pos_dim,z.n_res3);

    for(int nr: range(z.n_res1)) for(int d: range(  BT::n_pos_dim)) z.pos1[0](d,nr) = pos1_[nr*BT::n_pos_dim+d];
    for(int nr: range(z.n_res3)) for(int d: range(3*BT::n_pos_dim)) z.pos3[0](d,nr) = pos3_[nr*3*BT::n_pos_dim+d];

    int max_edges11 = z.n_res1 * (z.n_res1-1) / 2; z.edges11.resize(max_edges11);
    int max_edges13 = z.n_res1 *  z.n_res3;        z.edges13.resize(max_edges13);
    int max_edges33 = z.n_res3 * (z.n_res3-1) / 2; z.edges33.resize(max_edges33);

    z.edge_indices.resize(2*max_edges33);

    z.node_prob.reset(1,3,z.n_res3); z.edge_prob.reset(1,3*3,max_edges33);

    z.node_belief     .reset(1,3,z.n_res3); z.     edge_belief.reset(1,2*3,max_edges33);
    z.temp_node_belief.reset(1,3,z.n_res3); z.temp_edge_belief.reset(1,2*3,max_edges33);

    z.node_marginal_prob.reset(1,3,z.n_res3); z.edge_marginal_prob.reset(1,3*3,max_edges33);

    if(fixed_rotamers3_) {
        z.fixed_rotamers3 = vector<int>(fixed_rotamers3_,fixed_rotamers3_+z.n_res3);
        for(auto i: range(z.fixed_rotamers3.size())) {
            if(z.fixed_rotamers3[i]<0 || z.fixed_rotamers3[i]>2 ) {
                fprintf(stderr,"failure at rotamer %i %i\n",
                        i,z.fixed_rotamers3[i]);
                throw 0;
            }
        }
    }

    z.damping = damping_;
    z.max_iter = max_iter_;
    z.tol = tol_;
    
    return rcas;
}


float free_energy_and_parameter_deriv(RotamerConstructAndSolve* rcas, 
        float* parameter_deriv, const float* interactions) {
    return rcas->free_energy_and_parameter_deriv(parameter_deriv, interactions);
}


void delete_rotamer_construct_and_solve(RotamerConstructAndSolve* rcas) {
    delete rcas;
}


void dump_factor_graph(const char* fname, 
        int n_node, VecArray node_prob,
        int n_edge, VecArray edge_prob, int* edge_indices) {
    auto f = fopen(fname, "w");
    fprintf(f, "%i\n", n_node+n_edge);  // number of factors

    for(int nn: range(n_node)) {
        fprintf(f,"\n");
        fprintf(f,"1\n"); // number of variables
        fprintf(f,"%i\n", nn); // label of variable
        fprintf(f,"3\n"); // number of states
        fprintf(f,"3\n"); // number of factor values
        for(int no: range(3))
            fprintf(f,"%i %f\n", no, node_prob(no,nn)); // factor graph entry
    }

    for(int ne: range(n_edge)) {
        fprintf(f,"\n");
        fprintf(f,"2\n"); // number of variables
        fprintf(f,"%i %i\n", edge_indices[2*ne+0], edge_indices[2*ne+1]); // label of variables
        fprintf(f,"3 3\n"); // number of states
        fprintf(f,"9\n"); // number of factor values
        // libdai works in column major ordering, but we work in row major ordering
        for(int no2: range(3))
            for(int no1: range(3))
                fprintf(f,"%i %f\n", no1 + no2*3, edge_prob(no1*3+no2,ne)); // factor graph entry
    }
    fclose(f);
}
