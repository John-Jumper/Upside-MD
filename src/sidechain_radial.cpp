#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "interaction_graph.h"
#include "spline.h"
#include "state_logger.h"

using namespace std;
using namespace h5;


namespace {
template <bool is_symmetric>
struct RadialHelper {
    // spline-based distance interaction
    // n_knot is the number of basis splines (including those required to get zero
    //   derivative in the clamped spline)
    // spline is constant over [0,dx] to avoid funniness at origin

    // Please obey these 4 conditions:
    // p[0] = 1./dx, that is the inverse of the knot spacing
    // should have p[1] == p[3] for origin clamp (p[0] is inv_dx)
    // should have p[-3] == p[-1] (negative indices from the end, Python-style) for terminal clamping
    // should have (1./6.)*p[-3] + (2./3.)*p[-2] + (1./6.)*p[-1] == 0. for continuity at cutoff

    constexpr static bool  symmetric = is_symmetric;
    constexpr static int   n_knot=16, n_param=1+n_knot, n_dim1=3, n_dim2=3, simd_width=1;

    static float cutoff(const float* p) {
        const float inv_dx = p[0];
        return (n_knot-2-1e-6)/inv_dx;  // 1e-6 just insulates us from round-off error
    }

    static bool is_compatible(const float* p1, const float* p2) {
        if(symmetric) for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
        return true;
    }

    static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
        auto sequence_exclude = Int4(2);
        return (sequence_exclude < id1-id2) | (sequence_exclude < id2-id1);
    }

    static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4], 
            const Vec<n_dim1,Float4> &x1, const Vec<n_dim2,Float4> &x2) {
        alignas(16) const float inv_dx_data[4] = {p[0][0], p[1][0], p[2][0], p[3][0]};

        auto inv_dx     = Float4(inv_dx_data, Alignment::aligned);
        auto disp       = x1-x2;
        auto dist2      = mag2(disp);
        auto inv_dist   = rsqrt(dist2+Float4(1e-7f));  // 1e-7 is divergence protection
        auto dist_coord = dist2*(inv_dist*inv_dx);

        const float* pp[4] = {p[0]+1, p[1]+1, p[2]+1, p[3]+1};
        auto en = clamped_deBoor_value_and_deriv(pp, dist_coord, n_knot);
        d1 = disp*(inv_dist*inv_dx*en.y());
        d2 = -d1;
        return en.x();
    }

    static void param_deriv(Vec<n_param> &d_param, const float* p,
            const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
        d_param = make_zero<n_param>();
        float inv_dx = p[0];
        auto dist_coord = inv_dx*mag(x1-x2); // to convert to spline coords of interger grid of knots
        auto dV_dinv_dx = clamped_deBoor_value_and_deriv(p+1, dist_coord, n_knot).y()*mag(x1-x2);
        d_param[0] = dV_dinv_dx;
 
        int starting_bin;
        float result[4];
        clamped_deBoor_coeff_deriv(&starting_bin, result, dist_coord, n_knot);
        for(int i: range(4)) d_param[1+starting_bin+i] = result[i];
   }
};



struct SidechainRadialPairs : public PotentialNode
{

    InteractionGraph<RadialHelper<true>> igraph;

    SidechainRadialPairs(hid_t grp, CoordNode& bb_point_):
        PotentialNode(),
        igraph(grp, &bb_point_)
    {};

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("radial_pairs"));

        igraph.compute_edges();
        for(int ne=0; ne<igraph.n_edge; ++ne) igraph.edge_sensitivity[ne] = 1.f;
        igraph.propagate_derivatives();

        if(mode==PotentialAndDerivMode) {
            potential = 0.f;
            for(int ne=0; ne<igraph.n_edge; ++ne) 
                potential += igraph.edge_value[ne];
        }
    }
};


struct HBondSidechainRadialPairs : public PotentialNode
{

    InteractionGraph<RadialHelper<false>> igraph;

    HBondSidechainRadialPairs(hid_t grp, CoordNode& hb_point_, CoordNode& bb_point_):
        PotentialNode(),
        igraph(grp, &hb_point_, &bb_point_)
    {};

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("hbond_sc_radial_pairs"));

        igraph.compute_edges();
        for(int ne=0; ne<igraph.n_edge; ++ne) igraph.edge_sensitivity[ne] = 1.f;
        igraph.propagate_derivatives();

        if(mode==PotentialAndDerivMode) {
            potential = 0.f;
            for(int ne=0; ne<igraph.n_edge; ++ne) 
                potential += igraph.edge_value[ne];
        }
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
};


struct ContactEnergy : public PotentialNode
{
    struct Param {
        index_t   loc[2];
        float     dist;
        float     scale;  // 1.f/width
        float     cutoff;
    };
    struct Pij {
        float one_minus_p_ij;
        float one_minus_p_i_after;
        Vec<3> deriv;
    };

    int n_group;
    int n_contact;

    CoordNode& bead_pos;
    vector<vector<Param>> params;
    vector<float> group_energy;
    vector<Pij> p_ij;

    ContactEnergy(hid_t grp, CoordNode& bead_pos_):
        PotentialNode(),
        n_group(get_dset_size(1, grp, "group_energy")[0]),
        n_contact(get_dset_size(2, grp, "id")[0]),
        bead_pos(bead_pos_)
    {
        check_size(grp, "group_energy", n_group);
        check_size(grp, "group_id",     n_contact);
        check_size(grp, "id",           n_contact, 2);
        check_size(grp, "energy",       n_contact);
        check_size(grp, "distance",     n_contact);
        check_size(grp, "width",        n_contact);

        vector<size_t> group_id;
        vector<size_t> loc_within_group;
        params.resize(n_group);

        traverse_dset<1,float>(grp, "group_energy", [&](size_t ng, float x){
                group_energy.push_back(x);});

        traverse_dset<1,int>(grp, "group_id", [&](size_t nc, int gid){
                group_id.push_back(gid);
                loc_within_group.push_back(params[gid].size());
                params[gid].emplace_back();
                });

        traverse_dset<2,int  >(grp, "id",       [&](size_t nc, size_t i, int x){
                params[group_id[nc]][loc_within_group[nc]].loc[i] = x;});

        traverse_dset<1,float>(grp, "distance", [&](size_t nc, float x){
                params[group_id[nc]][loc_within_group[nc]].dist = x;});

        traverse_dset<1,float>(grp, "width",    [&](size_t nc, float x){
                params[group_id[nc]][loc_within_group[nc]].scale = 1.f/x;});

        // update all of the cutoffs and count largest group
        int n_largest_group = 0;
        for(auto &ps: params) {
            n_largest_group = max(int(ps.size()), n_largest_group);
            for(auto& p: ps) p.cutoff = p.dist + 1.f/p.scale;
        }
        p_ij.resize(n_largest_group);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("contact_energy"));
        VecArray pos  = bead_pos.output;
        VecArray sens = bead_pos.sens;
        potential = 0.f;

        for(int ng=0; ng<n_group; ++ng) {
            auto& ps = params[ng];
            float one_minus_pi = 1.f;
            for(size_t nc=0; nc<ps.size(); ++nc) {
                const auto& p = ps[nc];
                auto disp = load_vec<3>(pos, p.loc[0]) - load_vec<3>(pos, p.loc[1]);
                auto dist = mag(disp);
                Vec<2> contact = (dist>=p.cutoff)
                    ? make_vec2(0.f,0.f)
                    : compact_sigmoid(dist-p.dist, p.scale);
                p_ij[nc].one_minus_p_ij = 1.f - contact.x();
                p_ij[nc].deriv = (contact.y()*rcp(dist)) * disp;
                one_minus_pi *= p_ij[nc].one_minus_p_ij;
            }
            potential += group_energy[ng] * (1.f-one_minus_pi);

            {
                float one_minus_p_i_after = 1.f;
                for(int nc=int(ps.size()-1); nc>=0; --nc) {
                    p_ij[nc].one_minus_p_i_after = one_minus_p_i_after;
                    one_minus_p_i_after *= p_ij[nc].one_minus_p_ij;
                }
            }

            float one_minus_p_i_before = 1.f;
            for(size_t nc=0; nc<ps.size(); ++nc) {
                float one_minus_pi_other = one_minus_p_i_before * p_ij[nc].one_minus_p_i_after;
                one_minus_p_i_before *= p_ij[nc].one_minus_p_ij;

                auto scaled_deriv = (group_energy[ng] * one_minus_pi_other) * p_ij[nc].deriv;
                update_vec(sens, ps[nc].loc[0],  scaled_deriv);
                update_vec(sens, ps[nc].loc[1], -scaled_deriv);
            }
        }
    }
};
}

static RegisterNodeType<ContactEnergy,1>             contact_node("contact");
static RegisterNodeType<SidechainRadialPairs,1>      radial_node ("radial");
static RegisterNodeType<HBondSidechainRadialPairs,2> hbond_sc_radial_node ("hbond_sc_radial");
