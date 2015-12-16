#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "interaction_graph.h"
#include "spline.h"

using namespace std;
using namespace h5;

struct ContactPair {
    index_t loc[2];
    float3    sc_ref_pos[2];
    float     r0;
    float     scale;
    float     energy;
};

namespace {
struct SidechainRadialPairs : public PotentialNode
{
    struct Helper {
        // spline-based distance interaction
        // n_knot is the number of basis splines (including those required to get zero
        //   derivative in the clamped spline)
        // spline is constant over [0,dx] to avoid funniness at origin

        // Please obey these 4 conditions:
        // p[0] = 1./dx, that is the inverse of the knot spacing
        // should have p[1] == p[3] for origin clamp (p[0] is inv_dx)
        // should have p[-3] == p[-1] (negative indices from the end, Python-style) for terminal clamping
        // should have (1./6.)*p[-3] + (2./3.)*p[-2] + (1./6.)*p[-1] == 0. for continuity at cutoff

        constexpr static bool  symmetric = true;
        constexpr static int   n_knot=16, n_param=1+n_knot, n_dim1=3, n_dim2=3, simd_width=1;

        static float cutoff(const float* p) {
            const float inv_dx = p[0];
            return (n_knot-2-1e-6)/inv_dx;  // 1e-6 just insulates us from round-off error
        }

        static bool is_compatible(const float* p1, const float* p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            auto sequence_cutoff = Int4(3);
            return (sequence_cutoff < id1-id2) | (sequence_cutoff < id2-id1);
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
            auto en = clamped_deBoor_value_and_deriv(pp, dist_coord, n_param);
            d1 = disp*(inv_dist*inv_dx*en.y());
            d2 = -d1;
            return en.x();
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {}
    };

    InteractionGraph<Helper> igraph;

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
}


/*
void contact_energy(
        float* potential,
        const CoordArray   rigid_body,
        const ContactPair* contact_param,
        int n_contacts, float cutoff)
{
    if(potential) potential[0] = 0.f;
    for(int nc=0; nc<n_contacts; ++nc) {
        ContactPair p = contact_param[nc];
        AffineCoord<> r1(rigid_body, p.loc[0]);
        AffineCoord<> r2(rigid_body, p.loc[1]);

        float3 x1 = r1.apply(p.sc_ref_pos[0]);
        float3 x2 = r2.apply(p.sc_ref_pos[1]);

        float3 disp = x1-x2;
        float  dist = mag(disp);
        float  reduced_coord = p.scale * (dist - p.r0);

        if(reduced_coord<cutoff) {
            float  z = expf(reduced_coord);
            float  w = 1.f / (1.f + z);
            if(potential) potential[0] += p.energy * w;
            float  deriv_over_r = -p.scale/dist * p.energy * z * (w*w);
            float3 deriv = deriv_over_r * disp;

            r1.add_deriv_at_location(x1,  deriv);
            r2.add_deriv_at_location(x2, -deriv);
        }

        r1.flush();
        r2.flush();
    }
}

struct ContactEnergy : public PotentialNode
{
    int n_contact;
    CoordNode& alignment;
    vector<ContactPair> params;
    float cutoff;

    ContactEnergy(hid_t grp, CoordNode& alignment_):
        PotentialNode(),
        n_contact(get_dset_size(2, grp, "id")[0]),
        alignment(alignment_), 
        params(n_contact),
        cutoff(read_attribute<float>(grp, ".", "cutoff"))
    {
        check_elem_width(alignment, 7);

        check_size(grp, "id",         n_contact, 2);
        check_size(grp, "sc_ref_pos", n_contact, 2, 3);
        check_size(grp, "r0",         n_contact);
        check_size(grp, "scale",      n_contact);
        check_size(grp, "energy",     n_contact);

        traverse_dset<2,int  >(grp, "id",         [&](size_t nc, size_t i, int x) {params[nc].loc[i] = x;});
        traverse_dset<3,float>(grp, "sc_ref_pos", [&](size_t nc, size_t i, size_t d, float x) {
                params[nc].sc_ref_pos[i][d] = x;});

        traverse_dset<1,float>(grp, "r0",     [&](size_t nc, float x) {params[nc].r0     = x;});
        traverse_dset<1,float>(grp, "scale",  [&](size_t nc, float x) {params[nc].scale  = x;});
        traverse_dset<1,float>(grp, "energy", [&](size_t nc, float x) {params[nc].energy = x;});

        for(int j=0; j<2; ++j) 
            for(size_t i=0; i<params.size(); ++i) 
                alignment.slot_machine.add_request(1, params[i].loc[j]);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("contact_energy"));
        contact_energy((mode==PotentialAndDerivMode ? &potential : nullptr),
                alignment.coords(), params.data(), 
                n_contact, cutoff);
    }

    virtual double test_value_deriv_agreement() {return -1.;}
};
static RegisterNodeType<ContactEnergy,1>        contact_node("contact");
*/
static RegisterNodeType<SidechainRadialPairs,1> radial_node ("radial");
