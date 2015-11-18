#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "coord.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "interaction_graph.h"

using namespace std;
using namespace h5;

struct ContactPair {
    CoordPair loc[2];
    float3    sc_ref_pos[2];
    float     r0;
    float     scale;
    float     energy;
};

struct SidechainRadialPairs : public PotentialNode
{
    struct Helper {
        // params are r0_squared, scale, energy
        constexpr static float base_cutoff = 8.f;  
        constexpr static bool  symmetric = true;
        constexpr static int   n_param=3, n_dim1=3, n_dim2=3, simd_width=1;

        static float cutoff(const float* p) {return sqrtf(p[0] + base_cutoff/p[1]);}
        static bool is_compatible(const float* p1, const float* p2) {
            for(int i: range(n_param)) if(p1[i]!=p2[i]) return false;
            return true;
        }

        static bool exclude_by_id(int id1, int id2) { return (id1-id2<2) & (id2-id1<2); } // no nearest neighbor

        static float compute_edge(Vec<n_dim1> &d1, Vec<n_dim2> &d2, const float* p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
                float3 disp = x1-x2;
                float  z = expf(p[1] * (mag2(disp) - p[0]));
                float  w = 1.f / (1.f + z);

                float  deriv_over_r = -2.f*p[1] * p[2] * z * (w*w);
                d1 = deriv_over_r * disp;
                d2 = -d1;
                return p[2]*w;
        };

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

    virtual double test_value_deriv_agreement() { return -1.f; }
};


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

        traverse_dset<2,int  >(grp, "id",         [&](size_t nc, size_t i, int x) {params[nc].loc[i].index = x;});
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
static RegisterNodeType<SidechainRadialPairs,1> radial_node ("radial");
