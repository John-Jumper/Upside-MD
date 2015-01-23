#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "coord.h"
#include "affine.h"
#include <cmath>
#include <vector>

using namespace std;
using namespace h5;

struct SidechainRadialParams {
    CoordPair loc;
    int       restype;
};


struct SidechainRadialInteraction {
    float r0_squared;
    float scale;
    float energy;
};


struct ContactPair {
    CoordPair loc[2];
    float3    sc_ref_pos[2];
    float     r0;
    float     scale;
    float     energy;
};

struct SidechainRadialResidue {
    int      restype;
    Coord<3> coord;

    SidechainRadialResidue(int restype_, const CoordArray &ca, int ns, const CoordPair &loc):
        restype(restype_),
        coord(ca, ns, loc) {}
};

void radial_pairs(
        float* potential,
        const CoordArray                  interaction_pos,
        const SidechainRadialParams*      residue_param,
        const SidechainRadialInteraction* interaction_params,
        int n_types, float cutoff,
        int n_res, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        if(potential) potential[ns] = 0.f;
        vector<SidechainRadialResidue> residues;  residues.reserve(n_res);

        for(int nr=0; nr<n_res; ++nr)
            residues.emplace_back(residue_param[nr].restype, interaction_pos, ns, residue_param[nr].loc); 

        for(int nr1=0; nr1<n_res; ++nr1) {
            SidechainRadialResidue &r1 = residues[nr1];

            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
                SidechainRadialResidue &r2 = residues[nr2];

                float3 disp = r1.coord.f3() - r2.coord.f3();
                SidechainRadialInteraction at = interaction_params[r1.restype*n_types + r2.restype];
                float dist2 = mag2(disp);
                float reduced_coord = at.scale * (dist2 - at.r0_squared);

                if(reduced_coord<cutoff) {
                    //printf("reduced_coord %.1f %.1f\n",reduced_coord,sqrtf(dist2));
                    float  z = expf(reduced_coord);
                    float  w = 1.f / (1.f + z);
                    if(potential) potential[ns] += w;
                    float  deriv_over_r = -2.f*at.scale * at.energy * z * (w*w);
                    float3 deriv = deriv_over_r * disp;

                    r1.coord.d[0][0] += deriv.x; r2.coord.d[0][0] += -deriv.x; 
                    r1.coord.d[0][1] += deriv.y; r2.coord.d[0][1] += -deriv.y; 
                    r1.coord.d[0][2] += deriv.z; r2.coord.d[0][2] += -deriv.z; 
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr)
            residues[nr].coord.flush();
    }
}


void contact_energy(
        float* potential,
        const CoordArray   rigid_body,
        const ContactPair* contact_param,
        int n_contacts, float cutoff, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        if(potential) potential[ns] = 0.f;
        for(int nc=0; nc<n_contacts; ++nc) {
            ContactPair p = contact_param[nc];
            AffineCoord<> r1(rigid_body, ns, p.loc[0]);
            AffineCoord<> r2(rigid_body, ns, p.loc[1]);

            float3 x1 = r1.apply(p.sc_ref_pos[0]);
            float3 x2 = r2.apply(p.sc_ref_pos[1]);

            float3 disp = x1-x2;
            float  dist = mag(disp);
            float  reduced_coord = p.scale * (dist - p.r0);

            if(reduced_coord<cutoff) {
                float  z = expf(reduced_coord);
                float  w = 1.f / (1.f + z);
                if(potential) potential[ns] += w;
                float  deriv_over_r = -p.scale/dist * p.energy * z * (w*w);
                float3 deriv = deriv_over_r * disp;

                r1.add_deriv_at_location(x1,  deriv);
                r2.add_deriv_at_location(x2, -deriv);
            }

            r1.flush();
            r2.flush();
        }
    }
}


struct SidechainRadialPairs : public PotentialNode
{
    map<string,int> name_map;
    int n_residue;
    int n_type;
    CoordNode& bb_point;

    vector<SidechainRadialParams> params;
    vector<SidechainRadialInteraction> interaction_params;
    float cutoff;

    SidechainRadialPairs(hid_t grp, CoordNode& bb_point_):
        PotentialNode(bb_point_.n_system),
        n_residue(get_dset_size(1, grp, "id"   )[0]), 
        n_type   (get_dset_size(1, grp, "data/names")[0]),
        bb_point(bb_point_), 

        params(n_residue), interaction_params(n_type*n_type),
        cutoff(read_attribute<float>(grp, "data", "cutoff"))
    {
        check_elem_width(bb_point, 3);

        check_size(grp, "id",      n_residue);
        check_size(grp, "restype", n_residue);

        check_size(grp, "data/names",      n_type);
        check_size(grp, "data/energy",     n_type, n_type);
        check_size(grp, "data/scale",      n_type, n_type);
        check_size(grp, "data/r0_squared", n_type, n_type);

        int i=0; 
        traverse_string_dset<1>(grp, "data/names", [&](size_t nt, std::string &s) {name_map[s]=i++;});
        if(i!=n_type) throw std::string("internal error");

        traverse_dset<2,float>(grp, "data/energy", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].energy = x;});
        traverse_dset<2,float>(grp, "data/scale", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].scale = x;});
        traverse_dset<2,float>(grp, "data/r0_squared", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].r0_squared = x;});

        traverse_dset<1,int   >(grp, "id",      [&](size_t nr, int x) {params[nr].loc.index = x;});
        traverse_string_dset<1>(grp, "restype", [&](size_t nr, std::string &s) {
                if(name_map.find(s) == end(name_map)) std::string("restype contains name not found in data/");
                params[nr].restype    = name_map[s];
            });

        for(size_t nr=0; nr<params.size(); ++nr) bb_point.slot_machine.add_request(1, params[nr].loc);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("radial_pairs"));
        radial_pairs((mode==PotentialAndDerivMode ? potential.data() : nullptr),
                bb_point.coords(), 
                params.data(), interaction_params.data(), n_type, cutoff, 
                n_residue, bb_point.n_system);
    }
};

struct ContactEnergy : public PotentialNode
{
    int n_contact;
    CoordNode& alignment;
    vector<ContactPair> params;
    float cutoff;

    ContactEnergy(hid_t grp, CoordNode& alignment_):
        PotentialNode(alignment_.n_system),
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
                component(params[nc].sc_ref_pos[i], d) = x;});

        traverse_dset<1,float>(grp, "r0",     [&](size_t nc, float x) {params[nc].r0     = x;});
        traverse_dset<1,float>(grp, "scale",  [&](size_t nc, float x) {params[nc].scale  = x;});
        traverse_dset<1,float>(grp, "energy", [&](size_t nc, float x) {params[nc].energy = x;});

        for(int j=0; j<2; ++j) 
            for(size_t i=0; i<params.size(); ++i) 
                alignment.slot_machine.add_request(1, params[i].loc[j]);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("contact_energy"));
        contact_energy((mode==PotentialAndDerivMode ? potential.data() : nullptr),
                alignment.coords(), params.data(), 
                n_contact, cutoff, alignment.n_system);
    }
};
static RegisterNodeType<ContactEnergy,1>        contact_node("contact");
static RegisterNodeType<SidechainRadialPairs,1> radial_node ("radial");
