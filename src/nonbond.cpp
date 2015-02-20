#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"
#include "affine.h"
#include <algorithm>

using namespace std;
using namespace h5;

struct AffineParams {
    CoordPair residue;
};

struct RefPos {
    int32_t n_atom;
    float3  pos[4];
};

namespace {

template <bool return_deriv>
inline float
nonbonded_kernel_or_deriv_over_r(float r_mag2)
{
    const float wall = 3.0f;  // corresponds to vdW *diameter*
    const float wall_squared = wall*wall;  
    const float width = 0.10f;
    const float sharpness = 1.f/(wall*width);  // ensure character

    const float2 V = compact_sigmoid(r_mag2-wall_squared, sharpness);
    return return_deriv ? 2.f*V.y : V.x;
}

// template <bool return_deriv>
// inline float
// nonbonded_kernel_or_deriv_over_r(float r_mag2)
// {
//     const float wall = 2.8f;  // corresponds to vdW *diameter*
//     const float wall_squared = wall*wall;  
//     const float width = 0.10f;
//     const float scale_factor = 1.f/(wall*width);  // ensure character
// 
//     // overflow protection prevents NaN
//     float z = fminf(expf(scale_factor * (r_mag2-wall_squared)), 1e12f);
//     float w = 1.f/(1.f + z);  // include protection from 0
//     printf("nbk % f % f   % f % f\n", sqrtf(r_mag2), wall, w, -scale_factor * z * (w*w));
// 
//     if(return_deriv) {
//         float deriv_over_r = -2.f*scale_factor * z * (w*w);
//         return deriv_over_r;
//     } else {
//         return w;
//     }
// }


template <typename AffineCoordT>
inline void backbone_pairs_body(
        float* pot_value,
        AffineCoordT &body1,
        AffineCoordT &body2,
        int n_atom1, const float3* restrict rpos1,
        int n_atom2, const float3* restrict rpos2)
{
    for(int i1=0; i1<n_atom1; ++i1) {
        const float3 x1 = rpos1[i1];

        for(int i2=0; i2<n_atom2; ++i2) {
            const float3 x2 = rpos2[i2];

            const float3 r = x1-x2;
            const float r_mag2 = mag2(r);
            if(r_mag2>4.0f*4.0f) continue;
            const float deriv_over_r  = nonbonded_kernel_or_deriv_over_r<true> (r_mag2);
            if(pot_value) *pot_value += nonbonded_kernel_or_deriv_over_r<false>(r_mag2);
            const float3 g = deriv_over_r*r;

            body1.add_deriv_at_location(x1,  g);
            body2.add_deriv_at_location(x2, -g);
        }
    }
}

}

void backbone_pairs(
        float* potential,
        const CoordArray rigid_body,
        const RefPos* restrict ref_pos,
        const AffineParams* restrict params,
        float energy_scale,
        float dist_cutoff,
        int n_res, int n_system)
{
#pragma omp parallel for schedule(static,1)
    for(int ns=0; ns<n_system; ++ns) {
        if(potential) potential[ns] = 0.f;
        float dist_cutoff2 = dist_cutoff*dist_cutoff;
        vector<AffineCoord<>> coords; coords.reserve(n_res);
        for(int nr=0; nr<n_res; ++nr) 
            coords.emplace_back(rigid_body, ns, params[nr].residue);

        vector<int>    ref_pos_atoms (n_res);
        vector<float3> ref_pos_coords(n_res*4);

        for(int nr=0; nr<n_res; ++nr) {
            ref_pos_atoms[nr] = ref_pos[nr].n_atom;
            for(int na=0; na<4; ++na) ref_pos_coords[nr*4+na] = coords[nr].apply(ref_pos[nr].pos[na]);
        }

        for(int nr1=0; nr1<n_res; ++nr1) {
            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // start interactions at i+3
                if(mag2(coords[nr1].tf3()-coords[nr2].tf3()) < 2.f*dist_cutoff2) { // FIXME debug
                    backbone_pairs_body(
                            (potential ? potential+ns : nullptr),
                            coords[nr1],        coords[nr2], 
                            ref_pos_atoms[nr1], &ref_pos_coords[nr1*4],
                            ref_pos_atoms[nr2], &ref_pos_coords[nr2*4]);
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr) {
            for(int d=0; d<6; ++d) coords[nr].d[0][d] *= energy_scale;
            coords[nr].flush();
        }
        if(potential) potential[ns] *= energy_scale;
    }
}


struct BackbonePairs : public PotentialNode
{
    int n_residue;
    CoordNode& alignment;
    vector<AffineParams> params;
    vector<RefPos> ref_pos;
    float energy_scale;
    float dist_cutoff;

    BackbonePairs(hid_t grp, CoordNode& alignment_):
        PotentialNode(alignment_.n_system),
        n_residue(get_dset_size(1, grp, "id")[0]), alignment(alignment_), 
        params(n_residue), ref_pos(n_residue),
        energy_scale(read_attribute<float>(grp, ".", "energy_scale")),
        dist_cutoff (read_attribute<float>(grp, ".", "dist_cutoff"))
    {
        check_elem_width(alignment, 7);

        check_size(grp, "id",      n_residue);
        check_size(grp, "n_atom",  n_residue);
        check_size(grp, "ref_pos", n_residue, 4, 3);

        traverse_dset<1,int>(grp, "id",     [&](size_t nr, int x) {params[nr].residue.index = x;});
        traverse_dset<1,int>(grp, "n_atom", [&](size_t nr, int x) {ref_pos[nr].n_atom = x;});

        traverse_dset<3,float>(grp, "ref_pos", [&](size_t nr, size_t na, size_t d, float x) {
                component(ref_pos[nr].pos[na], d) = x;});

        for(size_t nr=0; nr<params.size(); ++nr) alignment.slot_machine.add_request(1, params[nr].residue);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("backbone_pairs"));
        backbone_pairs(
                (mode==PotentialAndDerivMode ? potential.data() : nullptr),
                alignment.coords(), 
                ref_pos.data(), params.data(), energy_scale, dist_cutoff, n_residue, 
                alignment.n_system);
    }

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs(1);
        for(auto &p: params) coord_pairs.back().push_back(p.residue);
        return compute_relative_deviation_for_node<7,BackbonePairs,BODY_VALUE>(*this, alignment, coord_pairs);
    }
};
static RegisterNodeType<BackbonePairs,1> backbone_pairs_node("backbone_pairs");
