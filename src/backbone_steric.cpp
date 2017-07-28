#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"
#include <algorithm>
#include "interaction_graph.h"

using namespace std;
using namespace h5;

struct AffineParams {
    index_t residue;
};

namespace {

constexpr float nonbonded_atom_cutoff2 = 3.f*3.f + 0.1f*3.f;

template <bool return_deriv>
inline float
nonbonded_kernel_or_deriv_over_r(float r_mag2)
{
    const float energy_scale = 4.f;
    const float wall = 3.0f;  // corresponds to vdW *diameter*
    const float wall_squared = wall*wall;  
    const float width = 0.10f;
    const float sharpness = 1.f/(wall*width);  // ensure character

    const float2 V = energy_scale*compact_sigmoid(r_mag2-wall_squared, sharpness);
    return return_deriv ? 2.f*V.y() : V.x();
}

Int4 acceptable_backbone_pair(const Int4& id1, const Int4& id2) {
        auto sequence_exclude = Int4(1);
        return (sequence_exclude < id1-id2) | (sequence_exclude < id2-id1);
}
}

struct BackbonePairs : public PotentialNode
{
    struct RefPos {
        int32_t n_atom;
        float3  pos[4];
    };

    int n_residue;
    CoordNode& alignment;
    vector<AffineParams> params;
    vector<RefPos> ref_pos;
    PairlistComputation<true> pairlist;
    unique_ptr<int32_t[]> id;
    float dist_cutoff;

    BackbonePairs(hid_t grp, CoordNode& alignment_):
        PotentialNode(),
        n_residue(get_dset_size(1, grp, "id")[0]), alignment(alignment_), 
        params(n_residue), ref_pos(n_residue),
        pairlist(n_residue, n_residue, (n_residue*(n_residue-1))/2),
        id(new_aligned<int32_t>(n_residue,16))
    {
        check_elem_width(alignment, 7);

        check_size(grp, "id",      n_residue);
        check_size(grp, "n_atom",  n_residue);
        check_size(grp, "ref_pos", n_residue, 4, 3);

        traverse_dset<1,int>(grp, "id",     [&](size_t nr, int x) {params[nr].residue = x; id[nr]=x;});
        traverse_dset<1,int>(grp, "n_atom", [&](size_t nr, int x) {ref_pos[nr].n_atom = x;});

        traverse_dset<3,float>(grp, "ref_pos", [&](size_t nr, size_t na, size_t d, float x) {
                ref_pos[nr].pos[na][d] = x;});

        // set dist cutoff to largest distance that could possibly have an atom cutoff
        float max_atom_dev = 0.f;
        for(const auto& p: ref_pos)
            for(int na: range(p.n_atom))
                max_atom_dev = max(mag(p.pos[na]), max_atom_dev);

        dist_cutoff = 2*max_atom_dev + sqrtf(nonbonded_atom_cutoff2);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("backbone_pairs"));

        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        VecArrayStorage coords(3,round_up(n_residue,4));
        vector<int>    ref_pos_atoms (n_residue);
        vector<float3> ref_pos_coords(n_residue*4);

        if(pot) *pot = 0.f;
        for(int nr=0; nr<n_residue; ++nr) {
            auto aff = load_vec<7>(alignment.output, params[nr].residue);
            float U[9]; quat_to_rot(U, aff.v+3);
            auto t = extract<0,3>(aff);
            store_vec(coords,nr, t);

            ref_pos_atoms[nr] = ref_pos[nr].n_atom;
            for(int na=0; na<4; ++na) 
                ref_pos_coords[nr*4+na] = apply_affine(U,t, ref_pos[nr].pos[na]);
        }

        // acceptable_backbone_pair checks that nr2>=nr1+2
        pairlist.template find_edges<acceptable_backbone_pair>(dist_cutoff,
                coords.x.get(), coords.row_width, id.get(),
                coords.x.get(), coords.row_width, id.get());
        int n_edge = pairlist.n_edge;

        for(int ne=0; ne<n_edge; ne++) {
            int nr1 = pairlist.edge_indices1[ne];
            int nr2 = pairlist.edge_indices2[ne];

            auto d1 = make_zero<3>(); auto torque1 = make_zero<3>(); auto t1 = load_vec<3>(coords,nr1);
            auto d2 = make_zero<3>(); auto torque2 = make_zero<3>(); auto t2 = load_vec<3>(coords,nr2);

            int n_atom1 = ref_pos_atoms[nr1];
            int n_atom2 = ref_pos_atoms[nr2];

            bool hit = false;
            for(int i1=0; i1<n_atom1; ++i1) {
                const float3 x1 = ref_pos_coords[nr1*4+i1];

                for(int i2=0; i2<n_atom2; ++i2) {
                    const float3 x2 = ref_pos_coords[nr2*4+i2];

                    const float3 r = x1-x2;
                    const float r_mag2 = mag2(r);
                    if(r_mag2>nonbonded_atom_cutoff2) continue;
                    hit = true;
                    const float deriv_over_r  = nonbonded_kernel_or_deriv_over_r<true> (r_mag2);
                    if(pot)     *pot         += nonbonded_kernel_or_deriv_over_r<false>(r_mag2);
                    const float3 g = deriv_over_r*r;

                    d1 +=  g;  torque1 += cross(x1-t1,  g);
                    d2 += -g;  torque2 += cross(x2-t2, -g);
                }
            }

            if(hit) {
                Vec<6> combine_deriv1; store<0,3>(combine_deriv1, d1); store<3,6>(combine_deriv1, torque1);
                Vec<6> combine_deriv2; store<0,3>(combine_deriv2, d2); store<3,6>(combine_deriv2, torque2);

                update_vec(alignment.sens, params[nr1].residue, combine_deriv1);
                update_vec(alignment.sens, params[nr2].residue, combine_deriv2);
            }
        }
    }
};
static RegisterNodeType<BackbonePairs,1> backbone_pairs_node("backbone_pairs");
