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

using namespace std;
using namespace h5;


struct SidechainInteractionParams {
    float cutoff2;
    float radius_outer, scale_outer, energy_outer;
    float radius_inner, scale_inner, energy_inner;

    void update_cutoff2() {
        float cutoff2_inner = sqr(radius_inner + 1.f/scale_inner);
        float cutoff2_outer = sqr(radius_outer + 1.f/scale_outer);
        cutoff2 = cutoff2_inner<cutoff2_outer ? cutoff2_outer : cutoff2_inner;
    }

    bool operator==(const SidechainInteractionParams& other) const {
        return cutoff2      == other.cutoff2      && 
               radius_inner == other.radius_inner && radius_outer == other.radius_outer &&
               energy_inner == other.energy_inner && energy_outer == other.energy_outer &&
               scale_inner  == other.scale_inner  && scale_outer  == other.scale_outer;
    }

    bool operator!=(const SidechainInteractionParams& other) {return !(*this==other);}
};


struct ResidueLoc {
    int restype;
    CoordPair affine_idx;
    CoordPair rama_idx;
};

template <int n_rot>
struct RotamerPlacement {
    int n_res;

    vector<ResidueLoc> loc;
    vector<int> global_restype;

    LayeredPeriodicSpline2D<n_rot*4> spline; // include both location and potential
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
        pos      (n_system_, n_rot*4, loc.size()), // order xyzv xyzv ...
        pos_deriv(n_system_, n_rot*4, loc.size()), // to be filled later by propagate_derivatives
        phi_deriv(n_system_, n_rot*4, loc.size()),
        psi_deriv(n_system_, n_rot*4, loc.size()),
        n_system(n_system_)
    {
        if(int(loc.size()) != n_res || int(global_restype.size()) != n_res) throw string("Internal error");
        spline.fit_spline(spline_data);
    }


    void push_derivatives(SysArray s_affine_pos, SysArray s_affine_deriv, SysArray s_rama_deriv, int ns) {
        VecArray affine_pos   = s_affine_pos[ns];
        VecArray affine_deriv = s_affine_deriv[ns];
        VecArray rama_deriv   = s_rama_deriv[ns];
        VecArray v_pos = pos[ns];
        VecArray v_pos_deriv = pos_deriv[ns];

        for(int nr: range(n_res)) {
            auto d = load_vec<n_rot*4>(v_pos_deriv, nr);
            store_vec(rama_deriv,loc[nr].rama_idx.slot, make_vec2(
                        dot(d,load_vec<n_rot*4>(phi_deriv[ns],nr)),
                        dot(d,load_vec<n_rot*4>(psi_deriv[ns],nr))));

            Vec<6> z; z[0]=z[1]=z[2]=z[3]=z[4]=z[5]=0.f;
            for(int no: range(n_rot)) {
                auto dx = make_vec3(d[no*4+0], d[no*4+1], d[no*4+2]);
                auto t  = load_vec<3>(affine_pos,loc[nr].affine_idx.index);
                auto x  = load_vec<3>(v_pos.shifted(no*4),nr);
                auto tq = cross(x-t,dx);  // torque relative to the residue center
                z[0] += dx[0]; z[1] += dx[1]; z[2] += dx[2];
                z[3] += tq[0]; z[4] += tq[1]; z[5] += tq[2];
            }
            store_vec(affine_deriv,loc[nr].affine_idx.slot, z);
        }
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

            float germ[n_rot*4*3]; 
            spline.evaluate_value_and_deriv(germ, loc[nr].restype, 
                    (r[0]+shift)*scale_x, (r[1]+shift)*scale_y);

            for(int no: range(n_rot)) {
                float* val = germ+no*4*3;

                float3 dx_dphi = apply_rotation(U, make_vec3(val[0*3+0], val[1*3+0], val[2*3+0])) * scale_x;
                float3 dx_dpsi = apply_rotation(U, make_vec3(val[0*3+1], val[1*3+1], val[2*3+1])) * scale_y;
                float3  x      = apply_affine(U,t, make_vec3(val[0*3+2], val[1*3+2], val[2*3+2]));

                float  dv_dphi = val[3*3+0] * scale_x;
                float  dv_dpsi = val[3*3+1] * scale_y;
                float   v      = val[3*3+2];

                phi_d(4*no+0,nr) = dx_dphi[0];
                phi_d(4*no+1,nr) = dx_dphi[1];
                phi_d(4*no+2,nr) = dx_dphi[2];
                phi_d(4*no+3,nr) = dv_dphi;

                psi_d(4*no+0,nr) = dx_dpsi[0];
                psi_d(4*no+1,nr) = dx_dpsi[1];
                psi_d(4*no+2,nr) = dx_dpsi[2];
                psi_d(4*no+3,nr) = dv_dpsi;

                pos_s(4*no+0,nr) =  x[0];
                pos_s(4*no+1,nr) =  x[1];
                pos_s(4*no+2,nr) =  x[2];
                pos_s(4*no+3,nr) =  v;

                // printf("placed %2i %1i", nr, no);
                // for(auto &s: {phi_d, psi_d, pos_s}) printf(" (% 7.3f,% 7.3f,% 7.3f,%7.3f)",
                //         s(4*no+0,nr), s(4*no+1,nr), s(4*no+2,nr), s(4*no+3,nr));
                // printf("\n");
            }
        }
    }
};


template <int n_rot1, int n_rot2>
struct PairInteraction {
    int nr1,nr2;
    float  potential[n_rot1][n_rot2];
    float3 deriv    [n_rot1][n_rot2];  // deriv is deriv with respect to the first residue
};


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


template <int n_rot1, int n_rot2>  // must have n_dim1 <= n_dim2
void compute_graph_elements(
        int* n_edge,  int max_edges,// length n_system
        PairInteraction<n_rot1,n_rot2>* edges,  // must be large enough to fit all generated edges
        int n_res1, const int* restype1, SysArray pos1,  // dimensionality n_rot1*4
        int n_res2, const int* restype2, SysArray pos2,  // dimensionality n_rot2*4
        int n_restype, const SidechainInteractionParams* interactions,
        int ns, bool is_self_interaction) {
    n_edge[ns] = 0;
    VecArray pos1_s = pos1[ns];
    VecArray pos2_s = pos2[ns];

    for(int nr1: range(n_res1)) {
        int nt1 = restype1[nr1];
        float3 rot1[n_rot1]; 
        for(int no1: range(n_rot1)) 
            rot1[no1] = load_vec<3>(pos1_s.shifted(4*no1), nr1);

        for(int nr2: range((is_self_interaction ? nr1+1 : 0), n_res2)) {
            int nt2 = restype2[nr2];

            auto &p = interactions[nt1*n_restype+nt2];
            float cutoff2 = p.cutoff2;

            // FIXME introduce some sort of early cutoff to reduce the cost of 
            //   checking every rotamer when there cannot be a hit.
            bool within_cutoff = 0;
            for(int no2: range(n_rot2)) {
                float3 rot2 = load_vec<3>(pos2_s.shifted(4*no2), nr2);
                for(int no1: range(n_rot1))
                    within_cutoff |= mag2(rot1[no1]-rot2) < cutoff2;
            }

            if(within_cutoff) {
                // grab the next edge location available and increment the count of required slots
                auto& edge = edges[ns*max_edges + (n_edge[ns]++)];
                edge.nr1 = nr1;
                edge.nr2 = nr2;

                for(int no2: range(n_rot2)) {
                    float3 rot2 = load_vec<3>(pos2_s.shifted(4*no2), nr2);
                    for(int no1: range(n_rot1)) {
                        float3 disp = rot1[no1]-rot2;
                        float dist2     = mag2(disp);
                        float inv_dist  = rsqrt(dist2);
                        float dist      = dist2*inv_dist;

                        float2 en = p.energy_outer*compact_sigmoid(dist-p.radius_outer, p.scale_outer)
                            +p.energy_inner*compact_sigmoid(dist-p.radius_inner, p.scale_inner);
                        edge.potential[no1][no2] = en.x();
                        edge.deriv    [no1][no2] = en.y() * (disp*inv_dist);
                    }
                }
            }
        }
    }
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
template<int n_rot>
pair<int,float> solve_for_beliefs(
        VecArray node_belief,      VecArray edge_belief, 
        VecArray temp_node_belief, VecArray temp_edge_belief,
        int n_node, VecArray node_prob,
        int n_edge, VecArray edge_prob, int* edge_indices_this_system,
        float damping, // 0.f indicates no damping
        int max_iter, float tol, bool re_initialize_beliefs) {

    if(re_initialize_beliefs) {
        for(int d: range(  n_rot)) for(int nn: range(n_node)) node_belief(d,nn) = 1.f;
        for(int d: range(2*n_rot)) for(int ne: range(n_edge)) edge_belief(d,ne) = 1.f;
    }

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


void convert_potential_graph_to_probability_graph(
        SysArray node_prob, SysArray edge_prob, int* edge_indices,
        int n_res3, SysArray pos3, // last dimension is 1-residue potential
        int* n_edge13, int max_edges13, PairInteraction<1,3>* edges13,
        int* n_edge33, int max_edges33, PairInteraction<3,3>* edges33,
        int ns) {

    // float potential_shift = 0.f;
    VecArray ep = edge_prob[ns];

    for(int ne33: range(n_edge33[ns])) {
        auto &e = edges33[ns*max_edges33 +ne33];
        edge_indices[ns*2*max_edges33 + ne33*2 + 0] = e.nr1;
        edge_indices[ns*2*max_edges33 + ne33*2 + 1] = e.nr2;

        float min_pot = 1e10f;
        for(int no1: range(3))
            for(int no2: range(3))
                min_pot = min(e.potential[no1][no2], min_pot);
        // potential_shift += min_pot;

        for(int no1: range(3))
            for(int no2: range(3))
                ep(no1*3+no2,ne33) = expf(-(e.potential[no1][no2] - min_pot));  // shift to avoid underflow later
    }

    VecArray np = node_prob[ns];
    VecArray p  = pos3[ns];

    for(int no: range(3))
        for(int nr: range(n_res3)) 
            np(no,nr) = p(no*4+3,nr);

    for(int ne13: range(n_edge13[ns])) {
        auto &e = edges13[ns*max_edges13+ne13];
        int nr = e.nr2;

        for(int no: range(3)) np(no,nr) += e.potential[0][no];
    }

    for(int nr: range(n_res3)) {
        float min_pot = 1e10f;
        for(int no: range(3))
            min_pot = min(np(no,nr), min_pot);
        // potential_shift += min_pot;

        for(int no: range(3)) 
            np(no,nr) = expf(-(np(no,nr)-min_pot));
    }
    // printf("potential_shift %.4f\n", potential_shift);
}


void propagate_derivatives(
        float* potential,
        SysArray s_pos,
        SysArray s_deriv1, SysArray s_deriv3,
        int n_res1, int n_res3,
        SysArray s_node_belief, SysArray s_edge_belief,
        SysArray s_edge_prob,
        const int* n_edge11, int max_edges11, const PairInteraction<1,1>* edges11,
        const int* n_edge13, int max_edges13, const PairInteraction<1,3>* edges13,
        const int* n_edge33, int max_edges33, const PairInteraction<3,3>* edges33,
        int ns) {

    VecArray pos = s_pos[ns];
    VecArray node_belief = s_node_belief[ns];
    VecArray edge_belief = s_edge_belief[ns];
    VecArray edge_prob = s_edge_prob[ns];
    VecArray deriv1 = s_deriv1[ns];
    VecArray deriv3 = s_deriv3[ns];

    // start with zero derivative
    fill(deriv1, 1*4, n_res1, 0.f);
    fill(deriv3, 3*4, n_res3, 0.f);

    double free_energy = 0.f;

    // node beliefs couple directly to the 1-body potentials
    for(int nr: range(n_res3)) {
        auto b = load_vec<3>(node_belief, nr);
        b *= rcp(sum(b)); // normalize probability

        for(int no: range(3)) {
            deriv3(no*4+3,nr) = b[no];
            // potential is given by the 3th element of position (note that -S is p*log p with no minus)
            if(potential) free_energy += b[no]*(pos(no*4+3,nr) + logf(1e-10f+b[no])); // 1-body entropies
        }
    }

    // edge beliefs couple to the positions
    for(int ne11: range(n_edge11[ns])) {
        auto &e = edges11[ns*max_edges11 + ne11];
        float3 z = e.deriv[0][0];
        update_vec(deriv1, e.nr1,  z);
        update_vec(deriv1, e.nr2, -z);
        if(potential) free_energy += e.potential[0][0]; // no entropy since only 1 state for each
    }

    for(int ne13: range(n_edge13[ns])) {
        auto &e = edges13[ns*max_edges13 + ne13];
        Vec<3> b = load_vec<3>(node_belief, e.nr2);
        b *= rcp(sum(b)); // normalize probability

        if(potential) free_energy += b[0]*e.potential[0][0]+b[1]*e.potential[0][1]+b[2]*e.potential[0][2];

        float3 d[3] = {b[0]*e.deriv[0][0], b[1]*e.deriv[0][1], b[2]*e.deriv[0][2]};

        update_vec(deriv1,e.nr1,  d[0]+d[1]+d[2]);
        for(int no2: range(3)) update_vec(deriv3.shifted(4*no2),e.nr2, -d[no2]);
    }

    // The edge marginal distributions are given by p(x1,x2) *
    // node_belief_1(x1) * node_belief_2(x2) / (edge_belief_12(x1) *
    // edge_belief_21(x2)) up to normalization.
    for(int ne33: range(n_edge33[ns])) {
        auto &e = edges33[ns*max_edges33+ne33];
        float3 b1 = load_vec<3>(node_belief, e.nr1);
        float3 b2 = load_vec<3>(node_belief, e.nr2);

        // correct for self interaction
        float3 bc1 = b1 * vec_rcp(load_vec<3>(edge_belief,            ne33));
        float3 bc2 = b2 * vec_rcp(load_vec<3>(edge_belief.shifted(3), ne33));

        Vec<9> pair_distrib = load_vec<9>(edge_prob, ne33);
        for(int no1: range(3))
            for(int no2: range(3))
                pair_distrib[no1*3+no2] *= bc1[no1]*bc2[no2];
        pair_distrib *= rcp(sum(pair_distrib));

        // normalize beliefs to obtain node marginals again
        b1 *= rcp(sum(b1));
        b2 *= rcp(sum(b2));

        if(potential) {
            float local_free_energy = 0.f;
            for(int no1: range(3)) {
                for(int no2: range(3)) {
                    auto p = pair_distrib[no1*3+no2];
                    local_free_energy += p*(e.potential[no1][no2] + logf(1e-10f + p*rcp(b1[no1]*b2[no2])));
                }
            }
            free_energy += local_free_energy;
        }

        float3 d[3][3];
        for(int no1: range(3)) 
            for(int no2: range(3)) 
                d[no1][no2] = pair_distrib[no1*3+no2]*e.deriv[no1][no2];

        for(int no1: range(3)) update_vec(deriv3.shifted(4*no1),e.nr1,  d[no1][0]+d[no1][1]+d[no1][2]);
        for(int no2: range(3)) update_vec(deriv3.shifted(4*no2),e.nr2, -d[0][no2]-d[1][no2]-d[2][no2]);
    }

    if(potential) potential[ns] = free_energy;
}



struct RotamerSidechain: public PotentialNode {
    struct RotamerIndices {
        int start;
        int stop;
    };

    int n_restype;
    CoordNode& rama;
    CoordNode& alignment;
    vector<SidechainInteractionParams> interactions;
    map<string,int> index_from_restype;

    vector<RotamerIndices> rotamer_indices;  // start and stop

    vector<string> sequence;
    vector<int>    restype;

    unique_ptr<RotamerPlacement<1>> placement1;
    unique_ptr<RotamerPlacement<3>> placement3;

    int             max_edges11, max_edges13, max_edges33;
    vector<int>     n_edge11, n_edge13, n_edge33;
    vector<PairInteraction<1,1>> edges11;
    vector<PairInteraction<1,3>> edges13;
    vector<PairInteraction<3,3>> edges33;

    vector<int> edge_indices;
    SysArrayStorage node_prob, edge_prob;
    SysArrayStorage node_belief, edge_belief, temp_node_belief, temp_edge_belief;

    float damping;
    int   max_iter;
    float tol;

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
        tol     (read_attribute<float>(grp, ".", "tol"))

    {
        check_size(grp, "energy", n_restype, n_restype, 2);  // 2 is for inner or outer energy
        check_size(grp, "radius", n_restype, n_restype, 2); 
        check_size(grp, "width",  n_restype, n_restype, 2);

        traverse_dset<3,float>(grp, "energy", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.energy_outer : p.energy_inner) = x;});

        traverse_dset<3,float>(grp, "radius", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.radius_outer : p.radius_inner) = x;});

        traverse_dset<3,float>(grp, "width", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.scale_outer  : p.scale_inner)  = 1.f/x;});

        for(auto& p: interactions) 
            p.update_cutoff2();

        for(int rt1: range(n_restype))
            for(int rt2: range(n_restype))
                if(interactions[rt1*n_restype + rt2] != interactions[rt2*n_restype + rt1]) 
                    throw string("interaction matrix must be symmetric");

        traverse_string_dset<1>(grp, "restype_order", [&](size_t idx, string nm) {index_from_restype[nm] = idx;});

        check_size(grp, "rotamer_start_stop", n_restype, 2);
        traverse_dset<2,int>(grp, "rotamer_start_stop", [&](size_t rt, size_t is_stop, int x) {
                (is_stop ? rotamer_indices[rt].stop : rotamer_indices[rt].start) = x;});

        traverse_string_dset<1>(grp, "restype", [&](size_t nr, string resname) {
                sequence.push_back(resname);
                restype .push_back(index_from_restype[resname]);
                });

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

        check_size(grp, "rotamer_center", n_bin, n_bin, n_total_rot, 3);
        check_size(grp, "rotamer_prob",   n_bin, n_bin, n_total_rot);

        vector<double> all_data_to_fit(n_total_rot*n_bin*n_bin*4);
        traverse_dset<4,double>(grp, "rotamer_center", [&](size_t ix, size_t iy, size_t i_pt, size_t d, double x) {
                all_data_to_fit.at(((i_pt*n_bin + ix)*n_bin + iy)*4 + d) = x;});
        traverse_dset<3,double>(grp, "rotamer_prob",   [&](size_t ix, size_t iy, size_t i_pt,           double x) {
                all_data_to_fit.at(((i_pt*n_bin + ix)*n_bin + iy)*4 + 3) = -log(x);}); // last entry is potential, not prob

        // Copy the data into local arrays
        map<int, vector<double>> data_to_fit;
        for(auto &kv: local_to_global) {
            int n_rot = kv.first;
            data_to_fit[n_rot] = vector<double>(kv.second.size()*n_bin*n_bin*n_rot*4);
            auto &v = data_to_fit[n_rot];

            for(int local_rt: range(kv.second.size())) {
                int rt = kv.second[local_rt];
                int start = rotamer_indices[rt].start;
                int stop  = rotamer_indices[rt].stop;

                for(int no: range(stop-start))
                    for(int ix: range(n_bin))
                        for(int iy: range(n_bin))
                            for(int d: range(4))
                                v[(((local_rt*n_bin + ix)*n_bin + iy)*n_rot + no)*4 + d] = 
                                    all_data_to_fit[(((start+no)*n_bin + ix)*n_bin + iy)*4 + d];
            }
        }

        if(int(sequence.size()) != alignment.n_elem || int(sequence.size()) != rama.n_elem) 
            throw string("Excluded residues not allowed for Rama potential");

        if(alignment.n_system != rama.n_system) throw string("Internal error");

        map<int, vector<ResidueLoc>> local_loc;
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
        placement1 = unique_ptr<RotamerPlacement<1>>(new RotamerPlacement<1>(
                    local_loc[1], global_restype[1], local_to_global[1].size(), n_bin, n_bin, data_to_fit[1].data(), n_system));
        placement3 = unique_ptr<RotamerPlacement<3>>(new RotamerPlacement<3>(
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
    }


    virtual void compute_value(ComputeMode mode) {
        Timer timer("rotamer");
        auto &p1 = *placement1;
        auto &p3 = *placement3;

#pragma omp parallel for schedule(dynamic)
        for(int ns=0; ns<n_system; ++ns) {
            p1.place_rotamers(alignment.coords().value, rama.coords().value, ns);
            p3.place_rotamers(alignment.coords().value, rama.coords().value, ns);

            compute_graph_elements<1,1>(n_edge11.data(), max_edges11, edges11.data(), 
                    p1.n_res, p1.global_restype.data(), p1.pos.array(),
                    p1.n_res, p1.global_restype.data(), p1.pos.array(),
                    n_restype, interactions.data(), ns, true);
            compute_graph_elements<1,3>(n_edge13.data(), max_edges13, edges13.data(), 
                    p1.n_res, p1.global_restype.data(), p1.pos.array(),
                    p3.n_res, p3.global_restype.data(), p3.pos.array(),
                    n_restype, interactions.data(), ns, false);
            compute_graph_elements<3,3>(n_edge33.data(), max_edges33, edges33.data(), 
                    p3.n_res, p3.global_restype.data(), p3.pos.array(),
                    p3.n_res, p3.global_restype.data(), p3.pos.array(),
                    n_restype, interactions.data(), ns, true);

            convert_potential_graph_to_probability_graph(
                    node_prob.array(), edge_prob.array(), edge_indices.data(),
                    p3.n_res, p3.pos.array(),  // last dimension is 1-residue potential
                    n_edge13.data(), max_edges13, edges13.data(),
                    n_edge33.data(), max_edges33, edges33.data(),
                    ns);

            auto result = solve_for_beliefs<3>(
                    node_belief[ns], edge_belief[ns], 
                    temp_node_belief[ns], temp_edge_belief[ns],
                    p3.n_res, node_prob[ns],
                    n_edge33[ns], edge_prob[ns], edge_indices.data() + ns*2*max_edges33,
                    damping, max_iter, tol, true); // do re-initialize beliefs

            if(result.first >= max_iter-1) 
                printf("%2i solved in %i iterations with error of %f\n", ns, result.first, result.second);

            propagate_derivatives(
                    (mode==PotentialAndDerivMode ? potential.data() : nullptr),
                    p3.pos.array(),
                    p1.pos_deriv.array(), p3.pos_deriv.array(),
                    p1.n_res,             p3.n_res,
                    node_belief.array(), edge_belief.array(),
                    edge_prob.array(),
                    n_edge11.data(), max_edges11, edges11.data(),
                    n_edge13.data(), max_edges13, edges13.data(),
                    n_edge33.data(), max_edges33, edges33.data(),
                    ns);

            p1.push_derivatives(alignment.coords().value,alignment.coords().deriv, rama.coords().deriv, ns);
            p3.push_derivatives(alignment.coords().value,alignment.coords().deriv, rama.coords().deriv, ns);
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
static RegisterNodeType<RotamerSidechain,2> rotamer_node ("rotamer");
