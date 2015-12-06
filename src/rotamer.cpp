#include "bead_interaction.h"
#include "interaction_graph.h"
#include <vector>
#include <string>
#include "h5_support.h"
#include <algorithm>
#include "deriv_engine.h"
#include "timing.h"
#include <memory>
#include "state_logger.h"
#include <tuple>
#include <set>

using namespace std;
using namespace h5;

constexpr static int UPPER_ROT = 4;  // 1 more than the most possible rotamers (handle 0)

struct EdgeLocator {
    protected:
        int data_size;   // 2*max_partners, must be divisible by 8
        unique_ptr<int32_t[]> locs;

        void resize(int new_data_size) {
            auto new_locs = unique_ptr<int32_t[]>(new_aligned<int32_t>(n_elem1*new_data_size,4));

            int copy_size = min(data_size, new_data_size);
            for(int ne=0; ne<n_elem1; ++ne) {
                for(int i=0; i<new_data_size; ++i)
                    new_locs  [ne*new_data_size + i] = i<copy_size 
                        ? locs[ne*    data_size + i]
                        : -1;  // -1 indicates no partner
            }

            data_size = new_data_size;
            locs = std::move(new_locs);
        }

    public:
        int32_t n_edge;
        int     n_elem1;

        EdgeLocator(int n_elem1_):
            data_size(0),
            locs(),
            n_elem1(n_elem1_)
        {
            resize(40);
            clear();
        }

        void clear() {
            fill_n(locs.get(), n_elem1*data_size, -1u);
            n_edge = 0u;
        }

        bool find_or_insert(int32_t &result, int32_t i1, int32_t i2) {
            // return value is true if the result was an insert
            int* partner_array = locs + int(i1*data_size);
            auto constant_i2   = Int4(i2);
            auto constant_umax = Int4(-1);

            for(int j=0; j<data_size; j+=8) {
                // first 4 entries are i2 values and second 4 entries are hash locations
                auto candidate_i2 = Int4(partner_array+j);
                auto hits = (constant_i2==candidate_i2);
                if(hits.any()) {
                    // There can be at most one hit
                    auto hit_loc = Int4(partner_array+j+4) & hits;
                    // we have a vector where the only nonzero entry is the hit, so the sum will move it to the 
                    // front
                    result = hit_loc.sum_in_all_entries().x();
                    return false;
                }

                int end_of_array = (constant_umax == candidate_i2).movemask();
                if(end_of_array) {
                    // first entry is number of 3 minus 
                    const int offset = 4-popcnt_nibble(end_of_array);
                    partner_array[j  +offset] = i2;
                    partner_array[j+4+offset] = n_edge++;
                    result = partner_array[j+4+offset];
                    return true;
                }
            }
            // if we reach the end, we need to grow the table to accommodate the entry
            // then we might as well just search again (even though we know where it will be
            resize(2*data_size);
            return find_or_insert(result, i1,i2);
        }
};

    

struct NodeHolder {
        const int n_rot;
        const int n_elem;

        VecArrayStorage prob;

        VecArrayStorage cur_belief;
        VecArrayStorage old_belief;

        NodeHolder(int n_rot_, int n_elem_):
            n_rot(n_rot_),
            n_elem(n_elem_),
            prob      (ru(n_rot),n_elem),
            cur_belief(ru(n_rot),n_elem),
            old_belief(ru(n_rot),n_elem) 
        {
            fill(cur_belief, 1.f);
            fill(old_belief, 1.f);
            reset();
        }

        void reset() {
            for(int i=0; i<n_elem; ++i)
                for(int j=0; j<ru(n_rot); ++j)
                    prob(j,i) = float(j<n_rot);
        }
        void swap_beliefs() { swap(cur_belief, old_belief); }

        void standardize_probs() {
            for(int ne: range(n_elem)) {
                float max_prob = 1e-10f;
                for(int no: range(n_rot)) if(prob(no,ne)>max_prob) max_prob = prob(no,ne);
                float inv_max_prob = rcp(max_prob);
                for(int no: range(n_rot)) prob(no,ne) *= inv_max_prob;
            }
        }

        template <int N_ROT>
        void standardize_belief_update(float damping) {
            for(int ne: range(n_elem)) {
                auto b = load_vec<N_ROT>(cur_belief, ne);
                b = (1.f-damping)*rcp(max(b))*b + damping*load_vec<N_ROT>(old_belief, ne);
                store_vec(cur_belief, ne, b);
            }
        }

        float max_deviation() {
            float dev = 0.f;
            for(int d: range(n_rot)) 
                for(int nn: range(n_elem)) 
                    dev = max(cur_belief(d,nn)-old_belief(d,nn), dev);
            return dev;
        }

        template<int N_ROT>
        void calculate_marginals() {
            // marginals are stored in the same array in cur_belief but l1 normalized
            for(int nn: range(n_elem)) {
                auto b = load_vec<N_ROT>(cur_belief,nn);
                store_vec(cur_belief,nn, b*rcp(sum(b)));
            }
        }

        template <int N_ROT>
        float node_free_energy(int nn) {
            auto b = load_vec<N_ROT>(cur_belief,nn);
            b *= rcp(sum(b));
            auto pr = load_vec<N_ROT>(prob,nn);
            
            float en = 0.f;
            // free energy is average energy - entropy
            for(int no: range(N_ROT)) en += b[no] * logf((1e-10f+b[no])*rcp(1e-10f+pr[no]));
            return en;
        }
};

constexpr static int simd_width = 4;

// struct SimdVecArrayStorage {
//     // AoSoA structure for holding partially transposed data
//     // makes it very easy to read Vec<n_dim,Float4> data
// 
//     int elem_width, n_elem;
//     unique_ptr<float[]> x;
// 
//     SimdVecArrayStorage(int elem_width_, int n_elem_min_):
//         elem_width(elem_width_), n_elem(round_up(n_elem_min_,simd_width)), 
//         x(new_aligned<float>(n_elem*elem_width, simd_width)) {}
// 
//     float& operator()(int i_comp, int i_elem) {
//         return x[(i_elem-i_elem%simd_width)*elem_width + i_comp*simd_width + i_elem%simd_width];
//     }
//     const float& operator()(int i_comp, int i_elem) const {
//         return x[(i_elem-i_elem%simd_width)*elem_width + i_comp*simd_width + i_elem%simd_width];
//     }
// };
// 
// template <int D>
// inline Vec<D,Float4> load_whole_vec(const SimdVecArrayStorage& a, int idx) {
//     // index must be divisible by 4
//     // assert(a.elem_width == D)
//     Vec<D,Float4> r;
//     #pragma unroll
//     for(int d=0; d<D; ++d) r[d] = Float4(a.x + idx*D + d*4);
//     return r;
// }
// 
// template <int D>
// inline void store_whole_vec(SimdVecArrayStorage& a, int idx, const Vec<D,Float4>& r) {
//     // index must be divisible by 4
//     // assert(a.elem_width == D)
//     #pragma unroll
//     for(int d=0; d<D; ++d) r[d].store(a.x + idx*D + d*4);
// }
// 
// // FIXME remove these inefficient helper functions
// template <int D>
// inline Vec<D,float> load_vec(const SimdVecArrayStorage& a, int idx) {
//     Vec<D,float> r;
//     #pragma unroll
//     for(int d=0; d<D; ++d) r[d] = a(d,idx);
//     return r;
// }
// 
// template <int D>
// inline void store_vec(SimdVecArrayStorage& a, int idx, const Vec<D,float>& r) {
//     #pragma unroll
//     for(int d=0; d<D; ++d) a(d,idx) = r[d];
// }
// 
// static void fill(SimdVecArrayStorage& v, float fill_value) {
//     std::fill_n(v.x.get(), v.n_elem*v.elem_width, fill_value);
// }
// 
// template<int D>
// void node_update_scatter(float* data, const Int4& offsets, Vec<D,Float4>& v) {
//     return v.NOT_IMPLEMENTED_IF_D_IS_NOT_3;
// }
// 
// template<>
// void node_update_scatter<3>(float* data, const Int4& offsets, Vec<3,Float4>& v) {
//     // note that this function changes the vector v
//     constexpr int D=3;
// 
//     float* p0 = data+offsets.x();
//     float* p1 = data+offsets.y();
//     float* p2 = data+offsets.z();
//     float* p3 = data+offsets.w();
// 
//     Float4 e[3]; // scratch space to do the transpose
// 
//     #pragma unroll
//     for(int d=0; d<D; d+=4)
//         transpose4(
//                 v[d  ],
//                 (d+1<D ? v[d+1] : e[0]),
//                 (d+2<D ? v[d+2] : e[1]),
//                 (d+3<D ? v[d+3] : e[2]));
// 
// 
//     // this writes must be done sequentially in case some of the 
//     // offsets are equal (and hence point to the same memory location)
//     v[0] *= Float4(p0); (v[0] * rcp(v[0].sum_in_all_entries())).store(p0);
//     v[1] *= Float4(p1); (v[1] * rcp(v[1].sum_in_all_entries())).store(p1);
//     v[2] *= Float4(p2); (v[2] * rcp(v[2].sum_in_all_entries())).store(p2);
//     e[2] *= Float4(p3); (e[2] * rcp(e[2].sum_in_all_entries())).store(p3);
// }

struct EdgeHolder {
    public:
        int n_rot1, n_rot2;  // should have rot1 < rot2
        NodeHolder &nodes1;
        NodeHolder &nodes2;

    public:
        struct EdgeLoc {int edge_num, dim, ne;};

        // FIXME include numerical stability data (basically scale each probability in a sane way)
        VecArrayStorage prob;
        VecArrayStorage cur_belief;
        VecArrayStorage old_belief;
        VecArrayStorage marginal;

        unique_ptr<int[]> edge_indices1;
        unique_ptr<int[]> edge_indices2;
        // unordered_map<unsigned,unsigned> nodes_to_edge;
        EdgeLocator nodes_to_edge;
        vector<EdgeLoc> edge_loc;

        EdgeHolder(NodeHolder &nodes1_, NodeHolder &nodes2_, int max_n_edge):
            n_rot1(nodes1_.n_rot), n_rot2(nodes2_.n_rot),
            nodes1(nodes1_), nodes2(nodes2_),
            prob      (n_rot1*ru(n_rot2), max_n_edge),
            cur_belief(ru(n_rot1)+ru(n_rot2), max_n_edge),
            old_belief(ru(n_rot1)+ru(n_rot2), max_n_edge),
            marginal(n_rot1*n_rot2, max_n_edge+1), // the +1 ensures we can write past the end

            edge_indices1(new_aligned<int>(max_n_edge,simd_width)),
            edge_indices2(new_aligned<int>(max_n_edge,simd_width)),

            nodes_to_edge(nodes1.n_elem)
        {

            edge_loc.reserve(n_rot1*n_rot2*max_n_edge);
            fill(cur_belief, 0.f);
            fill(old_belief, 0.f);
            fill_n(edge_indices1, round_up(max_n_edge,simd_width), 0);
            fill_n(edge_indices2, round_up(max_n_edge,simd_width), 0);
            nodes_to_edge.n_edge = max_n_edge;
            reset();
        }

        void reset() {
            // reset the probabilities we wrote over
            for(int idx=0; idx<nodes_to_edge.n_edge; ++idx)
                for(int i: range(n_rot1)) 
                    for(int j: range(n_rot2)) 
                        prob(i*ru(n_rot2)+j,idx) = 1.f;

            nodes_to_edge.clear();
            edge_loc.clear();
        }
        void swap_beliefs() { swap(cur_belief, old_belief); }

        void add_to_edge(
                int ne, float prob_val,
                unsigned id1, unsigned rot1, 
                unsigned id2, unsigned rot2) {
            int32_t idx;
            if(nodes_to_edge.find_or_insert(idx,id1,id2)){
                edge_indices1[idx] = id1;
                edge_indices2[idx] = id2;
            }

            int j = rot1*ru(n_rot2)+rot2;
            prob(j, idx) *= prob_val;
            edge_loc.emplace_back(EdgeLoc{ne, int(rot1*n_rot2+rot2), int(idx)});
        }

        void move_edge_prob_to_node2() {
            // FIXME assert n_rot1 == 1
            VecArray p = nodes2.prob;
            for(int ne: range(nodes_to_edge.n_edge)) {
                int nr = edge_indices2[ne];
                for(int no: range(n_rot2)) p(no,nr) *= prob(no,ne);
            }
        }

        void standardize_probs() { // FIXME should accumulate the rescaled probability
            for(int ne: range(nodes_to_edge.n_edge)) {
                float max_prob = 1e-10f;
                for(int nd: range(n_rot1*n_rot2)) if(prob(nd,ne)>max_prob) max_prob = prob(nd,ne);
                float inv_max_prob = rcp(max_prob);
                for(int nd: range(n_rot1*n_rot2)) prob(nd,ne) *= inv_max_prob;
            }
        }

        // float max_deviation() {
        //     FIX for padding;
        //     float dev = 0.f;
        //     for(int d: range(n_rot1+n_rot2)) 
        //         for(int nn: range(nodes_to_edge.n_edge)) 
        //             dev = max(cur_belief(d,nn)-old_belief(d,nn), dev);
        //     return dev;
        // }

        template<int N_ROT1, int N_ROT2>
        void calculate_marginals() {
            // FIXME ASSERT(n_rot1 == N_ROT1)
            // FIXME ASSERT(n_rot2 == N_ROT2)  // kind of clunky but should improve performance by loop unrolling

            for(int ne: range(nodes_to_edge.n_edge)) {
                auto b1 = load_vec<N_ROT1>(nodes1.cur_belief, edge_indices1[ne]);
                auto b2 = load_vec<N_ROT2>(nodes2.cur_belief, edge_indices2[ne]);

                // correct for self interaction
                auto b = load_vec<ru(N_ROT1)+ru(N_ROT2)>(cur_belief, ne);
                auto bc1 = b1 * vec_rcp(1e-10f + extract<0,         N_ROT1>           (b));
                auto bc2 = b2 * vec_rcp(1e-10f + extract<ru(N_ROT1),ru(N_ROT1)+N_ROT2>(b));

                auto p = load_vec<N_ROT1*ru(N_ROT2)>(prob, ne);
                Vec<N_ROT1*N_ROT2> marg;

                for(int no1: range(N_ROT1))
                    for(int no2: range(N_ROT2))
                        marg[no1*N_ROT2+no2] = p[no1*ru(N_ROT2)+no2]*bc1[no1]*bc2[no2];
                marg *= rcp(sum(marg));

                store_vec(marginal, ne, marg);
            }
        }

        template<int N_ROT1, int N_ROT2>
        float edge_free_energy(int ne) {
            auto b1 = load_vec<N_ROT1>(nodes1.cur_belief, edge_indices1[ne]);  // really marginal
            auto b2 = load_vec<N_ROT2>(nodes2.cur_belief, edge_indices2[ne]);

            auto p  = load_vec<N_ROT1*   N_ROT2 >(marginal, ne);
            auto pr = load_vec<N_ROT1*ru(N_ROT2)>(prob, ne);

            float en = 0.f;
            for(int no1: range(N_ROT1)) {
                for(int no2: range(N_ROT2)) {
                    int i = no1*N_ROT2 + no2;
                    // this is the average potential energy plus the mutual information,
                    // which is what I want for calculating the overall energy
                    // the 1e-10f prevents NaN if some probabilities are exactly 0
                    en += p[i] * logf((1e-10f+p[i]) * rcp(1e-10f+pr[no1*ru(N_ROT2)+no2]*b1[no1]*b2[no2]));
                }
            }

            return en;
        }

        void update_beliefs33() {
            // horizontal SIMD implementation of update_beliefs for N_ROT1==N_ROT2==3

            float* vec_old_node_belief1 = nodes1.old_belief.x.get();
            float* vec_cur_node_belief1 = nodes1.cur_belief.x.get();

            float* vec_old_node_belief2 = nodes2.old_belief.x.get();
            float* vec_cur_node_belief2 = nodes2.cur_belief.x.get();

            int n_edge = nodes_to_edge.n_edge;

            for(int ne=0; ne<n_edge; ++ne) {
                int i1 = edge_indices1[ne]*4;
                int i2 = edge_indices2[ne]*4;

                auto old_edge_belief1 = Float4(old_belief.x + ne*4*2 + 0);
                auto old_edge_belief2 = Float4(old_belief.x + ne*4*2 + 4);

                auto old_node_belief1 = Float4(vec_old_node_belief1 + i1);
                auto old_node_belief2 = Float4(vec_old_node_belief2 + i2);

                auto v1 = old_node_belief1 * rcp(Float4(1e-10f) + old_edge_belief1);
                auto v2 = old_node_belief2 * rcp(Float4(1e-10f) + old_edge_belief2);

                // load the edge probability matrix
                auto ep_row1 = Float4(prob.x + ne*4*3 + 0);
                auto ep_row2 = Float4(prob.x + ne*4*3 + 4);
                auto ep_row3 = Float4(prob.x + ne*4*3 + 8);

                // cur_edge_belief1 =  left_multiply_matrix(ep, v2));
                auto cur_edge_belief1 =  left_multiply_3x3(    ep_row1,ep_row2,ep_row3, v2);
                auto cur_edge_belief2 = right_multiply_3x3(v1, ep_row1,ep_row2,ep_row3);

                auto cur_node_belief1 = cur_edge_belief1 * Float4(vec_cur_node_belief1 + i1);
                auto cur_node_belief2 = cur_edge_belief2 * Float4(vec_cur_node_belief2 + i2);

                // // let's approximately l1 normalize everything to avoid any numerical problems later
                // auto scales_for_unit_l1 = Float4(3.f)*approx_rcp(horizontal_add(
                //         horizontal_add(cur_edge_belief1, cur_edge_belief2),
                //         horizontal_add(cur_node_belief1, cur_node_belief2)));
                // 
                // cur_edge_belief1 *= scales_for_unit_l1.broadcast<0>();
                // cur_edge_belief2 *= scales_for_unit_l1.broadcast<1>();
                // cur_node_belief1 *= scales_for_unit_l1.broadcast<2>();
                // cur_node_belief2 *= scales_for_unit_l1.broadcast<3>();

                // I might be able to do the normalization only every few steps if I wanted
                cur_edge_belief1.store(cur_belief.x + ne*4*2 + 0);
                cur_edge_belief2.store(cur_belief.x + ne*4*2 + 4);
                cur_node_belief1.store(vec_cur_node_belief1 + i1);
                cur_node_belief2.store(vec_cur_node_belief2 + i2);
            }
        }

        // template <int N_ROT1, int N_ROT2>
        // void update_beliefs(float damping) {
        //     Timer timer(std::string("update_beliefs"));
        //     // FIXME ASSERT(n_rot1 == N_ROT1)
        //     // FIXME ASSERT(n_rot2 == N_ROT2)  // kind of clunky but should improve performance by loop unrolling
        //     VecArray vec_old_node_belief1 = nodes1.old_belief;
        //     VecArray vec_cur_node_belief1 = nodes1.cur_belief;

        //     VecArray vec_old_node_belief2 = nodes2.old_belief;
        //     VecArray vec_cur_node_belief2 = nodes2.cur_belief;

        //     // note that edges in [n_edge,round_up(n_edge,4)) must be set to point to
        //     //   the dummy node
        //     for(int ne=0; ne<n_edge; ne+=4) {
        //         auto offset1 = Int4(edge_indices1+ne) * Int4(round_up(N_ROT1,4));
        //         auto offset2 = Int4(edge_indices2+ne) * Int4(round_up(N_ROT2,4));

        //         auto old_node_belief1 = aligned_gather_vec<N_ROT1>(vec_old_node_belief1.x, offset1);
        //         auto old_node_belief2 = aligned_gather_vec<N_ROT2>(vec_old_node_belief2.x, offset2);

        //         auto ep = load_whole_vec<N_ROT1*N_ROT2>(prob, ne);

        //         auto b = load_whole_vec<N_ROT1+N_ROT2>(old_belief, ne);
        //         auto old_edge_belief1 = extract<0,     N_ROT1>       (b);
        //         auto old_edge_belief2 = extract<N_ROT1,N_ROT1+N_ROT2>(b);

        //         auto cur_edge_belief1 =  left_multiply_matrix(ep, old_node_belief2 * vec_rcp(old_edge_belief2));
        //         auto cur_edge_belief2 = right_multiply_matrix(    old_node_belief1 * vec_rcp(old_edge_belief1), ep);
        //         cur_edge_belief1 *= rcp(max(cur_edge_belief1)); // rescale to avoid underflow in the future
        //         cur_edge_belief2 *= rcp(max(cur_edge_belief2));

        //         // store edge beliefs
        //         Vec<N_ROT1+N_ROT2,Float4> neb;
        //         store<0,     N_ROT1>       (neb, cur_edge_belief1);
        //         store<N_ROT1,N_ROT1+N_ROT2>(neb, cur_edge_belief2);
        //         store_whole_vec(cur_belief,ne,neb);

        //         // Update our beliefs about nodes (normalization keeps us near 1)
        //         node_update_scatter(vec_cur_node_belief1.x, offset1, cur_edge_belief1);
        //         node_update_scatter(vec_cur_node_belief2.x, offset2, cur_edge_belief2);
        //     }
        // }
};

template<>
void EdgeHolder::calculate_marginals<3,3>() {
    // FIXME ASSERT(n_rot1 == N_ROT1)
    // FIXME ASSERT(n_rot2 == N_ROT2)  // kind of clunky but should improve performance by loop unrolling

    int n_edge = nodes_to_edge.n_edge;
    for(int ne=0; ne<n_edge; ++ne) {
        auto b1 = Float4(nodes1.cur_belief.x + 4*edge_indices1[ne]);
        auto b2 = Float4(nodes2.cur_belief.x + 4*edge_indices2[ne]);

        // correct for self interaction
        auto bc1 = b1 * rcp(Float4(1e-10f) + Float4(cur_belief.x + 8*ne));
        auto bc2 = b2 * rcp(Float4(1e-10f) + Float4(cur_belief.x + 8*ne + 4));

        auto ep_row1 = Float4(prob.x + ne*4*3 + 0);
        auto ep_row2 = Float4(prob.x + ne*4*3 + 4);
        auto ep_row3 = Float4(prob.x + ne*4*3 + 8);

        // we want the element-wise product of ep and outer(bc1,bc2)
        auto marg_row1 = bc1.broadcast<0>() * (ep_row1 * bc2);
        auto marg_row2 = bc1.broadcast<1>() * (ep_row2 * bc2);
        auto marg_row3 = bc1.broadcast<2>() * (ep_row3 * bc2);
        
        // normalize marginal so that total probability is 1
        auto marg_scale = rcp((marg_row1 + marg_row2 + marg_row3).sum_in_all_entries());
        marg_row1 *= marg_scale;
        marg_row2 *= marg_scale;
        marg_row3 *= marg_scale;

        marg_row1.store(&marginal(0,ne), Alignment::  aligned);
        marg_row2.store(&marginal(3,ne), Alignment::unaligned);
        marg_row3.store(&marginal(6,ne), Alignment::unaligned);
    }
}


template <typename BT>
array<int,UPPER_ROT> calculate_n_elem(InteractionGraph<BT> &igraph) {
    array<int,UPPER_ROT> result; // 0-rot is included
    for(int& i: result) i=0;

    unordered_map<unsigned,set<unsigned>> unique_ids;
    for(int ne: range(igraph.n_elem1)) {
        unsigned id = igraph.id1[ne];
        unsigned selector = (1u<<n_bit_rotamer) - 1u;
        unsigned   rot = id & selector; id >>= n_bit_rotamer;
        unsigned n_rot = id & selector; id >>= n_bit_rotamer;
        if(rot>=n_rot) throw string("invalid rotamer number");
        unique_ids[n_rot].insert(id);
    }
    for(auto& kv: unique_ids) {
        if(kv.first >= UPPER_ROT) throw string("invalid rotamer count ")+to_string(kv.first);
        result.at(kv.first) = kv.second.size();
    }
    return result;
}


template <typename BT>
struct RotamerSidechain: public PotentialNode {
    vector<CoordNode*> prob_nodes;
    int n_prob_nodes;
    InteractionGraph<BT> igraph;
    array<int,UPPER_ROT> n_elem_rot;

    NodeHolder* node_holders_matrix[UPPER_ROT];
    NodeHolder  nodes1, nodes3; // initialize these with sane max_n_edge

    EdgeHolder* edge_holders_matrix[UPPER_ROT][UPPER_ROT];
    EdgeHolder edges11, edges13, edges33; // initialize these with sane max_n_edge

    float damping;
    int   max_iter;
    float tol;
    int   iteration_chunk_size;

    bool energy_fresh_relative_to_derivative;

    RotamerSidechain(hid_t grp, CoordNode &pos_node_, vector<CoordNode*> prob_nodes_):
        PotentialNode(),
        prob_nodes(prob_nodes_),
        n_prob_nodes(prob_nodes.size()),
        igraph(open_group(grp,"pair_interaction").get(), &pos_node_),
        n_elem_rot(calculate_n_elem(igraph)),

        nodes1(1,n_elem_rot[1]),
        nodes3(3,n_elem_rot[3]),

        edges11(nodes1,nodes1,n_elem_rot[1]*(n_elem_rot[1]+1)/2),
        edges13(nodes1,nodes3,n_elem_rot[1]* n_elem_rot[3]),
        edges33(nodes3,nodes3,n_elem_rot[3]*(n_elem_rot[3]+1)/2),

        damping (read_attribute<float>(grp, ".", "damping")),
        max_iter(read_attribute<int  >(grp, ".", "max_iter")),
        tol     (read_attribute<float>(grp, ".", "tol")),
        iteration_chunk_size(read_attribute<int>(grp, ".", "iteration_chunk_size")),

        energy_fresh_relative_to_derivative(false)
    {
        for(int i: range(UPPER_ROT)) node_holders_matrix[i] = nullptr;
        node_holders_matrix[1] = &nodes1;
        node_holders_matrix[3] = &nodes3;

        for(int i: range(UPPER_ROT)) for(int j: range(UPPER_ROT)) edge_holders_matrix[i][j] = nullptr;
        edge_holders_matrix[1][1] = &edges11;
        edge_holders_matrix[1][3] = &edges13;
        edge_holders_matrix[3][3] = &edges33;

        for(int i: range(prob_nodes.size())) 
            if(igraph.pos_node1->n_elem != prob_nodes[i]->n_elem)
                throw string("rotamer positions have " + to_string(igraph.pos_node1->n_elem) +
                        " elements but the " + to_string(i) + "-th (0-indexed) probability node has only " +
                        to_string(prob_nodes[i]->n_elem) + " elements.");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("rotamer_free_energy", {nodes1.n_elem+nodes3.n_elem}, 
                    [&](float* buffer) {
                       auto en = residue_free_energies();
                       copy(begin(en), end(en), buffer);});

            for(int npn: range(n_prob_nodes))
                default_logger->add_logger<float>(("rotamer_1body_energy" + to_string(npn)).c_str(),
                        {nodes1.n_elem+nodes3.n_elem}, [npn,this](float* buffer) {
                            auto en = this->rotamer_1body_energy(npn);
                            copy(begin(en), end(en), buffer);});
        }
    }


    void ensure_fresh_energy() {
        if(!energy_fresh_relative_to_derivative) compute_value(PotentialAndDerivMode);
    }

    virtual void compute_value(ComputeMode mode) {
        energy_fresh_relative_to_derivative = mode==PotentialAndDerivMode;

        fill_holders();
        auto solve_results = solve_for_marginals();
        if(solve_results.first >= max_iter - iteration_chunk_size - 1)
            printf("solved in %i iterations with an error of %f\n", 
                    solve_results.first, solve_results.second);

        propagate_derivatives();
        if(mode==PotentialAndDerivMode) potential = calculate_energy_from_marginals();
    }

    virtual double test_value_deriv_agreement() {return -1.;}

    void fill_holders()
    {
        Timer timer(std::string("rotamer_fill"));
        edges11.reset();
        for(int n_rot1: range(UPPER_ROT))
            for(int n_rot2: range(UPPER_ROT))
                if(edge_holders_matrix[n_rot1][n_rot2])
                    edge_holders_matrix[n_rot1][n_rot2]->reset();

        for(int n_rot: range(UPPER_ROT))
            if(node_holders_matrix[n_rot])
                node_holders_matrix[n_rot]->reset();

        vector<VecArray> energy_1body;
        energy_1body.reserve(n_prob_nodes);
        for(int i: range(n_prob_nodes)) 
            energy_1body.emplace_back(prob_nodes[i]->output);

        for(int n: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;
            int index = igraph.loc1[n].index;

            float energy = 0.f;
            for(auto &a: energy_1body) energy += a(0,index);

            node_holders_matrix[n_rot]->prob(rot,id) *= expf(-energy);
        }

        // Fill edge probabilities
        igraph.compute_edges();

        const unsigned selector = (1u<<n_bit_rotamer) - 1u;
        for(int ne=0; ne<igraph.n_edge; ++ne) {
            int   id1  = igraph.edge_id1[ne];
            int   id2  = igraph.edge_id2[ne];
            float prob = expf(-igraph.edge_value[ne]);  // value of edge is potential

            if((id1&(selector<<n_bit_rotamer)) > (id2&(selector<<n_bit_rotamer))) swap(id1,id2);

            unsigned   rot1 = id1 & selector; id1 >>= n_bit_rotamer;
            unsigned   rot2 = id2 & selector; id2 >>= n_bit_rotamer;

            unsigned n_rot1 = id1 & selector; id1 >>= n_bit_rotamer;
            unsigned n_rot2 = id2 & selector; id2 >>= n_bit_rotamer;

            edge_holders_matrix[n_rot1][n_rot2]->add_to_edge(ne, prob, id1, rot1, id2, rot2);
        }

        // for edges with a 1, we can just move it
        for(int n_rot: range(2,UPPER_ROT))
            if(edge_holders_matrix[1][n_rot])
                edge_holders_matrix[1][n_rot]->move_edge_prob_to_node2();
    }

    float calculate_energy_from_marginals() {
        // marginals must already have been solved
        // since edges1x were folded into the node probabilites, they should not be accumulated here
        float en = 0.f;
        for(int nn: range(nodes1 .n_elem)) en += nodes1 .node_free_energy<1>  (nn);
        for(int nn: range(nodes3 .n_elem)) en += nodes3 .node_free_energy<3>  (nn);
        for(int ne: range(edges11.nodes_to_edge.n_edge)) en += -logf(edges11.prob(0,ne));
        for(int ne: range(edges33.nodes_to_edge.n_edge)) en += edges33.edge_free_energy<3,3>(ne);

//        for(int nn: range(nodes1 .n_elem)) printf("en1  %4i % .2f\n",nn, nodes1 .node_free_energy<1>  (nn));
//        for(int nn: range(nodes3 .n_elem)) printf("en3  %4i % .2f\n",nn, nodes3 .node_free_energy<3>  (nn));
//        for(int ne: range(edges11.nodes_to_edge.n_edge)) printf("en11 %4i % .2f\n",ne, -logf(edges11.prob(0,ne)));
//        for(int ne: range(edges33.nodes_to_edge.n_edge)) printf("en33 %4i % .2f\n",ne, edges33.edge_free_energy<3,3>(ne));
        return en;
    }

    vector<float> residue_free_energies() {
        vector<float> e1(nodes1.n_elem, 0.f);
        vector<float> e3(nodes3.n_elem, 0.f);

        for(int nn: range(nodes1 .n_elem)) {float en = nodes1.node_free_energy<1>(nn); e1[nn] += en;}
        for(int nn: range(nodes3 .n_elem)) {float en = nodes3.node_free_energy<3>(nn); e3[nn] += en;}

        for(int ne: range(edges11.nodes_to_edge.n_edge)) {
            float         en = -logf(edges11.prob(0,ne));
            e1[edges11.edge_indices1[ne]] += 0.5*en;
            e1[edges11.edge_indices2[ne]] += 0.5*en;
        }

        for(int ne: range(edges33.nodes_to_edge.n_edge)) {
            float         en = edges33.edge_free_energy<3,3>(ne);
            e3[edges33.edge_indices1[ne]] += 0.5*en;
            e3[edges33.edge_indices2[ne]] += 0.5*en;
        }

        return arrange_energies(e1,e3);
    }

    vector<float> rotamer_1body_energy(int prob_node_index) {
        vector<float> e1(nodes1.n_elem, 0.f);
        vector<float> e3(nodes3.n_elem, 0.f);

        VecArray energy_1body = prob_nodes[prob_node_index]->output;
        for(int n: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;
            int index = igraph.loc1[n].index;

            switch(n_rot) {
                case 1: e1[id] += nodes1.cur_belief(rot,id) * energy_1body(0,index); break;
                case 3: e3[id] += nodes3.cur_belief(rot,id) * energy_1body(0,index); break;
                default: throw string("impossible");
            }
        }

        return arrange_energies(e1,e3);
    }

    vector<float> arrange_energies(const vector<float>& e1, const vector<float>& e3) {
        vector<float> energies(n_elem_rot[1]+n_elem_rot[3]);
        auto en_loc = begin(energies);

        set<unsigned> known_ids;

        for(int ne: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[ne];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            if(id&selector) continue; // only count on the 0th rotamer
            if(known_ids.find(id)!=known_ids.end()) continue; //may be multiple beads
            known_ids.insert(id);
            id>>=n_bit_rotamer;
            unsigned n_rot = id & selector;
            id>>=n_bit_rotamer;  // now id contains the local alignment

            switch(n_rot) {
                case 1: *en_loc = e1[id]; ++en_loc; break;
                case 3: *en_loc = e3[id]; ++en_loc; break;
                default: throw string("impossible");
            }
        }

        if(en_loc != end(energies)) throw string("wrong number of residues");
        return energies;
    }

    void propagate_derivatives() {
        for(auto &el: edges11.edge_loc)
            igraph.edge_sensitivity[el.edge_num] = 1.f;
        for(auto &el: edges13.edge_loc)
            igraph.edge_sensitivity[el.edge_num] = nodes3 .cur_belief(el.dim, edges13.edge_indices2[el.ne]);
        for(auto &el: edges33.edge_loc)
            igraph.edge_sensitivity[el.edge_num] = edges33.marginal  (el.dim, el.ne);
        igraph.propagate_derivatives();

        vector<VecArray> sens_1body;
        sens_1body.reserve(n_prob_nodes);
        for(int i: range(n_prob_nodes)) 
            sens_1body.emplace_back(prob_nodes[i]->sens);

        for(int n: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;

            for(int i: range(n_prob_nodes))
                sens_1body[i](0,igraph.loc1[n].index) += node_holders_matrix[n_rot]->cur_belief(rot,id);
        }
    }


    void calculate_new_beliefs(float damping_for_this_iteration) {
        copy(nodes3.prob, nodes3.cur_belief);
        edges33.update_beliefs33();
        nodes3.standardize_belief_update<3>(damping_for_this_iteration);
    }
    

    pair<int,float> solve_for_marginals() {
        Timer timer(std::string("rotamer_solve"));
        // first initialize old node beliefs to just be probability to speed convergence
        for(auto nh: node_holders_matrix)
            if(nh)
                for(int no: range(nh->n_rot))
                    for(int ne: range(nh->n_elem))
                        nh->old_belief(no,ne) = nh->prob(no,ne);

        alignas(16) float start_belief[4] = {1.f,1.f,1.f,0.f};
        auto sb = Float4(start_belief);
        for(int ne=0; ne<edges33.nodes_to_edge.n_edge; ++ne) {
            sb.store(edges33.old_belief.x+ne*8+0);
            sb.store(edges33.old_belief.x+ne*8+4);
        }
        // fill(edges33.old_belief, 1.f);

        // this will fix consistent values in cur_belief for edges but put poor values in cur_belief for the nodes
        copy(nodes3.prob, nodes3.old_belief);
        edges33.update_beliefs33();  // put good values in the edge beliefs
        nodes3.swap_beliefs();  // we want the "old" values here
        nodes3.standardize_belief_update<3>(0.f); // ensure normalization here

        float max_deviation = 1e10f;
        int iter = 0;

        for(; max_deviation>tol && iter<max_iter; iter+=iteration_chunk_size) {
            for(int j=0; j<iteration_chunk_size; ++j) {
                nodes3 .swap_beliefs();
                edges33.swap_beliefs();
                calculate_new_beliefs(damping);
            }

            // compute max deviation
            max_deviation = nodes3.max_deviation(); // max(nodes3.max_deviation(), edges33.max_deviation());
        }

        nodes3 .calculate_marginals<3>  ();
        edges33.calculate_marginals<3,3>();
        return make_pair(iter, max_deviation);
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const {return igraph.get_param();}
    virtual std::vector<float> get_param_deriv() const {return igraph.get_param_deriv();}
    virtual void set_param(const std::vector<float>& new_param) {igraph.set_param(new_param);}
#endif
};

template <typename BT>
struct RegisterRotamerSidechain {
    RegisterRotamerSidechain(string name_prefix) {
        NodeCreationFunction f = [name_prefix](hid_t grp, const ArgList& args) {
            if(args.size()<1u) throw string("node " + name_prefix + " needs at least 1 arg");
            ArgList args_rest;
            for(int i: range(1,args.size())) args_rest.push_back(args[i]);
            return new RotamerSidechain<BT>(grp, *args[0], args_rest);};
        add_node_creation_function(name_prefix, f);
    }
};

static RegisterRotamerSidechain<preferred_bead_type> rotamer_node ("rotamer");
