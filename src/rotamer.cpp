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
#include "Float4.h"
#include <functional>

using namespace std;
using namespace h5;

constexpr static int UPPER_ROT = 7;  // 1 more than the most possible rotamers (handle 0)

template <int N_Float4>
Vec<N_Float4,Float4> read4vec(float* address, Alignment align=Alignment::aligned) {
    Vec<N_Float4,Float4> ret;
    #pragma unroll
    for(int i=0; i<N_Float4; ++i) ret[i] = Float4(address+4*i,align);
    return ret;
}

template <int N_Float4>
Vec<N_Float4,Float4> store4vec(float* address, const Vec<N_Float4,Float4>& a, Alignment align=Alignment::aligned) {
    #pragma unroll
    for(int i=0; i<N_Float4; ++i) a[i].store(address+4*i,align);
    return a;
}


template <int D1, int D2>
struct PaddedMatrix {
    static constexpr int w1=(D1+3)/4;
    static constexpr int w2=(D2+3)/4;

    // I am having some weird type errors that I don't understand related to
    //   things like v[0].broadcast<0>() that do not correctly interpret as a method call.
    // This function will fix that will fix them, but I do not know why it is necessary.
    // C++ can be humbling some times.  I am sure that my mental block is something silly.
    template<int i_bcast>
    static Float4 bcast(const Float4& v) {return v.broadcast<i_bcast>();}

    Float4 row[D1][w2];

    PaddedMatrix(float* address) {
        for(int nr=0; nr<D1; ++nr)
            for(int i2=0; i2<w2; ++i2)
                row[nr][i2] = Float4(address + nr*4*w2 + 4*i2);
    }

    Vec<w1,Float4> apply_left(const Vec<w2,Float4>& v) {
        Vec<w1,Float4> z;
        if(D1==3 && D2==3) {
            z[0] = dp<1,0,0,0, 1,1,1,0>(row[0][0],v[0]) |
                   dp<0,1,0,0, 1,1,1,0>(row[1][0],v[0]) | 
                   dp<0,0,1,0, 1,1,1,0>(row[2][0],v[0]);
        } else if(D1==3 && D2==6) {
            auto a = dp<1,0,0,0, 1,1,1,1>(row[0][0],v[0]) | 
                     dp<0,1,0,0, 1,1,1,1>(row[1][0],v[0]) | 
                     dp<0,0,1,0, 1,1,1,1>(row[2][0],v[0]);

            auto b = dp<1,0,0,0, 1,1,0,0>(row[0][1],v[1]) | 
                     dp<0,1,0,0, 1,1,0,0>(row[1][1],v[1]) | 
                     dp<0,0,1,0, 1,1,0,0>(row[2][1],v[1]);
            z[0] = a+b;
        } else if(D1==6 && D2==6) {
            auto a1 = dp<1,0,0,0, 1,1,1,1>(row[0][0],v[0]) | 
                      dp<0,1,0,0, 1,1,1,1>(row[1][0],v[0]) | 
                      dp<0,0,1,0, 1,1,1,1>(row[2][0],v[0]) |
                      dp<0,0,0,1, 1,1,1,1>(row[3][0],v[0]);
            auto a2 = dp<1,0,0,0, 1,1,1,1>(row[4][0],v[0]) | 
                      dp<0,1,0,0, 1,1,1,1>(row[5][0],v[0]);

            auto b1 = dp<1,0,0,0, 1,1,0,0>(row[0][1],v[1]) | 
                      dp<0,1,0,0, 1,1,0,0>(row[1][1],v[1]) | 
                      dp<0,0,1,0, 1,1,0,0>(row[2][1],v[1]) |
                      dp<0,0,0,1, 1,1,0,0>(row[3][1],v[1]);
            auto b2 = dp<1,0,0,0, 1,1,0,0>(row[4][1],v[1]) | 
                      dp<0,1,0,0, 1,1,0,0>(row[5][1],v[1]);

            z[0] = a1+b1;
            z[1] = a2+b2;
        }
        return z;
    }

    Vec<w2,Float4> apply_right(const Vec<w1,Float4>& v) {
        Vec<w2,Float4> z;
        if(D1==3 && D2==3) {
            auto tmp0 =       bcast<0>(v[0])*row[0][0];
            auto tmp1 = fmadd(bcast<1>(v[0]),row[1][0], tmp0);
            auto tmp2 = fmadd(bcast<2>(v[0]),row[2][0], tmp1);
            z[0] = tmp2;
        } else if(D1==3 && D2==6) {
            // input in length 3 and output is length 6
            auto tmp0a =       bcast<0>(v[0])*row[0][0];
            auto tmp1a = fmadd(bcast<1>(v[0]),row[1][0], tmp0a);
            auto tmp2a = fmadd(bcast<2>(v[0]),row[2][0], tmp1a);
            auto tmp0b =       bcast<0>(v[0])*row[0][1];
            auto tmp1b = fmadd(bcast<1>(v[0]),row[1][1], tmp0b);
            auto tmp2b = fmadd(bcast<2>(v[0]),row[2][1], tmp1b);
            z[0] = tmp2a;
            z[1] = tmp2b;
        } else if(D1==6 && D2==6) {
            // FIXME I could tree out the additions to expose more instruction parallelism
            auto tmp0a =       bcast<0>(v[0])*row[0][0];
            auto tmp1a = fmadd(bcast<1>(v[0]),row[1][0], tmp0a);
            auto tmp2a = fmadd(bcast<2>(v[0]),row[2][0], tmp1a);
            auto tmp3a = fmadd(bcast<3>(v[0]),row[3][0], tmp2a);
            auto tmp4a = fmadd(bcast<0>(v[1]),row[4][0], tmp3a);
            auto tmp5a = fmadd(bcast<1>(v[1]),row[5][0], tmp4a);

            auto tmp0b =       bcast<0>(v[0])*row[0][1];
            auto tmp1b = fmadd(bcast<1>(v[0]),row[1][1], tmp0b);
            auto tmp2b = fmadd(bcast<2>(v[0]),row[2][1], tmp1b);
            auto tmp3b = fmadd(bcast<3>(v[0]),row[3][1], tmp2b);
            auto tmp4b = fmadd(bcast<0>(v[1]),row[4][1], tmp3b);
            auto tmp5b = fmadd(bcast<1>(v[1]),row[5][1], tmp4b);

            z[0] = tmp5a;
            z[1] = tmp5b;
        }
        return z;
    }
};


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

    unique_ptr<float[]> energy_offset;

    NodeHolder(int n_rot_, int n_elem_):
        n_rot(n_rot_),
        n_elem(n_elem_),
        prob         (n_rot,n_elem),
        cur_belief   (n_rot,n_elem),
        old_belief   (n_rot,n_elem),

        energy_offset(new_aligned<float>(n_elem,4))
        {
            fill(cur_belief, 1.f);
            fill(old_belief, 1.f);
            reset();
        }

    void reset() { fill(prob, 0.f); } // prob array initially contains energy
    void swap_beliefs() { swap(cur_belief, old_belief); }

    void convert_energy_to_prob() {
        // prob array should initially contain energy
        // prob array is not normalized at the end (one of the entries will be 1.),
        //   but should be sanely scaled to resist underflow/overflow
        // It might be more effective to l1 normalize the probabilities at the end,
        //   but then I would need an extra logf to add to the offset if we are in energy mode

        for(int ne: range(n_elem)) {
            auto e_offset = prob(0,ne);
            for(int d=1; d<n_rot; ++d)
                e_offset = min(e_offset, prob(d,ne));

            for(int d=0; d<n_rot; ++d)
                prob(d,ne) = expf(e_offset-prob(d,ne));

            energy_offset[ne] = e_offset;
        }
    }

    template <int N_ROT>
        void standardize_belief_update(float damping) {
            if(damping != 0.f) {
                for(int ne: range(n_elem)) {
                    auto b = load_vec<N_ROT>(cur_belief, ne);
                    b = (1.f-damping)*rcp(max(b))*b + damping*load_vec<N_ROT>(old_belief, ne);
                    store_vec(cur_belief, ne, b);
                }
            } else {  // zero damping should not keep any info, even NaN from previous iteration
                for(int ne: range(n_elem)) {
                    auto b = load_vec<N_ROT>(cur_belief, ne);
                    b = rcp(max(b))*b;
                    store_vec(cur_belief, ne, b);
                }
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

            float en = energy_offset[nn];
            // free energy is average energy - entropy
            for(int no: range(N_ROT)) en += b[no] * logf((1e-10f+b[no])*rcp(1e-10f+pr[no]));
            return en;
        }
};

constexpr static int simd_width = 4;

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
            prob      (n_rot1*ru(n_rot2),     max_n_edge+3),
            cur_belief(ru(n_rot1)+ru(n_rot2), max_n_edge+3),
            old_belief(ru(n_rot1)+ru(n_rot2), max_n_edge+3),
            marginal(n_rot1*n_rot2,           max_n_edge+3), // the +1 ensures we can write past the end

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

        float& insert_or_get_loc(
                int ne,
                unsigned id1, unsigned rot1, 
                unsigned id2, unsigned rot2) {
            int32_t idx;
            if(nodes_to_edge.find_or_insert(idx,id1,id2)){
                edge_indices1[idx] = id1;
                edge_indices2[idx] = id2;
            }

            int j = rot1*ru(n_rot2)+rot2;
            edge_loc.emplace_back(EdgeLoc{ne, int(rot1*n_rot2+rot2), int(idx)});
            return prob(j,idx);
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

        template <int N_ROT1, int N_ROT2>
        void update_beliefs() {
            constexpr const int w1 = (N_ROT1+3)/4;
            constexpr const int w2 = (N_ROT2+3)/4;
            constexpr const int ws = w1+w2;
            // horizontal SIMD implementation of update_beliefs for N_ROT1==N_ROT2==3

            float* vec_old_node_belief1 = nodes1.old_belief.x.get();
            float* vec_cur_node_belief1 = nodes1.cur_belief.x.get();

            float* vec_old_node_belief2 = nodes2.old_belief.x.get();
            float* vec_cur_node_belief2 = nodes2.cur_belief.x.get();

            int n_edge = nodes_to_edge.n_edge;

            for(int ne=0; ne<n_edge; ++ne) {
                int i1 = edge_indices1[ne]*4*w1;
                int i2 = edge_indices2[ne]*4*w2;

                auto old_edge_belief1 = read4vec<w1>(old_belief.x + ne*4*ws + 0);
                auto old_edge_belief2 = read4vec<w2>(old_belief.x + ne*4*ws + 4*w1);

                auto old_node_belief1 = read4vec<w1>(vec_old_node_belief1 + i1);
                auto old_node_belief2 = read4vec<w2>(vec_old_node_belief2 + i2);

                auto v1 = old_node_belief1 * vec_rcp(Float4(1e-10f) + old_edge_belief1);
                auto v2 = old_node_belief2 * vec_rcp(Float4(1e-10f) + old_edge_belief2);

                // store v1 and v2 onto old edge beliefs
                store4vec<w1>(old_belief.x + ne*4*ws + 0,    v1);
                store4vec<w2>(old_belief.x + ne*4*ws + 4*w1, v2);
            }

            for(int ne=0; ne<n_edge; ++ne) {
                int i1 = edge_indices1[ne]*4*w1;
                int i2 = edge_indices2[ne]*4*w2;

                auto v1 = read4vec<w1>(old_belief.x + ne*4*ws + 0);
                auto v2 = read4vec<w2>(old_belief.x + ne*4*ws + 4*w1);

                // load the edge probability matrix
                auto eprob = PaddedMatrix<N_ROT1,N_ROT2>(prob.x + ne*N_ROT1*4*w2);
                auto cur_edge_belief1 = eprob.apply_left (v2);
                auto cur_edge_belief2 = eprob.apply_right(v1);

                auto cur_node_belief1 = cur_edge_belief1 * read4vec<w1>(vec_cur_node_belief1 + i1);
                auto cur_node_belief2 = cur_edge_belief2 * read4vec<w2>(vec_cur_node_belief2 + i2);
                
                // // node normalization is needed for avoid NaN
                // // FIXME investigate edge scalings that could obviate this
                // // FIXME investigate scaling the edges only every N somethings to reduce expense
                // cur_node_belief1 *= rcp(sum(cur_node_belief1).sum_in_all_entries());
                // cur_node_belief2 *= rcp(sum(cur_node_belief2).sum_in_all_entries());

                store4vec<w1>(cur_belief.x + ne*4*ws + 0,    cur_edge_belief1);
                store4vec<w2>(cur_belief.x + ne*4*ws + 4*w1, cur_edge_belief2);
                store4vec<w1>(vec_cur_node_belief1 + i1,     cur_node_belief1);
                store4vec<w2>(vec_cur_node_belief2 + i2,     cur_node_belief2);
            }

            // Perform edge normalization for all edges
            // We could perform it in the loop above, but it would insert a long dependency chain in the 
            // middle of the algorithm.  The hope is that the processor will expose much more instruction
            // parallelism in this loop.  The loop process 2 edges at a time to fully utilize the horizontal
            // adds.
            for(int ne=0; ne<n_edge; ne+=2) {
                auto cb11 = read4vec<w1>(cur_belief.x + ne*4*ws + 0);
                auto cb12 = read4vec<w2>(cur_belief.x + ne*4*ws + 4*w1);
                auto cb21 = read4vec<w1>(cur_belief.x + ne*4*ws + 4*ws);
                auto cb22 = read4vec<w2>(cur_belief.x + ne*4*ws + 4*(ws+w1));

                // let's approximately l1 normalize everything edges to avoid any numerical problems later
                Float4 scales_for_unit_l1 = approx_rcp(horizontal_add(
                            horizontal_add(sum(cb11), sum(cb12)),
                            horizontal_add(sum(cb21), sum(cb22))));

                store4vec<w1>(cur_belief.x + ne*4*ws + 0,         cb11*scales_for_unit_l1.broadcast<0>());
                store4vec<w2>(cur_belief.x + ne*4*ws + 4*w1,      cb12*scales_for_unit_l1.broadcast<1>());
                store4vec<w1>(cur_belief.x + ne*4*ws + 4*ws,      cb21*scales_for_unit_l1.broadcast<2>());
                store4vec<w2>(cur_belief.x + ne*4*ws + 4*(ws+w1), cb22*scales_for_unit_l1.broadcast<3>());
            }
        }
};


// template<>
// void EdgeHolder::calculate_marginals<3,3>() {
//     int n_edge = nodes_to_edge.n_edge;
//     for(int ne=0; ne<n_edge; ++ne) {
//         auto b1 = Float4(nodes1.cur_belief.x + 4*edge_indices1[ne]);
//         auto b2 = Float4(nodes2.cur_belief.x + 4*edge_indices2[ne]);
// 
//         // correct for self interaction
//         auto bc1 = b1 * rcp(Float4(1e-10f) + Float4(cur_belief.x + 8*ne));
//         auto bc2 = b2 * rcp(Float4(1e-10f) + Float4(cur_belief.x + 8*ne + 4));
// 
//         auto ep_row1 = Float4(prob.x + ne*4*3 + 0);
//         auto ep_row2 = Float4(prob.x + ne*4*3 + 4);
//         auto ep_row3 = Float4(prob.x + ne*4*3 + 8);
// 
//         // we want the element-wise product of ep and outer(bc1,bc2)
//         auto marg_row1 = bc1.broadcast<0>() * (ep_row1 * bc2);
//         auto marg_row2 = bc1.broadcast<1>() * (ep_row2 * bc2);
//         auto marg_row3 = bc1.broadcast<2>() * (ep_row3 * bc2);
//         
//         // normalize marginal so that total probability is 1
//         auto marg_scale = rcp((marg_row1 + marg_row2 + marg_row3).sum_in_all_entries());
//         marg_row1 *= marg_scale;
//         marg_row2 *= marg_scale;
//         marg_row3 *= marg_scale;
// 
//         marg_row1.store(&marginal(0,ne), Alignment::  aligned);
//         marg_row2.store(&marginal(3,ne), Alignment::unaligned);
//         marg_row3.store(&marginal(6,ne), Alignment::unaligned);
//     }
// }


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
    NodeHolder  nodes1, nodes3, nodes6; // FIXME initialize these with sane max_n_edge

    EdgeHolder* edge_holders_matrix[UPPER_ROT][UPPER_ROT];
    EdgeHolder edges11, edges13, edges16, edges33, edges36, edges66; // FIXME initialize these with sane max_n_edge

    int max_edge_cell_assignment;
    unique_ptr<float*[]> edge_cell_assignment;

    float damping;
    int   max_iter;
    float tol;
    int   iteration_chunk_size;

    bool energy_fresh_relative_to_derivative;

    long n_bad_solve;

    RotamerSidechain(hid_t grp, CoordNode &pos_node_, vector<CoordNode*> prob_nodes_):
        PotentialNode(),
        prob_nodes(prob_nodes_),
        n_prob_nodes(prob_nodes.size()),
        igraph(open_group(grp,"pair_interaction").get(), &pos_node_),
        n_elem_rot(calculate_n_elem(igraph)),

        nodes1(1,n_elem_rot[1]),
        nodes3(3,n_elem_rot[3]),
        nodes6(6,n_elem_rot[6]),

        edges11(nodes1,nodes1,n_elem_rot[1]*(n_elem_rot[1]+1)/2),
        edges13(nodes1,nodes3,n_elem_rot[1]* n_elem_rot[3]),
        edges16(nodes1,nodes6,n_elem_rot[1]* n_elem_rot[6]),
        edges33(nodes3,nodes3,n_elem_rot[3]*(n_elem_rot[3]+1)/2),
        edges36(nodes3,nodes6,n_elem_rot[3]* n_elem_rot[6]),
        edges66(nodes6,nodes6,n_elem_rot[6]*(n_elem_rot[6]+1)/2),

        max_edge_cell_assignment(0),
        edge_cell_assignment(nullptr),

        damping (read_attribute<float>(grp, ".", "damping")),
        max_iter(read_attribute<int  >(grp, ".", "max_iter")),
        tol     (read_attribute<float>(grp, ".", "tol")),
        iteration_chunk_size(read_attribute<int>(grp, ".", "iteration_chunk_size")),

        energy_fresh_relative_to_derivative(false),
        n_bad_solve(0)
    {
        for(int i: range(UPPER_ROT)) node_holders_matrix[i] = nullptr;
        node_holders_matrix[1] = &nodes1;
        node_holders_matrix[3] = &nodes3;
        node_holders_matrix[6] = &nodes6;

        for(int i: range(UPPER_ROT)) for(int j: range(UPPER_ROT)) edge_holders_matrix[i][j] = nullptr;
        edge_holders_matrix[1][1] = &edges11;
        edge_holders_matrix[1][3] = &edges13;
        edge_holders_matrix[1][6] = &edges16;
        edge_holders_matrix[3][3] = &edges33;
        edge_holders_matrix[3][6] = &edges36;
        edge_holders_matrix[6][6] = &edges66;

        for(int i: range(prob_nodes.size())) 
            if(igraph.pos_node1->n_elem != prob_nodes[i]->n_elem)
                throw string("rotamer positions have " + to_string(igraph.pos_node1->n_elem) +
                        " elements but the " + to_string(i) + "-th (0-indexed) probability node has only " +
                        to_string(prob_nodes[i]->n_elem) + " elements.");

        if(logging(LOG_BASIC))
            default_logger->add_logger<long>("rotamer_bad_solves_cumulative", {1},
                    [&](long* buffer) {buffer[0]=n_bad_solve;});

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("rotamer_free_energy", {nodes1.n_elem+nodes3.n_elem+nodes6.n_elem}, 
                    [&](float* buffer) {
                       auto en = residue_free_energies();
                       copy(begin(en), end(en), buffer);});

            for(int npn: range(n_prob_nodes))
                default_logger->add_logger<float>(("rotamer_1body_energy" + to_string(npn)).c_str(),
                        {nodes1.n_elem+nodes3.n_elem+nodes6.n_elem}, [npn,this](float* buffer) {
                            auto en = this->rotamer_1body_energy(npn);
                            copy(begin(en), end(en), buffer);});
        }
    }

    virtual vector<float> get_value_by_name(const char* log_name) override {
        int n_node = nodes1.n_elem + nodes3.n_elem + nodes6.n_elem;

        if(!strcmp(log_name, "rotamer_free_energy")) {
            return residue_free_energies();
        } else if(!strcmp(log_name, "rotamer_1body_energy")) {
            int n_elem = 0; for(auto& i: n_elem_rot) n_elem += i;
            vector<float> result(n_elem*n_prob_nodes);

            for(int npn: range(n_prob_nodes)) {
                auto energies = rotamer_1body_energy(npn);
                if(energies.size() != size_t(n_elem)) throw string("impossible");
                for(int i: range(n_elem))
                    result[i*n_prob_nodes + npn] = energies[i];
            }

            return result;
        } else if(!strcmp(log_name, "count_edges_by_type")) {
            return igraph.count_edges_by_type();
        } else if(!strcmp(log_name, "n_node")) {
            vector<float> ret(1, float(n_node));
            return ret;
        } else if(!strcmp(log_name, "node_energy")) {
            vector<float> node_energy(n_node*6);

            int nn = 0; // global node counter
            vector<int> node_starts(7,0);
            for(const NodeHolder& nodes: {cref(nodes1), cref(nodes3), cref(nodes6)}) {
                int n_rot = nodes.n_rot;
                node_starts[n_rot] = nn;
                for(int ne=0; ne<nodes.n_elem; ++ne, ++nn)  // increment both global and local counters
                    for(int nr=0; nr<6; ++nr)
                        node_energy[nn*6+nr] = (nr<n_rot)
                            ? /*nodes.energy_offset[ne]*/-logf(nodes.prob(nr,ne))
                            : 1e5f;
            }
            return node_energy;
        } else if(!strcmp(log_name, "edge_energy") || !strcmp(log_name, "edge_marginal_in_graph_order")) {
            vector<float> edge_value(n_node*n_node*6*6, 0.f);

            vector<int> node_starts(7,0);
            node_starts[1] = 0;
            node_starts[3] = nodes1.n_elem;
            node_starts[6] = nodes1.n_elem+nodes3.n_elem;

            bool do_marginal = !strcmp(log_name, "edge_marginal_in_graph_order");

            if(do_marginal) {
                // We pre-fill with the uncorrelated marginals because this is the assumption 
                // for all zero-edges in belief propagation
                
                vector<float> node_marginal(n_node*6,0.f);
                int nn=0;
                for(const NodeHolder& nodes: {cref(nodes1), cref(nodes3), cref(nodes6)}) {
                    int n_rot = nodes.n_rot;
                    for(int ne=0; ne<nodes.n_elem; ++ne, ++nn)  // increment both global and local counters
                        for(int nr=0; nr<6; ++nr)
                            node_marginal[nn*6+nr] = (nr<n_rot) ? nodes.cur_belief(nr,ne) : 0.f;
                }

                for(int i1=0; i1<n_node; ++i1)
                    for(int i2=0; i2<n_node; ++i2)
                        for(int nr1=0; nr1<6; ++nr1)
                            for(int nr2=0; nr2<6; ++nr2)
                                edge_value[((i1*n_node + i2)*6 + nr1)*6 + nr2] =
                                    (i1==i2)
                                        ? node_marginal[i1*6+nr1] * (nr1==nr2)
                                        : node_marginal[i1*6+nr1] * node_marginal[i2*6+nr2];
            }

            for(const EdgeHolder& edges: {cref(edges11), cref(edges33), cref(edges36), cref(edges66)}) {
                int n_rot1 = edges.n_rot1;
                int n_rot2 = edges.n_rot2;

                for(int ne=0; ne<edges.nodes_to_edge.n_edge; ++ne) {
                    int i1 = node_starts[n_rot1] + edges.edge_indices1[ne];
                    int i2 = node_starts[n_rot2] + edges.edge_indices2[ne];

                    for(int nr1=0; nr1<n_rot1; ++nr1)
                        for(int nr2=0; nr2<n_rot2; ++nr2)
                            edge_value[((i1*n_node + i2)*6 + nr1)*6 + nr2] =
                                edge_value[((i2*n_node + i1)*6 + nr2)*6 + nr1] =
                                    do_marginal
                                        ? edges.marginal(nr1*n_rot2 + nr2, ne)
                                        : -logf(edges.prob(nr1*ru(n_rot2) + nr2, ne));
                }
            }

            return edge_value;
        } else if(!strcmp(log_name, "read n_bad_solve")) {
            return vector<float>(1, float(n_bad_solve));
        } else if(!strcmp(log_name, "read n_bad_solve and reset")) {
            vector<float> ret(1, float(n_bad_solve));
            n_bad_solve = 0;
            return ret;
        } else {
            throw string("Value ") + log_name + string(" not implemented");
        }
    }

    void ensure_fresh_energy() {
        // FIXME this won't be correct if there are subtasks
        if(!energy_fresh_relative_to_derivative) compute_value(0, PotentialAndDerivMode);
    }

    virtual void compute_value_subtask(int n_round, int task_idx, int n_subtask) override {
        if(n_round==0) {
            // printf("starting parallel %i\n", task_idx);
            fill_holders_subtask(task_idx, n_subtask);
            // printf("finishing parallel %i\n", task_idx);
        }
    }
    virtual int compute_value(int n_round, ComputeMode mode) override {
        if(n_round==0) {
            energy_fresh_relative_to_derivative = mode==PotentialAndDerivMode;
            fill_holders_init();
            return 3; // number of threads
        } else if(n_round==1) {
            fill_holders_finish();
            auto solve_results = solve_for_marginals();
            if(solve_results.first >= max_iter - iteration_chunk_size - 1)
                n_bad_solve++;

            propagate_derivatives();
            if(mode==PotentialAndDerivMode) potential = calculate_energy_from_marginals();
            return 0; // node is finished
        } else {
            return -1;
        }
    }

    void fill_holders_init() {
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
            int index = igraph.loc1[n];

            float energy = 0.f;
            for(auto &a: energy_1body) energy += a(0,index);

            // at this point, the node "prob" is really energy
            node_holders_matrix[n_rot]->prob(rot,id) += energy;
        }
        for(int n_rot: range(1,UPPER_ROT))
            if(node_holders_matrix[n_rot])
                node_holders_matrix[n_rot]->convert_energy_to_prob();
        igraph.compute_edges_init();

        if(igraph.n_edge > max_edge_cell_assignment) {
            max_edge_cell_assignment = igraph.n_edge;
            edge_cell_assignment.reset(new float*[max_edge_cell_assignment]);
        }
    }

    void fill_holders_subtask(int task_idx, int n_subtasks) {
        if(task_idx==n_subtasks-1) {
            // special subtask to find the write locations
            // this is typically fairly expensive but not trivially parallelized
            // even better would be to make it fully parallel -- it takes about 30us on ubiquitin
            const unsigned selector = (1u<<n_bit_rotamer) - 1u;
            for(int ne=0; ne < igraph.n_edge; ++ne) {
                int   id1  = igraph.edge_id1[ne];
                int   id2  = igraph.edge_id2[ne];

                if((id1&(selector<<n_bit_rotamer)) > (id2&(selector<<n_bit_rotamer))) swap(id1,id2);

                unsigned   rot1 = id1 & selector; id1 >>= n_bit_rotamer;
                unsigned   rot2 = id2 & selector; id2 >>= n_bit_rotamer;

                unsigned n_rot1 = id1 & selector; id1 >>= n_bit_rotamer;
                unsigned n_rot2 = id2 & selector; id2 >>= n_bit_rotamer;

                edge_cell_assignment[ne] = &edge_holders_matrix[n_rot1][n_rot2]->insert_or_get_loc(ne, id1, rot1, id2, rot2);
            }
        } else {
            tuple<int,int> my_range = igraph.compute_edges_run(task_idx, n_subtasks-1);
            for(int ne=get<0>(my_range); ne < get<1>(my_range); ++ne)
                igraph.edge_value[ne] = expf(-igraph.edge_value[ne]);  // value of edge is potential
        }
    }

    void fill_holders_finish()
    {
        for(int ne=0; ne < igraph.n_edge; ++ne)
            *edge_cell_assignment[ne] *= igraph.edge_value[ne];  // value of edge is potential

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
        for(int nn: range(nodes6 .n_elem)) en += nodes6 .node_free_energy<6>  (nn);
        for(int ne: range(edges11.nodes_to_edge.n_edge)) en += -logf(edges11.prob(0,ne));
        for(int ne: range(edges33.nodes_to_edge.n_edge)) en += edges33.edge_free_energy<3,3>(ne);
        for(int ne: range(edges36.nodes_to_edge.n_edge)) en += edges36.edge_free_energy<3,6>(ne);
        for(int ne: range(edges66.nodes_to_edge.n_edge)) en += edges66.edge_free_energy<6,6>(ne);
        return en;
    }

    vector<float> residue_free_energies() {
        vector<float> e1(nodes1.n_elem, 0.f);
        vector<float> e3(nodes3.n_elem, 0.f);
        vector<float> e6(nodes6.n_elem, 0.f);

        for(int nn: range(nodes1 .n_elem)) {float en = nodes1.node_free_energy<1>(nn); e1[nn] += en;}
        for(int nn: range(nodes3 .n_elem)) {float en = nodes3.node_free_energy<3>(nn); e3[nn] += en;}
        for(int nn: range(nodes6 .n_elem)) {float en = nodes6.node_free_energy<6>(nn); e6[nn] += en;}

        for(int ne: range(edges11.nodes_to_edge.n_edge)) {
            float en = -logf(edges11.prob(0,ne));
            e1[edges11.edge_indices1[ne]] += 0.5*en;
            e1[edges11.edge_indices2[ne]] += 0.5*en;
        }

        for(int ne: range(edges33.nodes_to_edge.n_edge)) {
            float en = edges33.edge_free_energy<3,3>(ne);
            e3[edges33.edge_indices1[ne]] += 0.5*en;
            e3[edges33.edge_indices2[ne]] += 0.5*en;
        }

        for(int ne: range(edges36.nodes_to_edge.n_edge)) {
            float en = edges36.edge_free_energy<3,6>(ne);
            e3[edges36.edge_indices1[ne]] += 0.5*en;
            e6[edges36.edge_indices2[ne]] += 0.5*en;
        }

        for(int ne: range(edges66.nodes_to_edge.n_edge)) {
            float en = edges66.edge_free_energy<6,6>(ne);
            e6[edges66.edge_indices1[ne]] += 0.5*en;
            e6[edges66.edge_indices2[ne]] += 0.5*en;
        }

        return arrange_energies(e1,e3,e6);
    }

    vector<float> rotamer_1body_energy(int prob_node_index) {
        vector<float> e1(nodes1.n_elem, 0.f);
        vector<float> e3(nodes3.n_elem, 0.f);
        vector<float> e6(nodes6.n_elem, 0.f);

        VecArray energy_1body = prob_nodes[prob_node_index]->output;
        for(int n: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;
            int index = igraph.loc1[n];

            switch(n_rot) {
                case 1: e1[id] += nodes1.cur_belief(rot,id) * energy_1body(0,index); break;
                case 3: e3[id] += nodes3.cur_belief(rot,id) * energy_1body(0,index); break;
                case 6: e6[id] += nodes6.cur_belief(rot,id) * energy_1body(0,index); break;
                default: throw string("impossible");
            }
        }

        return arrange_energies(e1,e3,e6);
    }

    vector<float> arrange_energies(const vector<float>& e1, const vector<float>& e3, const vector<float>& e6) {
        vector<float> energies(n_elem_rot[1]+n_elem_rot[3]+n_elem_rot[6]);
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
                case 6: *en_loc = e6[id]; ++en_loc; break;
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
        for(auto &el: edges16.edge_loc)
            igraph.edge_sensitivity[el.edge_num] = nodes6 .cur_belief(el.dim, edges16.edge_indices2[el.ne]);
        for(auto edge_set: {&edges33,&edges36,&edges66})
            for(auto &el: edge_set->edge_loc)
                igraph.edge_sensitivity[el.edge_num] = edge_set->marginal(el.dim, el.ne);
        igraph.propagate_derivatives();

        vector<VecArray> sens_1body;
        sens_1body.reserve(n_prob_nodes);
        for(int i: range(n_prob_nodes)) 
            sens_1body.emplace_back(prob_nodes[i]->sens.acquire());

        for(int n: range(igraph.n_elem1)) {
            unsigned id = igraph.id1[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;

            for(int i: range(n_prob_nodes)) {
                NodeHolder* nh = node_holders_matrix[n_rot];
                sens_1body[i](0,igraph.loc1[n]) += nh->cur_belief(rot,id);
            }
        }
        for(int i: range(n_prob_nodes)) 
            prob_nodes[i]->sens.release(sens_1body[i]);
    }


    void calculate_new_beliefs(float damping_for_this_iteration, bool do_swap_for_initial=false) {
        copy(nodes3.prob, nodes3.cur_belief);
        copy(nodes6.prob, nodes6.cur_belief);
        edges33.update_beliefs<3,3>();
        edges36.update_beliefs<3,6>();
        edges66.update_beliefs<6,6>();

        if(do_swap_for_initial) {
            // we want the "old" values here
            nodes3.swap_beliefs();
            nodes6.swap_beliefs();
        }
        nodes3.standardize_belief_update<3>(damping_for_this_iteration);
        nodes6.standardize_belief_update<6>(damping_for_this_iteration);
    }
    

    pair<int,float> solve_for_marginals() {
        Timer timer(std::string("rotamer_solve"));
        // first initialize old node beliefs to just be probability
        // this may affect the final answer since belief propagation is minimizing a non-convex function
        for(auto nh: node_holders_matrix)
            if(nh)
                for(int no: range(nh->n_rot))
                    for(int ne: range(nh->n_elem))
                        nh->old_belief(no,ne) = nh->prob(no,ne);

        alignas(16) float start_belief3 [4] = {1.f,1.f,1.f,0.f}; auto sb3  = Float4(start_belief3);
        alignas(16) float start_belief6a[4] = {1.f,1.f,1.f,1.f}; auto sb6a = Float4(start_belief6a);
        alignas(16) float start_belief6b[4] = {1.f,1.f,0.f,0.f}; auto sb6b = Float4(start_belief6b);
        for(int ne=0; ne<edges33.nodes_to_edge.n_edge; ++ne) {
            sb3 .store(edges33.old_belief.x+ne*8+0);
            sb3 .store(edges33.old_belief.x+ne*8+4);
        }
        for(int ne=0; ne<edges36.nodes_to_edge.n_edge; ++ne) {
            sb3 .store(edges36.old_belief.x+ne*12+0);
            sb6a.store(edges36.old_belief.x+ne*12+4);
            sb6b.store(edges36.old_belief.x+ne*12+8);
        }
        for(int ne=0; ne<edges66.nodes_to_edge.n_edge; ++ne) {
            sb6a.store(edges66.old_belief.x+ne*16+ 0);
            sb6b.store(edges66.old_belief.x+ne*16+ 4);
            sb6a.store(edges66.old_belief.x+ne*16+ 8);
            sb6b.store(edges66.old_belief.x+ne*16+12);
        }

        calculate_new_beliefs(0.f, true);
        float max_deviation = 1e10f;
        int iter = 0;

        for(; max_deviation>tol && iter<max_iter; iter+=iteration_chunk_size) {
            for(int j=0; j<iteration_chunk_size; ++j) {
                nodes3 .swap_beliefs();
                nodes6 .swap_beliefs();
                edges33.swap_beliefs();
                edges36.swap_beliefs();
                edges66.swap_beliefs();
                calculate_new_beliefs(damping);
            }

            // compute max deviation
            // printf("(%i,%.3f,%.3f)\n", iter, nodes3.max_deviation(), nodes6.max_deviation());
            max_deviation = max(nodes3.max_deviation(), nodes6.max_deviation());
        }

        nodes1 .calculate_marginals<1>  ();
        nodes3 .calculate_marginals<3>  ();
        nodes6 .calculate_marginals<6>  ();
        edges11.calculate_marginals<1,1>();
        edges33.calculate_marginals<3,3>();
        edges36.calculate_marginals<3,6>();
        edges66.calculate_marginals<6,6>();
        return make_pair(iter, max_deviation);
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
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
