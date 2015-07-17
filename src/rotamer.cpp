#include "bead_interaction.h"
#include "interaction_graph.h"
#include "rotamer.h"
#include <vector>
#include <string>
#include "h5_support.h"
#include <algorithm>
#include <map>
#include "deriv_engine.h"
#include "timing.h"
#include <memory>
#include "state_logger.h"
#include <tuple>

using namespace std;
using namespace h5;

constexpr const int UPPER_ROT = 4;  // 1 more than the most possible rotamers (handle 0)


struct NodeHolder {
    public:
        const int n_rot;
        const int n_elem;
    protected:
        SysArrayStorage s_prob;
        SysArrayStorage s_belief1;
        SysArrayStorage s_belief2;

    public:
        VecArray prob;

        VecArray cur_belief;
        VecArray old_belief;

        NodeHolder(int n_rot_, int n_elem_):
            n_rot(n_rot_),
            n_elem(n_elem_),
            s_prob(1,n_rot,n_elem),
            s_belief1(1,n_rot,n_elem),
            s_belief2(1,n_rot,n_elem),
            prob(s_prob[0]),
            cur_belief(s_belief1[0]),
            old_belief(s_belief2[0]) 
        {
            fill(cur_belief, n_rot, n_elem, 1.f);
            fill(old_belief, n_rot, n_elem, 1.f);
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
        void finish_belief_update(float damping) {
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


// FIXME specialize edge_holder when rot1 is 1
// FIXME implement beliefs and swapping system
struct EdgeHolder {
    public:
        int n_rot1, n_rot2;  // should have rot1 < rot2
        int n_edge;
        NodeHolder &nodes1;
        NodeHolder &nodes2;

    protected:
        SysArrayStorage s_prob;
        SysArrayStorage s_belief1;
        SysArrayStorage s_belief2;

    public:
        // FIXME include numerical stability data  (basically scale each probability in a sane way)
        VecArray prob;  // stored here
        VecArray cur_belief;
        VecArray old_belief;

        vector<pair<int,int>> edge_indices;
        unordered_map<unsigned,unsigned> nodes_to_edge;

        EdgeHolder(NodeHolder &nodes1_, NodeHolder &nodes2_, int max_n_edge):
            n_rot1(nodes1_.n_rot), n_rot2(nodes2_.n_rot),
            nodes1(nodes1_), nodes2(nodes2_),
            s_prob(1, n_rot1*n_rot2, max_n_edge),
            s_belief1(1, n_rot1+n_rot2, max_n_edge),
            s_belief2(1, n_rot1+n_rot2, max_n_edge),

            prob(s_prob[0]),
            cur_belief(s_belief1[0]),
            old_belief(s_belief2[0]),

            edge_indices(max_n_edge)
        {
            fill(cur_belief, n_rot1*n_rot2, n_edge, 1.f);
            fill(old_belief, n_rot1*n_rot2, n_edge, 1.f);
            reset();
        }

        void reset() {n_edge=0; nodes_to_edge.clear();}
        void swap_beliefs() { swap(cur_belief, old_belief); }

        void add_to_edge(
                float prob_val,
                unsigned id1, unsigned rot1, 
                unsigned id2, unsigned rot2) {
            // really I could take n_rot1 and n_rot2 as parameters so I didn't have to do a read 
            // of a number I already know
            unsigned pr = (id1<<16) + id2;  // this limits me to 65k residues, but that is enough I think
            auto ei = nodes_to_edge.find(pr);

            unsigned idx;
            if(ei == nodes_to_edge.end()) {
                idx = n_edge;
                nodes_to_edge[pr] = n_edge;
                edge_indices[idx] = make_pair(int(id1),int(id2));
                ++n_edge;

                for(int j: range(n_rot1*n_rot2)) 
                    prob(j,idx) = 1.f;
            } else {
                idx = ei->second;
            }

            prob(rot1*n_rot2+rot2, idx) *= prob_val;
        }

        void move_edge_prob_to_node2() {
            // FIXME assert n_rot1 == 1
            VecArray p = nodes2.prob;
            for(int ne: range(n_edge)) {
                int nr = edge_indices[ne].second;
                for(int no: range(n_rot2)) p(no,nr) *= prob(no,ne);
            }
        }

        void standardize_probs() { // FIXME should accumulate the rescaled probability
            for(int ne: range(n_edge)) {
                float max_prob = 1e-10f;
                for(int nd: range(n_rot1*n_rot2)) if(prob(nd,ne)>max_prob) max_prob = prob(nd,ne);
                float inv_max_prob = rcp(max_prob);
                for(int nd: range(n_rot1*n_rot2)) prob(nd,ne) *= inv_max_prob;
            }
        }

        float max_deviation() {
            float dev = 0.f;
            for(int d: range(n_rot1+n_rot2)) 
                for(int nn: range(n_edge)) 
                    dev = max(cur_belief(d,nn)-old_belief(d,nn), dev);
            return dev;
        }


        template<int N_ROT1, int N_ROT2>
        Vec<N_ROT1*N_ROT2> prob_matrix(int ne) {
            // FIXME ASSERT(n_rot1 == N_ROT1)
            // FIXME ASSERT(n_rot2 == N_ROT2)  // kind of clunky but should improve performance by loop unrolling

            auto e = edge_indices[ne];
            auto b1 = load_vec<N_ROT1>(nodes1.cur_belief, e.first);
            auto b2 = load_vec<N_ROT2>(nodes2.cur_belief, e.second);

            // correct for self interaction
            auto bc1 = b1 * vec_rcp(1e-10f + load_vec<N_ROT1>(cur_belief,                 ne));
            auto bc2 = b2 * vec_rcp(1e-10f + load_vec<N_ROT2>(cur_belief.shifted(N_ROT1), ne));

            auto p = load_vec<N_ROT1*N_ROT2>(prob, ne);
            for(int no1: range(N_ROT1))
                for(int no2: range(N_ROT2))
                    p[no1*N_ROT1+no2] *= bc1[no1]*bc2[no2];
            p *= rcp(sum(p));
            return p;
        }

        template<int N_ROT1, int N_ROT2>
        float edge_free_energy(int ne) {
            auto e = edge_indices[ne];
            auto b1 = load_vec<N_ROT1>(nodes1.cur_belief, e.first);
            auto b2 = load_vec<N_ROT2>(nodes2.cur_belief, e.second);

            auto p = prob_matrix<N_ROT1,N_ROT2>(ne);
            auto pr = load_vec<N_ROT1*N_ROT2>(prob, ne);

            b1 *= rcp(sum(b1)); // l1 normalization
            b2 *= rcp(sum(b2));

            float en = 0.f;
            for(int no1: range(N_ROT1)) {
                for(int no2: range(N_ROT2)) {
                    int i = no1*N_ROT1 + no2;
                    // this is the average potential energy plus the mutual information,
                    // which is what I want for calculating the overall energy
                    // the 1e-10f prevents NaN if some probabilities are exactly 0
                    en += p[i] * logf((1e-10f+p[i]) * rcp(1e-10f+pr[i]*b1[no1]*b2[no2]));
                }
            }

            return en;
        }

        template <int N_ROT1, int N_ROT2>
        void update_beliefs(float damping) {
            // FIXME ASSERT(n_rot1 == N_ROT1)
            // FIXME ASSERT(n_rot2 == N_ROT2)  // kind of clunky but should improve performance by loop unrolling
            VecArray vec_old_node_belief1 = nodes1.old_belief;
            VecArray vec_cur_node_belief1 = nodes1.cur_belief;

            VecArray vec_old_node_belief2 = nodes2.old_belief;
            VecArray vec_cur_node_belief2 = nodes2.cur_belief;

            for(int ne: range(n_edge)) {
                int n1 = edge_indices[ne].first;
                int n2 = edge_indices[ne].second;

                auto old_node_belief1 = load_vec<N_ROT1>(vec_old_node_belief1, n1);
                auto old_node_belief2 = load_vec<N_ROT2>(vec_old_node_belief2, n2);

                auto ep = load_vec<N_ROT1*N_ROT2>(prob, ne);

                auto old_edge_belief1 = load_vec<N_ROT1>(old_belief                ,ne);
                auto old_edge_belief2 = load_vec<N_ROT2>(old_belief.shifted(N_ROT1),ne);

                auto cur_edge_belief1 =  left_multiply_matrix(ep, old_node_belief2 * vec_rcp(old_edge_belief2));
                auto cur_edge_belief2 = right_multiply_matrix(    old_node_belief1 * vec_rcp(old_edge_belief1), ep);
                cur_edge_belief1 *= rcp(max(cur_edge_belief1)); // rescale to avoid underflow in the future
                cur_edge_belief2 *= rcp(max(cur_edge_belief2));

                // FIXME the rescaling system should be made more careful / correct

                // store edge beliefs
                Vec<N_ROT1+N_ROT2> neb = damping*load_vec<N_ROT1+N_ROT2>(old_belief,ne);
                for(int i: range(N_ROT1)) neb[i]        = (1.f-damping)*cur_edge_belief1[i];
                for(int i: range(N_ROT2)) neb[i+N_ROT1] = (1.f-damping)*cur_edge_belief2[i];
                store_vec(cur_belief,ne, neb);

                // update our beliefs about nodes (normalization is L2, but this still keeps us near 1)
                store_vec(vec_cur_node_belief1, n1, normalized(cur_edge_belief1 * load_vec<N_ROT1>(vec_cur_node_belief1, n1)));
                store_vec(vec_cur_node_belief2, n2, normalized(cur_edge_belief2 * load_vec<N_ROT2>(vec_cur_node_belief2, n2)));
            }
        }
};




template <typename BT>
array<int,UPPER_ROT> calculate_n_elem(SymmetricInteractionGraph<BT> &igraph) {
    array<int,UPPER_ROT> result; // 0-rot is included
    for(int& i: result) i=0;

    for(unsigned id: igraph.id) {
        unsigned selector = (1u<<n_bit_rotamer) - 1u;
        if(id&selector) continue; // only count on the 0th rotamer
        unsigned n_rot = (id>>n_bit_rotamer) & selector;
        result.at(n_rot) += 1;
    }
    return result;
}


template <typename BT>
struct RotamerSidechain: public PotentialNode {
    CoordNode &prob_node;
    vector<slot_t> prob_slot;
    SymmetricInteractionGraph<BT> igraph;
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

    RotamerSidechain(hid_t grp, CoordNode &pos_node_, CoordNode &prob_node_):
        PotentialNode(pos_node_.n_system),
        prob_node(prob_node_),
        igraph(open_group(grp,"pair_interaction").get(), pos_node_),
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
        if(n_system > 1) throw string("multiple systems broken for the node_holders and edge_holders");

        for(int i: range(UPPER_ROT)) node_holders_matrix[i] = nullptr;
        node_holders_matrix[1] = &nodes1;
        node_holders_matrix[3] = &nodes3;

        for(int i: range(UPPER_ROT)) for(int j: range(UPPER_ROT)) edge_holders_matrix[i][j] = nullptr;
        edge_holders_matrix[1][1] = &edges11;
        edge_holders_matrix[1][3] = &edges13;
        edge_holders_matrix[3][3] = &edges33;

        // the index and the type information is already stored in the igraph
        for(auto &x: igraph.param) {
            CoordPair p; p.index = x.index;
            prob_node.slot_machine.add_request(1, p);
            prob_slot.push_back(p.slot);
        }
    }

    void ensure_fresh_energy() {
        if(!energy_fresh_relative_to_derivative) compute_value(PotentialAndDerivMode);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer("rotamer");  // Timer code is not thread-safe, so cannot be used within parallel for
        energy_fresh_relative_to_derivative = mode==PotentialAndDerivMode;

        fill_holders(0);
        auto solve_results = solve_for_beliefs();
        auto en = calculate_energy_from_beliefs();

        // nodes3.standardize_probs(); // FIXME debug
        // printf("\n");
        // for(int i: range(nodes3.n_elem)) {
        //     printf("node %3i  ", i);
        //     for(int j: range(nodes3.n_rot)) printf(" %.2f", nodes3.prob(j,i));
        //     printf("\n");
        // }
        // printf("\n");

        // edges33.standardize_probs(); // FIXME debug
        // printf("\n");
        // for(int i: range(edges33.n_edge)) {
        //     printf("edge %3i %3i  ", edges33.edge_indices[i].first, edges33.edge_indices[i].second);
        //     for(int j: range(9)) printf(" %.2f", edges33.prob(j,i));
        //     printf("\n");
        // }
        // printf("\n");

        // printf("final beliefs\n");
        // print(nodes3 .cur_belief, nodes3.n_rot, nodes3.n_elem, "node");
        // printf("\n");

        // for(int ne:range(edges33.n_edge)) {
        //     auto a = load_vec<3>(edges33.cur_belief.shifted(0),ne); a*=rcp(max(a));
        //     auto b = load_vec<3>(edges33.cur_belief.shifted(3),ne); b*=rcp(max(b));
        //     printf("edge %3i %3i  %.2f %.2f %.2f  %.2f %.2f %.2f\n",
        //             edges33.edge_indices[ne].first,
        //             edges33.edge_indices[ne].second,
        //             a[0],a[1],a[2], b[0],b[1],b[2]);
        // }
        // printf("\n");

        printf("\nsolved in %i iterations to tol %.5f\n", solve_results.first, solve_results.second);
        printf("final energy is %.3f\n", en);
    }

    virtual double test_value_deriv_agreement() {return -1.;}

    void fill_holders(int ns)
    {
        for(int n_rot1: range(UPPER_ROT))
            for(int n_rot2: range(UPPER_ROT))
                if(edge_holders_matrix[n_rot1][n_rot2])
                    edge_holders_matrix[n_rot1][n_rot2]->reset();

        // Fill node base probabilities
        VecArray energy_1body = prob_node.coords().value[ns];
        for(int n: range(igraph.n_elem)) {
            unsigned id = igraph.id[n];
            unsigned selector = (1u<<n_bit_rotamer) - 1u;
            unsigned rot      = id & selector; id >>= n_bit_rotamer;
            unsigned n_rot    = id & selector; id >>= n_bit_rotamer;

            node_holders_matrix[n_rot]->prob(rot,id) = expf(-energy_1body(0,igraph.param[n].index));
        }

        // Fill edge probabilities
        igraph.compute_edges(ns, [&](int ne, float pot,
                    int index1, unsigned type1, unsigned id1,
                    int index2, unsigned type2, unsigned id2) {
                unsigned selector = (1u<<n_bit_rotamer) - 1u;
                if((id1&(selector<<n_bit_rotamer)) > (id2&(selector<<n_bit_rotamer))) swap(id1,id2);
                // now id2 must have a >= n_rot than id1

                unsigned   rot1 = id1 & selector; id1 >>= n_bit_rotamer;
                unsigned   rot2 = id2 & selector; id2 >>= n_bit_rotamer;

                unsigned n_rot1 = id1 & selector; id1 >>= n_bit_rotamer;
                unsigned n_rot2 = id2 & selector; id2 >>= n_bit_rotamer;

                edge_holders_matrix[n_rot1][n_rot2]->add_to_edge(expf(-pot), id1, rot1, id2, rot2);
                });

        // for edges with a 1, we can just move it
        for(int n_rot: range(2,UPPER_ROT))
            if(edge_holders_matrix[1][n_rot])
                edge_holders_matrix[1][n_rot]->move_edge_prob_to_node2();
    }

    float calculate_energy_from_beliefs() {
        // beliefs must already have been solved
        // since edges1x were folded into the node probabilites, they should not be accumulated here
        float en = 0.f;
        for(int nn: range(nodes3 .n_elem)) en += nodes3 .node_free_energy<3>  (nn);
        printf("after en1 %.3f\n", en);
        for(int ne: range(edges33.n_edge)) en += edges33.edge_free_energy<3,3>(ne);
        printf("after en2 %.3f\n", en);
        for(int ne: range(edges11.n_edge)) en -= logf(edges11.prob(0,ne));
        printf("after en3 %.3f\n", en);
        return en;
    }

    void calculate_new_beliefs(float damping_for_this_iteration) {
        for(int no: range(nodes3.n_rot))
            copy_n(&nodes3.prob(no,0), nodes3.n_elem, &nodes3.cur_belief(no,0));

        edges33.update_beliefs<3,3>(damping_for_this_iteration);

        nodes3.finish_belief_update<3>(damping_for_this_iteration);
    }
    
    pair<int,float> solve_for_beliefs() {
        // first initialize old node beliefs to just be probability to speed convergence
        for(auto nh: node_holders_matrix)
            if(nh)
                for(int no: range(nh->n_rot))
                    copy_n(&nh->prob(no,0), nh->n_elem, &nh->old_belief(no,0));

        fill(edges33.old_belief, edges33.n_rot1+edges33.n_rot2, edges33.n_edge, 1.f);

        // this will fix consistent values in cur_belief for edges but put poor values in cur_belief for the nodes
        calculate_new_beliefs(0.1f);
        // swapping just nodes means that reasonable values are in cur_belief for both edges and nodes
        nodes3.swap_beliefs();

        float max_deviation = 1e10f;
        int iter = 0;

        for(; max_deviation>tol && iter<max_iter; iter+=iteration_chunk_size) {
            for(int j=0; j<iteration_chunk_size; ++j) {
                nodes3 .swap_beliefs();
                edges33.swap_beliefs();
                calculate_new_beliefs(damping);
            }

            // compute max deviation
            max_deviation = max(nodes3.max_deviation(), edges33.max_deviation());
        }

        return make_pair(iter, max_deviation);
    }
};
static RegisterNodeType<RotamerSidechain<preferred_bead_type>,2> rotamer_node ("rotamer");
