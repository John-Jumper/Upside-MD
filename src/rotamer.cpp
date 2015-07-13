#include "interaction_graph.h"
#include "rotamer.h"
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
#include "state_logger.h"
#include <tuple>

using namespace std;
using namespace h5;


// FIXME specialize edge_holder when rot1 is 1
struct EdgeHolder {
    int n_rot1, n_rot2;  // should have rot1 < rot2
    int n_edge;

    VecArray node_belief1, node_belief2; // shared by multiple EdgeHolders

    VecArray prob;  // stored here
    VecArray belief_new;  // stored here
    VecArray belief_old;  // stored here
    // FIXME include numerical stability data  (basically scale each probability in a sane way)

    vector<pair<int,int>> edge_indices;

    unordered_map<pair<unsigned,unsigned>,unsigned> nodes_to_edge;

    void reset() {n_edge = 0;}

    void add_to_edge(
            float prob,
            unsigned id1, unsigned rot1, 
            unsigned id2, unsigned rot2) {
        auto pr = make_pair(id1,id2);
        auto ei = = nodes_to_edge.find(pr);

        unsigned idx;
        if(ei == nodes_to_edge.end()) {
            idx = n_edge;
            nodes_to_edge[pr] = n_edge;
            edge_indices[idx] = pr;
            ++n_edge;

            for(int j: range(n_rot1*n_rot2)) 
                prob(j,idx) = 1.f;
        } else {
            idx = *ei;
        }

        prob(rot1*n_rot2+rot2, idx) *= prob;
    }

    void update_beliefs(int edge_belief_num, int node_belief_num, float damping) {
        // implement buffer alternation
        VecArray new_edge_belief = edge_belief_num ? edge_belief2 : edge_belief1;
        VecArray old_edge_belief = edge_belief_num ? edge_belief1 : edge_belief2;

        VecArray new_node_belief1 = node_belief_num ? node1_belief2 : node1_belief1;
        VecArray old_node_belief1 = node_belief_num ? node1_belief1 : node1_belief2;

        VecArray new_node_belief2 = node_belief_num ? node2_belief2 : node2_belief1;
        VecArray old_node_belief2 = node_belief_num ? node2_belief1 : node2_belief2;

        for(int ne: range(n_edge)) {
            int node1 = edge_indices[ne].first;
            int node2 = edge_indices[ne].second;

            auto old_node_belief1 = load_vec<n_rot1>(old_node_belief1, node1);
            auto old_node_belief2 = load_vec<n_rot2>(old_node_belief2, node2);

            auto ep = load_vec<n_rot1*n_rot2>(prob, ne);

            auto old_edge_belief1 = load_vec<n_rot1>(old_edge_belief                ,ne);
            auto old_edge_belief2 = load_vec<n_rot2>(old_edge_belief.shifted(n_rot1),ne);

            auto new_edge_belief1 =  left_multiply_matrix(ep, old_node_belief2 * vec_rcp(old_edge_belief2));
            auto new_edge_belief2 = right_multiply_matrix(    old_node_belief1 * vec_rcp(old_edge_belief1), ep);
            new_edge_belief1 *= rcp(max(new_edge_belief1)); // rescale to avoid underflow in the future
            new_edge_belief2 *= rcp(max(new_edge_belief2));

            // store edge beliefs
            Vec<n_rot1+n_rot2> neb = damping*load_vec<n_rot1+n_rot2>(old_edge_belief,ne);
            for(int i: range(n_rot1)) neb[i]        = (1.f-damping)*new_edge_belief1[i];
            for(int i: range(n_rot2)) neb[i+n_rot1] = (1.f-damping)*new_edge_belief2[i];
            store_vec(new_edge_belief,ne, neb);

            // update our beliefs about nodes (normalization is L2, but this still keeps us near 1)
            store_vec(new_node_belief1, node1, normalized(new_edge_belief1 * load_vec<n_rot1>(new_node_belief1, node1)));
            store_vec(new_node_belief2, node2, normalized(new_edge_belief2 * load_vec<n_rot2>(new_node_belief2, node2)));
        }
    }

};


void add_to_edge(
        int ne, float prob,
        unsigned id1,
        unsigned id2,
        ) {
    // First I need to find (id1,id2) in my handy-dandy hash table
    // I also need to deal with conversion of id's to local id's
    // this is also a mess
    // last few bits of the id indicate the n_rot
    unsigned real_id1 = id1 >> n_bit_rotamer;
    unsigned real_id2 = id2 >> n_bit_rotamer;

    unsigned rot1 = id1 & ((1u<<n_bit_rotamer)-1u);
    unsigned rot2 = id2 & ((1u<<n_bit_rotamer)-1u);

     // FIXME also extract rot1 and rot2 here
    unsigned n_rot1 = id1 & (1u<<n_bit_rotamer - 1);
    unsigned n_rot2 = id1 & (1u<<n_bit_rotamer - 1);

    unsigned which_array = wa(n_rot1,n_rot2);

    unsigned real_id1 = id1 >> n_bit_rotamer;
    unsigned real_id2 = id2 >> n_bit_rotamer;

    edge_holders[which_array].add_to_edge(
            prob,
            real_id1, rot1, 
            real_id2, rot2);
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

    edge_holders.update_beliefs();

    // normalize node beliefs to avoid underflow in the future
    for(int nn: range(n_node)) {
        auto b = load_vec<n_rot>(new_node_belief, nn);
        b *= rcp(max(b));
        store_vec(new_node_belief, nn, b);
    }

    for(int d: range(  n_rot)) for(int nn: range(n_node)) new_node_belief(d,nn) = new_node_belief(d,nn)*(1.f-damping) + old_node_belief(d,nn)*damping;
}


// return value is number of iterations completed
pair<int,float> solve_for_beliefs(
        VecArray node_belief,      VecArray edge_belief, 
        VecArray temp_node_belief, VecArray temp_edge_belief,
        int n_node, VecArray node_prob,
        int n_edge, VecArray edge_prob, int* edge_indices_this_system,
        float damping, // 0.f indicates no damping
        int max_iter, float tol, bool re_initialize_node_beliefs) {
    const int n_rot = 3;

    if(re_initialize_node_beliefs) {
        for(int d: range(  n_rot)) for(int nn: range(n_node)) node_belief(d,nn) = node_prob(d,nn);
    }
    for(int d: range(2*n_rot)) for(int ne: range(n_edge)) temp_edge_belief(d,ne) = 1.f;

    // now let's construct the edge beliefs that are correctly related to the node beliefs
    // since old_node_belief sets new_edge_belief and old_edge_belief sets new_node_belief, 
    //   we will do a weird mix to get node_belief sets edge_belief
    calculate_new_beliefs(
            temp_node_belief, edge_belief,
            node_belief,      temp_edge_belief,
            n_node, node_prob,
            n_edge, edge_prob, edge_indices_this_system,
            min(damping,0.1f));

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

template <typename BT>
compute_probability_graph(
        SymmetricInteractionGraph<BT> &igraph,
        VecArray node_prob, VecArray edge_prob, int* edge_indices,
        int* loc,  // local_index == -1 implies 1 rot
        int n_res3, VecArray pos3, // last dimension is 1-residue potential
        int &n_edge13, int &n_edge33, int ns) {

    //igraph.compute_edges(ns, [&](int ne, float pot,
    //            int index1, unsigned type1, unsigned id1,
    //            int index2, unsigned type2, unsigned id2) {


    //        }
    //        }

}

template <typename BT>
void convert_potential_graph_to_probability_graph(
        VecArray node_prob, VecArray edge_prob, int* edge_indices,
        int n_res3, VecArray pos3, // last dimension is 1-residue potential
        int n_edge13, PairInteraction<1,3,BT>* edges13,
        int n_edge33, PairInteraction<3,3,BT>* edges33) {

    // float potential_shift = 0.f;
    for(int ne33: range(n_edge33)) {
        auto &e = edges33[ne33];
        edge_indices[ne33*2 + 0] = e.nr1;
        edge_indices[ne33*2 + 1] = e.nr2;

        float min_pot = 1e10f;
        for(int no1: range(3))
            for(int no2: range(3))
                min_pot = min(e.potential[no1][no2], min_pot);
        // potential_shift += min_pot;

        for(int no1: range(3))
            for(int no2: range(3))
                edge_prob(no1*3+no2,ne33) = expf(-(e.potential[no1][no2] - min_pot));  // shift to avoid underflow later
    }
            for(int no2: range(3))
                edge_prob(no1*3+no2,ne33) = expf(-(e.potential[no1][no2] - min_pot));  // shift to avoid underflow later
    }

    for(int no: range(3))
        for(int nr: range(n_res3)) 
            node_prob(no,nr) = pos3(no*BT::n_pos_dim+BT::n_pos_dim-1,nr);

    for(int ne13: range(n_edge13)) {
        auto &e = edges13[ne13];
        int nr = e.nr2;

        for(int no: range(3)) node_prob(no,nr) += e.potential[0][no];
    }

    for(int nr: range(n_res3)) {
        float min_pot = 1e10f;
        for(int no: range(3))
            min_pot = min(node_prob(no,nr), min_pot);
        // potential_shift += min_pot;

        for(int no: range(3)) 
            node_prob(no,nr) = expf(-(node_prob(no,nr)-min_pot));
    }
    // printf("potential_shift %.4f\n", potential_shift);
}


template <typename BT>
void compute_free_energy_and_derivative(
        float* potential, VecArray node_marginal_prob, VecArray edge_marginal_prob,
        VecArray pos,
        VecArray deriv1, VecArray deriv3,
        int n_res1, int n_res3,
        VecArray node_belief, VecArray edge_belief,
        VecArray edge_prob,
        int n_edge11, const PairInteraction<1,1,BT>* edges11,
        int n_edge13, const PairInteraction<1,3,BT>* edges13,
        int n_edge33, const PairInteraction<3,3,BT>* edges33) {

    // start with zero derivative
    fill(deriv1, 1*BT::n_pos_dim, n_res1, 0.f);
    fill(deriv3, 3*BT::n_pos_dim, n_res3, 0.f);

    double free_energy = 0.f;

    // node beliefs couple directly to the 1-body potentials
    for(int nr: range(n_res3)) {
        auto b = load_vec<3>(node_belief, nr);
        b *= rcp(sum(b)); // normalize probability

        for(int no: range(3)) {
            deriv3(no*BT::n_pos_dim+3,nr) = b[no];
            // potential is given by the 3th element of position (note that -S is p*log p with no minus)
            if(potential) {
                float v =  b[no]*pos(no*BT::n_pos_dim+BT::n_pos_dim-1,nr);
                float s = -b[no]*logf(1e-10f+b[no]); // 1-body entropies
                free_energy += v-s;
                node_marginal_prob(no,nr) = b[no];
            }
        }
    }

    // edge beliefs couple to the positions
    for(int ne11: range(n_edge11)) {
        auto &e = edges11[ne11];
        update_vec(deriv1, e.nr1, e.deriv[0][0].d1());
        update_vec(deriv1, e.nr2, e.deriv[0][0].d2());
        if(potential) {
            float v = e.potential[0][0];
            free_energy += v; // no entropy since only 1 state for each
        }
    }

    for(int ne13: range(n_edge13)) {
        auto &e = edges13[ne13];
        Vec<3> b = load_vec<3>(node_belief, e.nr2);
        b *= rcp(sum(b)); // normalize probability

        if(potential) {
            float v = b[0]*e.potential[0][0]+b[1]*e.potential[0][1]+b[2]*e.potential[0][2];
            free_energy += v;  // no mutual information since one of the residues has only a single state
        }

        auto d1 = make_zero<BT::n_pos_dim-1>();
        for(int no2: range(3)) {
            d1 += b[no2]*e.deriv[0][no2].d1();
            update_vec(deriv3.shifted(BT::n_pos_dim*no2),e.nr2, b[no2]*e.deriv[0][no2].d2());
        }
        update_vec(deriv1,e.nr1, d1);
    }

    // The edge marginal distributions are given by p(x1,x2) *
    // node_belief_1(x1) * node_belief_2(x2) / (edge_belief_12(x1) *
    // edge_belief_21(x2)) up to normalization.
    for(int ne33: range(n_edge33)) {
        auto &e = edges33[ne33];
        float3 b1 = load_vec<3>(node_belief, e.nr1);
        float3 b2 = load_vec<3>(node_belief, e.nr2);

        // correct for self interaction
        float3 bc1 = b1 * vec_rcp(1e-10f + load_vec<3>(edge_belief,            ne33));
        float3 bc2 = b2 * vec_rcp(1e-10f + load_vec<3>(edge_belief.shifted(3), ne33));

        Vec<9> pair_distrib = load_vec<9>(edge_prob, ne33);
        for(int no1: range(3))
            for(int no2: range(3))
                pair_distrib[no1*3+no2] *= bc1[no1]*bc2[no2];
        pair_distrib *= rcp(sum(pair_distrib));

        // normalize beliefs to obtain node marginals again
        b1 *= rcp(sum(b1));
        b2 *= rcp(sum(b2));

        if(potential) {
            float v = 0.f;
            float s = 0.f;  // mutual information
            for(int no1: range(3)) {
                for(int no2: range(3)) {
                    auto p = pair_distrib[no1*3+no2];
                    v += p*e.potential[no1][no2];
                    s -= p*logf((1e-10f+p)*rcp((1e-10f+b1[no1]*b2[no2])));
                }
            }
            free_energy += v-s;
            store_vec(edge_marginal_prob, ne33, pair_distrib);
        }

        #define p(no1,no2) (pair_distrib[(no1)*3+(no2)])
        for(int no1: range(3)) 
            update_vec(deriv3.shifted(BT::n_pos_dim*no1),e.nr1,  
                p(no1,0)*e.deriv[no1][0].d1()+p(no1,1)*e.deriv[no1][1].d1()+p(no1,2)*e.deriv[no1][2].d1());
        for(int no2: range(3)) 
            update_vec(deriv3.shifted(BT::n_pos_dim*no2),e.nr2, 
                p(0,no2)*e.deriv[0][no2].d2()+p(1,no2)*e.deriv[1][no2].d2()+p(2,no2)*e.deriv[2][no2].d2());
        #undef p
    }

    if(potential) *potential = free_energy;
} 


template <typename BT>
struct RotamerSidechain: public PotentialNode {
    struct RotamerIndices {
        int start;
        int stop;
    };

    int n_restype;
    vector<typename BT::SidechainInteraction> interactions;
    map<string,int> index_from_restype;

    vector<RotamerIndices> rotamer_indices;  // start and stop

    vector<string> sequence;
    vector<int>    restype;

    SymmetricInteractionGraph<BT> igraph;
    int             max_edges11, max_edges13, max_edges33;
    vector<int>     n_edge11, n_edge13, n_edge33;
    vector<PairInteraction<1,1,BT>> edges11;
    vector<PairInteraction<1,3,BT>> edges13;
    vector<PairInteraction<3,3,BT>> edges33;

    vector<int> edge_indices;
    SysArrayStorage node_prob, edge_prob;
    SysArrayStorage node_belief, edge_belief, temp_node_belief, temp_edge_belief;

    SysArrayStorage s_residue_energy1;
    SysArrayStorage s_residue_energy3;

    SysArrayStorage node_marginal_prob, edge_marginal_prob;
    vector<int>     fixed_rotamers3;

    float damping;
    int   max_iter;
    float tol;

    float scale_final_energy FIXME implement in derivative;

    bool energy_fresh_relative_to_derivative;
    int n_res_all;
    map<int, vector<ResidueLoc>> local_loc;

    RotamerSidechain(hid_t grp, CoordNode& pos_node_):
        PotentialNode(pos_node_.n_system),
        igraph(open_group(grp,"interaction").get(), pos_node_)
        rotamer_indices(n_restype),

        n_edge11(n_system), n_edge13(n_system), n_edge33(n_system),

        damping (read_attribute<float>(grp, ".", "damping")),
        max_iter(read_attribute<int  >(grp, ".", "max_iter")),
        tol     (read_attribute<float>(grp, ".", "tol")),
        scale_final_energy(read_attribute<float>(grp, ".", "scale_final_energy")),

        energy_fresh_relative_to_derivative(false)

    {
        if(h5_exists(grp, "fixed_rotamers")) {
            // FIXME this needs updating
            check_size(grp, "fixed_rotamers", sequence.size());
            traverse_dset<1,int>(grp, "fixed_rotamers", [&](size_t nr, int no) {
                    int rt = restype[nr];
                    int n_rot = rotamer_indices[rt].stop - rotamer_indices[rt].start;
                    if(!(0<=no && no<n_rot)) throw string("Invalid fixed_rotamers");
                    if(n_rot==3) fixed_rotamers3.push_back(no);});
        }

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

        node_marginal_prob.reset(n_system, 3,   placement3->n_res);
        edge_marginal_prob.reset(n_system, 3*3, max_edges33);

        s_residue_energy1.reset(n_system, 1, placement1->n_res);
        s_residue_energy3.reset(n_system, 2, placement3->n_res);

        n_res_all = placement1->n_res + placement3->n_res;

        auto &p1 = *placement1;
        auto &p3 = *placement3;

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("rotamer_potential_entropy", {n_system, n_res_all, 2}, [&](float* buffer) {
                    this->ensure_fresh_energy();
                    this->calculate_per_residue_energies();

                    for(int ns: range(n_system)) {
                        VecArray residue_energy1 = s_residue_energy1[ns];
                        VecArray residue_energy3 = s_residue_energy3[ns];
                        
                        // copy into buffer
                        for(int nr1: range(p1.n_res)) {
                            buffer[ns*n_res_all*2 + local_loc[1][nr1].affine_idx.index*2 + 0] = residue_energy1(0,nr1);
                            buffer[ns*n_res_all*2 + local_loc[1][nr1].affine_idx.index*2 + 1] = 0.f; // no 1-state entropy
                        }
                        for(int nr3: range(p3.n_res)) {
                            buffer[ns*n_res_all*2 + local_loc[3][nr3].affine_idx.index*2 + 0] = residue_energy3(0,nr3);
                            buffer[ns*n_res_all*2 + local_loc[3][nr3].affine_idx.index*2 + 1] = residue_energy3(1,nr3);
                        }
                    }});

            default_logger->add_logger<float>("node_marginal_prob", {n_system,p3.n_res,3}, [&](float* buffer) {
                    for(int ns:range(n_system))
                        for(int nn: range(p3.n_res)) 
                            for(int no: range(3))
                                buffer[(ns*p3.n_res + nn)*3 + no] = node_marginal_prob[ns](no,nn);});
        }



        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("edge_marginal_prob", {n_system,max_edges33,3,3}, [&](float* buffer) {
                    for(int ns:range(n_system))
                        for(int ne: range(max_edges33))
                            for(int no1: range(3))
                                for(int no2: range(3))
                                    buffer[((ns*max_edges33 + ne)*3 + no1)*3 + no2] = (ne<n_edge33[ns])
                                        ? edge_marginal_prob[ns](no1*3+no2,ne)
                                        : 0.f;});
        }
    }

    void ensure_fresh_energy() {
        if(!energy_fresh_relative_to_derivative) compute_value(PotentialAndDerivMode);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer("rotamer");  // Timer code is not thread-safe, so cannot be used within parallel for
        energy_fresh_relative_to_derivative = mode==PotentialAndDerivMode;

        auto &p1 = *placement1;
        auto &p3 = *placement3;

        #pragma omp parallel for schedule(dynamic)
        for(int ns=0; ns<n_system; ++ns) {
            compute_all_graph_elements(
                    n_edge11[ns], edges11.data() + ns*max_edges11,
                    n_edge13[ns], edges13.data() + ns*max_edges13,
                    n_edge33[ns], edges33.data() + ns*max_edges33,
                    p1.n_res, p1.global_restype.data(), p1.pos[ns],
                    p3.n_res, p3.global_restype.data(), p3.pos[ns],
                    n_restype, interactions.data());

            convert_potential_graph_to_probability_graph(
                    node_prob[ns], edge_prob[ns], edge_indices.data() + ns*2*max_edges33,
                    p3.n_res, p3.pos[ns],  // last dimension is 1-residue potential
                    n_edge13[ns], edges13.data()+ns*max_edges13,
                    n_edge33[ns], edges33.data()+ns*max_edges33);

            if(!fixed_rotamers3.size()) {
                auto result = solve_for_beliefs(
                        node_belief[ns], edge_belief[ns], 
                        temp_node_belief[ns], temp_edge_belief[ns],
                        p3.n_res, node_prob[ns],
                        n_edge33[ns], edge_prob[ns], edge_indices.data() + ns*2*max_edges33,
                        damping, max_iter, tol, true); // do re-initialize beliefs

                if(result.first >= max_iter-1) 
                    printf("%2i solved in %i iterations with error of %f\n", ns, result.first, result.second);
            } else {
                // 0,1 beliefs are equivalent to fixed rotamers
                // just populate the beliefs with certainties
                fill(node_belief[ns], 3, p3.n_res, 0.f);
                for(int nn: range(p3.n_res))
                    node_belief[ns](fixed_rotamers3[nn], nn) = 1.f;

                fill(edge_belief[ns], 2*3, n_edge33[ns], 0.f);
                for(int ne: range(n_edge33[ns])) {
                    int nr1 = edge_indices[ns*2*max_edges33+ne*2+0];
                    int nr2 = edge_indices[ns*2*max_edges33+ne*2+1];
                    edge_belief[ns](fixed_rotamers3[nr1]  ,ne) = 1.f;
                    edge_belief[ns](fixed_rotamers3[nr2]+3,ne) = 1.f;
                }
            }

            compute_free_energy_and_derivative(
                    (mode==PotentialAndDerivMode ? potential.data()+ns : nullptr), 
                    node_marginal_prob[ns], edge_marginal_prob[ns],
                    p3.pos[ns],
                    p1.pos_deriv[ns], p3.pos_deriv[ns],
                    p1.n_res,         p3.n_res,
                    node_belief[ns], edge_belief[ns],
                    edge_prob[ns],
                    n_edge11[ns], edges11.data() + ns*max_edges11,
                    n_edge13[ns], edges13.data() + ns*max_edges13,
                    n_edge33[ns], edges33.data() + ns*max_edges33);

            if(mode==PotentialAndDerivMode) potential[ns] *= scale_final_energy;

            p1.push_derivatives(alignment.coords().value[ns],alignment.coords().deriv[ns],rama.coords().deriv[ns],ns,scale_final_energy);
            p3.push_derivatives(alignment.coords().value[ns],alignment.coords().deriv[ns],rama.coords().deriv[ns],ns,scale_final_energy);
        }
    }

    void calculate_per_residue_energies() {
        auto &p1 = *placement1;
        auto &p3 = *placement3;

        #pragma omp parallel for schedule(static,1)
        for(int ns=0; ns<n_system; ++ns) {
            VecArray residue_energy1 = s_residue_energy1[ns];
            VecArray residue_energy3 = s_residue_energy3[ns];
            fill(residue_energy1, 1, p1.n_res, 0.f);

            for(int nr: range(p3.n_res)) {
                auto b = load_vec<3>(node_marginal_prob[ns], nr);
                auto vs = make_vec2(0.f,0.f);
                for(int no: range(3)) {vs[0] += b[no]*p3.pos[ns](no*BT::n_pos_dim+3,nr); vs[1] += -b[no]*logf(1e-10f+b[no]);}
                store_vec(residue_energy3, nr, vs);
            }

            for(int ne11: range(n_edge11[ns])) {
                auto &e = edges11[ns*max_edges11+ne11];
                update_vec(residue_energy1, e.nr1, 0.5f*make_vec1(e.potential[0][0]));
                update_vec(residue_energy1, e.nr2, 0.5f*make_vec1(e.potential[0][0]));
            }

            for(int ne13: range(n_edge13[ns])) {
                auto &e = edges13[ns*max_edges13+ne13];
                Vec<3> b = load_vec<3>(node_marginal_prob[ns], e.nr2);
                auto v = make_vec1(0.f);  // 1-body entropies were already handled
                for(int no2: range(3)) v[0] += b[no2]*e.potential[0][no2];
                update_vec(residue_energy1, e.nr1, 0.5f*v);
                update_vec(residue_energy3, e.nr2, 0.5f*v);
            }

            for(int ne33: range(n_edge33[ns])) {
                auto &e = edges33[ns*max_edges33+ne33];
                Vec<9> bp = load_vec<9>(edge_marginal_prob[ns], ne33);
                Vec<3> b1 = load_vec<3>(node_marginal_prob[ns], e.nr1);
                Vec<3> b2 = load_vec<3>(node_marginal_prob[ns], e.nr2);
                auto vs = make_vec2(0.f,0.f);
                for(int no1: range(3)) for(int no2: range(3)) {
                    int i = no1*3+no2;
                    vs[0] += bp[i]*e.potential[no1][no2];
                    vs[1] +=-bp[i]*(logf((1e-10f+bp[i])*rcp((1e-10f+b1[no1]*b2[no2]))));
                }
                update_vec(residue_energy3, e.nr1, 0.5f*vs);
                update_vec(residue_energy3, e.nr2, 0.5f*vs);
            }
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

        double rama_dev   = compute_relative_deviation_for_node<2>(*this, rama, coord_pairs_rama);

        return rama_dev;
    }
};
static RegisterNodeType<RotamerSidechain<preferred_bead_type>,2> rotamer_node ("rotamer");


struct RotamerConstructAndSolve {
    typedef preferred_bead_type BT;

    int n_restype;
    vector<BT::SidechainInteraction> interactions;

    vector<int>     restype1,   restype3;
    int             n_res1,     n_res3;
    SysArrayStorage pos1,       pos3;
    SysArrayStorage pos_deriv1, pos_deriv3;

    int n_edge11, n_edge13, n_edge33;
    vector<PairInteraction<1,1,BT>> edges11;
    vector<PairInteraction<1,3,BT>> edges13;
    vector<PairInteraction<3,3,BT>> edges33;

    vector<int> edge_indices;

    SysArrayStorage node_prob, edge_prob;
    SysArrayStorage node_belief, edge_belief, temp_node_belief, temp_edge_belief;

    SysArrayStorage node_marginal_prob, edge_marginal_prob;
    vector<int>     fixed_rotamers3;

    float damping;
    int   max_iter;
    float tol;

    float free_energy_and_parameter_deriv(float* parameter_deriv, const float* interactions_) {
        for(int i: range(n_restype*n_restype)) {
            for(int ip: range(BT::n_param))
                interactions[i].params[ip] = interactions_[BT::n_param*i+ip];
            interactions[i].update_cutoff2();
        }

        compute_all_graph_elements(
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data(),
                n_res1, restype1.data(), pos1[0],
                n_res3, restype3.data(), pos3[0],
                n_restype, interactions.data());

        convert_potential_graph_to_probability_graph(
                node_prob[0], edge_prob[0], edge_indices.data(),
                n_res3, pos3[0],  // last dimension is 1-residue potential
                n_edge13, edges13.data(),
                n_edge33, edges33.data());

        if(!fixed_rotamers3.size()) {
            auto result = solve_for_beliefs(
                    node_belief[0], edge_belief[0], 
                    temp_node_belief[0], temp_edge_belief[0],
                    n_res3, node_prob[0],
                    n_edge33, edge_prob[0], edge_indices.data(),
                    damping, max_iter, tol, true); // do re-initialize beliefs

            if(result.first >= max_iter-1) 
                fprintf(stderr,"solved in %i iterations with error of %f\n", result.first, result.second);
        } else {
            // 0,1 beliefs are equivalent to fixed rotamers
            // just populate the beliefs with certainties
            fill(node_belief[0], 3, n_res3, 0.f);
            for(int nn: range(n_res3))
                node_belief[0](fixed_rotamers3[nn], nn) = 1.f;

            fill(edge_belief[0], 2*3, n_edge33, 0.f);
            for(int ne: range(n_edge33)) {
                int nr1 = edge_indices[ne*2+0];
                int nr2 = edge_indices[ne*2+1];
                edge_belief[0](fixed_rotamers3[nr1]  ,ne) = 1.f;
                edge_belief[0](fixed_rotamers3[nr2]+3,ne) = 1.f;
            }
        }

        float free_energy = 0.f;
        compute_free_energy_and_derivative(
                &free_energy,
                node_marginal_prob[0], edge_marginal_prob[0],
                pos3[0],
                pos_deriv1[0], pos_deriv3[0],
                n_res1,        n_res3,
                node_belief[0], edge_belief[0],
                edge_prob[0],
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data());

        compute_parameter_derivatives(
                parameter_deriv, 
                node_marginal_prob[0], edge_marginal_prob[0],
                n_edge11, edges11.data(),
                n_edge13, edges13.data(),
                n_edge33, edges33.data(),
                n_res1, restype1.data(), pos1[0],  // dimensionality 1*BT::n_pos_dim
                n_res3, restype3.data(), pos3[0],  // dimensionality 3*BT::n_pos_dim
                n_restype, interactions.data());

        return free_energy;
    }
};

// C-friendly interface, so we can connect to python
RotamerConstructAndSolve* new_rotamer_construct_and_solve(
             int n_restype_, 
             int n_res1_, int* restype1_, float* pos1_,
             int n_res3_, int* restype3_, float* pos3_,
             float damping_, int max_iter_, float tol_,
             int* fixed_rotamers3_) {
    typedef preferred_bead_type BT;

    auto rcas = new RotamerConstructAndSolve;
    auto &z = *rcas;

    z.n_restype = n_restype_;
    z.interactions.resize(z.n_restype*z.n_restype);

    z.n_res1 = n_res1_; z.restype1 = vector<int>(restype1_, restype1_+z.n_res1);
    z.n_res3 = n_res3_; z.restype3 = vector<int>(restype3_, restype3_+z.n_res3);

    z.pos1.reset(1,1*BT::n_pos_dim,z.n_res1); z.pos_deriv1.reset(1,1*BT::n_pos_dim,z.n_res1);
    z.pos3.reset(1,3*BT::n_pos_dim,z.n_res3); z.pos_deriv3.reset(1,3*BT::n_pos_dim,z.n_res3);

    for(int nr: range(z.n_res1)) for(int d: range(  BT::n_pos_dim)) z.pos1[0](d,nr) = pos1_[nr*BT::n_pos_dim+d];
    for(int nr: range(z.n_res3)) for(int d: range(3*BT::n_pos_dim)) z.pos3[0](d,nr) = pos3_[nr*3*BT::n_pos_dim+d];

    int max_edges11 = z.n_res1 * (z.n_res1-1) / 2; z.edges11.resize(max_edges11);
    int max_edges13 = z.n_res1 *  z.n_res3;        z.edges13.resize(max_edges13);
    int max_edges33 = z.n_res3 * (z.n_res3-1) / 2; z.edges33.resize(max_edges33);

    z.edge_indices.resize(2*max_edges33);
    z.node_belief     .reset(1,3,z.n_res3); z.     edge_belief.reset(1,2*3,max_edges33);
    z.temp_node_belief.reset(1,3,z.n_res3); z.temp_edge_belief.reset(1,2*3,max_edges33);

    z.node_marginal_prob.reset(1,3,z.n_res3); z.edge_marginal_prob.reset(1,3*3,max_edges33);

    if(fixed_rotamers3_) {
        z.fixed_rotamers3 = vector<int>(fixed_rotamers3_,fixed_rotamers3_+z.n_res3);
        for(auto i: range(z.fixed_rotamers3.size())) {
            if(z.fixed_rotamers3[i]<0 || z.fixed_rotamers3[i]>2 ) {
                fprintf(stderr,"failure at rotamer %i %i\n",
                        i,z.fixed_rotamers3[i]);
                throw 0;
            }
        }
    }

    z.damping = damping_;
    z.max_iter = max_iter_;
    z.tol = tol_;
    
    return rcas;
}


float free_energy_and_parameter_deriv(RotamerConstructAndSolve* rcas, 
        float* parameter_deriv, const float* interactions) {
    return rcas->free_energy_and_parameter_deriv(parameter_deriv, interactions);
}


void delete_rotamer_construct_and_solve(RotamerConstructAndSolve* rcas) {
    delete rcas;
}


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
