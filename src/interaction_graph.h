#ifndef INTERACTION_GRAPH_H
#define INTERACTION_GRAPH_H

#include "deriv_engine.h"
#include "vector_math.h"
#include "h5_support.h"

template <typename IType, bool compute_param_deriv = false>
struct SymmetricInteractionGraph {
    CoordNode &pos_node;
    int n_system;
    int n_type;
    int n_elem;
    int max_n_edge;

    std::vector<CoordPair> param;

    // FIXME group these values as both must be read whenever either is read
    std::vector<unsigned> types;   // pair type is type[0]*n_types2 + type[1]
    std::vector<unsigned> id;  // used to avoid self-interaction

    // FIXME consider padding for vector loads
    std::vector<Vec<IType::n_param>> interaction_param;
    SysArrayStorage interaction_param_deriv;
    std::vector<float> cutoff2;

    SysArrayStorage edge_deriv;  // size (n_system, n_deriv, max_n_edge)
    SysArrayStorage edge_param_deriv;
    SysArrayStorage pos_deriv;
    std::vector<int> edge_indices;

    std::vector<int> n_edge;

    // It is easiest if I just get my own subgroup of the .h5 file
    SymmetricInteractionGraph(hid_t grp, CoordNode& pos_node_):
        pos_node(pos_node_),
        n_system(pos_node.n_system), 
        n_type(h5::get_dset_size(3,grp,"interaction_param")[0]),
        n_elem(h5::get_dset_size(1,grp,"index")[0]),
        max_n_edge((n_elem*(n_elem-1))/2),
        interaction_param(n_type*n_type),
        interaction_param_deriv(n_system, IType::n_param, (compute_param_deriv ? n_type*n_type : 0)),
        cutoff2(n_type*n_type),
        edge_deriv(n_system, IType::n_deriv, max_n_edge),
        edge_param_deriv(n_system, IType::n_param, (compute_param_deriv ? max_n_edge : 0)),
        pos_deriv (n_system, IType::n_dim,   n_elem),
        edge_indices(n_system*2*max_n_edge),
        n_edge(n_system)
    {
        using namespace h5;

        check_size(grp, "interaction_param", n_type, n_type, IType::n_param);
        traverse_dset<3,float>(grp, "interaction_param", [&](size_t nt1, size_t nt2, size_t np, float x) {
                interaction_param[nt1*n_type+nt2][np] = x;});

        for(int nt1: range(n_type)) {
            for(int nt2: range(n_type)) {
                cutoff2[nt1*n_type+nt2] = sqr(IType::cutoff(interaction_param[nt1*n_type+nt2]));
                if(!IType::is_compatible(interaction_param[nt1*n_type+nt2], 
                                         interaction_param[nt2*n_type+nt1])) {
                    throw std::string("incompatibile parameters");
                }
            }
        }

        check_size(grp, "index", n_elem);
        check_size(grp, "type",  n_elem);
        check_size(grp, "id",    n_elem);

        traverse_dset<1,int>(grp, "index", [&](size_t nr, int x) {param.emplace_back(x,0u);});
        traverse_dset<1,int>(grp, "type",  [&](size_t nr, int x) {types.push_back(x);});
        traverse_dset<1,int>(grp, "id",    [&](size_t nr, int x) {id   .push_back(x);});

        for(auto &p: param) pos_node.slot_machine.add_request(1, p);

    }


    // I need to be hooked to a data source of fixed size
    template <typename F>
    void compute_edges(int ns, F f) {
        VecArray pos = pos_node.coords().value[ns];
        fill(pos_deriv[ns], IType::n_dim, n_elem, 0.f);
        if(compute_param_deriv) fill(interaction_param_deriv[ns], IType::n_param, n_type*n_type, 0.f);

        int ne = 0;
        for(int i1: range(n_elem)) {
            auto index1 = param[i1].index;
            auto coord1 = load_vec<IType::n_dim>(pos, index1);
            auto x1     = extract<0,3>(coord1);
            auto type1  = types[i1];
            auto id1    = id[i1];

            for(int i2: range(i1+1, n_elem)) {
                auto index2 = param[i2].index;
                auto x2 = load_vec<3>(pos, index2);
                auto type2 = types[i2];

                int interaction_type = type1*n_type + type2;

                if(!(mag2(x1-x2) < cutoff2[interaction_type])) continue;

                auto id2 = id[i2];
                if(IType::exclude_by_id(id1,id2)) continue;  // avoid self-interaction in a user defined way
                auto coord2 = load_vec<IType::n_dim>(pos, index2);

                Vec<IType::n_deriv> deriv;
                auto value = IType::compute_edge(deriv, interaction_param[interaction_type], coord1, coord2);

                store_vec(edge_deriv[ns], ne, deriv);
                edge_indices[ns*2*max_n_edge + 2*ne + 0] = i1;
                edge_indices[ns*2*max_n_edge + 2*ne + 1] = i2;

                f(ne, value, index1,type1,id1, index2,type2,id2);

                if(compute_param_deriv) {
                    Vec<IType::n_param> dp;
                    IType::param_deriv(dp, interaction_param[interaction_type], coord1, coord2);
                    store_vec(edge_param_deriv[ns], ne, dp);
                }

                ++ne;
            }
        }
        n_edge[ns] = ne;
    }

    void use_derivative(int ns, int edge_idx, float sensitivity) {
        auto deriv = sensitivity*load_vec<IType::n_deriv>(edge_deriv[ns], edge_idx);
        Vec<IType::n_dim> d1,d2;
        IType::expand_deriv(d1,d2, deriv);
        update_vec(pos_deriv[ns], edge_indices[ns*2*max_n_edge + 2*edge_idx + 0], d1);
        update_vec(pos_deriv[ns], edge_indices[ns*2*max_n_edge + 2*edge_idx + 1], d2);

        if(compute_param_deriv) {
            auto d = sensitivity*load_vec<IType::n_param>(edge_param_deriv[ns], edge_idx);
            auto type1 = types[edge_indices[ns*2*max_n_edge + 2*edge_idx + 0]];
            auto type2 = types[edge_indices[ns*2*max_n_edge + 2*edge_idx + 1]];
            update_vec(interaction_param_deriv[ns], type1*n_type+type2, d);
        }
    }

    void propagate_derivatives(int ns) {
        // Finally put the data where it is needed.
        // This function must be called exactly once after the user has finished calling 
        // use_derivative for the round.

       VecArray pos_accum = pos_node.coords().deriv[ns];
       for(int i: range(n_elem))
           store_vec(pos_accum, param[i].slot, load_vec<IType::n_dim>(pos_deriv[ns],i));
    }
};

#endif
