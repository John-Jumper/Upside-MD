#include "force.h"
#include "md_export.h"
#include "coord.h"
#include "timing.h"
#include <map>
#include <algorithm>

using namespace h5;

using namespace std;

void add_node_creation_function(std::string name_prefix, NodeCreationFunction fcn)
{
    auto& m = node_creation_map();

    // No string in m can be a prefix of any other string in m, since 
    //   the function node to call is determined by checking string prefixes
    for(const auto& kv : m) {
        if(is_prefix(kv.first, name_prefix)) {
            auto s = std::string("Internal error.  Type name ") + kv.first + " is a prefix of " + name_prefix + ".";
            fprintf(stderr, "%s\n", s.c_str());
            throw s;
        }
        if(is_prefix(name_prefix, kv.first)) {
            auto s = std::string("Internal error.  Type name ") + name_prefix + " is a prefix of " + kv.first + ".";
            fprintf(stderr, "%s\n", s.c_str());
            throw s;
        }
    }

    m[name_prefix] = fcn;
}

bool is_prefix(const std::string& s1, const std::string& s2) {
    return s1 == s2.substr(0,s1.size());
}

void check_elem_width(const CoordNode& node, int expected_elem_width) {
    if(node.elem_width != expected_elem_width) 
        throw std::string("expected argument with width ") + std::to_string(expected_elem_width) + 
            " but received argument with width " + std::to_string(node.elem_width);
}

void check_arguments_length(const ArgList& arguments, int n_expected) {
    if(int(arguments.size()) != n_expected) 
        throw std::string("expected ") + std::to_string(n_expected) + 
            " arguments but got " + std::to_string(arguments.size());
}

void Pos::propagate_deriv() {
    Timer timer(string("pos_deriv"));
    deriv_accumulation(SysArray(deriv.data(),n_atom*3), 
            slot_machine.accum_array(), slot_machine.deriv_tape.data(), 
            slot_machine.deriv_tape.size(), n_atom, n_system);
}

void DerivEngine::add_node(
        const string& name, 
        unique_ptr<DerivComputation>&& fcn, 
        vector<string> argument_names) 
{
    if(any_of(nodes.begin(), nodes.end(), [&](const Node& n) {return n.name==name;})) 
        throw string("name conflict in DerivEngine");

    nodes.emplace_back(name, move(fcn));
    auto& node = nodes.back();

    for(auto& nm: argument_names) { 
        int parent_idx = get_idx(nm);
        node.parents.push_back(parent_idx);
        nodes[parent_idx].children.push_back(nodes.size()-1);
    }
}

DerivEngine::Node& DerivEngine::get(const string& name) {
    auto loc = find_if(begin(nodes), end(nodes), [&](const Node& n) {return n.name==name;});
    if(loc == nodes.end()) throw string("name not found");
    return *loc;
}

int DerivEngine::get_idx(const string& name, bool must_exist) {
    auto loc = find_if(begin(nodes), end(nodes), [&](const Node& n) {return n.name==name;});
    if(must_exist && loc == nodes.end()) throw string("name not found");
    return loc != nodes.end() ? loc-begin(nodes) : -1;
}

void DerivEngine::compute() {
    // FIXME add CUDA streams support
    // for each BFS depth, each computation gets the stream ID of its parent
    // if the parents have multiple streams or another (sibling) computation has already taken the stream id,
    //   the computation gets a new stream ID and a synchronization wait event
    // if the node has multiple children, it must record a compute_germ event for synchronization of the children
    // the compute_deriv must run in the same stream as compute_germ
    // if a node has multiple parents, it must record an event for compute_deriv
    // pos_node is special since its compute_germ is empty

    for(auto& n: nodes) n.germ_exec_level = n.deriv_exec_level = -1;

    // BFS traversal
    for(int lvl=0, not_finished=1; ; ++lvl, not_finished=0) {
        for(auto& n: nodes) {
            if(n.germ_exec_level == -1) {
                not_finished = 1;
                bool all_parents = all_of(begin(n.parents), end(n.parents), [&] (int ip) {
                        int exec_lvl = nodes[ip].germ_exec_level;
                        return exec_lvl!=-1 && exec_lvl!=lvl; // do not execute at same level as your parents
                        });
                if(all_parents) {
                    n.computation->compute_germ();
                    n.germ_exec_level = lvl;
                }
            }

            if(n.deriv_exec_level == -1 && n.germ_exec_level != -1) {
                not_finished = 1;
                bool all_children = all_of(begin(n.children), end(n.children), [&] (int ip) {
                        int exec_lvl = nodes[ip].deriv_exec_level;
                        return exec_lvl!=-1 && exec_lvl!=lvl; // do not execute at same level as your children
                        });
                if(all_children) {
                    n.computation->propagate_deriv();
                    n.deriv_exec_level = lvl;
                }
            }
        }
        if(!not_finished) break;
    }
}


void DerivEngine::integration_cycle(float* mom, float dt, float max_force, IntegratorType type) {
    // integrator from Predescu et al., 2012
    // http://dx.doi.org/10.1080/00268976.2012.681311

    float a = (type==Predescu) ? 0.108991425403425322 : 1./6.;
    float b = (type==Predescu) ? 0.290485609075128726 : 1./3.;

    float mom_update[] = {1.5f-3.f*a, 1.5f-3.f*a, 6.f*a};
    float pos_update[] = {     3.f*b, 3.0f-6.f*b, 3.f*b};

    for(int stage=0; stage<3; ++stage) {
        compute();   // compute derivatives
        Timer timer(string("integration"));
        integration_stage( 
                SysArray(mom,                pos->n_atom*3), 
                SysArray(pos->output.data(), pos->n_atom*3), 
                SysArray(pos->deriv.data(),  pos->n_atom*3),
                dt*mom_update[stage], dt*pos_update[stage], max_force, 
                pos->n_atom, pos->n_system);
    }
}

template <int ndim>
struct ComputeMyDeriv {
    const float* accum;
    vector<StaticCoord<ndim>> deriv;

    ComputeMyDeriv(const std::vector<float> &accum_, int n_atom):
        accum(accum_.data()), deriv(n_atom)
    {zero();}

    void add_pair(const CoordPair &cp) { deriv.at(cp.index) += StaticCoord<ndim>(accum, cp.slot); }
    void zero() { for(auto &a: deriv) for(int i=0; i<ndim; ++i) a.v[i] = 0.f; }
    void print(const char* nm) { 
        for(int nr=0; nr<(int)deriv.size(); ++nr) {
            printf("%-12s %2i", nm,nr);
            for(int i=0; i<ndim; ++i) {
                if(i%3==0) printf(" ");
                printf(" % 6.2f", deriv[nr].v[i]);
            }
            printf("\n");
        }
        printf("\n");
    }
};



template <typename ComputationT, typename ArgumentT> 
void attempt_add_node(
        DerivEngine& engine,
        hid_t force_group,
        const string  name,
        const string  argument_name,
        const string* table_name_if_different = nullptr) 
{
    auto table_name = table_name_if_different ? *table_name_if_different : name;
    if(!h5_noerr(H5LTpath_valid(force_group, table_name.c_str(), 1))) return;
    auto grp = h5_obj(H5Gclose, H5Gopen2(force_group, table_name.c_str(), H5P_DEFAULT));
    printf("initializing %s\n", name.c_str());

    // note that the unique_ptr holds the object by a superclass pointer
    try {
        auto& argument = dynamic_cast<ArgumentT&>(*engine.get(argument_name).computation);
        auto computation = unique_ptr<DerivComputation>(new ComputationT(grp.get(), argument));
        engine.add_node(name, move(computation), {argument_name});
    } catch(const string &e) {
        throw "while adding '" + name + "', " + e;
    }
}

template <typename ComputationT, typename Argument1T, typename Argument2T> 
void attempt_add_node(
        DerivEngine& engine,
        hid_t force_group,
        const string  name,
        const string  argument_name1,
        const string  argument_name2,
        const string* table_name_if_different = nullptr) 
{
    auto table_name = table_name_if_different ? *table_name_if_different : name;
    if(!h5_noerr(H5LTpath_valid(force_group, table_name.c_str(), 1))) return;
    auto grp = h5_obj(H5Gclose, H5Gopen2(force_group, table_name.c_str(), H5P_DEFAULT));
    printf("initializing %s\n", name.c_str());

    // note that the unique_ptr holds the object by a superclass pointer
    try {
        auto& argument1 = dynamic_cast<Argument1T&>(*engine.get(argument_name1).computation);
        auto& argument2 = dynamic_cast<Argument2T&>(*engine.get(argument_name2).computation);
        auto computation = unique_ptr<DerivComputation>(new ComputationT(grp.get(), argument1, argument2));

        engine.add_node(name, move(computation), {argument_name1, argument_name2});
    } catch(const string &e) {
        throw "while adding '" + name + "', " + e;
    }
}


DerivEngine initialize_engine_from_hdf5(int n_atom, int n_system, hid_t force_group)
{
    auto engine = DerivEngine(n_atom, n_system);
    auto& m = node_creation_map();

    map<string, vector<string>> dep_graph;
    dep_graph["pos"] = vector<string>();
    for(const auto &name : node_names_in_group(force_group, "."))
        dep_graph[name] = read_attribute<vector<string>>(force_group, name.c_str(), "arguments");

    for(auto &kv : dep_graph) {
        for(auto& dep_name : kv.second) {
            if(dep_graph.find(dep_name) == end(dep_graph)) 
                throw string("Node ") + kv.first + " takes " + dep_name + 
                    " as an argument, but no node of that name can be found.";
        }
    }

    vector<string> topo_order;
    auto in_topo = [&](const string &name) {
        return find(begin(topo_order), end(topo_order), name) != end(topo_order);};

    int graph_size = dep_graph.size();
    for(int round_num=0; round_num<graph_size; ++round_num) {
        for(auto it=begin(dep_graph); it!=end(dep_graph); ++it) {
            if(all_of(begin(it->second), end(it->second), in_topo)) {
                topo_order.push_back(it->first);
                dep_graph.erase(it);
            }
        }
    }
    if(dep_graph.size()) throw string("Unsatisfiable dependency in force computation");

    // using topo_order here ensures that a node is only parsed after all its arguments
    for(auto &nm : topo_order) {
        if(nm=="pos") continue;  // pos node is added specially
        // some name in the node_creation_map must be a prefix of this name
        string node_type_name = "";
        for(auto &kv : m) {
            if(is_prefix(kv.first, nm))
                node_type_name = kv.first;
        }
        if(node_type_name == "") throw string("No node type found for name '") + nm + "'";
        NodeCreationFunction& node_func = m[node_type_name];

        auto argument_names = read_attribute<vector<string>>(force_group, nm.c_str(), "arguments");
        ArgList arguments;
        for(const auto& arg_name : argument_names)  {
            // if the node is not a CoordNode, a null pointer will be returned from dynamic_cast
            arguments.push_back(dynamic_cast<CoordNode*>(engine.get(arg_name).computation.get()));
            if(!arguments.back()) 
                throw arg_name + " is not an intermediate value, but it is an argument of " + nm;
        }

        auto grp = open_group(force_group,nm.c_str());
        auto computation = unique_ptr<DerivComputation>(node_func(grp.get(),arguments));
        engine.add_node(nm, move(computation), argument_names);
    }

    return engine;
}

NodeCreationMap& node_creation_map() 
{
    static NodeCreationMap m;
    if(!m.size()) {
        m[string("pos")] = NodeCreationFunction([](hid_t grp, const ArgList& arguments) {
                throw string("Cannot create pos-type node");
                return nullptr; });
    }
    return m;
}


double get_n_hbond(DerivEngine &engine) {
    HBondCounter* hbond_comp = engine.get_idx("hbond_energy",false)==-1 ? 
        nullptr : 
        &engine.get_computation<HBondCounter>("hbond_energy");
    return hbond_comp?hbond_comp->n_hbond:-1.f;
}
