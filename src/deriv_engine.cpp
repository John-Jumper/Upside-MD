#include "deriv_engine.h"
#include "timing.h"
#include <map>
#include <algorithm>
#include <memory>

using namespace h5;

using namespace std;

void
integration_stage(
        VecArray mom,
        VecArray pos,
        const VecArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom)
{
    for(int na=0; na<n_atom; ++na) {
        // assumes unit mass for all particles

        auto d = load_vec<3>(deriv, na);
        if(max_force) {
            float f_mag = mag(d)+1e-6f;  // ensure no NaN when mag(deriv)==0.
            float scale_factor = atan(f_mag * ((0.5f*M_PI_F) / max_force)) * (max_force/f_mag * (2.f/M_PI_F));
            d *= scale_factor;
        }

        auto p = load_vec<3>(mom, na) - vel_factor*d;
        store_vec (mom, na, p);
        update_vec(pos, na, pos_factor*p);
    }
}

void
recenter(VecArray pos, bool xy_recenter_only, int n_atom)
{
    float3 center = make_vec3(0.f, 0.f, 0.f);
    for(int na=0; na<n_atom; ++na) center += load_vec<3>(pos,na);
    center /= float(n_atom);

    if(xy_recenter_only) center.z() = 0.f;

    for(int na=0; na<n_atom; ++na)
        update_vec(pos, na, -center);
}

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

void check_elem_width_lower_bound(const CoordNode& node, int elem_width_lower_bound) {
    if(node.elem_width < elem_width_lower_bound) 
        throw std::string("expected argument with width at least ") + std::to_string(elem_width_lower_bound) + 
            " but received argument with width " + std::to_string(node.elem_width);
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

void DerivEngine::add_node(
        const string& name, 
        unique_ptr<DerivComputation> fcn, 
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

void DerivEngine::compute(ComputeMode mode) {
    // FIXME depth-first traversal would be simpler and more cache-friendly
    for(auto& n: nodes) n.germ_exec_level = n.deriv_exec_level = -1;

    if(mode == PotentialAndDerivMode) potential = 0.f;

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
                    n.computation->compute_value(mode);
                    n.germ_exec_level = lvl;
                    if(mode == PotentialAndDerivMode && n.computation->potential_term) {
                        auto pot_node = static_cast<PotentialNode*>(n.computation.get());
                        potential += pot_node->potential;
                    }
                    if(!n.computation->potential_term) {
                        // ensure zero sensitivity for later derivative writing
                        CoordNode* coord_node = static_cast<CoordNode*>(n.computation.get());
                        fill(coord_node->sens, 0.f);
                    }
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


void DerivEngine::integration_cycle(VecArray mom, float dt, float max_force, IntegratorType type) {
    // integrator from Predescu et al., 2012
    // http://dx.doi.org/10.1080/00268976.2012.681311

    float a = (type==Predescu) ? 0.108991425403425322 : 1./6.;
    float b = (type==Predescu) ? 0.290485609075128726 : 1./3.;

    float mom_update[] = {1.5f-3.f*a, 1.5f-3.f*a, 6.f*a};
    float pos_update[] = {     3.f*b, 3.0f-6.f*b, 3.f*b};

    for(int stage=0; stage<3; ++stage) {
        compute(DerivMode);   // compute derivatives
        Timer timer(string("integration"));
        integration_stage( 
                mom,
                pos->output,
                pos->sens,
                dt*mom_update[stage], dt*pos_update[stage], max_force, 
                pos->n_atom);
    }
}


DerivEngine initialize_engine_from_hdf5(int n_atom, hid_t potential_group, bool quiet)
{
    DerivEngine engine(n_atom);
    auto& m = node_creation_map();

    map<string, pair<bool,vector<string>>> dep_graph;  // bool indicates node is active
    dep_graph["pos"] = make_pair(true, vector<string>());
    for(const auto &name : node_names_in_group(potential_group, "."))
        dep_graph[name] = make_pair(true, read_attribute<vector<string>>(potential_group, name.c_str(), "arguments"));

    for(auto &kv : dep_graph) {
        for(auto& dep_name : kv.second.second) {
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
            if(!it->second.first) continue;
            if(all_of(begin(it->second.second), end(it->second.second), in_topo)) {
                topo_order.push_back(it->first);
                it->second.first = false;  // make node inactive
            }
        }
    }
    for(auto &kv : dep_graph) if(kv.second.first) 
        throw string("Unsatisfiable dependency ") + kv.first + " in potential computation";

    // using topo_order here ensures that a node is only parsed after all its arguments
    for(auto &nm : topo_order) {
        if(!quiet)  printf("initializing %-27s%s", nm.c_str(), nm=="pos" ? "\n" : ""); 
        if(nm=="pos") continue;  // pos node is added specially
        // some name in the node_creation_map must be a prefix of this name
        string node_type_name = "";
        for(auto &kv : m) {
            if(is_prefix(kv.first, nm))
                node_type_name = kv.first;
        }
        if(node_type_name == "") throw string("No node type found for name '") + nm + "'";
        NodeCreationFunction& node_func = m[node_type_name];

        auto argument_names = read_attribute<vector<string>>(potential_group, nm.c_str(), "arguments");
        ArgList arguments;
        for(const auto& arg_name : argument_names)  {
            // if the node is not a CoordNode, a null pointer will be returned from dynamic_cast
            arguments.push_back(dynamic_cast<CoordNode*>(engine.get(arg_name).computation.get()));
            if(!arguments.back()) 
                throw arg_name + " is not an intermediate value, but it is an argument of " + nm;
        }
        if(!quiet) {
            printf(" with %sargument%s ", 
                    argument_names.size() == 0 ? "no " : "",
                    argument_names.size() == 1 ? " "   : "s");
            for(size_t na=0; na<argument_names.size(); ++na) 
                printf("%s%s", na ? " and " : "", argument_names[na].c_str());
            printf("\n");
        }

        try {
            auto grp = open_group(potential_group,nm.c_str());
            auto computation = unique_ptr<DerivComputation>(node_func(grp.get(),arguments));
            engine.add_node(nm, move(computation), argument_names);
        } catch(const string &e) {
            throw "while adding '" + nm + "', " + e;
        }
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
    return engine.get_idx("hbond_energy",false) == -1 
        ? -1.f
        : engine.get_computation<HBondCounter>("hbond_energy").n_hbond;
}


vector<float> central_difference_deriviative(
        const function<void()>& compute_value, vector<float>& input, vector<float>& output, float eps,
        ValueType value_type) 
{
    // FIXME only handles single systems
    auto old_input = input;
    compute_value();
    auto output_minus_eps = output;

    vector<float> jacobian(input.size()*output.size(), -10101.f);

    for(unsigned ni=0; ni<input.size(); ++ni) {
        copy(begin(old_input), end(old_input), begin(input));

        input[ni] = old_input[ni] - eps; 
        compute_value();
        copy(begin(output), end(output), begin(output_minus_eps));

        input[ni] = old_input[ni] + eps; 
        compute_value();

        if(value_type == BODY_VALUE) {
            if(output.size() % 7) throw "impossible";
            for(unsigned no=0; no<output.size(); no+=7){
                const float* o = &output[no+3];
                float* ome = &output_minus_eps[no+3];

                float4 qo   = make_vec4(o  [0],o  [1],o  [2],o  [3]);
                float4 qome = make_vec4(ome[0],ome[1],ome[2],ome[3]);

                // resolve whether q or -q is closer
                if(mag2(qo+qome) < mag2(qo-qome)) for(int d=0; d<4; ++d) ome[d] *= -1.f;
            }
        }

        for(unsigned no=0; no<output.size(); ++no){
            float diff = output[no]-output_minus_eps[no];
            if(value_type == ANGULAR_VALUE) {
                //printf("diff %f\n", diff/M_PI_F*180.f);
                if(diff> M_PI_F) diff -= 2.f*M_PI_F;
                if(diff<-M_PI_F) diff += 2.f*M_PI_F;
            }
            jacobian[no*input.size() + ni] = diff * (0.5f/eps);
        }
    }

    // restore input and recompute the output so the caller is not surprised
    copy(begin(old_input), end(old_input), begin(input));  
    compute_value();

    return jacobian;
}
