#ifndef FORCE_H
#define FORCE_H

#include "h5_support.h"
#include "md.h"
#include <vector>
#include <memory>
#include <string>
#include "coord.h"
#include <functional>
#include <map>

struct SlotMachine
{
    const int width;
    const int n_elem;
    const int n_system;
    int offset;

    std::vector<DerivRecord> deriv_tape;
    std::vector<float>       accum;

    SlotMachine(int width_, int n_elem_, int n_system_): 
        width(width_), n_elem(n_elem_), n_system(n_system_), offset(0) {}

    void add_request(int output_width, CoordPair &pair) { 
        DerivRecord prev_record = deriv_tape.size() ? deriv_tape.back() : DerivRecord(-1,0,0);
        deriv_tape.emplace_back(pair.index, prev_record.loc+prev_record.output_width, output_width);
        pair.slot = deriv_tape.back().loc;
        for(int i=0; i<output_width*width*n_system; ++i) accum.push_back(0.f);
        offset += output_width*width;
    }

    SysArray accum_array() { return SysArray(accum.data(), offset); }
};

struct DerivComputation 
{
    virtual void compute_germ()    {};
    virtual void propagate_deriv() {};
};

struct CoordNode : public DerivComputation
{
    int n_system;
    int n_elem;
    int elem_width;
    std::vector<float> output;
    SlotMachine slot_machine;
    CoordNode(int n_system_, int n_elem_, int elem_width_):
        n_system(n_system_), n_elem(n_elem_), elem_width(elem_width_), 
        output(n_system*n_elem*elem_width, 0.f), 
        slot_machine(elem_width, n_elem, n_system) {}
    virtual CoordArray coords() {
        return CoordArray(SysArray(output.data(), n_elem*elem_width), slot_machine.accum_array());
    }
};


struct HBondCounter : public DerivComputation {
    float n_hbond;
    HBondCounter(): n_hbond(-1.f) {};
};


struct Pos : public CoordNode
{
    int n_atom;
    std::vector<float> deriv;

    Pos(int n_atom_, int n_system_):
        CoordNode(n_system_, n_atom_, 3), 
        n_atom(n_atom_), deriv(3*n_atom*n_system, 0.f)
    {}

    virtual void propagate_deriv();
    CoordArray coords() {
        return CoordArray(SysArray(output.data(), n_atom*3), slot_machine.accum_array());
    }
};


struct DerivEngine
{
    struct Node 
    {
        std::string name;
        std::unique_ptr<DerivComputation> computation;
        std::vector<size_t> parents;  // cannot hold pointer to vector contents, so just store index
        std::vector<size_t> children;

        int germ_exec_level;
        int deriv_exec_level;

        Node(std::string name_, std::unique_ptr<DerivComputation>&& computation_):
            name(name_), computation(move(computation_)) {};
    };

    std::vector<Node> nodes;  // nodes[0] is the pos node
    Pos* pos;

    DerivEngine(int n_atom, int n_system)
        {
            nodes.emplace_back("pos", std::unique_ptr<DerivComputation>(new Pos(n_atom, n_system)));
            pos = dynamic_cast<Pos*>(nodes[0].computation.get());
        }

    void add_node(
            const std::string& name, 
            std::unique_ptr<DerivComputation>&& fcn, 
            std::vector<std::string> argument_names);

    Node& get(const std::string& name);
    int get_idx(const std::string& name, bool must_exist=true);

    template <typename T>
    T& get_computation(const std::string& name) {
        return *dynamic_cast<T*>(get(name).computation.get());
    }


    void compute();
    enum IntegratorType {Verlet=0, Predescu=1};
    void integration_cycle(float* mom, float dt, float max_force, IntegratorType type = Verlet);
};

double get_n_hbond(DerivEngine &engine);
DerivEngine initialize_engine_from_hdf5(int n_atom, int n_system, hid_t force_group);

// note that there are no null points in the vector of CoordNode*
typedef std::vector<CoordNode*> ArgList;
typedef std::function<DerivComputation*(hid_t, const ArgList&)> NodeCreationFunction;
typedef std::map<std::string, NodeCreationFunction> NodeCreationMap;
NodeCreationMap& node_creation_map(); 

static bool is_prefix(const std::string& s1, const std::string& s2) {
    return s1 == s2.substr(0,s1.size());
}

static void add_node_creation_function(std::string name_prefix, NodeCreationFunction fcn) 
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

static void check_elem_width(const CoordNode& node, int expected_elem_width) {
    if(node.elem_width != expected_elem_width) 
        throw std::string("expected argument with width ") + std::to_string(expected_elem_width) + 
            " but received argument with width " + std::to_string(node.elem_width);
}

static void check_arguments_length(const ArgList& arguments, int n_expected) {
    if(int(arguments.size()) != n_expected) 
        throw std::string("expected ") + std::to_string(n_expected) + 
            " arguments but got " + std::to_string(arguments.size());
}

template <typename NodeClass, int n_args>
struct RegisterNodeType {
    RegisterNodeType(std::string name_prefix);
};

template <typename NodeClass>
struct RegisterNodeType<NodeClass,0> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,0); 
            return new NodeClass(grp);};
        add_node_creation_function(name_prefix, f);
    }
};

template <typename NodeClass>
struct RegisterNodeType<NodeClass,1> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,1); 
            return new NodeClass(grp, *args[0]);};
        add_node_creation_function(name_prefix, f);
    }
};

template <typename NodeClass>
struct RegisterNodeType<NodeClass,2> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,2); 
            return new NodeClass(grp, *args[0], *args[1]);};
        add_node_creation_function(name_prefix, f);
    }
};
// Pos should not be registered, since it is of a special type

/*
#include "random.h"
using namespace std;


template <typename CoordT, typename FuncT>
void finite_difference(FuncT& f, CoordT& x, float* expected, float eps = 1e-2) 
{
    int ndim_output = decltype(f(x))::n_dim;
    auto y = x;
    int ndim_input  = decltype(y)::n_dim;

    vector<float> ret(ndim_output*ndim_input);
    for(int d=0; d<ndim_input; ++d) {
        CoordT x_prime1 = x; x_prime1.v[d] += eps;
        CoordT x_prime2 = x; x_prime2.v[d] -= eps;

        auto val1 = f(x_prime1);
        auto val2 = f(x_prime2);
        for(int no=0; no<ndim_output; ++no) ret[no*ndim_input+d] = (val1.v[no]-val2.v[no]) / (2*eps);
    }
    float z = 0.f;
    for(int no=0; no<ndim_output; ++no) {
        printf("exp:");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", expected[no*ndim_input+ni]);
        printf("\n");

        printf("fd: ");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", ret     [no*ndim_input+ni]);
        printf("\n\n");
        for(int ni=0; ni<ndim_input; ++ni) {
            float t = expected[no*ndim_input+ni]-ret[no*ndim_input+ni];
            z += t*t;
        }
    }
    printf("rmsd % f\n\n\n", sqrtf(z/ndim_output/ndim_input));

}
*/

#endif
