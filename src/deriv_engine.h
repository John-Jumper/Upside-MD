#ifndef DERIV_ENGINE_H
#define DERIV_ENGINE_H

#include "h5_support.h"
#include <vector>
#include <string>
#include <functional>
#include <initializer_list>
#include <map>
#include "vector_math.h"

inline void copy_vec_array_to_buffer(VecArray arr, int n_elem, int n_dim, float* buffer) {
        for(int i=0; i<n_elem; ++i)
            for(int d=0; d<n_dim; ++d) 
                buffer[i*n_dim+d] = arr(d,i);
}

typedef int index_t;  //!< type of coordinate indices

void
integration_stage(
        VecArray mom,
        VecArray pos,
        const VecArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom);

void
recenter(VecArray pos, bool xy_recenter_only, int n_atom);

enum ComputeMode { DerivMode = 0, PotentialAndDerivMode = 1 };

struct DerivComputation 
{
    const bool potential_term;
    DerivComputation(bool potential_term_):
        potential_term(potential_term_) {}
    virtual ~DerivComputation() {}

    virtual int  compute_value(int round, ComputeMode mode)=0;
    virtual void compute_value_subtask(int round, int task_num, int n_subtask){}

    virtual int  propagate_deriv(int round)=0;
    virtual void propagate_deriv_subtask(int round, int task_num, int n_subtask) {}

    virtual std::vector<float> get_param() const {return std::vector<float>();}
    virtual void set_param(const std::vector<float>& new_params) {}

    #ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() {return std::vector<float>();}
    #endif

    virtual std::vector<float> get_value_by_name(const char* log_name) {
        throw std::string("No values implemented");
    }
};

struct CoordNode : public DerivComputation
{
    int n_elem;
    int elem_width;
    VecArrayStorage output;
    VecArrayAccum   sens;

    CoordNode(int n_elem_, int elem_width_):
        DerivComputation(false),
        n_elem(n_elem_), elem_width(elem_width_), 
        output(elem_width, round_up(n_elem,4)),
        sens  (elem_width, round_up(n_elem,4)) {}
};


struct PotentialNode : public DerivComputation
{
    float potential;
    PotentialNode():
        DerivComputation(true) {}
    virtual int  propagate_deriv(int round){return 0;}
};


struct HBondCounter : public PotentialNode {
    float n_hbond;
    HBondCounter(): PotentialNode(), n_hbond(-1.f) {};
};


struct Pos : public CoordNode
{
    int n_atom;

    Pos(int n_atom_):
        CoordNode(n_atom_, 3), 
        n_atom(n_atom_)
    {}

    virtual int compute_value(int round, ComputeMode mode) {return 0;};
    virtual int propagate_deriv(int round) {return 0;};
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

        Node(std::string name_, std::unique_ptr<DerivComputation> computation_):
            name(name_), computation(std::move(computation_)) {};
        Node(std::string name_, DerivComputation* computation_):
            name(name_), computation(computation_) {};
        Node(const Node& other) = delete;
        Node(Node&& other):
            name(std::move(other.name)),
            computation(std::move(other.computation)),
            parents(std::move(other.parents)),
            children(std::move(other.children)),
            germ_exec_level(other.germ_exec_level),
            deriv_exec_level(other.deriv_exec_level)
        {}
    };

    std::vector<Node> nodes;  // nodes[0] is the pos node
    Pos* pos;
    float potential;

    DerivEngine() {}
    DerivEngine(int n_atom): 
        potential(0.f)
    {
        nodes.emplace_back("pos", new Pos(n_atom));
        pos = dynamic_cast<Pos*>(nodes[0].computation.get());
    }

    void add_node(
            const std::string& name, 
            std::unique_ptr<DerivComputation> fcn, 
            std::vector<std::string> argument_names);

    Node& get(const std::string& name);
    int get_idx(const std::string& name, bool must_exist=true);

    template <typename T>
    T& get_computation(const std::string& name) {
        auto computation = get(name).computation.get();
        if(!computation) throw std::string("impossible pointer value");
        return dynamic_cast<T&>(*computation);
    }

    void compute(ComputeMode mode);
    enum IntegratorType {Verlet=0, Predescu=1};
    void integration_cycle(VecArray mom, float dt, float max_force, IntegratorType type = Verlet);
};

double get_n_hbond(DerivEngine &engine);
DerivEngine initialize_engine_from_hdf5(int n_atom, hid_t potential_group, bool quiet=false);

// note that there are no null points in the vector of CoordNode*
typedef std::vector<CoordNode*> ArgList;
typedef std::function<DerivComputation*(hid_t, const ArgList&)> NodeCreationFunction;
typedef std::map<std::string, NodeCreationFunction> NodeCreationMap;
NodeCreationMap& node_creation_map(); 

bool is_prefix(const std::string& s1, const std::string& s2);
void add_node_creation_function(std::string name_prefix, NodeCreationFunction fcn);
void check_elem_width(const CoordNode& node, int expected_elem_width);
void check_elem_width_lower_bound(const CoordNode& node, int elem_width_lower_bound);
void check_arguments_length(const ArgList& arguments, int n_expected);

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

template <typename NodeClass>
struct RegisterNodeType<NodeClass,3> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,3); 
            return new NodeClass(grp, *args[0], *args[1], *args[2]);};
        add_node_creation_function(name_prefix, f);
    }
};

enum ValueType {CARTESIAN_VALUE=0, ANGULAR_VALUE=1, BODY_VALUE=2};

std::vector<float> central_difference_deriviative(
        const std::function<void()> &compute_value, std::vector<float> &input, std::vector<float> &output,
        float eps=1e-2f, ValueType value_type = CARTESIAN_VALUE);


static double relative_rms_deviation(
        const std::vector<float> &reference, const std::vector<float> &actual) {
    if(reference.size() != actual.size()) 
        throw std::string("impossible size mismatch ") + 
            std::to_string(reference.size()) + " " + std::to_string(actual.size());
    double diff_mag2  = 0.;
    double value1_mag2 = 0.;
    for(size_t i=0; i<reference.size(); ++i) {
        diff_mag2  += sqr(reference[i]-actual[i]);
        value1_mag2 += sqr(reference[i]);
    }
    return sqrt(diff_mag2/value1_mag2);
}


#endif
