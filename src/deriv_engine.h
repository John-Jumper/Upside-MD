#ifndef DERIV_ENGINE_H
#define DERIV_ENGINE_H

#include "h5_support.h"
#include <vector>
#include <string>
#include <functional>
#include <initializer_list>
#include <map>
#include "vector_math.h"

//!\brief Copy VecArray to a flat float* array
inline void copy_vec_array_to_buffer(VecArray arr, int n_elem, int n_dim, float* buffer) {
        for(int i=0; i<n_elem; ++i)
            for(int d=0; d<n_dim; ++d) 
                buffer[i*n_dim+d] = arr(d,i);
}

typedef int index_t;  //!< Type of coordinate indices

//! \brief Update position and momentum
void
integration_stage(
        VecArray mom, //!< [inout] momentum
        VecArray pos, //!< [inout] position
        const VecArray deriv, //!< [in] derivative of potential with respect to position
        float vel_factor, //!< [in] fraction of force to add to momentum (integration dependent)
        float pos_factor,//!< [in] fraction of momentum to add to position (integration dependent)
        float max_force, //!< [in] clip forces so that they do not exceed maxforce (increase stability)
        int n_atom //!<[in] number of atoms
        );

//! \brief Recenter position array to origin
void
recenter(
        VecArray pos, //!< [inout] position
        bool xy_recenter_only, //!< if true, do not recenter in z-direction (useful for membrane)
        int n_atom //!< number of atoms
        );

//! \brief Whether to compute potential value as well as its derivative
enum ComputeMode {
    DerivMode = 0, //!< Only derivative must be computed correctly (potential may not be correct)
    PotentialAndDerivMode = 1 //!< Compute potential and derivative correctly
};

//! \brief Differentiable computation node
struct DerivComputation 
{
    //! \brief True if output represents a potential energy rather than a new coordinate
    const bool potential_term;

    //! \brief Construct only noting if node is a potential_node
    DerivComputation(bool potential_term_):
        potential_term(potential_term_) {}

    //! \brief Trivial destructor
    virtual ~DerivComputation() {}

    //! \brief Reads inputs and computes output value
    virtual void compute_value(ComputeMode mode)=0;

    //! \brief Uses its sensitivity to its output to add to sensivities of its inputs
    virtual void propagate_deriv() =0;

    //! \brief Return arbitrary subset of parameters
    virtual std::vector<float> get_param() const {return std::vector<float>();}

    //! \brief Set arbitrary subset of parameters (same as get_param)
    virtual void set_param(const std::vector<float>& new_params) {}
#ifdef PARAM_DERIV
    //! \brief Param deriv of arbitrary subset of parameters (same as get_param)
    virtual std::vector<float> get_param_deriv() {return std::vector<float>();}
#endif

    //! \brief Compute a named quantity and return vector of floats (arbitrary behavior)
    virtual std::vector<float> get_value_by_name(const char* log_name) {
        throw std::string("No values implemented");
    }
};

//! Specialization of DerivComputation for derived coordinates
struct CoordNode : public DerivComputation
{
    int n_elem;  //!< number of output elements
    int elem_width;  //!< number of dimensions for each output element
    VecArrayStorage output; //!< output values
    VecArrayStorage sens; //!< sensitivity of the overall potential to each output value

    //! Initialize from n_elem and elem_width
    CoordNode(int n_elem_, int elem_width_):
        DerivComputation(false),
        n_elem(n_elem_), elem_width(elem_width_), 
        output(elem_width, round_up(n_elem,4)),
        sens  (elem_width, round_up(n_elem,4)) {}
};


//! Specialization of DerivComputation for potential terms
struct PotentialNode : public DerivComputation
{
    float potential; //!< output potential value
    //! \brief Trivial constructor
    PotentialNode():
        DerivComputation(true) {}
    //! \brief (unused)
    //!
    //! The propagate_deriv function is never called for potential nodes
    virtual void propagate_deriv() {};
};


//! Specialization of PotentialNode to report number of hbonds
struct HBondCounter : public PotentialNode {
    float n_hbond; //!< number of hbonds
    //! Trivial constructor
    HBondCounter(): PotentialNode(), n_hbond(-1.f) {};
};


//! CoordNode to store/output atom positions
struct Pos : public CoordNode
{
    int n_atom; //!< number of atoms in system

    //! Construct from number of atoms
    Pos(int n_atom_):
        CoordNode(n_atom_, 3), 
        n_atom(n_atom_)
    {}

    //! \brief Compute value
    //!
    //! compute_value is empty because DerivEngine directly overwrites the output
    virtual void compute_value(ComputeMode mode) {};

    //! \brief Propagate deriv
    //!
    //! propagate_deriv is empty because DerivEngine directly reads from it
    virtual void propagate_deriv() {};
};


//! Main class to represent differentiable computational graph
struct DerivEngine
{
    //! Class for managing the automatic differentiation
    struct Node 
    {
        std::string name; //!< HDF5 name of the group for the node
        std::unique_ptr<DerivComputation> computation; //!< underlying DerivComputation for calculations
        std::vector<size_t> parents;  //!< list of indices for parent nodes (a.k.a. inputs)
        std::vector<size_t> children; //!< list of indices for child nodes (a.k.a. consumers of output)

        int germ_exec_level; //!< Directed acyclic graph height of compute_value computation
        int deriv_exec_level;//!< Directed acyclic graph height of propagate_deriv computation

        //! \brief Construct from name and unique_ptr to computation
        Node(std::string name_, std::unique_ptr<DerivComputation> computation_):
            name(name_), computation(std::move(computation_)) {};
        //! \brief Construct from name and raw pointer to computation
        Node(std::string name_, DerivComputation* computation_):
            name(name_), computation(computation_) {};
        Node(const Node& other) = delete;
        //! \brief Move constructor (Node's are not copyable)
        Node(Node&& other):
            name(std::move(other.name)),
            computation(std::move(other.computation)),
            parents(std::move(other.parents)),
            children(std::move(other.children)),
            germ_exec_level(other.germ_exec_level),
            deriv_exec_level(other.deriv_exec_level)
        {}
    };

    //! \brief vector of all Node's in the computation graph
    //!
    //! nodes[0] is guaranteed to be the Pos node
    std::vector<Node> nodes;
    //! \brief Pointer to pos node (used for position input and derivative output)
    Pos* pos;
    //! \brief potential energy output of the computation graph
    //!
    //! The potential should only be read after calling compute(PotentialAndDerivMode)
    //! and may be any value after the completion of compute(DerivMode)
    float potential;

    //! \brief Default constructor (not used)
    DerivEngine() {}
    //! \brief Construct from number of atoms
    DerivEngine(int n_atom): 
        potential(0.f)
    {
        nodes.emplace_back("pos", new Pos(n_atom));
        pos = dynamic_cast<Pos*>(nodes[0].computation.get());
    }

    //! \brief Add nodes to computation graph
    void add_node(
            const std::string& name, 
            std::unique_ptr<DerivComputation> fcn, 
            std::vector<std::string> argument_names);

    //! \brief Get Node by name
    //!
    //! Throws exception if node does not exist
    Node& get(const std::string& name); //!< get Node by name

    //! \brief Get node index by name
    //!
    //! If node name does not exist, if must_exist then exception else return -1
    int get_idx(const std::string& name, bool must_exist=true); 

    //! \brief Get DerivComputation by name
    //!
    //! Looks up the node by name, then dynamic_casts to the template return type.
    template <typename T>
    T& get_computation(const std::string& name) {
        auto computation = get(name).computation.get();
        if(!computation) throw std::string("impossible pointer value");
        return dynamic_cast<T&>(*computation);
    }

    //! \brief Execute computational graph
    //!
    //! See ComputeMode for details.
    void compute(ComputeMode mode);

    //! \brief Integration scheme (i.e. position and velocity update weights) to use
    enum IntegratorType {Verlet=0, Predescu=1};

    //! \brief Perform a full integration cycle (3 time steps)
    //!
    //! See integration_stage for details.
    void integration_cycle(VecArray mom, float dt, float max_force,
            IntegratorType type = Verlet);
};

//! \brief Count the number hbonds for a system
double get_n_hbond(DerivEngine &engine);

//! \brief Construct DerivEngine from potential group
DerivEngine initialize_engine_from_hdf5(int n_atom, hid_t potential_group, bool quiet=false);

//! \brief Vector of non-null CoordNode pointers
typedef std::vector<CoordNode*> ArgList;

//! \brief DerivComputation factory
//!
//! Construction of a DerivComputation always proceeds from an HDF5 group
//! reference and an ArgList of inputs
typedef std::function<DerivComputation*(hid_t, const ArgList&)> NodeCreationFunction;

//! \brief Mapping from names to creation functions
typedef std::map<std::string, NodeCreationFunction> NodeCreationMap;

//! \brief Obtain the dynamically-registered node creation map
NodeCreationMap& node_creation_map(); 

//! \brief Returns true if string s1 is prefix of s2
bool is_prefix(const std::string& s1, const std::string& s2);

//! \brief Register a NodeCreationFunction with node_creation_map
void add_node_creation_function(std::string name_prefix, NodeCreationFunction fcn);

//! \brief Throw exception if elem_width of node is not expected_elem_width
void check_elem_width(const CoordNode& node, int expected_elem_width);

//! \brief Throw exception if elem_width of node is not at least elem_width_lower_bound
void check_elem_width_lower_bound(const CoordNode& node, int elem_width_lower_bound);

//! \brief Throw except if ArgList is not length n_expected
void check_arguments_length(const ArgList& arguments, int n_expected);

//! \brief Register class that takes n_args
template <typename NodeClass, int n_args>
struct RegisterNodeType {
    RegisterNodeType(std::string name_prefix);
};


//! \brief Register class that takes 1+ args (i.e. variadic nodes)
template <typename NodeClass>
struct RegisterNodeType<NodeClass,-1> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            if(!args.size()) throw std::string("Expected at least 1 arg");
            return new NodeClass(grp, args);};
        add_node_creation_function(name_prefix, f);
    }
};

//! \brief Register class that takes no args
template <typename NodeClass>
struct RegisterNodeType<NodeClass,0> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,0); 
            return new NodeClass(grp);};
        add_node_creation_function(name_prefix, f);
    }
};

//! \brief Register class that takes 1 arg
template <typename NodeClass>
struct RegisterNodeType<NodeClass,1> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,1); 
            return new NodeClass(grp, *args[0]);};
        add_node_creation_function(name_prefix, f);
    }
};

//! \brief Register class that takes 2 args
template <typename NodeClass>
struct RegisterNodeType<NodeClass,2> {
    RegisterNodeType(std::string name_prefix){
        NodeCreationFunction f = [](hid_t grp, const ArgList& args) {
            check_arguments_length(args,2); 
            return new NodeClass(grp, *args[0], *args[1]);};
        add_node_creation_function(name_prefix, f);
    }
};

//! \brief Register class that takes 3 args
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

//! \brief Compute central difference approximation to derivative
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
