#ifndef FORCE_H
#define FORCE_H

#include "h5_support.h"
#include <vector>
#include <string>
#include "coord.h"
#include <functional>
#include <initializer_list>
#include <map>

struct DerivRecord {
    index_t atom;
    slot_t  loc, output_width;
    DerivRecord(index_t atom_, slot_t loc_, slot_t output_width_):
        atom(atom_), loc(loc_), output_width(output_width_) {}
};


struct SlotMachine
{
    const int width;
    const int n_elem;

    int n_slot;
    std::vector<DerivRecord> deriv_tape;
    std::vector<float>       accum;

    SlotMachine(int width_, int n_elem_): 
        width(width_), n_elem(n_elem_), n_slot(0) {}

    void add_request(int output_width, CoordPair &pair) { 
        DerivRecord prev_record = deriv_tape.size() ? deriv_tape.back() : DerivRecord(-1,0,0);
        deriv_tape.emplace_back(pair.index, prev_record.loc+prev_record.output_width, output_width);
        pair.slot = deriv_tape.back().loc;
        for(int i=0; i<output_width*width; ++i) accum.push_back(0.f);
        n_slot += output_width;
    }

    VecArray accum_array() { 
        return VecArray(accum.data(), n_slot); 
    }
};


struct AutoDiffParams {
    unsigned char  n_slots1, n_slots2;
    slot_t slots1[6];      
    slot_t slots2[5];        

    AutoDiffParams(
            const std::initializer_list<slot_t> &slots1_,
            const std::initializer_list<slot_t> &slots2_)
    {
        unsigned loc1=0;
        for(auto i: slots1_) if(i!=(slot_t)(-1)) slots1[loc1++] = i;
        n_slots1 = loc1;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        for(auto i: slots2_) if(i!=(slot_t)(-1)) slots2[loc2++] = i;
        n_slots2 = loc2;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

    explicit AutoDiffParams(const std::initializer_list<slot_t> &slots1_)
    { 
        unsigned loc1=0;
        for(auto i: slots1_) if(i!=(slot_t)(-1)) slots1[loc1++] = i;
        n_slots1 = loc1;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        // for(auto i: slots2_) if(i!=(slot_t)(-1)) slots1[loc2++] = i;
        n_slots2 = loc2;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

} ;  // struct for implementing reverse autodifferentiation

enum ComputeMode { DerivMode = 0, PotentialAndDerivMode = 1 };

struct DerivComputation 
{
    const bool potential_term;
    DerivComputation(bool potential_term_):
        potential_term(potential_term_) {}
    virtual ~DerivComputation() {}
    virtual void compute_value(ComputeMode mode)=0;
    virtual void propagate_deriv() =0;
    virtual double test_value_deriv_agreement() {return -1.;}

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const {return std::vector<float>();}
    virtual void set_param(const std::vector<float>& new_params) {}
    virtual std::vector<float> get_param_deriv() const {return std::vector<float>();}
#endif
};

struct CoordNode : public DerivComputation
{
    int n_elem;
    int elem_width;
    std::vector<float> output;
    SlotMachine slot_machine;
    CoordNode(int n_elem_, int elem_width_):
        DerivComputation(false), n_elem(n_elem_), elem_width(elem_width_), 
        output(n_elem*elem_width), 
        slot_machine(elem_width, n_elem) {}
    virtual CoordArray coords() {
        return CoordArray(VecArray(output.data(), n_elem), slot_machine.accum_array());
    }
};


struct PotentialNode : public DerivComputation
{
    std::vector<float> potential;
    PotentialNode():
        DerivComputation(true), potential(1) {}
    virtual void propagate_deriv() {};
};


struct HBondCounter : public PotentialNode {
    float n_hbond;
    HBondCounter(): PotentialNode(), n_hbond(-1.f) {};
};


struct Pos : public CoordNode
{
    int n_atom;
    std::vector<float> deriv;

    Pos(int n_atom_):
        CoordNode(n_atom_, 3), 
        n_atom(n_atom_), deriv(3*n_atom, 0.f)
    {}

    virtual void compute_value(ComputeMode mode) {};
    virtual void propagate_deriv();
    virtual double test_value_deriv_agreement() {return 0.;};
    CoordArray coords() {
        return CoordArray(VecArray(output.data(), n_atom), slot_machine.accum_array());
    }
    VecArray deriv_array() {
        return VecArray(deriv.data(), n_atom);
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
            deriv_exec_level(other.deriv_exec_level) {}
    };

    std::vector<Node> nodes;  // nodes[0] is the pos node
    Pos* pos;
    std::vector<float> potential;

    DerivEngine() {}
    DerivEngine(int n_atom): 
        potential(1)
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


template <int my_width, int width1, int width2>
void reverse_autodiff(
        const VecArray accum,
        VecArray deriv1,
        VecArray deriv2,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_atom)
{
    std::vector<TempCoord<my_width>> sens(n_atom);
    for(int nt=0; nt<n_tape; ++nt) {
        auto tape_elem = tape[nt];
        for(int rec=0; rec<int(tape_elem.output_width); ++rec) {
            auto val = StaticCoord<my_width>(accum, tape_elem.loc + rec);
            for(int d=0; d<my_width; ++d)
                sens[tape_elem.atom].v[d] += val.v[d];
        }
    }

    for(int na=0; na<n_atom; ++na) {
        if(width1) {
            for(int nsl=0; nsl<p[na].n_slots1; ++nsl) {
                for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                    MutableCoord<width1> c(deriv1, p[na].slots1[nsl]+sens_dim);
                    for(int d=0; d<width1; ++d) c.v[d] *= sens[na].v[sens_dim];
                    c.flush();
                }
            }
        }

        if(width2) {
            for(int nsl=0; nsl<p[na].n_slots2; ++nsl) {
                for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                    MutableCoord<width2> c(deriv2, p[na].slots2[nsl]+sens_dim);
                    for(int d=0; d<width2; ++d) c.v[d] *= sens[na].v[sens_dim];
                    c.flush();
                }
            }
        }
    }
}


enum ValueType {CARTESIAN_VALUE=0, ANGULAR_VALUE=1, BODY_VALUE=2};

std::vector<float> central_difference_deriviative(
        const std::function<void()> &compute_value, std::vector<float> &input, std::vector<float> &output,
        float eps=1e-2f, ValueType value_type = CARTESIAN_VALUE);


template <int NDIM_INPUT>
std::vector<float> extract_jacobian_matrix( const std::vector<std::vector<CoordPair>>& coord_pairs,
        int elem_width_output, const std::vector<AutoDiffParams>* ad_params, 
        CoordNode &input_node, int n_arg)
{
    using namespace std;
    // First validate coord_pairs consistency with ad_params
    if(ad_params) {
        vector<slot_t> slots;
        if(ad_params->size() != coord_pairs.size()) throw string("internal error");
        for(unsigned no=0; no<ad_params->size(); ++no) {
            slots.resize(0);
            auto p = (*ad_params)[no];
            if     (n_arg==0) slots.insert(begin(slots), p.slots1, p.slots1+p.n_slots1);
            else if(n_arg==1) slots.insert(begin(slots), p.slots2, p.slots2+p.n_slots2);
            else throw string("internal error");

            if(slots.size() != coord_pairs[no].size()) 
                throw string("size mismatch (") + to_string(slots.size()) + " != " + to_string(coord_pairs[no].size()) + ")";
            for(unsigned i=0; i<slots.size(); ++i) if(slots[i] != coord_pairs[no][i].slot) throw string("inconsistent");
        }
    }

    int output_size = coord_pairs.size()*elem_width_output;
    int input_size  = input_node.n_elem*NDIM_INPUT;
    // special case handling for rigid bodies, since torques have 3 elements but quats have 4
    if(input_node.elem_width != NDIM_INPUT && input_node.elem_width!=7) 
        throw string("dimension mismatch ") + to_string(input_node.elem_width) + " " + to_string(NDIM_INPUT);

    vector<float> jacobian(output_size * input_size,0.f);
    VecArray accum_array = input_node.coords().deriv;

    for(unsigned no=0; no<coord_pairs.size(); ++no) {
        for(auto cp: coord_pairs[no]) {
            for(int eo=0; eo<elem_width_output; ++eo) {
                StaticCoord<NDIM_INPUT> d(accum_array, cp.slot+eo);
                for(int i=0; i<NDIM_INPUT; ++i) {
                    jacobian[eo*coord_pairs.size()*input_size + no*input_size + i*(input_size/NDIM_INPUT) + cp.index] += d.v[i];
                }
            }
        }
    }

    return jacobian;
}

template <typename T>
void dump_matrix(int nrow, int ncol, const char* name, T matrix) {
    if(int(matrix.size()) != nrow*ncol) throw std::string("impossible matrix sizes");
    FILE* f = fopen(name, "w");
    for(int i=0; i<nrow; ++i) {
	for(int j=0; j<ncol; ++j)
	    fprintf(f, "%f ", matrix[i*ncol+j]);
	fprintf(f, "\n");
    }
    fclose(f);
}

#if !defined(IDENT_NAME)
#define IDENT_NAME atom
#endif
template <typename T>
std::vector<std::vector<CoordPair>> extract_pairs(const std::vector<T>& params, bool is_potential) {
    std::vector<std::vector<CoordPair>> coord_pairs;
    int n_slot = sizeof(params[0].IDENT_NAME)/sizeof(params[0].IDENT_NAME[0]);
    for(int ne=0; ne<int(params.size()); ++ne) {
        if(ne==0 || !is_potential) coord_pairs.emplace_back();
        for(int nsl=0; nsl<n_slot; ++nsl) {
            CoordPair s = params[ne].IDENT_NAME[nsl];
            if(s.slot != slot_t(-1)) coord_pairs.back().push_back(s);
        }
    }
    return coord_pairs;
}




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
