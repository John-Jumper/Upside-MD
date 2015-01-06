#ifndef FORCE_H
#define FORCE_H

#include "h5_support.h"
#include <vector>
#include <memory>
#include <string>
#include "coord.h"
#include <functional>
#include <initializer_list>
#include <map>

struct DerivRecord {
    unsigned short atom, loc, output_width, unused;
    DerivRecord(unsigned short atom_, unsigned short loc_, unsigned short output_width_):
        atom(atom_), loc(loc_), output_width(output_width_) {}
};


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


struct AutoDiffParams {
    unsigned char  n_slots1, n_slots2;
    unsigned short slots1[6];      
    unsigned short slots2[5];        

    AutoDiffParams(
            const std::initializer_list<unsigned short> &slots1_,
            const std::initializer_list<unsigned short> &slots2_):
        n_slots1(slots1_.size()), n_slots2(slots2_.size()) 
    {
        unsigned loc1=0;
        for(auto i: slots1_) slots1[loc1++] = i;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        for(auto i: slots2_) slots1[loc2++] = i;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

    explicit AutoDiffParams(const std::initializer_list<unsigned short> &slots1_):
        n_slots1(slots1_.size()), n_slots2(0u)
    { 
        unsigned loc1=0;
        for(auto i: slots1_) slots1[loc1++] = i;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        // for(auto i: slots2_) slots1[loc2++] = i;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

} ;  // struct for implementing reverse autodifferentiation

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
        return dynamic_cast<T&>(*get(name).computation.get());
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
        const SysArray accum,
        SysArray deriv1,
        SysArray deriv2,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_atom, 
        int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        std::vector<TempCoord<my_width>> sens(n_atom);
        for(int nt=0; nt<n_tape; ++nt) {
            auto tape_elem = tape[nt];
            for(int rec=0; rec<tape_elem.output_width; ++rec) {
                auto val = StaticCoord<my_width>(accum, ns, tape_elem.loc + rec);
                for(int d=0; d<my_width; ++d)
                    sens[tape_elem.atom].v[d] += val.v[d];
            }
        }

        for(int na=0; na<n_atom; ++na) {
            if(width1) {
                for(int nsl=0; nsl<p[na].n_slots1; ++nsl) {
                    for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                        MutableCoord<width1> c(deriv1, ns, p[na].slots1[nsl]+sens_dim);
                        for(int d=0; d<width1; ++d) c.v[d] *= sens[na].v[sens_dim];
                        c.flush();
                    }
                }
            }

            if(width2) {
                for(int nsl=0; nsl<p[na].n_slots2; ++nsl) {
                    for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                        MutableCoord<width2> c(deriv2, ns, p[na].slots2[nsl]+sens_dim);
                        for(int d=0; d<width2; ++d) c.v[d] *= sens[na].v[sens_dim];
                        c.flush();
                    }
                }
            }
        }
    }
}

#endif
