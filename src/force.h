#ifndef FORCE_H
#define FORCE_H

#include "h5_support.h"
#include "md.h"
#include <vector>
#include <memory>
#include <string>

struct SlotMachine
{
    const int width;
    const int n_elem;
    const int n_system;

    std::vector<DerivRecord> deriv_tape;
    std::vector<float>       accum;

    SlotMachine(int width_, int n_elem_, int n_system_): width(width_), n_elem(n_elem_), n_system(n_system_) {}

    void add_request(int output_width, CoordPair &pair) { 
        DerivRecord prev_record = deriv_tape.size() ? deriv_tape.back() : DerivRecord(-1,0,0);
        deriv_tape.emplace_back(pair.index, prev_record.loc+prev_record.output_width, output_width);
        pair.slot = deriv_tape.back().loc;
        for(int i=0; i<output_width*width; ++i) accum.push_back(0.f);
    }
};

struct DerivComputation 
{
    virtual void compute_germ()    {};
    virtual void propagate_deriv() {};
};

struct Pos : public DerivComputation
{
    int n_atom;
    int n_system;
    SlotMachine slot_machine;
    std::vector<float> output;
    std::vector<float> deriv;

    Pos(int n_atom_, int n_system_):
        n_atom(n_atom_), n_system(n_system_), slot_machine(3, n_atom, n_system), 
        output(3*n_atom*n_system, 0.f), deriv(3*n_atom*n_system, 0.f)
    {}

    virtual void propagate_deriv();
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
            std::initializer_list<std::string> argument_names);

    Node& get(const std::string& name);
    int get_idx(const std::string& name, bool must_exist=true);

    template <typename T>
    T& get_computation(const std::string& name) {
        return *dynamic_cast<T*>(get(name).computation.get());
    }


    void compute();
    enum IntegratorType {Verlet=0, Predescu=1};
    void integration_cycle(float* mom, float dt, IntegratorType type = Verlet);
};

double get_n_hbond(DerivEngine &engine);
DerivEngine initialize_engine_from_hdf5(int n_atom, int n_system, hid_t force_group);
#endif
