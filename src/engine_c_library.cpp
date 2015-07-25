#include "engine_c_library.h"
#include "deriv_engine.h"
#include <algorithm>

using namespace h5;
using namespace std;

DerivEngine* construct_deriv_engine(int n_atom, const char* potential_file, bool quiet) try {
    H5Obj config = h5_obj(H5Fclose, H5Fopen(potential_file, H5F_ACC_RDWR, H5P_DEFAULT));
    auto potential_group = open_group(config.get(), "/input/potential");
    
    auto engine = new DerivEngine(std::move(initialize_engine_from_hdf5(n_atom, 1, potential_group.get(), quiet)));
    return engine;
} catch(const string& e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
    return 0;
} catch(...) {
    return 0;
}


void free_deriv_engine(DerivEngine* engine) {
    delete engine;
}


// 0 indicates success, anything else is failure
int evaluate_energy(float* energy, DerivEngine* engine, const float* pos) try {
    VecArray a = engine->pos->coords().value[0];
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            a(d,na) = pos[na*3+d];
    engine->compute(PotentialAndDerivMode);
    *energy = engine->potential[0];
    return 0;
} catch(...) {
    return 1;
}

// 0 indicates success, anything else is failure
int evaluate_deriv(float* deriv, DerivEngine* engine, const float* pos) try {
    // result is size (n_atom,3)
    VecArray a = engine->pos->coords().value[0];
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            a(d,na) = pos[na*3+d];

    engine->compute(PotentialAndDerivMode);

    VecArray b = engine->pos->deriv_array()[0];
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            deriv[na*3+d] = b(d,na);
    return 0;
} catch(...) {
    return 1;
}


// // calling with deriv==nullptr just gives the size
// int  get_param_deriv(float* deriv, DerivEngine* engine, const char* node_name);
// void set_param(const float* param, DerivEngine* engine, const char* node_name);
