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


int set_param(int n_param, const float* param, DerivEngine* engine, const char* node_name) try {
#ifdef PARAM_DERIV
    vector<float> param_v(param, param+n_param);
    engine->get(string(node_name)).computation->set_param(param_v);
    return 0;
#else
    return -1;
#endif
} catch(...) {
    return 1;
}

int get_param(int n_param, float* param, DerivEngine* engine, const char* node_name) try {
#ifdef PARAM_DERIV
    auto param_v = engine->get(string(node_name)).computation->get_param();
    if(param_v.size() != size_t(n_param)) 
        throw string("Wrong number of parameters, expected ") + to_string(param_v.size()) + " but got " + 
            to_string(n_param);
    copy(begin(param_v), end(param_v), param);
    return 0;
#else
    return -1;
#endif
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}

int get_param_deriv(int n_param, float* deriv, DerivEngine* engine, const char* node_name) try {
#ifdef PARAM_DERIV
    auto deriv_v = engine->get(string(node_name)).computation->get_param_deriv();
    if(deriv_v.size() != size_t(n_param)) 
        throw string("Wrong number of parameters, expected ") + to_string(deriv_v.size()) + " but got " + 
            to_string(n_param);
    copy(begin(deriv_v), end(deriv_v), deriv);
    return 0;
#else
    return -1;
#endif
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}
