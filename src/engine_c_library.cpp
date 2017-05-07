#include "engine_c_library.h"
#include "deriv_engine.h"
#include <algorithm>
#include "spline.h"

using namespace h5;
using namespace std;

DerivEngine* construct_deriv_engine(int n_atom, const char* potential_file, bool quiet) try {
    H5Obj config = h5_obj(H5Fclose, H5Fopen(potential_file, H5F_ACC_RDONLY, H5P_DEFAULT));
    auto potential_group = open_group(config.get(), "/input/potential");
    
    auto engine = new DerivEngine(initialize_engine_from_hdf5(n_atom, potential_group.get(), quiet));
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
    VecArray a = engine->pos->output;
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            a(d,na) = pos[na*3+d];
    engine->compute(PotentialAndDerivMode);
    *energy = engine->potential;
    return 0;
} catch(const char* e) {
    fprintf(stderr, "\n\nERROR: %s\n", e);
    return 1;
} catch(const string& e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
    return 1;
} catch(...) {
    return 1;
}

// 0 indicates success, anything else is failure
int evaluate_deriv(float* deriv, DerivEngine* engine, const float* pos) try {
    // result is size (n_atom,3)
    VecArray a = engine->pos->output;
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            a(d,na) = pos[na*3+d];

    engine->compute(PotentialAndDerivMode);

    VecArray b = engine->pos->sens;
    for(int na: range(engine->pos->n_atom))
        for(int d: range(3))
            deriv[na*3+d] = b(d,na);
    return 0;
} catch(...) {
    return 1;
}


int set_param(int n_param, const float* param, DerivEngine* engine, const char* node_name) try {
    vector<float> param_v(param, param+n_param);
    engine->get(string(node_name)).computation->set_param(param_v);
    return 0;
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    fprintf(stderr, "ERROR: %s\n", "unknown error");
    return 1;
}

int get_param(int n_param, float* param, DerivEngine* engine, const char* node_name) try {
    auto param_v = engine->get(string(node_name)).computation->get_param();
    if(param_v.size() != size_t(n_param)) 
        throw string("Wrong number of parameters, expected ") + to_string(param_v.size()) + " but got " + 
            to_string(n_param);
    copy(begin(param_v), end(param_v), param);
    return 0;
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


int get_sens(int n_output, float* sens, DerivEngine* engine, const char* node_name) try {
    auto& dc = engine->get_computation<DerivComputation&>(string(node_name));

    if(dc.potential_term) {
        auto& p = dynamic_cast<PotentialNode&>(dc);
        if(n_output!=1) throw string("wrong size for potential node");
        *sens = p.potential;
    } else {
        auto& c = dynamic_cast<CoordNode&>(dc);
        if(n_output != c.n_elem*c.elem_width) throw string("wrong size for CoordNode");
        VecArray a = c.sens;
        for(int ne: range(c.n_elem))
            for(int d: range(c.elem_width))
                sens[ne*c.elem_width + d] = a(d,ne);
    }
    return 0;
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}


int get_output(int n_output, float* output, DerivEngine* engine, const char* node_name) try {
    auto& dc = engine->get_computation<DerivComputation&>(string(node_name));

    if(dc.potential_term) {
        auto& p = dynamic_cast<PotentialNode&>(dc);
        if(n_output!=1) throw string("wrong size for potential node");
        *output = p.potential;
    } else {
        auto& c = dynamic_cast<CoordNode&>(dc);
        if(n_output != c.n_elem*c.elem_width) throw string("wrong size for CoordNode");
        VecArray a = c.output;
        for(int ne: range(c.n_elem))
            for(int d: range(c.elem_width))
                output[ne*c.elem_width + d] = a(d,ne);
    }
    return 0;
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}


int get_output_dims(int* n_elem, int* elem_width, DerivEngine* engine, const char* node_name) try {
    auto& dc = engine->get_computation<DerivComputation&>(string(node_name));

    if(dc.potential_term) {
        *n_elem = 1;
        *elem_width = 1;
    } else {
        auto& c = dynamic_cast<CoordNode&>(dc);
        *n_elem = c.n_elem;
        *elem_width = c.elem_width;
    }
    return 0;
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}


int get_value_by_name(int n_output, float* output, DerivEngine* engine, const char* node_name, const char* log_name)
try {
    auto& dc = engine->get_computation<DerivComputation&>(string(node_name));
    auto value = dc.get_value_by_name(log_name);
    if(n_output != int(value.size()))
        throw string("expected size (") + to_string(n_output) +
            " elements) inconsistent with actual size (" + to_string(value.size()) + ")";
    copy(begin(value),end(value), output);
    return 0;
} catch(const string& s) {
    fprintf(stderr, "ERROR: %s\n", s.c_str());
    return 1;
} catch(...) {
    return 1;
}

int clamped_spline_solve(int N_coeff, float* bspline_coeff, const float* values) {
    vector<double> temp(3*N_coeff);
    vector<double> bspline_coeff_d(N_coeff);
    vector<double> values_d(values, values+(N_coeff-2));

    solve_clamped_1d_spline_for_bsplines(N_coeff, bspline_coeff_d.data(), values_d.data(), temp.data());
    for(int i: range(N_coeff)) bspline_coeff[i] = bspline_coeff_d[i];
    return 0;
}


int clamped_spline_value(int N, float* result, const float* bspline_coeff, int nx, float* x) {
    const float* coeff_ptrs[4] = {bspline_coeff, bspline_coeff, bspline_coeff, bspline_coeff};
    int i;
    for(i=0; i<nx-3; i+=4) {
        auto en = clamped_deBoor_value_and_deriv(coeff_ptrs, Float4(x+i, Alignment::unaligned), N);
        en.x().store(result+i, Alignment::unaligned);
    }

    if(i<nx) {
        // handle ragged end in non-multiple of 4 cases
        alignas(16) float last_data[4] = {0.f,0.f,0.f,0.f};
        for(int j=0; j<nx-i; ++j) last_data[j] = x[i+j];
        auto en = clamped_deBoor_value_and_deriv(coeff_ptrs, Float4(last_data), N);
        en.x().store(last_data);
        for(int j=0; j<nx-i; ++j) result[i+j] = last_data[j];
    }

    return 0;
}

int get_clamped_value_and_deriv(int N, float* result, const float* bspline_coeff, int nx, float* x) {
    for(int i: range(nx)) {
        auto en = clamped_deBoor_value_and_deriv(bspline_coeff, x[i], N);
        result[i*2+0] = en.x();
        result[i*2+1] = en.y();
    }
    return 0;
}


int get_bounded_value_and_deriv_2d(int N, float* result, const float* bspline_coeff, int nx, const float* x) {
    // this is a rather convoluted function because there is no scalar interface to the 2d splines
    const float* p[4] = {bspline_coeff, bspline_coeff, bspline_coeff, bspline_coeff};

    for(int i: range(0,nx,4)) {
        alignas(16) float x4[4] = {0.f,0.f,0.f,0.f};
        alignas(16) float y4[4] = {0.f,0.f,0.f,0.f};
        for(int j=0; j<4; ++j) {
            x4[j] = j<min(4,nx-i) ? x[2*(i+j)  ] : 1.f;
            y4[j] = j<min(4,nx-i) ? x[2*(i+j)+1] : 1.f;

            if((x4[j]<1.f)|(x4[j]>=float(N-2))|(y4[j]<1.f)|(y4[j]>=float(N-2))) return 1;
        }

        auto v = deBoor2d_value_and_deriv(N, p, Float4(x4), Float4(y4));

        alignas(16) float val[4] = {0.f,0.f,0.f,0.f};
        alignas(16) float der_x[4] = {0.f,0.f,0.f,0.f};
        alignas(16) float der_y[4] = {0.f,0.f,0.f,0.f};
        v[0].store(val);  
        v[1].store(der_x);
        v[2].store(der_y);

        for(int j=0; j<min(4,nx-i); ++j) {
            result[(i+j)*3+0] = val[j];
            result[(i+j)*3+1] = der_x[j];
            result[(i+j)*3+2] = der_y[j];
        }
    }
    return 0;
}

int get_clamped_coeff_deriv(int N, float* result, const float* bspline_coeff, float x) {
    int starting_bin;
    float data[4];
    clamped_deBoor_coeff_deriv(&starting_bin, data, x, N);
    for(int i: range(N)) result[i] = 0.f;
    for(int i: range(4)) result[starting_bin+i] = data[i];
    return 0;
}
