#ifndef ENGINE_C_LIBRARY_H
#define ENGINE_C_LIBRARY_H

extern "C" {
    struct DerivEngine;

    DerivEngine* construct_deriv_engine(int n_atom, const char* potential_file, bool quiet);
    void free_deriv_engine(DerivEngine* engine);

    int evaluate_energy(float* energy, DerivEngine* engine, const float* pos);
    int evaluate_deriv (float* deriv,  DerivEngine* engine, const float* pos);

    int set_param      (int n_param, const  float* param,  DerivEngine* engine, const char* node_name);

    int get_param_deriv(int n_param,  float* deriv,  DerivEngine* engine, const char* node_name);
    int get_param      (int n_param,  float* param,  DerivEngine* engine, const char* node_name);
    int get_output     (int n_output, float* output, DerivEngine* engine, const char* node_name);
    int get_output_dims(int* n_elem, int* elem_width, DerivEngine* engine, const char* node_name);
}

#endif
