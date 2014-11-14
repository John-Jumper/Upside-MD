#ifndef STRING_H
#define STRING_H

#include "coord.h"
#include <vector>

struct DihedralStringSystem {
    std::vector<int> id;
    int n_dihe;
    int atom_space;

    std::vector<double> string_pos;
    int n_string_coords;
    int n_system_per_string_coord;

    float string_mass;
    float spring_constant;

    DihedralStringSystem(const std::vector<int> &id_, 
            int n_string_coords_, int n_system_per_string_coord_,
            float string_mass_, float spring_constant_);

    void string_deriv(float* deriv, SysArray pos);
    void update(const float* deriv, float dt);

    void compute_deriv_and_update(SysArray pos, float dt) {
        std::vector<float> deriv(string_pos.size());
        string_deriv(deriv.data(), pos);
        update      (deriv.data(), dt);
    }
};

#endif
