#ifndef STERIC_H
#define STERIC_H

#include "coord.h"

struct StericPoint {
    float3 pos;
    float  weight;
    int    type;
};


struct StericResidue {
    float3 center;
    float  radius;   // all points are within radius of the center
    int    start_point;
    int    n_pts;
};


struct StericParams {
    CoordPair loc;
    int       restype;
};


struct Interaction {
    float   largest_cutoff;
    int     n_types;
    int     n_bin;
    float   inv_dx;

    float*  cutoff2;
    float4* germ_arr;

    Interaction(int n_types_, int n_bin_, float dx_):
        largest_cutoff(0.f),
        n_types(n_types_),
        n_bin(n_bin_),
        inv_dx(1.f/dx_),
        cutoff2 (new float [n_types*n_types]),
        germ_arr(new float4[n_types*n_types*n_bin]) {
            for(int nb=0; nb<n_bin; ++nb) 
                germ_arr[nb] = make_float4(0.f,0.f,0.f,0.f);
        };


    float2 germ(int loc, float r_mag) const {
        float coord = inv_dx*r_mag;
        int   coord_bin = int(coord);

        float4 vals = germ_arr[loc*n_types*n_types + coord_bin]; 
        float r_excess = coord - coord_bin;
        float l_excess = 1.f-r_excess;

        return make_float2(l_excess * vals.x + r_excess * vals.y,
                           l_excess * vals.z + r_excess * vals.w);
    }

    ~Interaction() {
        delete [] cutoff2;
        delete [] germ_arr;
    }
};

void steric_pairs(
        const CoordArray     rigid_body,
        const StericParams*  residues,  //  size (n_res,)
        const StericResidue* ref_res,
        const StericPoint*   ref_point,
        const Interaction &interaction,
        const int*           point_starts,  // size (n_res+1,)
        int n_res, int n_system);

#endif
