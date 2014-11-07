#ifndef SIDECHAIN_H
#define SIDECHAIN_H
#include "coord.h"
#include <vector>
#include <string>

struct SidechainParams {
    CoordPair res;
    int restype; // index into sidechain array
};


struct Density3D {
    const float3  corner;
    const float3  side_length;
    const float   bin_scale;
    const int     nx,ny,nz;
    const std::vector<float4> data;

    Density3D(
            const float3 corner_,
            const float bin_scale_,
            const int nx_, const int ny_, const int nz_,
            const std::vector<float4> data_):
       corner(corner_), 
       side_length(make_float3((nx_-1)/bin_scale_*(1-1e-6),(ny_-1)/bin_scale_*(1-1e-6),(nz_-1)/bin_scale_*(1-1e-6))), 
       bin_scale(bin_scale_), nx(nx_), ny(ny_), nz(nz_), data(data_)
    { if(nx*ny*nz != (int)data.size()) throw std::string("improper side length"); }

    float4 read_value(float3 point) const; 
};


struct Sidechain {
    // .w component is weight/probability of each center (sum to number of sidechain atoms)
    std::vector<float4> density_kernel_centers;
    Density3D interaction_pot;

    Sidechain(const std::vector<float4>& density_kernel_centers_, const Density3D& interaction_pot_):
        density_kernel_centers(density_kernel_centers_), interaction_pot(interaction_pot_) {};
};

void sidechain_pairs(
        const CoordArray rigid_body,

        Sidechain* restrict sidechains,
        SidechainParams* restrict params,
        
        float dist_cutoff,  // pair lists will subsume this
        int n_res, int n_system);
#endif
