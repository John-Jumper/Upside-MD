#ifndef SIDECHAIN_H
#define SIDECHAIN_H
#include "coord.h"
#include <vector>
#include <string>
#include <cmath>

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
    float3 density_center;     float density_radius;
    float3 interaction_center; float interaction_radius;

    // .w component is weight/probability of each center (sum to number of sidechain atoms)
    std::vector<float4> density_kernel_centers;
    Density3D interaction_pot;

    Sidechain(const std::vector<float4>& density_kernel_centers_, const Density3D& interaction_pot_, float energy_cutoff=0.f):
        density_kernel_centers(density_kernel_centers_), interaction_pot(interaction_pot_) {
            density_center = make_float3(0.f,0.f,0.f);
            for(auto& x: density_kernel_centers) density_center += xyz(x);
            if(density_kernel_centers.size()) density_center *= 1.f/density_kernel_centers.size();

            density_radius = 0.f;
            for(auto& x: density_kernel_centers) {
                float r = sqrtf(mag2(density_center-xyz(x)));
                if(r>density_radius) density_radius = r;
            }

            auto &p = interaction_pot;
            interaction_center = p.corner + 0.5*p.side_length;
            interaction_radius = 0.f;

            for(int ix=0; ix<p.nx; ++ix) {
                for(int iy=0; iy<p.ny; ++iy) {
                    for(int iz=0; iz<p.nz; ++iz) {
                        auto pt = p.corner + make_float3(ix,iy,iz) * (1./p.bin_scale);
                        float r = sqrtf(mag2(pt-interaction_center));
                        if(r > interaction_radius && fabsf(p.data[ix*p.ny*p.nz + iy*p.nz + iz].w) > energy_cutoff)
                            interaction_radius = r;
                    }
                }
            }
        };
};

void sidechain_pairs(
        const CoordArray rigid_body,

        Sidechain* restrict sidechains,
        SidechainParams* restrict params,
        
        float dist_cutoff,  // pair lists will subsume this
        int n_res, int n_system);
#endif
