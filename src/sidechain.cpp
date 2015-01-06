#include "force.h"
#include <string>
#include "timing.h"
#include "coord.h"
#include "md_export.h"
#include "affine.h"
#include <cmath>
#include "md.h"
#include "affine.h"

#include <random>

using namespace std;
using namespace h5;

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

float4 Density3D::read_value(float3 point) const {
    float3 shifted_point = point - corner;
    if((     shifted_point.x < 0.f) | (shifted_point.x >= side_length.x) |
            (shifted_point.y < 0.f) | (shifted_point.y >= side_length.y) |
            (shifted_point.z < 0.f) | (shifted_point.z >= side_length.z)) return make_float4(0.f,0.f,0.f,0.f);

    shifted_point *= bin_scale;

    int ix = int(shifted_point.x);
    int iy = int(shifted_point.y);
    int iz = int(shifted_point.z);

    float rx = shifted_point.x - ix;
    float ry = shifted_point.y - iy;
    float rz = shifted_point.z - iz;

    float lx = 1.f - rx;
    float ly = 1.f - ry;
    float lz = 1.f - rz;

    #define f(i,j,k) (data[(i)*(ny*nz) + (j)*nz + (k)])
    return
        lx * (
                ly * (lz*f(ix  ,iy  ,iz)+rz*f(ix  ,iy  ,iz+1)) + 
                ry * (lz*f(ix  ,iy+1,iz)+rz*f(ix  ,iy+1,iz+1))) +
        rx * (
                ly * (lz*f(ix+1,iy  ,iz)+rz*f(ix+1,iy  ,iz+1)) + 
                ry * (lz*f(ix+1,iy+1,iz)+rz*f(ix+1,iy+1,iz+1)));
    #undef f
}

void test_density3d() {
    auto corner = make_float3(-1.f, 2.f,4.f);
    auto nx = 50;
    auto ny = 40;
    auto nz = 30;
    auto dx = 0.1;
    auto bin_scale = 1.f/dx;

    auto data_ = vector<float4>(nx*ny*nz);
    auto data  = [&](size_t i, size_t j, size_t k)->float4& {return data_[i*ny*nz + j*nz + k];};

    auto f = [](float3 x) { return make_float4(2.*x.x,4.*x.y,x.z, x.x*x.x + 2.*x.y*x.y + 1.*x.z*x.z); };

    for(int ix=0; ix<nx; ++ix)
        for(int iy=0; iy<ny; ++iy)
            for(int iz=0; iz<nz; ++iz)  {
                data(ix,iy,iz) = f(corner + make_float3(ix,iy,iz)*dx);
            }

    auto density = Density3D(corner, bin_scale, nx,ny,nz, data_);

    random_device rd;
    mt19937 gen; //(rd());
    auto ran = [&]() {return generate_canonical<double,64>(gen);};
    
    for(int i=0; i<4; ++i) {
        auto pt = corner+make_float3(ran()*(nx-1)*dx, ran()*(ny-1)*dx, ran()*(nz-1)*dx);
        auto a = f(pt);
        auto b = density.read_value(pt);
        printf("% 8.3f % 8.3f % 8.3f   % 8.3f % 8.3f % 8.3f % 8.3f   % 8.3f % 8.3f % 8.3f % 8.3f\n",
                pt.x,pt.y,pt.z,
                a.x,a.y,a.z,a.w,
                b.x,b.y,b.z,b.w);
    }

}



template <typename AffineCoordT>
float sidechain_interaction(
        AffineCoordT& restrict body1,
        AffineCoordT& restrict body2,

        const Density3D &density1,
        const int       n_density_centers2,
        const float4*   density_kernel_centers2)
{
    float3 com_deriv = make_float3(0.f,0.f,0.f);
    float3 torque    = make_float3(0.f,0.f,0.f);

    float value = 0.f;
    for(int nc=0; nc<n_density_centers2; ++nc) {
        float4 dc2 = density_kernel_centers2[nc];
        float3 dc2_in_ref1_frame = body1.apply_inverse(body2.apply(float3_from_float4(dc2)));

        float4 germ = dc2.w * density1.read_value(dc2_in_ref1_frame);
        float3 der = float3_from_float4(germ); // float3 is derivative components

        value     += germ.w;
        com_deriv += der;
        torque    += cross(dc2_in_ref1_frame, der);
    }

    // now rotate derivative and torque to lab frame
    com_deriv = body1.apply_rotation(com_deriv);
    torque    = body1.apply_rotation(torque);

    // for body1, derivatives must be reversed (since the derivatives are for the points
    // for body2, torque must be recentered to its origin, instead of body1's
    body1.add_deriv_and_torque(-com_deriv, -torque);
    body2.add_deriv_and_torque( com_deriv,  torque + cross(body1.tf3()-body2.tf3(), com_deriv));
    return value;
}


void sidechain_pairs(
        const CoordArray rigid_body,

        Sidechain* restrict sidechains,
        SidechainParams* restrict params,
        
        float dist_cutoff,  // pair lists will subsume this
        int n_res,
        int n_system)
{
    float dist_cutoff2 = dist_cutoff*dist_cutoff;

    for(int ns=0; ns<n_system; ++ns) {
        vector<AffineCoord<>> coords;
        coords.reserve(n_res);
        for(int nr=0; nr<n_res; ++nr) 
            coords.emplace_back(rigid_body, ns, params[nr].res);

        for(int nr1=0; nr1<n_res; ++nr1) {
            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
                if(mag2(coords[nr1].tf3()-coords[nr2].tf3()) < dist_cutoff2) {
                    int rt1 = params[nr1].restype;
                    int rt2 = params[nr2].restype;

                    Sidechain &sc1 = sidechains[rt1];
                    Sidechain &sc2 = sidechains[rt2];

                    float cutoff12 = sc1.interaction_radius + sc2.density_radius;
                    if(mag2(sc1.interaction_center-sc2.density_center) < cutoff12*cutoff12) {
                        sidechain_interaction(
                                coords[nr1], coords[nr2],
                                sc1.interaction_pot,
                                sc2.density_kernel_centers.size(),
                                sc2.density_kernel_centers.data());
                    }


                    float cutoff21 = sc2.interaction_radius + sc1.density_radius;
                    if(mag2(sc2.interaction_center-sc1.density_center) < cutoff21*cutoff21) {
                        sidechain_interaction(
                                coords[nr2], coords[nr1],
                                sc2.interaction_pot,
                                sc1.density_kernel_centers.size(),
                                sc1.density_kernel_centers.data());
                    }
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr) {
            coords[nr].flush();
        }
    }
}


struct SidechainInteraction : public DerivComputation 
{
    int n_residue;
    CoordNode&  alignment;
    vector<Sidechain> sidechain_params;
    float             dist_cutoff;
    float             energy_cutoff;
    map<string,int>   name_map;
    vector<float>     density_data;
    vector<float4>    center_data;
    vector<SidechainParams> params;

    SidechainInteraction(hid_t grp, CoordNode& alignment_):
        n_residue(h5::get_dset_size<1>(grp, "restype")[0]), alignment(alignment_),
        energy_cutoff(h5::read_attribute<float>(grp, "./sidechain_data", "energy_cutoff")),
        params(n_residue)
    {
        traverse_string_dset<1>(grp, "restype", [&](size_t nr, std::string &s) {
                if(name_map.find(s) == end(name_map)) {
                    sidechain_params.push_back(
                        parse_sidechain(energy_cutoff,
                            h5_obj(H5Gclose, 
                                H5Gopen2(grp, (string("sidechain_data/")+s).c_str(), H5P_DEFAULT)).get()));
                    name_map[s] = sidechain_params.size()-1;
                }
                params[nr].res.index  = nr;
                params[nr].restype = name_map[s];
            });

        // find longest possible interaction
        float max_interaction_radius = 0.f;
        float max_density_radius = 0.f;
        for(auto &sc: sidechain_params) {
            // FIXME is the sidechain data in the correct reference frame
            float interaction_maxdist = sc.interaction_radius+mag(sc.interaction_center);
            float density_maxdist = sc.density_radius+mag(sc.density_center);
            max_interaction_radius = max(max_interaction_radius, interaction_maxdist);
            max_density_radius = max(max_density_radius, density_maxdist);
            printf("%4.1f %4.1f %4.1f %4.1f\n", 
                    mag(sc.interaction_center), sc.interaction_radius, 
                    mag(sc.density_center), sc.density_radius);
        }
        dist_cutoff = max_interaction_radius + max_density_radius;
        printf("total_cutoff: %4.1f\n", dist_cutoff);

        if(n_residue != alignment.n_elem) throw string("invalid restype array");
        for(int nr=0; nr<n_residue; ++nr) alignment.slot_machine.add_request(1,params[nr].res);
    }

    static Sidechain parse_sidechain(float energy_cutoff, hid_t grp) 
    {
        using namespace h5;
        int nkernel = get_dset_size<2>(grp, "kernels")[0];
        check_size(grp, "kernels", nkernel, 4);

        auto kernels = vector<float4>(nkernel);

        traverse_dset<2,float>(grp, "kernels", [&](size_t nk, size_t dim, float v) {
                switch(dim) {
                case 0: kernels[nk].x = v; break;
                case 1: kernels[nk].y = v; break;
                case 2: kernels[nk].z = v; break;
                case 3: kernels[nk].w = v; break;
                }});

        auto dims = get_dset_size<4>(grp, "interaction");
        check_size(grp, "interaction", dims[0], dims[1], dims[2], 4);

        auto data = vector<float4>(dims[0]*dims[1]*dims[2]);
        traverse_dset<4,float>(grp, "interaction", [&](size_t i, size_t j, size_t k, size_t dim, float v) {
                int idx = i*dims[1]*dims[2] + j*dims[2] + k;
                switch(dim) {
                case 0: data[idx].x = v; break;
                case 1: data[idx].y = v; break;
                case 2: data[idx].z = v; break;
                case 3: data[idx].w = v; break;
                }});

        check_size(grp, "corner_location", 3);
        float3 corner;
        traverse_dset<1,float>(grp, "corner_location", [&](size_t dim, float v) {
                switch(dim) {
                case 0: corner.x = v; break;
                case 1: corner.y = v; break;
                case 2: corner.z = v; break;
                }});

        auto bin_side_length = read_attribute<float>(grp, "interaction", "bin_side_length");

        return Sidechain(
                kernels, 
                Density3D(corner, 1.f/bin_side_length, dims[0],dims[1],dims[2], data),
                energy_cutoff);
    }

    virtual void compute_germ() {
        Timer timer(string("sidechain_pairs"));
        sidechain_pairs(
                alignment.coords(), 
                sidechain_params.data(), params.data(), 
                dist_cutoff, n_residue, alignment.n_system);
    }
};
static RegisterNodeType<SidechainInteraction,1> sidechain_node("sidechain");
