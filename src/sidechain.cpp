#include "md_export.h"
#include "affine.h"
#include <cmath>
#include "md.h"
#include "affine.h"
#include "sidechain.h"

#include <random>

using namespace std;

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
        const float* restrict rigid_body,
        float*       restrict rigid_body_deriv,

        Sidechain* restrict sidechains,
        SidechainParams* restrict params,
        
        float dist_cutoff,  // pair lists will subsume this
        int n_res)
{
    float dist_cutoff2 = dist_cutoff*dist_cutoff;

    vector<AffineCoord> coords;
    coords.reserve(n_res);
    for(int nr=0; nr<n_res; ++nr) 
        coords.emplace_back(rigid_body, rigid_body_deriv, params[nr].res);

    for(int nr1=0; nr1<n_res; ++nr1) {
        for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
            if(mag2(coords[nr1].tf3()-coords[nr2].tf3()) < dist_cutoff2) {
                int rt1 = params[nr1].restype;
                int rt2 = params[nr2].restype;
                sidechain_interaction(
                        coords[nr1], coords[nr2],
                        sidechains[rt1].interaction_pot,
                        sidechains[rt2].density_kernel_centers.size(),
                        sidechains[rt2].density_kernel_centers.data());

                sidechain_interaction(
                        coords[nr2], coords[nr1],
                        sidechains[rt2].interaction_pot,
                        sidechains[rt1].density_kernel_centers.size(),
                        sidechains[rt1].density_kernel_centers.data());
            }
        }
    }

    for(int nr=0; nr<n_res; ++nr) {
        coords[nr].flush();
    }
    // test_density3d();
}
