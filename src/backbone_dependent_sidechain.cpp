#include "backbone_dependent_sidechain.h"
#include "affine.h"
#include <vector>

void read_backbone_dependent_point(
        float3& com, float3& dcom_dphi, float3 &dcom_dpsi, 
        BackbonePointMap &map, int layer, float2 rama_pos) {
    int n_bin      = map.n_bin;
    float* germ     = map.germ;

    float phi = rama_pos.x;
    float psi = rama_pos.y;

    // map phi and psi into dimension 0 to n_bin
    float phi_coord = (phi+M_PI_F) * (0.5*M_1_PI_F*n_bin) - 0.5f;
    float psi_coord = (psi+M_PI_F) * (0.5*M_1_PI_F*n_bin) - 0.5f;

    if(phi_coord<0.f) phi_coord += n_bin;
    if(psi_coord<0.f) psi_coord += n_bin;

    // order of comparisons is correct in case of NaN
    if(!((phi_coord>=0.f) & (phi_coord<=n_bin) & (psi_coord>=0.f) & (psi_coord<=n_bin))) {
        com = dcom_dphi = dcom_dpsi = make_float3(NAN, NAN, NAN);
        return;
    }

    int phi_l_bin = phi_coord;
    int psi_l_bin = psi_coord;

    // handle periodicity
    int phi_r_bin = phi_l_bin==n_bin-1 ? 0 : phi_l_bin+1;
    int psi_r_bin = psi_l_bin==n_bin-1 ? 0 : psi_l_bin+1;

    int i_ll = layer*n_bin*n_bin + phi_l_bin*n_bin + psi_l_bin;
    int i_lr = layer*n_bin*n_bin + phi_l_bin*n_bin + psi_r_bin;
    int i_rl = layer*n_bin*n_bin + phi_r_bin*n_bin + psi_l_bin;
    int i_rr = layer*n_bin*n_bin + phi_r_bin*n_bin + psi_r_bin;

    float phi_r_weight = phi_coord - phi_l_bin;
    float psi_r_weight = psi_coord - psi_l_bin;

    float phi_l_weight = 1.f - phi_r_weight;
    float psi_l_weight = 1.f - psi_r_weight;

    float c_ll = phi_l_weight * psi_l_weight;
    float c_lr = phi_l_weight * psi_r_weight;
    float c_rl = phi_r_weight * psi_l_weight;
    float c_rr = phi_r_weight * psi_r_weight;

    float data[9];
    for(int d=0; d<9; ++d) 
        data[d] = c_ll*germ[i_ll*9+d] + c_lr*germ[i_lr*9+d] + c_rl*germ[i_rl*9+d] + c_rr*germ[i_rr*9+d];

    com       = make_float3(data[0], data[1], data[2]);
    dcom_dphi = make_float3(data[3], data[4], data[5]);
    dcom_dpsi = make_float3(data[6], data[7], data[8]);
}


void backbone_dependent_point(
        SysArray   output,
        CoordArray rama,
        CoordArray alignment,
        BackboneSCParam* param,
        BackbonePointMap map,
        int n_term, int n_system) {
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_term; ++nt) {
            Coord<2,3>     r(rama, ns, param[nt].rama_residue);
            AffineCoord<3> body(alignment, ns, param[nt].alignment_residue);

            float3 com, dcom_dphi, dcom_dpsi;
            read_backbone_dependent_point(com, dcom_dphi, dcom_dpsi, 
                    map, param[nt].restype, make_float2(r.v[0], r.v[1]));

            // rotate reference derivatives into the body frame
            float3 com_rotated = body.apply(com);
            float3 dcom_dphi_rotated = body.apply_rotation(dcom_dphi);
            float3 dcom_dpsi_rotated = body.apply_rotation(dcom_dpsi);

            r.d[0][0] = dcom_dphi_rotated.x;   r.d[0][1] = dcom_dpsi_rotated.x;
            r.d[1][0] = dcom_dphi_rotated.y;   r.d[1][1] = dcom_dpsi_rotated.y;
            r.d[2][0] = dcom_dphi_rotated.z;   r.d[2][1] = dcom_dpsi_rotated.z;

            body.add_deriv_at_location(0, com_rotated, make_float3(1.f, 0.f, 0.f));
            body.add_deriv_at_location(1, com_rotated, make_float3(0.f, 1.f, 0.f));
            body.add_deriv_at_location(2, com_rotated, make_float3(0.f, 0.f, 1.f));

            body.flush();
            r.flush();
        }
    }
}
