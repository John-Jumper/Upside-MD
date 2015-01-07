#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"
#include <vector>

using namespace std;
using namespace h5;

struct BackboneSCParam {
    CoordPair rama_residue;
    CoordPair alignment_residue;
    int       restype;
};

struct BackbonePointMap {
    int    n_bin;
    float* germ;
};


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


struct BackboneDependentPoint : public CoordNode
{
    CoordNode& rama;
    CoordNode& alignment;

    int n_restype;
    int n_bin;
    vector<float> backbone_point_map_data;

    vector<BackboneSCParam> params;
    vector<AutoDiffParams> autodiff_params;

    BackboneDependentPoint(hid_t grp, CoordNode& rama_, CoordNode& alignment_):
        CoordNode(rama_.n_system, get_dset_size(1, grp, "restype")[0], 3),
        rama(rama_), alignment(alignment_),
        n_restype(get_dset_size(5, grp, "backbone_point_map")[0]),
        n_bin    (get_dset_size(5, grp, "backbone_point_map")[1]),
        backbone_point_map_data(n_restype*n_bin*n_bin*3*3, 0.f),
        params(n_elem)
    {
        check_elem_width(rama,     2);
        check_elem_width(alignment, 7);

        check_size(grp, "rama_residue",       n_elem);
        check_size(grp, "alignment_residue",  n_elem);
        check_size(grp, "restype",            n_elem);
        check_size(grp, "backbone_point_map", n_restype, n_bin, n_bin, 3, 3);


        traverse_dset<1,int>(grp, "rama_residue",      [&](size_t nr, int x) {params[nr].rama_residue.index = x;});
        traverse_dset<1,int>(grp, "alignment_residue", [&](size_t nr, int x) {params[nr].alignment_residue.index = x;});
        traverse_dset<1,int>(grp, "restype",           [&](size_t nr, int x) {params[nr].restype = x;});

        traverse_dset<5,float>(grp, "backbone_point_map", 
                [&](size_t rt, size_t nb1, size_t nb2, size_t val_or_dphi_or_dpsi, size_t d, float x) {
                backbone_point_map_data[(((rt*n_bin + nb1)*n_bin + nb2)*3 + val_or_dphi_or_dpsi)*3 + d] = x;});

        for(size_t i=0; i<params.size(); ++i) rama     .slot_machine.add_request(3, params[i].rama_residue);
        for(size_t i=0; i<params.size(); ++i) alignment.slot_machine.add_request(3, params[i].alignment_residue);

        for(auto &p: params) autodiff_params.push_back(
                AutoDiffParams({p.rama_residue.slot}, {p.alignment_residue.slot}));
    }

    virtual void compute_value() {
        Timer timer(string("backbone_point"));
        BackbonePointMap bb_map;
        bb_map.n_bin = n_bin;
        bb_map.germ = backbone_point_map_data.data();

        backbone_dependent_point(
                coords().value, rama.coords(), alignment.coords(),
                params.data(), bb_map, n_elem, alignment.n_system);
    }

    virtual void propagate_deriv() {
        Timer timer(string("backbone_point_deriv"));
        reverse_autodiff<3,2,6>(
                slot_machine.accum_array(), 
                rama.slot_machine.accum_array(), alignment.slot_machine.accum_array(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_elem, alignment.n_system);
    }
};
static RegisterNodeType<BackboneDependentPoint,2> backbone_dependent_point_node("backbone_dependent_point");
