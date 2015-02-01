#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"
#include <vector>
#include "spline.h"

using namespace std;
using namespace h5;

struct BackboneSCParam {
    CoordPair rama_residue;
    CoordPair alignment_residue;
    int       restype;
};


void backbone_dependent_point(
        SysArray   output,
        CoordArray rama,
        CoordArray alignment,
        BackboneSCParam* params,
        const LayeredPeriodicSpline2D<3> &map,
        int n_term, int n_system) {
    const float scale = map.nx * (0.5f/M_PI_F - 1e-7f);
    const float shift = M_PI_F;

#pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_term; ++nt) {
            MutableCoord<3> com_rotated(output, ns, nt);
            Coord<2,3>      r    (rama,     ns, params[nt].rama_residue);
            AffineCoord<3>  body(alignment, ns, params[nt].alignment_residue);

            float v[9];
            map.evaluate_value_and_deriv(v, params[nt].restype,
                    (r.v[0]+shift)*scale, (r.v[1]+shift)*scale);
            float3 dcom_dphi = make_float3(v[0], v[3], v[6]) * scale;
            float3 dcom_dpsi = make_float3(v[1], v[4], v[7]) * scale;
            float3 com       = make_float3(v[2], v[5], v[8]);

            // rotate reference derivatives into the body frame
            com_rotated.set_value(body.apply(com));
            float3 dcom_dphi_rotated = body.apply_rotation(dcom_dphi);
            float3 dcom_dpsi_rotated = body.apply_rotation(dcom_dpsi);

            r.d[0][0] = dcom_dphi_rotated.x;  r.d[0][1] = dcom_dpsi_rotated.x;
            r.d[1][0] = dcom_dphi_rotated.y;  r.d[1][1] = dcom_dpsi_rotated.y;
            r.d[2][0] = dcom_dphi_rotated.z;  r.d[2][1] = dcom_dpsi_rotated.z;

            body.add_deriv_at_location(0, com_rotated.f3(), make_float3(1.f, 0.f, 0.f));
            body.add_deriv_at_location(1, com_rotated.f3(), make_float3(0.f, 1.f, 0.f));
            body.add_deriv_at_location(2, com_rotated.f3(), make_float3(0.f, 0.f, 1.f));

            com_rotated.flush();
            body.flush();
            r.flush();
        }
    }
}


struct BackboneDependentPoint : public CoordNode
{
    CoordNode& rama;
    CoordNode& alignment;

    LayeredPeriodicSpline2D<3> backbone_point_map;
    vector<BackboneSCParam> params;
    vector<AutoDiffParams> autodiff_params;

    BackboneDependentPoint(hid_t grp, CoordNode& rama_, CoordNode& alignment_):
        CoordNode(rama_.n_system, get_dset_size(1, grp, "restype")[0], 3),
        rama(rama_), alignment(alignment_),
        backbone_point_map(
                get_dset_size(4, grp, "backbone_point_map")[0],
                get_dset_size(4, grp, "backbone_point_map")[1],
                get_dset_size(4, grp, "backbone_point_map")[2]),
        params(n_elem)
    {
        auto &bbm = backbone_point_map;

        check_elem_width(rama,     2);
        check_elem_width(alignment, 7);

        check_size(grp, "rama_residue",       n_elem);
        check_size(grp, "alignment_residue",  n_elem);
        check_size(grp, "restype",            n_elem);
        check_size(grp, "backbone_point_map", bbm.n_layer, bbm.nx, bbm.ny, 3);

        if(bbm.nx != bbm.ny) throw string("must have same x and y grid spacing for Rama maps");
        vector<double> backbone_point_map_data(bbm.n_layer * bbm.nx * bbm.ny * 3);

        traverse_dset<1,int>(grp, "rama_residue",      [&](size_t nr, int x) {params[nr].rama_residue.index = x;});
        traverse_dset<1,int>(grp, "alignment_residue", [&](size_t nr, int x) {params[nr].alignment_residue.index = x;});
        traverse_dset<1,int>(grp, "restype",           [&](size_t nr, int x) {params[nr].restype = x;});

        traverse_dset<4,float>(grp, "backbone_point_map", 
                [&](size_t rt, size_t nb1, size_t nb2, size_t d, float x) {
                backbone_point_map_data.at(((rt*bbm.nx + nb1)*bbm.ny + nb2)*3 + d) = x;});
        bbm.fit_spline(backbone_point_map_data.data());

        for(size_t i=0; i<params.size(); ++i) rama     .slot_machine.add_request(3, params[i].rama_residue);
        for(size_t i=0; i<params.size(); ++i) alignment.slot_machine.add_request(3, params[i].alignment_residue);

        for(auto &p: params) autodiff_params.push_back(
                AutoDiffParams({p.rama_residue.slot}, {p.alignment_residue.slot}));
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("backbone_point"));
        backbone_dependent_point(
                coords().value, rama.coords(), alignment.coords(),
                params.data(), backbone_point_map, n_elem, alignment.n_system);
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

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs;
        for(auto &p: params) coord_pairs.push_back(vector<CoordPair>{{p.rama_residue}});
        double rama_rel_error = compute_relative_deviation_for_node<2>(*this, rama, coord_pairs);

        coord_pairs.clear();
        for(auto &p: params) coord_pairs.push_back(vector<CoordPair>{{p.alignment_residue}});
        double align_rel_error = compute_relative_deviation_for_node<7,BackboneDependentPoint,BODY_VALUE>(
                *this, alignment, coord_pairs);

        // printf("both relative errors %f %f\n", rama_rel_error, align_rel_error);
        return align_rel_error + rama_rel_error;
    }
};
static RegisterNodeType<BackboneDependentPoint,2> backbone_dependent_point_node("backbone_dependent_point");
