#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "affine.h"
#include <cmath>
#include "h5_support.h"
#include <vector>
#include "spline.h"

using namespace h5;
using namespace std;

struct MembranePotential : public PotentialNode
{
    struct MembraneResidueParams {
        // Logically the two residues are the same, but they may have different
        // indices in the CB and Env outputs respectively
        index_t cb_index;
        index_t env_index;
        int restype;
    };

    struct MembranePotentialCBParams {
        float cov_midpoint;
        float cov_sharpness;
    };

    int n_elem;        // number of residues to process
    int n_restype;
    int n_donor, n_acceptor;

    CoordNode& res_pos;  // CB atom
    CoordNode& environment_coverage;
    CoordNode& protein_hbond;

    vector<MembraneResidueParams> res_params;
    vector<MembranePotentialCBParams> pot_params;

    LayeredClampedSpline1D<1> membrane_energy_cb_spline;
    LayeredClampedSpline1D<1> membrane_energy_uhb_spline;

    // shift and scale to convert z coordinates to spline coordinates
    float cb_z_shift, cb_z_scale;
    float uhb_z_shift, uhb_z_scale;

    MembranePotential(hid_t grp, CoordNode& res_pos_,
                                 CoordNode& environment_coverage_,
                                 CoordNode& protein_hbond_):
        PotentialNode(),

        n_elem    (get_dset_size(1, grp,             "cb_index")[0]),
        n_restype (get_dset_size(2, grp,            "cb_energy")[0]),
        n_donor   (get_dset_size(1, grp,    "donor_residue_ids")[0]),
        n_acceptor(get_dset_size(1, grp, "acceptor_residue_ids")[0]),

        res_pos(res_pos_),
        environment_coverage(environment_coverage_),
        protein_hbond(protein_hbond_),

        res_params(n_elem),
        pot_params(n_restype),

        membrane_energy_cb_spline(
                get_dset_size(2, grp, "cb_energy")[0],
                get_dset_size(2, grp, "cb_energy")[1]),

        membrane_energy_uhb_spline(
                get_dset_size(2, grp, "uhb_energy")[0],
                get_dset_size(2, grp, "uhb_energy")[1]),

        cb_z_shift(-read_attribute<float>(grp, "cb_energy", "z_min")),
        cb_z_scale((membrane_energy_cb_spline.nx-1)/(read_attribute<float>(grp, "cb_energy", "z_max")+cb_z_shift)),

        uhb_z_shift(-read_attribute<float>(grp, "uhb_energy", "z_min")),
        uhb_z_scale((membrane_energy_uhb_spline.nx-1)/(read_attribute<float>(grp, "uhb_energy", "z_max")+uhb_z_shift))
    {
        check_elem_width_lower_bound(res_pos, 3);
        check_elem_width_lower_bound(environment_coverage, 1);

        check_size(grp,      "cb_index",    n_elem);
        check_size(grp,     "env_index",    n_elem);
        check_size(grp,  "residue_type",    n_elem);
        check_size(grp,  "cov_midpoint", n_restype);
        check_size(grp, "cov_sharpness", n_restype);
        check_size(grp,     "cb_energy", n_restype, membrane_energy_cb_spline.nx);
        check_size(grp,    "uhb_energy",         2, membrane_energy_uhb_spline.nx); // type 0 for unpaird donor, type 1 for unpaird acceptor

        traverse_dset<1,  int>(grp,      "cb_index", [&](size_t nr,   int  x) {res_params[nr].cb_index  = x;});
        traverse_dset<1,  int>(grp,     "env_index", [&](size_t nr,   int  x) {res_params[nr].env_index = x;});
        traverse_dset<1,  int>(grp,  "residue_type", [&](size_t nr,   int rt) {res_params[nr].restype   = rt;});
        traverse_dset<1,float>(grp,  "cov_midpoint", [&](size_t rt, float bc) {pot_params[rt].cov_midpoint  = bc;});
        traverse_dset<1,float>(grp, "cov_sharpness", [&](size_t rt, float bw) {pot_params[rt].cov_sharpness = bw;});

        vector<double> cb_energy_data;
        traverse_dset<2,double>(grp, "cb_energy", [&](size_t rt, size_t z_index, double value) {
                cb_energy_data.push_back(value);});
        membrane_energy_cb_spline.fit_spline(cb_energy_data.data());

        vector<double> uhb_energy_data;
        traverse_dset<2,double>(grp, "uhb_energy", [&](size_t rt, size_t z_index, double value) {
                uhb_energy_data.push_back(value);});
        membrane_energy_uhb_spline.fit_spline(uhb_energy_data.data());
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("membrane_potential"));

        VecArray cb_pos       = res_pos.output;
        VecArray cb_pos_sens  = res_pos.sens;
        VecArray env_cov      = environment_coverage.output;
        VecArray env_cov_sens = environment_coverage.sens;
        VecArray hb_pos       = protein_hbond.output;
        VecArray hb_sens      = protein_hbond.sens;

        potential = 0.f;

        for(int nr=0; nr<n_elem; ++nr) {
            auto &p = res_params[nr];
            float cb_z = cb_pos(2, p.cb_index);

            float result[2];    // deriv then value
            membrane_energy_cb_spline.evaluate_value_and_deriv(result, p.restype, (cb_z + cb_z_shift) * cb_z_scale);
            float spline_value = result[1];
            float spline_deriv = result[0]*cb_z_scale;

            auto cover_sig = compact_sigmoid(
              env_cov(0, p.env_index)-pot_params[p.restype].cov_midpoint,
              pot_params[p.restype].cov_sharpness);

            potential                    += spline_value*cover_sig.x();
            cb_pos_sens (2, p.cb_index)  += spline_deriv*cover_sig.x();
            env_cov_sens(0, p.env_index) += spline_value*cover_sig.y();
        }

        int n_virtual = n_donor+n_acceptor;
        for(int nv=0; nv<n_virtual; ++nv) {
            float hb_z    = hb_pos(2, nv);
            float hb_prob = hb_pos(6, nv);  // probability of forming hbond

            float result[2];
            membrane_energy_uhb_spline.evaluate_value_and_deriv(result, int(nv>=n_donor), (hb_z + uhb_z_shift) * uhb_z_scale);
            float spline_value = result[1];
            float spline_deriv = result[0]*uhb_z_scale;

            float uhb_prob  = 1-hb_prob;
            float uhb_prob2 = uhb_prob*uhb_prob;

            potential      += spline_value*uhb_prob2;
            hb_sens(2, nv) += spline_deriv*uhb_prob2;
            hb_sens(6, nv) += spline_value*(-2)*uhb_prob; 
        }
    }
};

static RegisterNodeType<MembranePotential, 3> membrane_potential_node("membrane_potential");
