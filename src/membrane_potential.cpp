#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include "coord.h"
#include "affine.h"
#include <cmath>
#include "h5_support.h"
#include <vector>
#include "spline.h"

using namespace h5;
using namespace std;

struct MembraneResidueParams {
    CoordPair residue;
    int restype; 
};

void membrane_potential(
        float* restrict potential,
        const CoordArray sc_com_pos,
        const MembraneResidueParams* params,
        const LayeredClampedSpline1D<1>& energy_spline,
        float z_shift, float z_scale,
        int n_residue)
{
    if(potential) potential[0] = 0.f;

    for(int nr=0; nr<n_residue; ++nr) {	                                                         
        Coord<3> pos1(sc_com_pos, params[nr].residue); 
        float result[2];  // deriv then value
        energy_spline.evaluate_value_and_deriv(result, params[nr].restype, (pos1.f3().z()+z_shift)*z_scale);
        pos1.set_deriv(make_vec3(0.f, 0.f, result[0]*z_scale));
        if(potential) potential[0] += result[1];
        pos1.flush();
    }
}


struct MembranePotential : public PotentialNode
{
    int n_elem;        // number of residues to process
    CoordNode& sidechain_pos;
    vector<MembraneResidueParams> params;
    LayeredClampedSpline1D<1> membrane_energy_spline;

    // shift and scale to convert z coordinates to spline coordinates
    float z_shift;
    float z_scale;

    MembranePotential(hid_t grp, CoordNode& sidechain_pos_):
        PotentialNode(),
        n_elem(get_dset_size(1, grp, "residue_id")[0]), 
        sidechain_pos(sidechain_pos_), 
        params(n_elem),
        membrane_energy_spline(
                get_dset_size(2, grp, "energy")[0],
                get_dset_size(2, grp, "energy")[1]),

        z_shift(-read_attribute<float>(grp, "energy", "z_min")),
        z_scale((membrane_energy_spline.nx-1)/(read_attribute<float>(grp, "energy", "z_max")+z_shift))
    {
        check_size(grp, "residue_id", n_elem);
        check_size(grp, "restype",    n_elem);

        traverse_dset<1,int>(grp, "residue_id", [&](size_t nda, int x ) {params[nda].residue.index = x;});
        traverse_dset<1,int>(grp, "restype",    [&](size_t nr,  int rt) {params[nr].restype = rt;});

        vector<double> energy_data;
        traverse_dset<2,double>(grp, "energy", [&](size_t rt, size_t z_index, double value) {
                energy_data.push_back(value);});
        membrane_energy_spline.fit_spline(energy_data.data());

        for(size_t i=0; i<params.size(); ++i)
            sidechain_pos.slot_machine.add_request(1, params[i].residue);

    }
    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("membrane_potential"));
        membrane_potential((mode==PotentialAndDerivMode ? potential.data() : nullptr),
                sidechain_pos.coords(), params.data(),
                membrane_energy_spline, z_shift, z_scale,
                n_elem);
    }
    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs(1);
        for(auto &p: params) coord_pairs.back().push_back(p.residue);
        return -1.; // compute_relative_deviation_for_node<3>(*this, sidechain_pos, coord_pairs);
    }

};

static RegisterNodeType<MembranePotential,1> membrane_potential_node("membrane_potential");
