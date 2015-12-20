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
        index_t residue;
        int restype; 
    };

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

        traverse_dset<1,int>(grp, "residue_id", [&](size_t nda, int x ) {params[nda].residue = x;});
        traverse_dset<1,int>(grp, "restype",    [&](size_t nr,  int rt) {params[nr].restype = rt;});

        vector<double> energy_data;
        traverse_dset<2,double>(grp, "energy", [&](size_t rt, size_t z_index, double value) {
                energy_data.push_back(value);});
        membrane_energy_spline.fit_spline(energy_data.data());
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("membrane_potential"));

        VecArray pos      = sidechain_pos.output;
        VecArray pos_sens = sidechain_pos.sens;
        potential = 0.f;

        for(int nr=0; nr<n_elem; ++nr) {
            auto &p = params[nr];
            float z = pos(2,p.residue);
            float result[2];    // deriv then value
            membrane_energy_spline.evaluate_value_and_deriv(result, p.restype, (z+z_shift)*z_scale);
            pos_sens(2,p.residue) += result[0];
            potential             += result[1];
        }
    }
};

static RegisterNodeType<MembranePotential,1> membrane_potential_node("membrane_potential");
