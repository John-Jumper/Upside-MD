#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"
#include "spline.h"
#include "state_logger.h"

using namespace h5;
using namespace std;

struct RamaMapParams {
    CoordPair residue;
    int       rama_map_id;
};

namespace {


void rama_map_pot(
        float* restrict potential,
        SysArray s_residue_potential,
        const CoordArray rama,
        const RamaMapParams* restrict params,
        const LayeredPeriodicSpline2D<1>& rama_map_data,
        int n_residue, int n_system) 
{
    // add a litte paranoia to make sure there are no rounding problems
    const float scale = rama_map_data.nx * (0.5f/M_PI_F - 1e-7f);
    const float shift = M_PI_F;

    for(int ns=0; ns<n_system; ++ns) {
        VecArray residue_potential = s_residue_potential[ns];
        if(potential) potential[ns] = 0.f;

        for(int nr=0; nr<n_residue; ++nr) {
            Coord<2> r(rama, ns, params[nr].residue);

            float map_value[3];
            rama_map_data.evaluate_value_and_deriv(map_value, params[nr].rama_map_id, 
                    (r.v[0]+shift)*scale, (r.v[1]+shift)*scale);

            if(potential) {potential[ns] += map_value[2]; residue_potential(0,nr) = map_value[2];}
            r.d[0][0] = map_value[0] * scale;
            r.d[0][1] = map_value[1] * scale;
            r.flush();
        }
    }
}
}


struct RamaMapPot : public PotentialNode
{
    int n_residue;
    CoordNode& rama;
    vector<RamaMapParams> params;
    LayeredPeriodicSpline2D<1> rama_map_data;
    SysArrayStorage residue_potential;

    RamaMapPot(hid_t grp, CoordNode& rama_):
        PotentialNode(rama_.n_system),
        n_residue(get_dset_size(1, grp, "residue_id")[0]), 
        rama(rama_), 
        params(n_residue),
        rama_map_data(
                get_dset_size(3, grp, "rama_pot")[0], 
                get_dset_size(3, grp, "rama_pot")[1], 
                get_dset_size(3, grp, "rama_pot")[2]),
        residue_potential(n_system, 1, n_residue)
    {
        auto& r = rama_map_data;
        check_size(grp, "residue_id",     n_residue);
        check_size(grp, "rama_map_id",    n_residue);
        check_size(grp, "rama_pot",       r.n_layer, r.nx, r.ny);

        if(r.nx != r.ny) throw string("must have same x and y grid spacing for Rama maps");
        vector<double> raw_data(r.n_layer * r.nx * r.ny);

        traverse_dset<1,int>   (grp, "residue_id",  [&](size_t i, int x) {params[i].residue.index = x;});
        traverse_dset<1,int>   (grp, "rama_map_id", [&](size_t i, int x) {params[i].rama_map_id = x;});
        traverse_dset<3,double>(grp, "rama_pot",    [&](size_t il, size_t ix, size_t iy, double x) {
                raw_data[(il*r.nx + ix)*r.ny + iy] = x;});
        r.fit_spline(raw_data.data());

        for(size_t i=0; i<params.size(); ++i) rama.slot_machine.add_request(1, params[i].residue);

        if(default_logger) default_logger->add_logger<float>("rama_map_potential", {n_system,n_residue}, [&](float* buffer) {
                for(int ns: range(n_system)) 
                    for(int nr: range(n_residue)) 
                        buffer[ns*n_residue+nr] = residue_potential[ns](0,nr);
                        });
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("rama_map_pot"));
        rama_map_pot((mode==PotentialAndDerivMode ? potential.data() : nullptr), residue_potential.array(),
                rama.coords(), params.data(), rama_map_data, n_residue, n_system);
    }

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs(1);
        for(auto &p: params) coord_pairs.back().push_back(p.residue);
        return compute_relative_deviation_for_node<2>(*this, rama, coord_pairs);
    }
};
static RegisterNodeType<RamaMapPot,1> rama_map_pot_node("rama_map_pot");
