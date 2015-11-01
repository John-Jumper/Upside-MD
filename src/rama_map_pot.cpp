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


struct RamaMapPot : public PotentialNode
{
    int n_residue;
    CoordNode& rama;
    vector<RamaMapParams> params;
    LayeredPeriodicSpline2D<1> rama_map_data;
    vector<float> residue_potential;

    RamaMapPot(hid_t grp, CoordNode& rama_):
        PotentialNode(),
        n_residue(get_dset_size(1, grp, "residue_id")[0]), 
        rama(rama_), 
        params(n_residue),
        rama_map_data(
                get_dset_size(3, grp, "rama_pot")[0], 
                get_dset_size(3, grp, "rama_pot")[1], 
                get_dset_size(3, grp, "rama_pot")[2]),
        residue_potential(n_residue)
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

        if(logging(LOG_DETAILED)) 
            default_logger->add_logger<float>("rama_map_potential", {n_residue}, [&](float* buffer) {
                for(int nr: range(n_residue)) 
                    buffer[nr] = residue_potential[nr];
                    });
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("rama_map_pot"));

        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        auto ramac = rama.coords();
        if(pot) *pot = 0.f;

        // add a litte paranoia to make sure there are no rounding problems
        const float scale = rama_map_data.nx * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        if(pot) *pot = 0.f;
        for(int nr=0; nr<n_residue; ++nr) {
            Coord<2> r(ramac, params[nr].residue);

            float map_value[3];
            rama_map_data.evaluate_value_and_deriv(map_value, params[nr].rama_map_id, 
                    (r.v[0]+shift)*scale, (r.v[1]+shift)*scale);

            if(pot) {*pot += map_value[2]; residue_potential[nr] = map_value[2];}
            r.d[0][0] = map_value[0] * scale;
            r.d[0][1] = map_value[1] * scale;
            r.flush();
        }
    }

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs(1);
        for(auto &p: params) coord_pairs.back().push_back(p.residue);
        return -1.; // compute_relative_deviation_for_node<2>(*this, rama, coord_pairs);
    }

#ifdef PARAM_DERIV
    virtual void set_param(const std::vector<float>& new_param) {
        auto& r = rama_map_data;
        vector<double> raw_data(r.n_layer * r.nx * r.ny);
        if(raw_data.size() != new_param.size()) throw string("wrong number of parameters");
        for(size_t i=0u; i<size_t(r.n_layer * r.nx * r.ny); ++i) raw_data[i] = new_param[i];
        r.fit_spline(raw_data.data());
    }
#endif
};
static RegisterNodeType<RamaMapPot,1> rama_map_pot_node("rama_map_pot");
