#include "deriv_engine.h"
#include "timing.h"
#include "spline.h"
#include "state_logger.h"

using namespace h5;
using namespace std;

struct RamaMapParams {
    index_t residue;
    int       rama_map_id;
};


struct RamaMapPot : public PotentialNode
{
    int n_residue;
    CoordNode& rama;
    vector<RamaMapParams> params;
    LayeredPeriodicSpline2D<1> rama_map_data;
    vector<float> residue_potential;
    bool log_pot; // if false, never log potential

    RamaMapPot(hid_t grp, CoordNode& rama_):
        PotentialNode(),
        n_residue(get_dset_size(1, grp, "residue_id")[0]), 
        rama(rama_), 
        params(n_residue),
        rama_map_data(
                get_dset_size(3, grp, "rama_pot")[0], 
                get_dset_size(3, grp, "rama_pot")[1], 
                get_dset_size(3, grp, "rama_pot")[2]),
        residue_potential(n_residue),
        log_pot(read_attribute<int>(grp,".","log_pot",1))
    {
        auto& r = rama_map_data;
        check_size(grp, "residue_id",     n_residue);
        check_size(grp, "rama_map_id",    n_residue);
        check_size(grp, "rama_pot",       r.n_layer, r.nx, r.ny);

        if(r.nx != r.ny) throw string("must have same x and y grid spacing for Rama maps");
        vector<double> raw_data(r.n_layer * r.nx * r.ny);

        traverse_dset<1,int>   (grp, "residue_id",  [&](size_t i, int x) {params[i].residue = x;});
        traverse_dset<1,int>   (grp, "rama_map_id", [&](size_t i, int x) {params[i].rama_map_id = x;});
        traverse_dset<3,double>(grp, "rama_pot",    [&](size_t il, size_t ix, size_t iy, double x) {
                raw_data[(il*r.nx + ix)*r.ny + iy] = x;});
        r.fit_spline(raw_data.data());

        if(log_pot && logging(LOG_DETAILED))
            default_logger->add_logger<float>("rama_map_potential", {n_residue}, [&](float* buffer) {
                for(int nr: range(n_residue)) 
                    buffer[nr] = residue_potential[nr];
                    });
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("rama_map_pot"));

        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        VecArray ramac     = rama.output;
        VecArray rama_sens = rama.sens;
        if(pot) *pot = 0.f;

        // add a litte paranoia to make sure there are no rounding problems
        const float scale = rama_map_data.nx * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        if(pot) *pot = 0.f;
        for(int nr=0; nr<n_residue; ++nr) {
            const auto& p = params[nr];
            auto r = load_vec<2>(ramac, p.residue);

            float value,dx,dy;
            rama_map_data.evaluate_value_and_deriv(&value,&dx,&dy, p.rama_map_id, 
                    (r.v[0]+shift)*scale, (r.v[1]+shift)*scale);

            if(pot) {*pot += value; residue_potential[nr] = value;}
            rama_sens(0,p.residue) += dx * scale;
            rama_sens(1,p.residue) += dy * scale;
        }
    }

#ifdef PARAM_DERIV
    virtual void set_param(const std::vector<float>& new_param) override {
        auto& r = rama_map_data;
        vector<double> raw_data(r.n_layer * r.nx * r.ny);
        if(raw_data.size() != new_param.size()) throw string("wrong number of parameters");
        for(size_t i=0u; i<size_t(r.n_layer * r.nx * r.ny); ++i) raw_data[i] = new_param[i];
        r.fit_spline(raw_data.data());
    }
#endif
};
static RegisterNodeType<RamaMapPot,1> rama_map_pot_node("rama_map_pot");
