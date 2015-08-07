#include <cstdint>
#include <cmath>
#include "coord.h"

struct OrnsteinUhlenbeckThermostat
        // following the notation in Gillespie, 1996
{
    private: 
        uint64_t n_invocations;
        void update_parameters() {
            mom_scale   = exp(-delta_t/timescale);
            for(int ns=0; ns<n_system; ++ns) 
                noise_scale[ns] = sqrtf(temp[ns] * (1-mom_scale*mom_scale));
        }

    public:
        int n_system;
        uint32_t random_seed;
        float timescale;
        float delta_t;
        float mom_scale;

        std::vector<float> temp;
        std::vector<float> noise_scale;

        OrnsteinUhlenbeckThermostat(): n_system(1) {}
        OrnsteinUhlenbeckThermostat(uint32_t random_seed_, float timescale_, std::vector<float> temp_, float delta_t_):
            n_invocations(0), n_system(temp_.size()),
            random_seed(random_seed_), timescale(timescale_), delta_t(delta_t_), temp(temp_), noise_scale(n_system)
            {
                update_parameters();}

        OrnsteinUhlenbeckThermostat& set_timescale(float timescale_) {
            timescale = timescale_; update_parameters(); return *this;}
        OrnsteinUhlenbeckThermostat& set_temp     (std::vector<float> temp_)      {
            temp      = temp_;      update_parameters(); return *this;}
        OrnsteinUhlenbeckThermostat& set_delta_t  (float delta_t_)   {
            delta_t   = delta_t_;   update_parameters(); return *this;}

        void apply(SysArray mom, int n_atom); 
};
