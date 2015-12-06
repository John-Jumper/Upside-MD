#include <cstdint>
#include <cmath>

struct OrnsteinUhlenbeckThermostat
        // following the notation in Gillespie, 1996
{
    private: 
        uint64_t n_invocations;
        void update_parameters() {
            mom_scale   = exp(-delta_t/timescale);
            noise_scale = sqrtf(temp * (1-mom_scale*mom_scale));
        }

    public:
        uint32_t random_seed;
        float timescale;
        float delta_t;
        float mom_scale;

        float temp;
        float noise_scale;

        OrnsteinUhlenbeckThermostat() {}
        OrnsteinUhlenbeckThermostat(uint32_t random_seed_, float timescale_, float temp_, float delta_t_):
            n_invocations(0),
            random_seed(random_seed_), timescale(timescale_), delta_t(delta_t_), temp(temp_), noise_scale(0.f)
            {
                update_parameters();
            }

        OrnsteinUhlenbeckThermostat& set_timescale(float timescale_) {
            timescale = timescale_; update_parameters(); return *this;}
        OrnsteinUhlenbeckThermostat& set_temp     (float temp_)      {
            temp      = temp_;      update_parameters(); return *this;}
        OrnsteinUhlenbeckThermostat& set_delta_t  (float delta_t_)   {
            delta_t   = delta_t_;   update_parameters(); return *this;}

        void apply(VecArray mom, int n_atom); 
};
