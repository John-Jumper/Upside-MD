#include "deriv_engine.h"
#include "thermostat.h"
#include "md_export.h"
#include "timing.h"
#include "random.h"
#include <string>

using namespace std;


void
ornstein_uhlenbeck_thermostat(
        SysArray mom,
        const float  mom_scale,
        const float* noise_scale,
        uint32_t seed,
        int n_atoms,
        uint64_t n_round,
        int n_system)
{
#pragma omp parallel for
    for(uint32_t ns=0; ns<(uint32_t)n_system; ++ns) {
        for(int na=0; na<n_atoms; ++na) {
            RandomGenerator random(seed, THERMOSTAT_RANDOM_STREAM, na*n_system + ns, n_round);
            MutableCoord<3> p(mom, ns, na);
            p.set_value(mom_scale*p.f3() + noise_scale[ns]*random.normal3());
            p.flush();
        }
    }
}

void OrnsteinUhlenbeckThermostat::apply(SysArray mom, int n_atom) {
    Timer timer(string("thermostat"));
    ornstein_uhlenbeck_thermostat(mom, mom_scale, noise_scale.data(), random_seed, n_atom, 
            n_invocations++, n_system);
}
