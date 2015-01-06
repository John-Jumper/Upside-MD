#include "deriv_engine.h"
#include "thermostat.h"
#include "md_export.h"
#include "timing.h"
#include "random.h"
#include <string>

using namespace std;

template <typename MutableCoordT>
void ornstein_uhlenbeck_thermostat_body(
        RandomGenerator& random,
        MutableCoordT &mom,
        const float mom_scale,
        const float noise_scale)
{
    mom.set_value(mom_scale*mom.f3() + noise_scale*random.normal3());
}


void
ornstein_uhlenbeck_thermostat(
        SysArray mom,
        const float mom_scale,
        const float noise_scale,
        int seed,  // FIXME uint32_t
        int n_atoms,
        int n_round, // FIXME uint64_t
        int n_system)
{
#pragma omp parallel for
    for(uint32_t ns=0; ns<(uint32_t)n_system; ++ns) {
        for(int na=0; na<n_atoms; ++na) {
            RandomGenerator random(seed, 0u, na*n_system + ns, n_round);
            MutableCoord<3> p(mom, ns, na);
            ornstein_uhlenbeck_thermostat_body(random, p, mom_scale, noise_scale);
            p.flush();
        }
    }
}

void OrnsteinUhlenbeckThermostat::apply(SysArray mom, int n_atom, int n_system) {
    Timer timer(string("thermostat"));
    ornstein_uhlenbeck_thermostat(mom, mom_scale, noise_scale, random_seed, n_atom, 
            n_invocations++, n_system);
}
