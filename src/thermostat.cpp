#include "deriv_engine.h"
#include "thermostat.h"
#include "md_export.h"
#include "timing.h"
#include "random.h"
#include <string>

using namespace std;

void OrnsteinUhlenbeckThermostat::apply(SysArray mom, int n_atom) {
    Timer timer(string("thermostat"));

    for(int na=0; na<n_atom; ++na) {
        RandomGenerator random(random_seed, THERMOSTAT_RANDOM_STREAM, na, n_invocations);
        MutableCoord<3> p(mom, 0, na);
        p.set_value(mom_scale*p.f3() + noise_scale*random.normal3());
        p.flush();
    }
    n_invocations++;
}
