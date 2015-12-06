#include "deriv_engine.h"
#include "thermostat.h"
#include "timing.h"
#include "random.h"
#include <string>

using namespace std;

void OrnsteinUhlenbeckThermostat::apply(VecArray mom, int n_atom) {
    Timer timer(string("thermostat"));

    for(int na=0; na<n_atom; ++na) {
        RandomGenerator random(random_seed, THERMOSTAT_RANDOM_STREAM, na, n_invocations);
        auto p = load_vec<3>(mom, na);
        store_vec(mom, na, mom_scale*p + noise_scale*random.normal3());
    }
    n_invocations++;
}
