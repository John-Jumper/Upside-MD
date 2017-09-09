#include "deriv_engine.h"
#include "thermostat.h"
#include "timing.h"
#include "random.h"
#include <string>

using namespace std;

void OrnsteinUhlenbeckThermostat::apply(VecArray mom, int n_atom) {
    Timer timer(string("thermostat"));
    RandomGenerator random(random_seed, THERMOSTAT_RANDOM_STREAM, 0, n_invocations);

    if(max_randn<n_atom*3) {
        max_randn = n_atom*3;
        randn.reset(new float[max_randn]);
    }
    random.many_normal(randn.get(), n_atom*3);

    for(int na=0; na<n_atom; ++na) {
        auto p = load_vec<3>(mom, na);
        auto r = make_vec3(randn[na*3+0], randn[3*na+1], randn[3*na+2]);
        store_vec(mom, na, mom_scale*p + noise_scale*r);
    }
    n_invocations++;
}
