#include "thermostat.h"
#include "md_export.h"
#include "timing.h"
#include <string>

using namespace std;

void OrnsteinUhlenbeckThermostat::apply(float* mom, int n_atom) {
    Timer timer(string("thermostat"));
    ornstein_uhlenbeck_thermostat(mom, mom_scale, noise_scale, random_seed, n_atom, 
            n_invocations++);
}
