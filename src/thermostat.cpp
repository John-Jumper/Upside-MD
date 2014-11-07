#include "thermostat.h"
#include "md_export.h"
#include "timing.h"
#include <string>

using namespace std;

void OrnsteinUhlenbeckThermostat::apply(SysArray mom, int n_atom, int n_system) {
    Timer timer(string("thermostat"));
    ornstein_uhlenbeck_thermostat(mom, mom_scale, noise_scale, random_seed, n_atom, 
            n_invocations++, n_system);
}
