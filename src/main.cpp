#include "pivot_sampler.h"
#include "h5_support.h"
#include <tclap/CmdLine.h>
#include "deriv_engine.h"
#include "timing.h"
#include "thermostat.h"
#include <chrono>
#include "md_export.h"
#include <algorithm>
#include "random.h"
#include "state_logger.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;
using namespace h5;


void parallel_tempering_step(
        uint32_t seed, uint64_t round,
        vector<int>& coord_indices,
        const vector<float>& temperatures, DerivEngine& engine) {
    int n_system = engine.pos->n_system;
    if(int(temperatures.size()) != n_system) throw string("impossible");
    if(int(coord_indices.size()) != n_system) throw string("impossible");

    vector<float> beta(n_system);
    for(int i=0; i<n_system; ++i) beta[i] = 1.f/temperatures[i];

    // compute the boltzmann factors for everyone
    auto compute_log_boltzmann = [&]() {
        engine.compute(PotentialAndDerivMode);
        vector<float> result(n_system);
        for(int i=0; i<n_system; ++i) 
            result[i] = -beta[i]*engine.potential[i];
        return result;
    };

    // swap coordinates and the associated system indices
    auto coord_swap = [&](int ns1, int ns2) {
        SysArray pos = engine.pos->coords().value;
        swap_ranges(pos[ns1].v, pos[ns1+1].v, pos[ns2].v);
        swap(coord_indices[ns1], coord_indices[ns2]);
    };

    RandomGenerator random(seed, REPLICA_EXCHANGE_RANDOM_STREAM, 0u, round);

    // attempt neighbor swaps, first (2*i,2*i+1) swaps, then (2*i-1,2*i)
    int start = random.uniform_open_closed().x() < 0.5f;  // must start at 0 or 1 randomly

    auto old_lboltz = compute_log_boltzmann();
    for(int i=start; i+1<n_system; i+=2) coord_swap(i,i+1);
    auto new_lboltz  = compute_log_boltzmann();

    // reverse all swaps that should not occur by metropolis criterion
    int n_trial = 0; 
    int n_accept = 0;
    for(int i=start; i+1<n_system; i+=2) {
        n_trial++;
        float lboltz_diff = (new_lboltz[i] + new_lboltz[i+1]) - (old_lboltz[i]+old_lboltz[i+1]);
        // If we reject the swap, we must reverse it
        if(lboltz_diff < 0.f && expf(lboltz_diff) < random.uniform_open_closed().x()) {
            coord_swap(i,i+1);
        } else {
            n_accept++;
        }
    }
    // This function could probably do with fewer potential evaluations,
    // but it is called so rarely that it is unlikely to matter.
    // It is important that the energy is computed more than once in case
    // we are doing Hamiltonian parallel tempering rather than 
    // temperature parallel tempering
}


void deriv_matching(hid_t config, DerivEngine& engine, bool generate, double deriv_tol=1e-3) {
    auto &pos = *engine.pos;
    if(generate) {
        auto group = ensure_group(config, "/testing");
        ensure_not_exist(group.get(), "expected_deriv");
        auto tbl = create_earray(group.get(), "expected_deriv", H5T_NATIVE_FLOAT, 
                {pos.n_atom, 3, 0}, {pos.n_atom, 3, 1});
        vector<float> deriv_value(pos.n_system*pos.n_atom*3);
        for(int ns=0; ns<pos.n_system; ++ns) 
            for(int na=0; na<pos.n_atom; ++na)
                for(int d=0; d<3; ++d)
                    deriv_value[ns*pos.n_atom*3 + na*3 + d] = pos.deriv_array()[ns](d,na);
        append_to_dset(tbl.get(), deriv_value, 2);
    }

    if(h5_exists(config, "/testing/expected_deriv")) {
        check_size(config, "/testing/expected_deriv", pos.n_atom, 3, pos.n_system);
        double rms_error = 0.;
        traverse_dset<3,float>(config, "/testing/expected_deriv", [&](size_t na, size_t d, size_t ns, float x) {
                double dev = x - pos.deriv_array()[ns](d,na);
                rms_error += dev*dev;});
        rms_error = sqrtf(rms_error / pos.n_atom / pos.n_system);
        printf("RMS deriv difference: %.6f\n", rms_error);
        // if(rms_error > deriv_tol) throw string("inacceptable deriv deviation");
    }
}


vector<float> potential_deriv_agreement(DerivEngine& engine) {
    vector<float> relative_error;
    int n_atom = engine.pos->n_elem;
    SysArray pos_array = engine.pos->coords().value;

    for(int ns=0; ns<engine.pos->n_system; ++ns) {
        vector<float> input(n_atom*3);
        for(int na=0; na<n_atom; ++na)
            for(int d=0; d<3; ++d)
                input[na*3+d] = pos_array[ns](d,na);
        vector<float> output(1);

        auto do_compute = [&]() {
            for(int na=0; na<n_atom; ++na)
                for(int d=0; d<3; ++d)
                    pos_array[ns](d,na) = input[na*3+d];
            engine.compute(PotentialAndDerivMode);
            output[0] = engine.potential[ns];
        };

        for(auto &n: engine.nodes) {
            if(n.computation->potential_term) {
                auto &v = dynamic_cast<PotentialNode&>(*n.computation.get()).potential;
                printf("%s:", n.name.c_str());
                for(auto e: v) printf(" % 4.3f", e);
                printf("\n");
            }
        }
        printf("\n\n");

        auto central_diff_jac = central_difference_deriviative(do_compute, input, output);
        vector<float> deriv_array;
        for(int na=0; na<n_atom; ++na)
            for(int d=0; d<3; ++d)
                deriv_array.push_back(engine.pos->deriv_array()[ns](d,na));

        relative_error.push_back(
                relative_rms_deviation(central_diff_jac, deriv_array));
    }
    return relative_error;
}



int main(int argc, const char* const * argv)
try {
    using namespace TCLAP;  // Templatized C++ Command Line Parser (tclap.sourceforge.net)
    CmdLine cmd("Using Protein Statistical Information for Dynamics Estimation (UPSIDE)\n Author: John Jumper", 
            ' ', "0.1");

    ValueArg<string> config_arg("", "config", 
            "path to .h5 file from make_sys.py that contains the simulation configuration",
            true, "", "file path", cmd);
    ValueArg<double> time_step_arg("", "time-step", "time step for integration (default 0.01)", 
            false, 0.01, "float", cmd);
    ValueArg<double> duration_arg("", "duration", "duration of simulation", 
            true, -1., "float", cmd);
    ValueArg<unsigned long> seed_arg("", "seed", "random seed (default 42)", 
            false, 42l, "int", cmd);
    SwitchArg overwrite_output_arg("", "overwrite-output", 
            "overwrite the output group of the system file if present (default false)", 
            cmd, false);
    ValueArg<double> temperature_arg("", "temperature", "thermostat temperature (default 1.0)", 
            false, 1., "float", cmd);
    ValueArg<double> max_temperature_arg("", "max-temperature", "maximum thermostat temperature (useful for parallel tempering)", 
            false, -1., "float", cmd);
    ValueArg<double> frame_interval_arg("", "frame-interval", "simulation time between frames", 
            true, -1., "float", cmd);
    ValueArg<double> replica_interval_arg("", "replica-interval", 
            "simulation time between applications of replica exchange (0 means no replica exchange, default 0.)", 
            false, 0., "float", cmd);
    ValueArg<double> pivot_interval_arg("", "pivot-interval", 
            "simulation time between attempts to pivot move (0 means no pivot moves, default 0.)", 
            false, 0., "float", cmd);
    ValueArg<double> thermostat_interval_arg("", "thermostat-interval", 
            "simulation time between applications of the thermostat", 
            false, -1., "float", cmd);
    ValueArg<double> thermostat_timescale_arg("", "thermostat-timescale", "timescale for the thermostat", 
            false, 5., "float", cmd);
    SwitchArg disable_recenter_arg("", "disable-recentering", 
            "Disable all recentering of protein in the universe", 
            cmd, false);
    SwitchArg disable_z_recenter_arg("", "disable-z-recentering", 
            "Disable z-recentering of protein in the universe", 
            cmd, false);
    ValueArg<double> equilibration_duration_arg("", "equilibration-duration", 
            "duration to limit max force for equilibration (also decreases thermostat interval)", 
            false, 0., "float", cmd);
    SwitchArg generate_expected_deriv_arg("", "generate-expected-deriv", 
            "write an expected deriv to the input for later testing (developer only)", 
            cmd, false);
    ValueArg<string> re_energize_arg("", "re-energize", 
            "path to .h5 file with a trajectory under /output/pos for which the energy will be recalculated "
            "using the energy in --config.  Must have the same number of systems as --config and must be a "
            "different file than --config.", false, "", "file path", cmd);
    ValueArg<string> log_level_arg("", "log-level", 
            "Use this option to control which arrays are stored in /output.  Availabe levels are basic, detailed, "
            "or extensive.  Default is basic.",
            false, "", "basic, detailed, extensive", cmd);
    SwitchArg potential_deriv_agreement_arg("", "potential-deriv-agreement",
            "(developer use only) check the agreement of the derivative with finite differences "
            "of the potential for the initial structure.  This may give strange answers for native structures "
            "(no steric clashes may given an agreement of NaN) or random structures (where bonds and angles are "
            "exactly at their equilibrium values).  Interpret these results at your own risk.", cmd, false);
    cmd.parse(argc, argv);

    printf("invocation:");
    std::string invocation(argv[0]);
    for(auto arg=argv+1; arg!=argv+argc; ++arg) invocation += string(" ") + *arg;
    printf("%s\n", invocation.c_str());

    try {
        h5_noerr(H5Eset_auto(H5E_DEFAULT, nullptr, nullptr));
        H5Obj config;
        try {
            config = h5_obj(H5Fclose, H5Fopen(config_arg.getValue().c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
        } catch(string &s) {
            throw string("Unable to open configuration file at ") + config_arg.getValue();
        }

        if(h5_exists(config.get(), "/output", false)) {
            // Note that it is not possible in HDF5 1.8.x to reclaim space by deleting
            // datasets or groups.  Subsequent h5repack will reclaim space, however.
            if(overwrite_output_arg.getValue()) h5_noerr(H5Ldelete(config.get(), "/output", H5P_DEFAULT));
            else throw string("/output already exists and --overwrite-output was not specified");
        }

        {
            LogLevel log_level;
            if     (log_level_arg.getValue() == "")          log_level = LOG_BASIC;
            else if(log_level_arg.getValue() == "basic")     log_level = LOG_BASIC;
            else if(log_level_arg.getValue() == "detailed")  log_level = LOG_DETAILED;
            else if(log_level_arg.getValue() == "extensive") log_level = LOG_EXTENSIVE;
            else throw string("Illegal value for --log-level");

            default_logger = move(unique_ptr<H5Logger>(new H5Logger(config, "output", log_level)));
        }
        write_string_attribute(config.get(), "output", "invocation", invocation);

        auto pos_shape = get_dset_size(3, config.get(), "/input/pos");
        int  n_atom   = pos_shape[0];
        int  n_system = pos_shape[2];
        if(pos_shape[1]!=3) throw string("invalid dimensions for initial position");

        #if defined(_OPENMP)
        // current we only use OpenMP parallelism over systems
        if(n_system < omp_get_max_threads()) omp_set_num_threads(n_system);
        if(omp_get_max_threads() > 1) 
            printf("Multi-threaded execution with %i threads\n\n", omp_get_max_threads());
        else
            printf("Single-threaded execution\n\n");
        #endif

        auto potential_group = open_group(config.get(), "/input/potential");
        auto engine = initialize_engine_from_hdf5(n_atom, n_system, potential_group.get());
        traverse_dset<3,float>(config.get(), "/input/pos", [&](size_t na, size_t d, size_t ns, float x) { 
                engine.pos->coords().value[ns](d,na) = x;});
        printf("\nn_atom %i\nn_system %i\n", engine.pos->n_atom, engine.pos->n_system);

        engine.compute(PotentialAndDerivMode);
        printf("Initial potential energy:");
        for(float e: engine.potential) printf(" %.2f", e);
        printf("\n");

        deriv_matching(config.get(), engine, generate_expected_deriv_arg.getValue());
        if(potential_deriv_agreement_arg.getValue()){
            if(n_system>1) throw string("Testing code does not support n_system > 1");
            printf("Initial agreement:\n");
            for(auto &n: engine.nodes)
                printf("%24s %f\n", n.name.c_str(), n.computation->test_value_deriv_agreement());
            printf("\n");

            auto relative_error = potential_deriv_agreement(engine);
            printf("overall potential relative error: ");
            for(auto r: relative_error) printf(" %.5f", r);
            printf("\n");
        }

        float dt = time_step_arg.getValue();
        double duration = duration_arg.getValue();
        uint64_t n_round = round(duration / (3*dt));
        int thermostat_interval = max(1.,round(thermostat_interval_arg.getValue() / (3*dt)));
        int frame_interval = max(1.,round(frame_interval_arg.getValue() / (3*dt)));

        unsigned long big_prime = 4294967291ul;  // largest prime smaller than 2^32
        uint32_t random_seed = uint32_t(seed_arg.getValue() % big_prime);

        int pivot_interval = pivot_interval_arg.getValue() > 0. 
            ? max(1,int(pivot_interval_arg.getValue()/(3*dt)))
            : 0;

        PivotSampler pivot_sampler;
        if(pivot_interval) {
            pivot_sampler = PivotSampler{open_group(config.get(), "/input/pivot_moves").get(), n_system};
            default_logger->add_logger<int>("pivot_stats", {n_system,2}, [&](int* stats_buffer) {
                    for(int ns=0; ns<n_system; ++ns) {
                        stats_buffer[2*ns+0] = pivot_sampler.pivot_stats[ns].n_success;
                        stats_buffer[2*ns+1] = pivot_sampler.pivot_stats[ns].n_attempt;
                        pivot_sampler.pivot_stats[ns].reset();
                    }});
        }

        vector<float> temperature(n_system);
        float max_temp = max_temperature_arg.getValue();
        float min_temp = temperature_arg.getValue();
        if(max_temp == -1.) max_temp = min_temp;

        if(max_temp != min_temp && n_system == 1) 
            throw string("--max-temperature cannot be specified for only a single system");
        if(n_system==1) {
            temperature[0] = temperature_arg.getValue();
        } else {
            // system 0 is the minimum temperature
            // tighter spacing at the low end of temperatures because the variance is decreasing
            for(int ns=0; ns<n_system; ++ns) 
                temperature[ns] = sqr((sqrt(min_temp)*(n_system-1-ns) + sqrt(max_temp)*ns)/(n_system-1));
        }
        printf("temperature:");
        for(auto t: temperature) printf(" %f", t);
        printf("\n");

        default_logger->log_once<double>("temperature", {n_system}, [&](double* temperature_buffer) {
                for(int ns=0; ns<n_system; ++ns) temperature_buffer[ns] = temperature[ns];});

        float equil_duration = equilibration_duration_arg.getValue();
        // equilibration_max_force is set so that the change in momentum should not be more than
        // 20% of the equilibration magnitude over a single dt interval
        float equil_avg_mom   = sqrtf(temperature[n_system-1]/1.5f);  // all particles have mass == 1.
        float equil_max_force = 0.8f*equil_avg_mom/dt;

        // initialize thermostat and thermalize momentum
        vector<float> mom(n_atom*n_system*3, 0.f);
        SysArray mom_sys(mom.data(), n_atom*3, n_atom);

        printf("random seed: %lu\n", (unsigned long)(random_seed));
        auto thermostat = OrnsteinUhlenbeckThermostat(
                random_seed,
                thermostat_timescale_arg.getValue(),
                temperature,
                1e8);
        thermostat.apply(mom_sys, n_atom); // initial thermalization
        thermostat.set_delta_t(thermostat_interval*3*dt);  // set true thermostat interval

        default_logger->add_logger<float>("pos", {n_system, n_atom, 3}, [&](float* pos_buffer) {
                SysArray pos_array = engine.pos->coords().value;
                for(int ns=0; ns<n_system; ++ns)
                    for(int na=0; na<n_atom; ++na) 
                        for(int d=0; d<3; ++d) 
                            pos_buffer[ns*n_atom*3 + na*3 + d] = pos_array[ns](d,na);
            });
        default_logger->add_logger<double>("kinetic", {n_system}, [&](double* kin_buffer) {
            for(int ns=0; ns<n_system; ++ns) {
                double sum_kin = 0.f;
                for(int na=0; na<n_atom; ++na) sum_kin += mag2(StaticCoord<3>(mom_sys, ns, na).f3());
                kin_buffer[ns] = (0.5/n_atom)*sum_kin;  // kinetic_energy = (1/2) * <mom^2>
            }});
        default_logger->add_logger<double>("potential", {n_system}, [&](double* pot_buffer) {
                engine.compute(PotentialAndDerivMode);
                for(int ns=0; ns<n_system; ++ns) pot_buffer[ns] = engine.potential[ns];});

        int duration_print_width = ceil(log(1+duration)/log(10));

        int replica_interval = 0;
        if(replica_interval_arg.getValue())
            replica_interval = max(1.,replica_interval_arg.getValue()/(3*dt));

        vector<int> coord_indices; 
        for(int ns=0; ns<n_system; ++ns) coord_indices.push_back(ns);
        default_logger->add_logger<int>("coord_indices", {n_system}, [&](int* indices_buffer) {
                copy_n(begin(coord_indices), n_system, indices_buffer);});

        bool do_recenter = !disable_recenter_arg.getValue();
        bool xy_recenter_only = do_recenter && disable_z_recenter_arg.getValue();

        // quick hack of a check for z-centering and membrane potential
        if(do_recenter && !xy_recenter_only) {
            for(auto &n: engine.nodes) {
                if(is_prefix(n.name, "membrane_potential") || is_prefix(n.name, "z_flat_bottom"))
                    throw string("You have z-centering and a z-dependent potential turned on.  "
                            "This is not what you want.  Consider --disable-z-recentering "
                            "or --disable-recentering.");
            }
        }

        if(do_recenter) {
            for(auto &n: engine.nodes) {
                if(is_prefix(n.name, "cavity_radial") || is_prefix(n.name, "tension"))
                    throw string("You have re-centering and a position-dependent potential turned on.  "
                            "This is not what you want.  Consider --disable-recentering.");
            }
        }

        double physical_time = 0.;
        default_logger->add_logger<double>("time", {}, [&](double* time_buffer) {*time_buffer=physical_time;});

        if(re_energize_arg.getValue() != "") {
            H5Obj coord_config;
            try {
                coord_config = h5_obj(H5Fclose, H5Fopen(re_energize_arg.getValue().c_str(), 
                            H5F_ACC_RDWR, H5P_DEFAULT));
            } catch(string &s) {
                throw string("Unable to open configuration file at ") + config_arg.getValue();
            }

            int n_frame = get_dset_size(1, coord_config.get(), "/output/time")[0];
            printf("number of frames to re-energize %i\n", n_frame);
            check_size(coord_config.get(), "/output/time", n_frame);
            check_size(coord_config.get(), "/output/pos",  n_frame, n_system, n_atom, 3);

            // first read the physical times
            vector<double> all_times(n_frame);
            traverse_dset<1,double>(coord_config.get(), "/output/time", [&](size_t nf, double x) {
                    all_times.push_back(x);});

            // now load the coordinate information, processing a frame as soon as it is ready
            SysArray pos_sys = engine.pos->coords().value;
            traverse_dset<4,float>(coord_config.get(), "/output/pos", 
                    [&](size_t nf, size_t ns, size_t na, size_t d, float x) {
                    pos_sys[ns](d,na) = x;
                    if(int(na)==n_atom-1 && int(d)==2) { 
                         // we are done loading a frame, so process it
                         physical_time = all_times[nf];
                         default_logger->collect_samples();
                    }});

            if(h5_exists(coord_config.get(), "/output/potential")) {
                check_size(coord_config.get(), "/output/potential", n_frame, n_system);
                vector<double> old_potential(n_frame*n_system);
                traverse_dset<2,double>(coord_config.get(), "/output/potential", [&](int nf, int ns, double x){
                        old_potential[nf*n_system + ns] = x;});

                default_logger->log_once<double>("old_potential", {n_frame, n_system}, 
                        [&](double* buffer) {for(int i=0; i<n_frame*n_system; ++i) buffer[i] = old_potential[i];});
            }

            return 0; // do not run dynamics
        }

        auto tstart = chrono::high_resolution_clock::now();
        for(uint64_t nr=0; nr<n_round; ++nr) {
            physical_time = nr*3*double(dt);

            if(pivot_interval && !(nr%pivot_interval)) 
                pivot_sampler.pivot_monte_carlo_step(random_seed, nr, temperature, engine);

            if(replica_interval && !(nr%replica_interval))
                parallel_tempering_step(random_seed, nr, coord_indices, temperature, engine);

            if(!frame_interval || !(nr%frame_interval)) {
                if(do_recenter) recenter(engine.pos->coords().value, xy_recenter_only, n_atom, n_system);
                default_logger->collect_samples();

                double Rg = 0.f;
                for(int ns=0; ns<n_system; ++ns) {
                    float3 com = make_vec3(0.f, 0.f, 0.f);
                    for(int na=0; na<n_atom; ++na)
                        com += StaticCoord<3>(engine.pos->coords().value, ns, na).f3();;
                    com *= 1.f/n_atom;

                    for(int na=0; na<n_atom; ++na) 
                        Rg += mag2(StaticCoord<3>(engine.pos->coords().value, ns, na).f3()-com);
                }
                Rg = sqrt(Rg/(n_atom*n_system));

                printf("%*.0f / %*.0f elapsed %5.1f hbonds, Rg %5.1f A, potential", 
                        duration_print_width, physical_time, 
                        duration_print_width, duration, 
                        get_n_hbond(engine)/n_system, Rg);
                for(float e: engine.potential) printf(" % 8.2f", e);
                printf("\n");
                fflush(stdout);
            }
            // To be cautious, apply the thermostat more often if in the equilibration phase
            bool in_equil = nr*3*dt < equil_duration;
            if(!(nr%thermostat_interval) || in_equil) 
                thermostat.apply(mom_sys, n_atom);
            engine.integration_cycle(mom_sys, dt, (in_equil ? equil_max_force : 0.f), DerivEngine::Verlet);
        }
        default_logger->flush();

        auto elapsed = chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count();
        printf("\n\nfinished in %.1f seconds (%.2f us/systems/step, %.4f seconds/simulation_time_unit)\n",
                elapsed, elapsed*1e6/n_system/n_round/3, elapsed/duration_arg.getValue());

        {
            auto sum_kin = vector<double>(n_system, 0.);
            auto n_kin   = vector<long>  (n_system, 0);
            traverse_dset<2,float>(config.get(),"/output/kinetic", [&](size_t i, size_t ns, float x){
                    if(i>n_round*0.5 / frame_interval){ sum_kin[ns]+=x; n_kin[ns]++; }
                    });
            printf("\navg_kinetic_energy/1.5kT");
            for(int ns=0; ns<n_system; ++ns) printf(" % .4f", sum_kin[ns]/n_kin[ns] / (1.5*temperature[ns]));
            printf("\n");
        }

        if(pivot_interval) {
            std::vector<int64_t> ps(2*n_system);
            traverse_dset<3,int>(config.get(), "/output/pivot_stats", [&](size_t nf, size_t ns, int d, int x) {
                    ps[2*ns+d] += x;});
            printf("pivot_success:");
            for(int ns=0; ns<n_system; ++ns) printf(" % .4f", double(ps[2*ns+0])/double(ps[2*ns+1]));
            printf("\n");
        }

        printf("\n");
        global_time_keeper.print_report(3*n_round+1);
        printf("\n");
    } catch(const string &e) {
        fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
        return 1;
    } catch(...) {
        fprintf(stderr, "\n\nERROR: unknown error\n");
        return 1;
    }

    return 0;
} catch(const TCLAP::ArgException &e) { 
    fprintf(stderr, "\n\nERROR: %s for argument %s\n", e.error().c_str(), e.argId().c_str());
    return 1;
}
