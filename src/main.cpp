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

struct System {
    int n_atom;
    uint32_t random_seed;
    float temperature;
    H5Obj config;
    shared_ptr<H5Logger> logger;
    DerivEngine engine;
    PivotSampler pivot_sampler;
    SysArrayStorage mom; // momentum
    OrnsteinUhlenbeckThermostat thermostat;
    uint64_t round_num;
    System(): round_num(0) {}
};


void parallel_tempering_step(
        uint32_t seed, uint64_t round,
        vector<System>& systems) {
    int n_system = systems.size();

    vector<float> beta(n_system);
    for(int i=0; i<n_system; ++i) beta[i] = 1.f/systems[i].temperature;

    // compute the boltzmann factors for everyone
    auto compute_log_boltzmann = [&]() {
        vector<float> result(n_system);
        for(int i=0; i<n_system; ++i) {
            systems[i].engine.compute(PotentialAndDerivMode);
            result[i] = -beta[i]*systems[i].engine.potential[0];
        }
        return result;
    };

    // swap coordinates and the associated system indices
    auto coord_swap = [&](int ns1, int ns2) {
        swap_ranges(
                systems[ns1].engine.pos->coords().value[0].v,  
                systems[ns1].engine.pos->coords().value[1].v,  
                systems[ns2].engine.pos->coords().value[0].v);
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


void deriv_matching(hid_t config, DerivEngine& engine, bool generate, double deriv_tol=1e-4) {
    auto &pos = *engine.pos;
    if(generate) {
        auto group = ensure_group(config, "/testing");
        ensure_not_exist(group.get(), "expected_deriv");
        auto tbl = create_earray(group.get(), "expected_deriv", H5T_NATIVE_FLOAT, 
                {pos.n_atom, 3, -1}, {pos.n_atom, 3, 1});
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

        auto central_diff_jac = central_difference_deriviative(do_compute, input, output, 1e-3);
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
    CmdLine cmd("Using Protein Statistical Information for Dynamics Estimation (Upside)\n Author: John Jumper", 
            ' ', "0.1");

    ValueArg<double> time_step_arg("", "time-step", "time step for integration (default 0.01)", 
            false, 0.01, "float", cmd);
    ValueArg<double> duration_arg("", "duration", "duration of simulation", 
            true, -1., "float", cmd);
    ValueArg<unsigned long> seed_arg("", "seed", "random seed (default 42)", 
            false, 42l, "int", cmd);
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
    SwitchArg generate_expected_deriv_arg("", "generate-expected-deriv", 
            "write an expected deriv to the input for later testing (developer only)", 
            cmd, false);
    ValueArg<string> log_level_arg("", "log-level", 
            "Use this option to control which arrays are stored in /output.  Availabe levels are basic, detailed, "
            "or extensive.  Default is basic.",
            false, "", "basic, detailed, extensive", cmd);
    SwitchArg potential_deriv_agreement_arg("", "potential-deriv-agreement",
            "(developer use only) check the agreement of the derivative with finite differences "
            "of the potential for the initial structure.  This may give strange answers for native structures "
            "(no steric clashes may given an agreement of NaN) or random structures (where bonds and angles are "
            "exactly at their equilibrium values).  Interpret these results at your own risk.", cmd, false);
    UnlabeledMultiArg<string> config_args("config_files","configuration .h5 files", true, "h5_files");
    cmd.add(config_args);
    cmd.parse(argc, argv);

    try {

        printf("invocation:");
        std::string invocation(argv[0]);
        for(auto arg=argv+1; arg!=argv+argc; ++arg) invocation += string(" ") + *arg;
        printf("%s\n", invocation.c_str());

        float dt = time_step_arg.getValue();
        double duration = duration_arg.getValue();
        uint64_t n_round = round(duration / (3*dt));
        int thermostat_interval = max(1.,round(thermostat_interval_arg.getValue() / (3*dt)));
        int frame_interval = max(1.,round(frame_interval_arg.getValue() / (3*dt)));

        unsigned long big_prime = 4294967291ul;  // largest prime smaller than 2^32
        uint32_t base_random_seed = uint32_t(seed_arg.getValue() % big_prime);

        // initialize thermostat and thermalize momentum
        printf("random seed: %lu\n", (unsigned long)(base_random_seed));

        float max_temp = max_temperature_arg.getValue();
        float min_temp = temperature_arg.getValue();
        if(max_temp == -1.) max_temp = min_temp;

        int pivot_interval = pivot_interval_arg.getValue() > 0. 
            ? max(1,int(pivot_interval_arg.getValue()/(3*dt)))
            : 0;

        int duration_print_width = ceil(log(1+duration)/log(10));

        int replica_interval = 0;
        if(replica_interval_arg.getValue())
            replica_interval = max(1.,replica_interval_arg.getValue()/(3*dt));

        bool do_recenter = !disable_recenter_arg.getValue();
        bool xy_recenter_only = do_recenter && disable_z_recenter_arg.getValue();

        h5_noerr(H5Eset_auto(H5E_DEFAULT, nullptr, nullptr));
        vector<string> config_paths = config_args.getValue();
        vector<System> systems(config_paths.size());

        #pragma omp critical
        for(int ns=0; ns<int(systems.size()); ++ns) {
            System* sys = &systems[ns];  // a pointer here makes later lambda's more natural
            sys->random_seed = base_random_seed + ns;

            try {
                sys->config = move(h5_obj(H5Fclose, H5Fopen(config_paths[ns].c_str(), H5F_ACC_RDWR, H5P_DEFAULT)));
            } catch(string &s) {
                throw string("Unable to open configuration file at ") + config_paths[ns];
            }

            if(h5_exists(sys->config.get(), "/output", false)) {
                // Note that it is not possible in HDF5 1.8.x to reclaim space by deleting
                // datasets or groups.  Subsequent h5repack will reclaim space, however.
                h5_noerr(H5Ldelete(sys->config.get(), "/output", H5P_DEFAULT));
            }

            LogLevel log_level;
            if     (log_level_arg.getValue() == "")          log_level = LOG_BASIC;
            else if(log_level_arg.getValue() == "basic")     log_level = LOG_BASIC;
            else if(log_level_arg.getValue() == "detailed")  log_level = LOG_DETAILED;
            else if(log_level_arg.getValue() == "extensive") log_level = LOG_EXTENSIVE;
            else throw string("Illegal value for --log-level");

            sys->logger = make_shared<H5Logger>(sys->config, "output", log_level);
            default_logger = sys->logger;  // FIXME kind of a hack for the ugly global variable

            write_string_attribute(sys->config.get(), "output", "invocation", invocation);

            auto pos_shape = get_dset_size(3, sys->config.get(), "/input/pos");
            sys->n_atom = pos_shape[0];
            sys->mom.reset(1, 3, sys->n_atom);
            for(int d: range(3)) for(int na: range(sys->n_atom)) sys->mom[0](d,na) = 0.f;

            if(pos_shape[1]!=3) throw string("invalid dimensions for initial position");
            if(pos_shape[2]!=1) throw string("must have n_system 1 from config");

            auto potential_group = open_group(sys->config.get(), "/input/potential");
            sys->engine = initialize_engine_from_hdf5(sys->n_atom, 1, potential_group.get());

            traverse_dset<3,float>(sys->config.get(), "/input/pos", [&](size_t na, size_t d, size_t ns, float x) { 
                    sys->engine.pos->coords().value[ns](d,na) = x;});
            printf("\nn_atom %i\n", sys->n_atom);

            deriv_matching(sys->config.get(), sys->engine, generate_expected_deriv_arg.getValue());
            if(potential_deriv_agreement_arg.getValue()){
                // if(n_system>1) throw string("Testing code does not support n_system > 1");
                sys->engine.compute(PotentialAndDerivMode);
                printf("Initial agreement:\n");
                for(auto &n: sys->engine.nodes)
                    printf("%24s %f\n", n.name.c_str(), n.computation->test_value_deriv_agreement());
                printf("\n");

                auto relative_error = potential_deriv_agreement(sys->engine);
                printf("overall potential relative error: ");
                for(auto r: relative_error) printf(" %.5f", r);
                printf("\n");
            }

            if(max_temp != min_temp && systems.size() == 1u) 
                throw string("--max-temperature cannot be specified for only a single system");
            if(systems.size()==1u) {
                systems.front().temperature = temperature_arg.getValue();
            } else {
                // system 0 is the minimum temperature
                // tighter spacing at the low end of temperatures because the variance is decreasing
                int n_system = systems.size();
                sys->temperature = sqr((sqrt(min_temp)*(n_system-1-ns) + sqrt(max_temp)*ns)/(n_system-1));
            }

            sys->thermostat = OrnsteinUhlenbeckThermostat(
                    sys->random_seed,
                    thermostat_timescale_arg.getValue(),
                    vector<float>(1,sys->temperature),
                    1e8);

            sys->thermostat.apply(sys->mom.array(), sys->n_atom); // initial thermalization
            sys->thermostat.set_delta_t(thermostat_interval*3*dt);  // set true thermostat interval

            // we must capture the sys pointer by value here so that it is available later
            sys->logger->add_logger<float>("pos", {1, sys->n_atom, 3}, [sys](float* pos_buffer) {
                    SysArray pos_array = sys->engine.pos->coords().value;
                    for(int na=0; na<sys->n_atom; ++na) 
                    for(int d=0; d<3; ++d) 
                    pos_buffer[na*3 + d] = pos_array[0](d,na);
                    });
            sys->logger->add_logger<double>("kinetic", {1}, [sys](double* kin_buffer) {
                    double sum_kin = 0.f;
                    for(int na=0; na<sys->n_atom; ++na) sum_kin += mag2(load_vec<3>(sys->mom[0],na));
                    kin_buffer[0] = (0.5/sys->n_atom)*sum_kin;  // kinetic_energy = (1/2) * <mom^2>
                    });
            sys->logger->add_logger<double>("potential", {1}, [sys](double* pot_buffer) {
                    sys->engine.compute(PotentialAndDerivMode);
                    pot_buffer[0] = sys->engine.potential[0];});
            sys->logger->add_logger<double>("time", {}, [sys,dt](double* time_buffer) {
                    *time_buffer=3*dt*sys->round_num;});

            if(pivot_interval) {
                sys->pivot_sampler = PivotSampler{open_group(sys->config.get(), "/input/pivot_moves").get()};

                sys->logger->add_logger<int>("pivot_stats", {2}, [ns,sys](int* stats_buffer) {
                        stats_buffer[0] = sys->pivot_sampler.pivot_stats.n_success;
                        stats_buffer[1] = sys->pivot_sampler.pivot_stats.n_attempt;
                        sys->pivot_sampler.pivot_stats.reset();
                        });
            }

            // quick hack of a check for z-centering and membrane potential
            if(do_recenter && !xy_recenter_only) {
                for(auto &n: sys->engine.nodes) {
                    if(is_prefix(n.name, "membrane_potential") || is_prefix(n.name, "z_flat_bottom") || is_prefix(n.name, "tension"))
                        throw string("You have z-centering and a z-dependent potential turned on.  "
                                "This is not what you want.  Consider --disable-z-recentering "
                                "or --disable-recentering.");
                }
            }

            if(do_recenter) {
                for(auto &n: sys->engine.nodes) {
                    if(is_prefix(n.name, "cavity_radial"))
                        throw string("You have re-centering and a radial potential turned on.  "
                                "This is not what you want.  Consider --disable-recentering.");
                }
            }
        }
        default_logger = shared_ptr<H5Logger>();  // FIXME kind of a hack for the ugly global variable

        if(replica_interval) {
            int n_atom = systems[0].n_atom;
            for(System& sys: systems) 
                if(sys.n_atom != n_atom) 
                    throw string("Replica exchange requires all systems have the same number of atoms");
        }

        // #if defined(_OPENMP)
        // // current we only use OpenMP parallelism over systems
        // if(n_system < omp_get_max_threads()) omp_set_num_threads(n_system);
        // if(omp_get_max_threads() > 1) 
        //     printf("Multi-threaded execution with %i threads\n\n", omp_get_max_threads());
        // else
        //     printf("Single-threaded execution\n\n");
        // #endif

        printf("\n");
        for(int ns: range(systems.size())) {
            printf("temperature %2i: %.3f\n", ns, systems[ns].temperature);
            systems[ns].logger->log_once<double>("temperature", {1}, [&](double* temperature_buffer) {
                    temperature_buffer[0] = systems[ns].temperature;});
        }
        printf("\n");

        printf("Initial potential energy:");
        for(System& sys: systems) {
            sys.engine.compute(PotentialAndDerivMode);
            printf(" %.2f", sys.engine.potential[0]);
        }
        printf("\n");

        // we need to run everyone until the next synchronization event
        // a little care is needed if we are multiplexing the events
        auto tstart = chrono::high_resolution_clock::now();
        while(systems[0].round_num < n_round) {
            int last_start = systems[0].round_num;
            #pragma omp parallel for schedule(static,1)
            for(int ns=0; ns<int(systems.size()); ++ns) {
                System& sys = systems[ns];
                for(bool do_break=false; (!do_break) && (sys.round_num<n_round); ++sys.round_num) {
                    int nr = sys.round_num;
                    if(pivot_interval && !(nr%pivot_interval)) 
                        sys.pivot_sampler.pivot_monte_carlo_step(sys.random_seed, nr, sys.temperature, sys.engine);

                    if(!frame_interval || !(nr%frame_interval)) {
                        if(do_recenter) recenter(sys.engine.pos->coords().value, xy_recenter_only, sys.n_atom, 1);
                        sys.engine.compute(PotentialAndDerivMode);
                        sys.logger->collect_samples();

                        double Rg = 0.f;
                        float3 com = make_vec3(0.f, 0.f, 0.f);
                        for(int na=0; na<sys.n_atom; ++na)
                            com += load_vec<3>(sys.engine.pos->coords().value[0], na);
                        com *= 1.f/sys.n_atom;

                        for(int na=0; na<sys.n_atom; ++na) 
                            Rg += mag2(load_vec<3>(sys.engine.pos->coords().value[0],na)-com);
                        Rg = sqrtf(Rg/sys.n_atom);

                        printf("%*.0f / %*.0f elapsed %2i system %5.1f hbonds, Rg %5.1f A, potential % 8.2f\n", 
                                duration_print_width, nr*3*double(dt), 
                                duration_print_width, duration, 
                                ns,
                                get_n_hbond(sys.engine), Rg, sys.engine.potential[0]);
                        fflush(stdout);
                    }
                    if(!(nr%thermostat_interval)) 
                        sys.thermostat.apply(sys.mom.array(), sys.n_atom);
                    sys.engine.integration_cycle(sys.mom.array(), dt, 0.f, DerivEngine::Verlet);

                    do_break = nr>last_start && replica_interval && !((nr+1)%replica_interval);
                }
            }

            if(replica_interval && !(systems[0].round_num % replica_interval)) {
                parallel_tempering_step(base_random_seed, systems[0].round_num, systems);
            }
        }
        for(auto& sys: systems) sys.logger = shared_ptr<H5Logger>(); // release shared_ptr

        auto elapsed = chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count();
        printf("\n\nfinished in %.1f seconds (%.2f us/systems/step, %.4f seconds/simulation_time_unit)\n",
                elapsed, elapsed*1e6/systems.size()/n_round/3, elapsed/duration_arg.getValue());

        /*
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
        */

        try {
            if(pivot_interval) {
                printf("pivot_success:");
                for(auto& sys: systems) {
                    std::vector<int64_t> ps(2,0);
                    traverse_dset<2,int>(sys.config.get(), "/output/pivot_stats", [&](size_t nf, int d, int x) {
                            ps[d] += x;});
                    printf(" % .4f", double(ps[0])/double(ps[1]));
                }
                printf("\n");
            }
        } catch(...) {}  // stats reporting is optional

#ifdef COLLECT_PROFILE
        printf("\n");
        global_time_keeper.print_report(3*n_round+1);
        printf("\n");
#endif
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
} catch(const string &e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
    return 1;
}
