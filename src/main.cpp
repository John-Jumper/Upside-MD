#include "pivot_sampler.h"
#include "h5_support.h"
#include <tclap/CmdLine.h>
#include "deriv_engine.h"
#include "timing.h"
#include "thermostat.h"
#include <chrono>
#include <algorithm>
#include <set>
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
    float initial_temperature;
    float temperature;
    H5Obj config;
    shared_ptr<H5Logger> logger;
    DerivEngine engine;
    PivotSampler pivot_sampler;
    VecArrayStorage mom; // momentum
    OrnsteinUhlenbeckThermostat thermostat;
    uint64_t round_num;
    System(): round_num(0) {}

    void set_temperature(float new_temp) {
        temperature = new_temp;
        thermostat.set_temp(temperature);
    }
};


double stod_strict(const std::string& s) {
    size_t nchar = -1u;
    double x = stod(s, &nchar);
    if(nchar != s.size()) throw("invalid float '" + s + "'");
    return x;
}
int stoi_strict(const std::string& s) {
    size_t nchar = -1u;
    int i = stoi(s, &nchar);
    if(nchar != s.size()) throw("invalid integer '" + s + "'");
    return i;
}

vector<string> split_string(const string& src, const string& sep) {
    vector<string> ret;

    for(auto curr_pos = begin(src); curr_pos-begin(src) < ptrdiff_t(src.size()); ) {
        auto next_match_end = search(curr_pos, end(src), begin(sep), end(sep));
        ret.emplace_back(curr_pos, next_match_end);
        curr_pos = next_match_end + sep.size();
    }

    return ret;
}


struct ReplicaExchange {
    struct SwapPair {int sys1; int sys2; uint64_t n_attempt; uint64_t n_success;};
    vector<vector<SwapPair>> swap_sets;
    vector<int> replica_indices;
    vector<vector<SwapPair*>> participating_swaps;

    ReplicaExchange(vector<System>& systems, vector<string> swap_sets_strings) {
        int n_system = systems.size();
        for(int ns: range(n_system)) {
            replica_indices.push_back(ns);
            participating_swaps.emplace_back();
        }

        for(string& set_string: swap_sets_strings) {
            swap_sets.emplace_back();
            auto& set = swap_sets.back();
            
            for(auto pair_string: split_string(set_string,",")) {
                set.emplace_back();
                auto& s = set.back();
                auto p = split_string(pair_string, "-");
                if(p.size() != 2u) throw string("invalid swap pair, because it contains " +
                        to_string(p.size()) + "elements but should contain 2");

                s.sys1 = stoi_strict(p[0]);
                s.sys2 = stoi_strict(p[1]);

                s.n_attempt = 0u;
                s.n_success = 0u;

                int n_system = systems.size();
                if(s.sys1 >= n_system || s.sys2 >= n_system) throw string("invalid system");
            }
        }

        for(auto& ss: swap_sets) {
            set<int> systems_in_set;
            for(auto& sw: ss) {
                if(systems_in_set.count(sw.sys1) ||
                   systems_in_set.count(sw.sys2) ||
                   sw.sys1==sw.sys2) {
                    throw string("Overlapping indices in swap set.  "
                            "No replica index can appear more than once in a swap set.  "
                            "You probably (but maybe not; I didn't look that closely) need more "
                            "swap sets to get non-overlapping pairs.");
                }
                systems_in_set.insert(sw.sys1);
                systems_in_set.insert(sw.sys2);

                participating_swaps[sw.sys1].push_back(&sw);
                participating_swaps[sw.sys2].push_back(&sw);
            }
        }

        // enable logging of replica events
        for(int ns: range(n_system)) {
            auto logger = systems[ns].logger;
            if(!logger) continue;
            if(static_cast<int>(logger->level) < static_cast<int>(LOG_BASIC)) continue;

            auto* replica_ptr = &replica_indices[ns];
            logger->add_logger<int>("replica_index", {1}, [replica_ptr](int* buffer){buffer[0] = *replica_ptr;});

            auto* swap_vectors = &participating_swaps[ns];
            logger->log_once<int>("replica_swap_partner", {int(swap_vectors->size())},[swap_vectors,ns](int* buffer){
                    for(int i: range(swap_vectors->size())) {
                        auto sw = (*swap_vectors)[i];
                        // write down the system that is *not* this system
                        buffer[i] = (sw->sys1!=ns ? sw->sys1 : sw->sys2);
                    }});

            logger->add_logger<int>("replica_cumulative_swaps", {int(swap_vectors->size()), 2}, 
                  [swap_vectors](int* buffer) {
                    for(int i: range(swap_vectors->size())) {
                        auto sw = (*swap_vectors)[i];
                        buffer[2*i+0] = sw->n_success;
                        buffer[2*i+1] = sw->n_attempt;
                    }});
        }
    }

    void reset_stats() {
        for(auto& ss: swap_sets)
            for(auto& sw: ss)
                sw.n_success = sw.n_attempt = 0u;
    }

    void attempt_swaps(uint32_t seed, uint64_t round, vector<System>& systems) {
        int n_system = systems.size();

        vector<float> beta;
        for(auto &sys: systems) beta.push_back(1.f/sys.temperature);

        // compute the boltzmann factors for everyone
        auto compute_log_boltzmann = [&]() {
            vector<float> result(n_system);
            for(int i=0; i<n_system; ++i) {
                systems[i].engine.compute(PotentialAndDerivMode);
                result[i] = -beta[i]*systems[i].engine.potential;
            }
            return result;
        };

        // swap coordinates and the associated system indices
        auto coord_swap = [&](int ns1, int ns2) {
            swap(systems[ns1].engine.pos->output, systems[ns2].engine.pos->output);
            swap(replica_indices[ns1], replica_indices[ns2]);
        };

        RandomGenerator random(seed, REPLICA_EXCHANGE_RANDOM_STREAM, 0u, round);

        for(auto& set: swap_sets) {
            // FIXME the first energy computation is unnecessary if we are not on the first swap set
            // It is important that the energy is computed more than once in case
            // we are doing Hamiltonian parallel tempering rather than 
            // temperature parallel tempering
            
            auto old_lboltz = compute_log_boltzmann();
            for(auto& swap_pair: set) coord_swap(swap_pair.sys1, swap_pair.sys2);
            auto new_lboltz  = compute_log_boltzmann();

            // reverse all swaps that should not occur by metropolis criterion
            for(auto& swap_pair: set) {
                auto s1 = swap_pair.sys1; 
                auto s2 = swap_pair.sys2;
                swap_pair.n_attempt++;
                float lboltz_diff = (new_lboltz[s1] + new_lboltz[s2]) - (old_lboltz[s1]+old_lboltz[s2]);
                // If we reject the swap, we must reverse it
                if(lboltz_diff < 0.f && expf(lboltz_diff) < random.uniform_open_closed().x()) {
                    coord_swap(s1,s2);
                } else {
                    swap_pair.n_success++;
                }
            }
        }
    }
};


void deriv_matching(hid_t config, DerivEngine& engine, bool generate, double deriv_tol=1e-4) {
    auto &pos = *engine.pos;
    if(generate) {
        auto group = ensure_group(config, "/testing");
        ensure_not_exist(group.get(), "expected_deriv");
        auto tbl = create_earray(group.get(), "expected_deriv", H5T_NATIVE_FLOAT, 
                {pos.n_atom, 3, -1}, {pos.n_atom, 3, 1});
        vector<float> deriv_value(pos.n_atom*3);
        for(int na=0; na<pos.n_atom; ++na)
            for(int d=0; d<3; ++d)
                deriv_value[na*3 + d] = pos.sens(d,na);
        append_to_dset(tbl.get(), deriv_value, 2);
    }

    if(h5_exists(config, "/testing/expected_deriv")) {
        check_size(config, "/testing/expected_deriv", pos.n_atom, 3, 1);
        double rms_error = 0.;
        traverse_dset<3,float>(config, "/testing/expected_deriv", [&](size_t na, size_t d, size_t ns, float x) {
                double dev = x - pos.sens(d,na);
                rms_error += dev*dev;});
        rms_error = sqrtf(rms_error / pos.n_atom);
        printf("RMS deriv difference: %.6f\n", rms_error);
        // if(rms_error > deriv_tol) throw string("inacceptable deriv deviation");
    }
}


vector<float> potential_deriv_agreement(DerivEngine& engine) {
    vector<float> relative_error;
    int n_atom = engine.pos->n_elem;
    VecArray pos_array = engine.pos->output;

    vector<float> input(n_atom*3);
    for(int na=0; na<n_atom; ++na)
        for(int d=0; d<3; ++d)
            input[na*3+d] = pos_array(d,na);
    std::vector<float> output(1);

    auto do_compute = [&]() {
        for(int na=0; na<n_atom; ++na)
            for(int d=0; d<3; ++d)
                pos_array(d,na) = input[na*3+d];
        engine.compute(PotentialAndDerivMode);
        output[0] = engine.potential;
    };

    for(auto &n: engine.nodes) {
        if(n.computation->potential_term) {
            auto &v = dynamic_cast<PotentialNode&>(*n.computation.get()).potential;
            printf("%s: % 4.3f\n", n.name.c_str(), v);
        }
    }
    printf("\n\n");

    auto central_diff_jac = central_difference_deriviative(do_compute, input, output, 1e-3);
    vector<float> deriv_array;
    for(int na=0; na<n_atom; ++na)
        for(int d=0; d<3; ++d)
            deriv_array.push_back(engine.pos->sens(d,na));

    relative_error.push_back(
            relative_rms_deviation(central_diff_jac, deriv_array));
    return relative_error;
}



int main(int argc, const char* const * argv)
try {
    using namespace TCLAP;  // Templatized C++ Command Line Parser (tclap.sourceforge.net)
    CmdLine cmd("Using Protein Statistical Information for Dynamics Estimation (Upside)\n Author: John Jumper", 
            ' ', "0.1");

    ValueArg<double> time_step_arg("", "time-step", "time step for integration (default 0.009)", 
            false, 0.009, "float", cmd);
    ValueArg<double> duration_arg("", "duration", "duration of simulation", 
            true, -1., "float", cmd);
    ValueArg<unsigned long> seed_arg("", "seed", "random seed (default 42)", 
            false, 42l, "int", cmd);
    ValueArg<string> temperature_arg("", "temperature", "thermostat temperature (default 1.0, "
            "should be comma-separated list of temperatures). If running a single system, a single temperature is fine)", 
            false, "", "temperature_list", cmd);
    MultiArg<string> swap_set_args("","swap-set", "list like 0-1,2-3,6-7,4-8 of non-overlapping swaps for a replica "
            "exchange.  May be specified multiple times for multiple swap sets (non-overlapping is only required "
            "within a single swap set).", false, "h5_files");
    cmd.add(swap_set_args);
    ValueArg<double> anneal_factor_arg("", "anneal-factor", "annealing factor (0.1 means the final temperature "
            "will be 10% of the initial temperature)", 
            false, 1., "float", cmd);
    ValueArg<double> anneal_duration_arg("", "anneal-duration", "duration of annealing phase "
            "(default is duration of simulation)", 
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
            "Use this option to control which arrays are stored in /output.  Available levels are basic, detailed, "
            "or extensive.  Default is detailed.",
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

        int pivot_interval = pivot_interval_arg.getValue() > 0. 
            ? max(1,int(pivot_interval_arg.getValue()/(3*dt)))
            : 0;

        int duration_print_width = ceil(log(1+duration)/log(10));

        bool do_recenter = !disable_recenter_arg.getValue();
        bool xy_recenter_only = do_recenter && disable_z_recenter_arg.getValue();

        h5_noerr(H5Eset_auto(H5E_DEFAULT, nullptr, nullptr));
        vector<string> config_paths = config_args.getValue();
        vector<System> systems(config_paths.size());

        auto temperature_strings = split_string(temperature_arg.getValue(), ",");
        if(temperature_strings.size() != 1u && temperature_strings.size() != systems.size()) 
            throw string("Received "+to_string(temperature_strings.size())+" temperatures but have "
                    +to_string(systems.size())+" systems");

        for(int ns: range(systems.size())) {
            float T = stod_strict(temperature_strings.size()>1u ? temperature_strings[ns] : temperature_strings[0]);
            systems[ns].initial_temperature = T;
        }

        double anneal_factor = anneal_factor_arg.getValue();
        double anneal_duration = anneal_duration_arg.getValue();
        if(anneal_duration == -1.) anneal_duration = duration;
        double anneal_start = duration - anneal_duration;

        // tighter spacing at the low end of temperatures because the variance is decreasing
        auto anneal_temp = [=](double initial_temperature, double time) {
            auto fraction = max(0., (time-anneal_start) / anneal_duration);
            double T0 = initial_temperature;
            double T1 = initial_temperature*anneal_factor;
            return sqr(sqrt(T0)*(1.-fraction) + sqrt(T1)*fraction);
        };

        int replica_interval = 0;
        if(replica_interval_arg.getValue())
            replica_interval = max(1.,replica_interval_arg.getValue()/(3*dt));

        // system 0 is the minimum temperature
        int n_system = systems.size();

        #pragma omp critical
        for(int ns=0; ns<n_system; ++ns) try {
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
            if     (log_level_arg.getValue() == "")          log_level = LOG_DETAILED;
            else if(log_level_arg.getValue() == "basic")     log_level = LOG_BASIC;
            else if(log_level_arg.getValue() == "detailed")  log_level = LOG_DETAILED;
            else if(log_level_arg.getValue() == "extensive") log_level = LOG_EXTENSIVE;
            else throw string("Illegal value for --log-level");

            sys->logger = make_shared<H5Logger>(sys->config, "output", log_level);
            default_logger = sys->logger;  // FIXME kind of a hack for the ugly global variable

            write_string_attribute(sys->config.get(), "output", "invocation", invocation);

            auto pos_shape = get_dset_size(3, sys->config.get(), "/input/pos");
            sys->n_atom = pos_shape[0];
            sys->mom.reset(3, sys->n_atom);
            for(int d: range(3)) for(int na: range(sys->n_atom)) sys->mom(d,na) = 0.f;

            if(pos_shape[1]!=3) throw string("invalid dimensions for initial position");
            if(pos_shape[2]!=1) throw string("must have n_system 1 from config");

            auto potential_group = open_group(sys->config.get(), "/input/potential");
            sys->engine = initialize_engine_from_hdf5(sys->n_atom, potential_group.get());

            traverse_dset<3,float>(sys->config.get(), "/input/pos", [&](size_t na, size_t d, size_t ns, float x) { 
                    sys->engine.pos->output(d,na) = x;});

            printf("%s\nn_atom %i\n\n", config_paths[ns].c_str(), sys->n_atom);

            deriv_matching(sys->config.get(), sys->engine, generate_expected_deriv_arg.getValue());
            if(potential_deriv_agreement_arg.getValue()){
                // if(n_system>1) throw string("Testing code does not support n_system > 1");
                sys->engine.compute(PotentialAndDerivMode);
                printf("Initial potential:\n");
                auto relative_error = potential_deriv_agreement(sys->engine);
                printf("overall potential relative error: ");
                for(auto r: relative_error) printf(" %.5f", r);
                printf("\n");
            }

            sys->thermostat = OrnsteinUhlenbeckThermostat(
                    sys->random_seed,
                    thermostat_timescale_arg.getValue(),
                    1.,
                    1e8);
            sys->set_temperature(sys->initial_temperature);

            sys->thermostat.apply(sys->mom, sys->n_atom); // initial thermalization
            sys->thermostat.set_delta_t(thermostat_interval*3*dt);  // set true thermostat interval

            // we must capture the sys pointer by value here so that it is available later
            sys->logger->add_logger<float>("pos", {1, sys->n_atom, 3}, [sys](float* pos_buffer) {
                    VecArray pos_array = sys->engine.pos->output;
                    for(int na=0; na<sys->n_atom; ++na) 
                    for(int d=0; d<3; ++d) 
                    pos_buffer[na*3 + d] = pos_array(d,na);
                    });
            sys->logger->add_logger<double>("kinetic", {1}, [sys](double* kin_buffer) {
                    double sum_kin = 0.f;
                    for(int na=0; na<sys->n_atom; ++na) sum_kin += mag2(load_vec<3>(sys->mom,na));
                    kin_buffer[0] = (0.5/sys->n_atom)*sum_kin;  // kinetic_energy = (1/2) * <mom^2>
                    });
            sys->logger->add_logger<double>("potential", {1}, [sys](double* pot_buffer) {
                    sys->engine.compute(PotentialAndDerivMode);
                    pot_buffer[0] = sys->engine.potential;});
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
        } catch(const string &e) {
            fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
            exit(1);
        } catch(...) {
            fprintf(stderr, "\n\nERROR: unknown error\n");
            exit(1);
        }
        default_logger = shared_ptr<H5Logger>();  // FIXME kind of a hack for the ugly global variable

        unique_ptr<ReplicaExchange> replex;
        if(replica_interval) {
            printf("initializing replica exchange\n");
            replex.reset(new ReplicaExchange(systems, swap_set_args.getValue()));
            if(!replex->swap_sets.size()) throw string("replica exchange requested but no swap sets proposed");
        }


        if(replica_interval) {
            int n_atom = systems[0].n_atom;
            for(System& sys: systems) 
                if(sys.n_atom != n_atom) 
                    throw string("Replica exchange requires all systems have the same number of atoms");
        }

        printf("\n");
        for(int ns: range(systems.size())) {
            printf("%i %.2f\n", ns, systems[ns].temperature);
            float* temperature_pointer = &(systems[ns].temperature);
            systems[ns].logger->add_logger<double>("temperature", {1}, [temperature_pointer](double* temperature_buffer) {
                    temperature_buffer[0] = *temperature_pointer;});
        }
        printf("\n");

        printf("Initial potential energy:");
        for(System& sys: systems) {
            sys.engine.compute(PotentialAndDerivMode);
            printf(" %.2f", sys.engine.potential);
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
                    // don't pivot at t=0 so that a partially strained system may relax before the
                    // first pivot
                    if(nr && pivot_interval && !(nr%pivot_interval)) 
                        sys.pivot_sampler.pivot_monte_carlo_step(sys.random_seed, nr, sys.temperature, sys.engine);

                    if(!frame_interval || !(nr%frame_interval)) {
                        if(do_recenter) recenter(sys.engine.pos->output, xy_recenter_only, sys.n_atom);
                        sys.engine.compute(PotentialAndDerivMode);
                        sys.logger->collect_samples();

                        double Rg = 0.f;
                        float3 com = make_vec3(0.f, 0.f, 0.f);
                        for(int na=0; na<sys.n_atom; ++na)
                            com += load_vec<3>(sys.engine.pos->output, na);
                        com *= 1.f/sys.n_atom;

                        for(int na=0; na<sys.n_atom; ++na) 
                            Rg += mag2(load_vec<3>(sys.engine.pos->output,na)-com);
                        Rg = sqrtf(Rg/sys.n_atom);

                        printf("%*.0f / %*.0f elapsed %2i system %.2f temp %5.1f hbonds, Rg %5.1f A, potential % 8.2f\n", 
                                duration_print_width, nr*3*double(dt), 
                                duration_print_width, duration, 
                                ns, sys.temperature,
                                get_n_hbond(sys.engine), Rg, sys.engine.potential);
                        fflush(stdout);
                    }

                    if(!(nr%thermostat_interval)) {
                        // Handle simulated annealing if applicable
                        if(anneal_factor != 1.)
                            sys.set_temperature(anneal_temp(sys.initial_temperature, 3*dt*(sys.round_num+1)));
                        sys.thermostat.apply(sys.mom, sys.n_atom);
                    }
                    sys.engine.integration_cycle(sys.mom, dt, 0.f, DerivEngine::Verlet);

                    do_break = nr>last_start && replica_interval && !((nr+1)%replica_interval);
                }
            }

            if(replica_interval && !(systems[0].round_num % replica_interval))
                replex->attempt_swaps(base_random_seed, systems[0].round_num, systems);
        }
        for(auto& sys: systems) sys.logger = shared_ptr<H5Logger>(); // release shared_ptr

        auto elapsed = chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count();
        printf("\n\nfinished in %.1f seconds (%.2f us/systems/step, %.1e simulation_time_unit/hour)\n",
                elapsed, elapsed*1e6/systems.size()/n_round/3, 
                n_round*3*dt/elapsed * 3600.);

        printf("\navg_kinetic_energy/1.5kT");
        for(auto& sys: systems) {
            double sum_kinetic = 0.;
            long n_kinetic = 0l;
            size_t tot_frames = get_dset_size(2, sys.config.get(), "/output/kinetic")[0];

            traverse_dset<2,float>(sys.config.get(),"/output/kinetic", [&](size_t nf, size_t ns, float x){
                    if(nf>tot_frames/2){ sum_kinetic+=x; n_kinetic++; }
                    });
            printf(" % .3f", sum_kinetic/n_kinetic / (1.5*sys.temperature));
        }
        printf("\n");

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
