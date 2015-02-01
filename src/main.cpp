#include "h5_support.h"
#include <tclap/CmdLine.h>
#include "deriv_engine.h"
#include "timing.h"
#include "thermostat.h"
#include <chrono>
#include "md_export.h"
#include <algorithm>
#include "random.h"

using namespace std;
using namespace h5;


void parallel_tempering_step(
        uint32_t seed, uint64_t round,
        vector<int>& current_system_indices,
        const vector<float>& temperatures, DerivEngine& engine) {
    int n_system = engine.pos->n_system;
    if(int(temperatures.size()) != n_system) throw string("impossible");
    if(int(current_system_indices.size()) != n_system) throw string("impossible");

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
        swap_ranges(pos.x+ns1*pos.offset, pos.x+(ns1+1)*pos.offset, pos.x+ns2*pos.offset);
        swap(current_system_indices[ns1], current_system_indices[ns2]);
    };

    RandomGenerator random(seed, 1u, 0u, round);

    // attempt neighbor swaps, first (2*i,2*i+1) swaps, then (2*i-1,2*i)
    int start = random.uniform_open_closed().x < 0.5f;  // must start at 0 or 1 randomly

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
        if(lboltz_diff < 0.f && expf(lboltz_diff) < random.uniform_open_closed().x) {
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


struct StateLogger
{
    int n_atom;
    int n_sys;
    int n_chunk;

    hid_t config;
    H5Obj pos_tbl;
    H5Obj kin_tbl;
    H5Obj pot_tbl;
    H5Obj time_tbl;

    vector<float>  pos_buffer;
    vector<double> kin_buffer;
    vector<double> pot_buffer;
    vector<double> time_buffer;

    // destructor would close tables inappropriately after naive copy
    StateLogger(const StateLogger &o) = delete;
    StateLogger& operator=(const StateLogger &o) = delete;
    // move constructor could be defined if necessary

    StateLogger(int n_atom_, hid_t config_, hid_t output_grp, int n_chunk_, int n_sys_):
        n_atom(n_atom_), n_sys(n_sys_), n_chunk(n_chunk_), config(config_),
        pos_tbl (create_earray(output_grp, "pos",     H5T_NATIVE_FLOAT,  {0,n_sys,n_atom,3}, {n_chunk,1,n_atom,3})),
        kin_tbl (create_earray(output_grp, "kinetic", H5T_NATIVE_DOUBLE, {0,n_sys},          {n_chunk,1})),
        pot_tbl (create_earray(output_grp, "potential",H5T_NATIVE_DOUBLE,{0,n_sys},          {n_chunk,1})),
        time_tbl(create_earray(output_grp, "time",    H5T_NATIVE_DOUBLE, {0},                {n_chunk}))
    {
        pos_buffer .reserve(n_chunk * n_sys * n_atom * 3);
        kin_buffer .reserve(n_chunk * n_sys);
        pot_buffer .reserve(n_chunk * n_sys);
        time_buffer.reserve(n_chunk);
    }

    void log(double sim_time, const float* pos, const float* mom, const float* potential) {
        Timer timer(string("state_logger"));
        time_buffer.push_back(sim_time);

        pos_buffer.resize(pos_buffer.size()+n_sys*n_atom*3);
        std::copy_n(pos, n_sys*n_atom*3, &pos_buffer[pos_buffer.size()-n_sys*n_atom*3]);

        pot_buffer.resize(pot_buffer.size()+n_sys);
        std::copy_n(potential, n_sys, &pot_buffer[pot_buffer.size()-n_sys]);

        for(int ns=0; ns<n_sys; ++ns) {
            double sum_kin = 0.f;
            for(int i=0; i<n_atom*3; ++i) sum_kin += mom[ns*n_atom*3 + i]*mom[ns*n_atom*3 + i];
            kin_buffer.push_back((0.5/n_atom)*sum_kin);  // kinetic_energy = (1/2) * <mom^2>
        }

        if(time_buffer.size() == (size_t)n_chunk) flush();
    }

    void flush() {
        // the buffer sizes should stay in sync in normal operation, but they could get out of sync
        // if the user catches an exception (exception paranoia seems prudent when dealing with
        // I/O on NFS)
        if(pos_buffer .size()) {append_to_dset(pos_tbl.get(),  pos_buffer,  0); pos_buffer .resize(0);}
        if(kin_buffer .size()) {append_to_dset(kin_tbl.get(),  kin_buffer,  0); kin_buffer .resize(0);}
        if(pot_buffer .size()) {append_to_dset(pot_tbl.get(),  pot_buffer,  0); pot_buffer .resize(0);}
        if(time_buffer.size()) {append_to_dset(time_tbl.get(), time_buffer, 0); time_buffer.resize(0);}
        H5Fflush(config, H5F_SCOPE_LOCAL);
    }

    virtual ~StateLogger() {
        try {flush();} catch(...) {}  // destructors should never throw an exception
    }
};


void deriv_matching(hid_t config, DerivEngine& engine, bool generate, double deriv_tol=1e-3) {
    auto &pos = *engine.pos;
    if(generate) {
        auto group = ensure_group(config, "/testing");
        ensure_not_exist(group.get(), "expected_deriv");
        auto tbl = create_earray(group.get(), "expected_deriv", H5T_NATIVE_FLOAT, 
                {pos.n_atom, 3, 0}, {pos.n_atom, 3, 1});
        append_to_dset(tbl.get(), pos.deriv, 2);
    }

    if(h5_exists(config, "/testing/expected_deriv")) {
        check_size(config, "/testing/expected_deriv", pos.n_atom, 3, pos.n_system);
        double rms_error = 0.;
        traverse_dset<3,float>(config, "/testing/expected_deriv", [&](size_t na, size_t d, size_t ns, float x) {
                double dev = x - pos.deriv.at(na*3*pos.n_system + d*pos.n_system + ns);
                rms_error += dev*dev;});
        rms_error = sqrtf(rms_error / pos.n_atom / pos.n_system);
        printf("RMS deriv difference: %.6f\n", rms_error);
        if(rms_error > deriv_tol) throw string("inacceptable deriv deviation");
    }
}


vector<float> potential_deriv_agreement(DerivEngine& engine) {
    vector<float> relative_error;
    int n_atom = engine.pos->n_elem;
    SysArray pos_array = engine.pos->coords().value;

    for(int ns=0; ns<engine.pos->n_system; ++ns) {
        vector<float> input(n_atom*3);
        copy_n(pos_array.x + ns*pos_array.offset, n_atom*3, begin(input));
        vector<float> output(1);

        auto do_compute = [&]() {
            copy_n(begin(input), n_atom*3, pos_array.x + ns*pos_array.offset);
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

        relative_error.push_back(
                relative_rms_deviation(
                    central_diff_jac, 
                    vector<float>(&engine.pos->deriv[ns*n_atom*3], &engine.pos->deriv[(ns+1)*n_atom*3])));
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
    ValueArg<double> thermostat_interval_arg("", "thermostat-interval", 
            "simulation time between applications of the thermostat", 
            false, -1., "float", cmd);
    ValueArg<double> thermostat_timescale_arg("", "thermostat-timescale", "timescale for the thermostat", 
            false, 5., "float", cmd);
    SwitchArg disable_recenter_arg("", "disable-recentering", 
            "Disable recentering of protein in the universe", 
            cmd, false);
    ValueArg<double> equilibration_duration_arg("", "equilibration-duration", 
            "duration to limit max force for equilibration (also decreases thermostat interval)", 
            false, 0., "float", cmd);
    SwitchArg generate_expected_deriv_arg("", "generate-expected-deriv", 
            "write an expected deriv to the input for later testing (developer only)", 
            cmd, false);
    cmd.parse(argc, argv);

    printf("invocation:");
    for(auto arg=argv; arg!=argv+argc; ++arg) printf(" %s", *arg);
    printf("\n");


    try {
        h5_noerr(H5Eset_auto(H5E_DEFAULT, nullptr, nullptr));
        H5Obj config;
        try {
            config = h5_obj(H5Fclose, H5Fopen(config_arg.getValue().c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
        } catch(string &s) {
            throw string("Unable to open configuration file at ") + config_arg.getValue();
        }

        auto pos_shape = get_dset_size(3, config.get(), "/input/pos");
        int  n_atom   = pos_shape[0];
        int  n_system = pos_shape[2];
        if(pos_shape[1]!=3) throw string("invalid dimensions for initial position");

        auto potential_group = open_group(config.get(), "/input/potential");
        auto engine = initialize_engine_from_hdf5(n_atom, n_system, potential_group.get());
        traverse_dset<3,float>(config.get(), "/input/pos", [&](size_t na, size_t d, size_t ns, float x) { 
                engine.pos->output.at(ns*n_atom*3 + na*3 + d) = x;});
        printf("\nn_atom %i\nn_system %i\n", engine.pos->n_atom, engine.pos->n_system);

        engine.compute(PotentialAndDerivMode);
        printf("Initial potential energy:");
        for(float e: engine.potential) printf(" %.2f", e);
        printf("\n");


        deriv_matching(config.get(), engine, generate_expected_deriv_arg.getValue());
        {
            // printf("Initial agreement:\n");
            // for(auto &n: engine.nodes) printf("%24s %f\n", n.name.c_str(), n.computation->test_value_deriv_agreement());
            // printf("\n");

            // auto relative_error = potential_deriv_agreement(engine);
            // printf("overall potential relative error: ");
            // for(auto r: relative_error) printf(" %.5f", r);
            // printf("\n");
        }

        float dt = time_step_arg.getValue();
        uint64_t n_round = round(duration_arg.getValue() / (3*dt));
        int thermostat_interval = max(1.,round(thermostat_interval_arg.getValue() / (3*dt)));
        int frame_interval = max(1.,round(frame_interval_arg.getValue() / (3*dt)));

        unsigned long big_prime = 4294967291ul;  // largest prime smaller than 2^32
        uint32_t random_seed = uint32_t(seed_arg.getValue() % big_prime);

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

        float equil_duration = equilibration_duration_arg.getValue();
        // equilibration_max_force is set so that the change in momentum should not be more than
        // 20% of the equilibration magnitude over a single dt interval
        float equil_avg_mom   = sqrtf(temperature[n_system-1]/1.5f);  // all particles have mass == 1.
        float equil_max_force = 0.8f*equil_avg_mom/dt;

        // initialize thermostat and thermalize momentum
        vector<float> mom(n_atom*n_system*3, 0.f);
        SysArray mom_sys(mom.data(), n_atom*3);

        printf("random seed: %lu\n", (unsigned long)(random_seed));
        auto thermostat = OrnsteinUhlenbeckThermostat(
                random_seed,
                thermostat_timescale_arg.getValue(),
                temperature,
                1e8);
        thermostat.apply(mom_sys, n_atom); // initial thermalization
        thermostat.set_delta_t(thermostat_interval*3*dt);  // set true thermostat interval

        if(h5_exists(config.get(), "/output", false)) {
            // Note that it is not possible in HDF5 1.8.x to reclaim space by deleting
            // datasets or groups.  Subsequent h5repack will reclaim space, however.
            if(overwrite_output_arg.getValue()) h5_noerr(H5Ldelete(config.get(), "/output", H5P_DEFAULT));
            else throw string("/output already exists and --overwrite-output was not specified");
        }

        auto output_grp = h5_obj(H5Gclose, 
                H5Gcreate2(config.get(), "output", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        StateLogger state_logger(n_atom, config.get(), output_grp.get(), 100, n_system);

        int round_print_width = ceil(log(n_round)/log(10));

        int replica_interval = 0;
        if(replica_interval_arg.getValue())
            replica_interval = max(1.,replica_interval_arg.getValue()/(3*dt));

        vector<int> current_system_indices; 
        for(int ns=0; ns<n_system; ++ns) current_system_indices.push_back(ns);
        printf("replica interval %i\n", replica_interval);

        bool do_recenter = !disable_recenter_arg.getValue();
        auto tstart = chrono::high_resolution_clock::now();
        for(uint64_t nr=0; nr<n_round; ++nr) {
            if(replica_interval && !(nr%replica_interval))
                parallel_tempering_step(random_seed, nr, current_system_indices, temperature, engine);

            if(!frame_interval || !(nr%frame_interval)) {
                if(do_recenter) recenter(engine.pos->coords().value, n_atom, n_system);
                engine.compute(PotentialAndDerivMode);
                state_logger.log(nr*3*dt, engine.pos->output.data(), mom_sys.x, engine.potential.data());

                if(replica_interval) {
                    printf("current_system_indices");
                    for(auto i: current_system_indices) printf(" %2i", i);
                    printf("\n");
                }

                double Rg = 0.f;
                for(int ns=0; ns<n_system; ++ns) {
                    float3 com = make_float3(0.f, 0.f, 0.f);
                    for(int na=0; na<n_atom; ++na) 
                        com += StaticCoord<3>(engine.pos->coords().value, ns, na).f3();
                    com *= 1.f/n_atom;

                    for(int na=0; na<n_atom; ++na) 
                        Rg += mag2(StaticCoord<3>(engine.pos->coords().value, ns, na).f3()-com);
                }
                Rg = sqrt(Rg/(n_atom*n_system));

                printf("%*lu / %*lu rounds %5.1f hbonds, Rg %5.1f A, potential", 
                        round_print_width, (unsigned long)nr, 
                        round_print_width, (unsigned long)n_round, 
                        get_n_hbond(engine)/n_system, Rg);
                for(float e: engine.potential) printf(" % 8.2f", e);
                printf("\n");
                fflush(stdout);
            }
            // To be cautious, apply the thermostat more often if in the equilibration phase
            bool in_equil = nr*3*dt < equil_duration;
            if(!(nr%thermostat_interval) || in_equil) 
                thermostat.apply(mom_sys, n_atom);
            engine.integration_cycle(mom_sys.x, dt, (in_equil ? equil_max_force : 0.f), DerivEngine::Verlet);
        }
        state_logger.flush();

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
