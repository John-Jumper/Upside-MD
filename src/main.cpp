#include "h5_support.h"
#include <tclap/CmdLine.h>
#include "force.h"
#include "timing.h"
#include "thermostat.h"
#include <chrono>
#include "md_export.h"

using namespace std;
using namespace h5;


struct StateLogger
{
    int n_atom;
    int n_sys;
    int n_chunk;

    hid_t config;
    H5Obj pos_tbl;
    H5Obj kin_tbl;
    H5Obj time_tbl;

    vector<float>  pos_buffer;
    vector<double> kin_buffer;
    vector<double> time_buffer;

    // destructor would close tables inappropriately after naive copy
    StateLogger(const StateLogger &o) = delete;
    StateLogger& operator=(const StateLogger &o) = delete;
    // move constructor could be defined if necessary

    StateLogger(int n_atom_, hid_t config_, hid_t output_grp, int n_chunk_, int n_sys_):
        n_atom(n_atom_), n_sys(n_sys_), n_chunk(n_chunk_), config(config_),
        pos_tbl (create_earray(output_grp, "pos",     H5T_NATIVE_FLOAT,  {0,n_sys,n_atom,3}, {n_chunk,1,n_atom,3})),
        kin_tbl (create_earray(output_grp, "kinetic", H5T_NATIVE_DOUBLE, {0,n_sys},          {n_chunk,1})),
        time_tbl(create_earray(output_grp, "time",    H5T_NATIVE_DOUBLE, {0},                {n_chunk}))
    {
        pos_buffer .reserve(n_chunk * n_sys * n_atom * 3);
        kin_buffer .reserve(n_chunk * n_sys);
        time_buffer.reserve(n_chunk);
    }

    void log(double sim_time, const float* pos, const float* mom) {
        Timer timer(string("state_logger"));
        time_buffer.push_back(sim_time);

        pos_buffer.resize(pos_buffer.size()+n_sys*n_atom*3);
        std::copy_n(pos, n_sys*n_atom*3, &pos_buffer[pos_buffer.size()-n_sys*n_atom*3]);

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
        if(time_buffer.size()) {append_to_dset(time_tbl.get(), time_buffer, 0); time_buffer.resize(0);}
        H5Fflush(config, H5F_SCOPE_LOCAL);
    }

    virtual ~StateLogger() {
        try {flush();} catch(...) {}  // destructors should never throw an exception
    }
};


void force_testing(hid_t config, DerivEngine& engine, bool generate, double force_tol=1e-3) {
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
        printf("RMS force difference: %.6f\n", rms_error);
        if(rms_error > force_tol) throw string("inacceptable force deviation");
    }
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
    ValueArg<int> seed_arg("", "seed", "random seed (default 42)", 
            false, 42, "int", cmd);
    SwitchArg overwrite_output_arg("", "overwrite-output", 
            "overwrite the output group of the system file if present (default false)", 
            cmd, false);
    ValueArg<double> temperature_arg("", "temperature", "thermostat temperature (default 1.0)", 
            false, 1., "float", cmd);
    ValueArg<double> frame_interval_arg("", "frame-interval", "simulation time between frames", 
            true, -1., "float", cmd);
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
    SwitchArg generate_expected_force_arg("", "generate-expected-force", 
            "write an expected force to the input for later testing (developer only)", 
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

        auto pos_shape = get_dset_size<3>(config.get(), "/input/pos");
        int  n_atom   = pos_shape[0];
        int  n_system = pos_shape[2];
        if(pos_shape[1]!=3) throw string("invalid dimensions for initial position");

        auto force_group = open_group(config.get(), "/input/force");
        auto engine = initialize_engine_from_hdf5(n_atom, n_system, force_group.get());
        traverse_dset<3,float>(config.get(), "/input/pos", [&](size_t na, size_t d, size_t ns, float x) { 
                engine.pos->output.at(ns*n_atom*3 + na*3 + d) = x;});
        printf("\nn_atom %i\nn_system %i\n", engine.pos->n_atom, engine.pos->n_system);

        engine.compute();  // just a test force computation
        force_testing(config.get(), engine, generate_expected_force_arg.getValue());

        float dt = time_step_arg.getValue();
        uint64_t n_round = round(duration_arg.getValue() / (3*dt));
        int thermostat_interval = max(1.,round(thermostat_interval_arg.getValue() / (3*dt)));
        int frame_interval = max(1.,round(frame_interval_arg.getValue() / (3*dt)));

        float equil_duration = equilibration_duration_arg.getValue();
        // equilibration_max_force is set so that the change in momentum should not be more than
        // 20% of the equilibration magnitude over a single dt interval
        float equil_avg_mom   = sqrtf(temperature_arg.getValue()/1.5f);  // all particles have mass == 1.
        float equil_max_force = 0.8f*equil_avg_mom/dt;

        // initialize thermostat and thermalize momentum
        vector<float> mom(n_atom*n_system*3, 0.f);
        auto thermostat = OrnsteinUhlenbeckThermostat(
                seed_arg.getValue(), 
                thermostat_timescale_arg.getValue(),
                temperature_arg.getValue(),
                1e8);
        thermostat.apply(SysArray(mom.data(),n_atom*3), n_atom, n_system); // initial thermalization
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

        bool do_recenter = !disable_recenter_arg.getValue();
        auto tstart = chrono::high_resolution_clock::now();
        for(uint64_t nr=0; nr<n_round; ++nr) {
            if(!frame_interval || !(nr%frame_interval)) {
                if(do_recenter) recenter(engine.pos->coords().value, n_atom, n_system);
                state_logger.log(nr*3*dt, engine.pos->output.data(), mom.data());
                printf("%*lu / %*lu rounds %5.1f hbonds\n", 
                        round_print_width, (unsigned long)nr, 
                        round_print_width, (unsigned long)n_round, 
                        get_n_hbond(engine)/n_system);
                fflush(stdout);
            }
            // To be cautious, apply the thermostat more often if in the equilibration phase
            bool in_equil = nr*3*dt < equil_duration;
            if(!(nr%thermostat_interval) || in_equil) 
                thermostat.apply(SysArray(mom.data(),n_atom*3), n_atom, n_system);
            engine.integration_cycle(mom.data(), dt, (in_equil ? equil_max_force : 0.f), DerivEngine::Verlet);
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
            printf("\navg kinetic energy");
            for(int ns=0; ns<n_system; ++ns) printf(" % .4f", sum_kin[ns]/n_kin[ns]);
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
