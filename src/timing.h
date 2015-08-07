#ifndef TIMING_H
#define TIMING_H

#include <unordered_map>
#include <string>
#include <chrono>

struct TimeKeeper {
    // ignore some number of initial timing events to avoid cache-warming and 
    // delayed-initialization effects as much as possible
    const int n_ignore;

    struct TimeRecord {
        long n_invoke = 0;
        double total_elapsed = 0.;
    };

    TimeKeeper(int n_ignore_ = -1): n_ignore(n_ignore_) {}

    std::unordered_map<std::string, TimeRecord> records;

    void add_time(const std::string &name, double t_elapsed) {
        TimeRecord& record = records[name];
        record.n_invoke++;
        if(record.n_invoke<n_ignore) return; 
        record.total_elapsed += t_elapsed;
    }

    void print_report(int n_steps);
};
extern TimeKeeper global_time_keeper;

#ifdef COLLECT_PROFILE
struct Timer {
    TimeKeeper &time_keeper;
    const std::string name;
    const std::chrono::time_point<std::chrono::high_resolution_clock> tstart;
    bool  active;

    Timer(const std::string &name_, TimeKeeper& time_keeper_ = global_time_keeper): 
        time_keeper(time_keeper_), name(name_), tstart(std::chrono::high_resolution_clock::now()), active(true) {}

    void stop() {
        if(active) {
            auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart).count();
            time_keeper.add_time(name, elapsed);
            active = false;
        }
    }

    void abort() {active = false;}

    ~Timer() {stop();}
};
#else
struct Timer {
    Timer(const std::string &name_, TimeKeeper& time_keeper_ = global_time_keeper) {}
    void stop () {}
    void abort() {}
};
#endif


#endif
