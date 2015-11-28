#include "timing.h"
#include <cstdio>
#include <algorithm>
#include <map>
#include <vector>

TimeKeeper global_time_keeper(1);

using namespace std;

void TimeKeeper::print_report(int n_steps) {
    struct S {
        string name; 
        TimeRecord rec;
        double avg_time;
        double steps_per_invocation;
        double contribution;
    };

    vector<S> sorted_records;

    double all_total = 0.;
    for(auto &p: records) {
        auto avg_time = p.second.total_elapsed / (p.second.n_invoke-n_ignore);
        auto steps_per_invocation = double(n_steps) / p.second.n_invoke;
        if(!(avg_time>0.)) avg_time = 0.;

        sorted_records.emplace_back(); 
        auto &s = sorted_records.back();

        s.name = p.first;
        s.rec = p.second;
        s.avg_time = avg_time;
        s.steps_per_invocation = steps_per_invocation;
        s.contribution = s.avg_time / s.steps_per_invocation;
        all_total += s.contribution;
    }
    
    sort(begin(sorted_records), end(sorted_records), [&](const S& s1, const S& s2) {
            return s1.contribution!=s2.contribution ? s1.contribution > s2.contribution : s1.name < s2.name;});

    int maxlen = 0;
    for(auto &p: sorted_records) maxlen = max(int(p.name.size()), maxlen);

    for(auto &p: sorted_records) {
        printf("%*s  %6.1f us/step  (%4.1f%%, %7.2f invocations/step, %7.1f us/invocation)\n", 
                maxlen, p.name.c_str(), 
                p.contribution*1e6, 
                p.contribution/all_total*100.,
                1.f/p.steps_per_invocation, 
                p.avg_time*1e6);
    }
    printf("%*s  %6.1f us/step\n", maxlen, "(total)", all_total*1e6);
}







