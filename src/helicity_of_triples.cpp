#include "h5_support.h"
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>

using namespace std;
using namespace h5;

int main(int argc, char* const * const argv) 
try {
    if(argc!=2) throw string("need to specify the rama file");

    auto input = h5_obj(H5Fclose, H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT));
    auto restypes  = read_attribute<vector<string>>(input.get(), "/rama", "restype");

    if(restypes.back() != string("ALL")) throw string("format not understood");
    restypes.pop_back();

    int  n_restype = restypes.size();   // -1 for the ALL type
    int  n_bin     = get_dset_size<5>(input.get(), "/rama")[3];

    check_size(input.get(), "/rama", n_restype, 2, n_restype+1, n_bin, n_bin);

    auto lprob    = vector<float>{};
    lprob.reserve(n_restype*2*(n_restype+1)*n_bin*n_bin);

    traverse_dset<5,float>(input.get(), "/rama", [&](int rt1, int left_right, int rt2,
                int phi_bin, int psi_bin, float x) {lprob.push_back(x);});

    auto lprob_get = [&] (int rt1, int left_right, int rt2, int phi_bin, int psi_bin) {
        return lprob[(((rt1*2 + left_right)*(n_restype+1) + rt2)*n_bin + phi_bin)*n_bin + psi_bin];};
    float bin_width = 360.f / n_bin;

    printf("rt1,rt2,rt3,helical_basin,tight_helix\n");
    for(int rt1=0; rt1<(int)restypes.size(); ++rt1) {
        for(int rt2=0; rt2<(int)restypes.size(); ++rt2) {
            for(int rt3=0; rt3<(int)restypes.size(); ++rt3) {
                // get left and right binds
                auto lt_lprob = [&](int phi_bin, int psi_bin) {return lprob_get(rt2,0,rt1, phi_bin,psi_bin);};
                auto rt_lprob = [&](int phi_bin, int psi_bin) {return lprob_get(rt2,1,rt3, phi_bin,psi_bin);};

                double total_prob = 0.;
                double helix_prob = 0.;
                double authentic_helix_prob = 0.;
                for(int phi=0; phi<n_bin; ++phi) {
                    for(int psi=0; psi<n_bin; ++psi) {
                        float prob = exp(double(-lt_lprob(phi,psi)-rt_lprob(phi,psi)));
                        total_prob += prob;
                        float phi_center = (phi+0.5f)*bin_width - 180.f;
                        float psi_center = (psi+0.5f)*bin_width - 180.f;
                        if(-180.f<phi_center && phi_center<0.f && -100.f<psi_center && psi_center<50.f)
                            helix_prob += prob;
                        if(- 85.f<phi_center && phi_center<-35.f && -65.f<psi_center && psi_center<-15.f)
                            authentic_helix_prob += prob;
                    }
                }
                helix_prob /= total_prob;
                authentic_helix_prob /= total_prob;
                printf("%s,%s,%s,%f,%f\n", 
                        restypes[rt1].c_str(), restypes[rt2].c_str(), restypes[rt3].c_str(), 
                        helix_prob, authentic_helix_prob);
            }
        }
    }

    return 0;
} catch (const string &e) {
    printf("\n\nERROR: %s\n", e.c_str());
}
