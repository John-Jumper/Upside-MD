#ifndef PIVOT_SAMPLER_H
#define PIVOT_SAMPLER_H

#include "coord.h"
#include "affine.h"
#include "random.h"
#include "h5_support.h"
#include "timing.h"
#include "deriv_engine.h"
#include <algorithm>


struct PivotLocation {
    int rama_atom[5];
    int pivot_range[2];
    int restype;
};

struct PivotStats {
    uint64_t n_attempt;
    uint64_t n_success;
    PivotStats() {reset();}
    void reset() {n_attempt = n_success = 0u;}
};

// FIXME take a set of temperatures and propose at the appropriate temperature
struct PivotSampler {
    int n_system;
    int n_layer;
    int n_bin;
    int n_pivot_loc;

    std::vector<PivotStats>    pivot_stats;
    std::vector<PivotLocation> pivot_loc;
    std::vector<float>         proposal_pot;
    std::vector<float>         proposal_prob_cdf;

    PivotSampler(): n_system(0), n_layer(0), n_bin(0), n_pivot_loc(0) {}

    PivotSampler(hid_t grp, int n_system_): 
        n_system(n_system_),
        n_layer(h5::get_dset_size(3, grp, "proposal_pot")[0]), 
        n_bin  (h5::get_dset_size(3, grp, "proposal_pot")[1]),
        n_pivot_loc(h5::get_dset_size(2, grp, "pivot_atom")[0]),

        pivot_stats(n_system),
        pivot_loc(n_pivot_loc),
        proposal_prob_cdf(n_layer*n_bin*n_bin)
    {
        using namespace h5;

        check_size(grp, "proposal_pot", n_layer,     n_bin, n_bin);
        check_size(grp, "pivot_atom",    n_pivot_loc, 5);
        check_size(grp, "pivot_range",   n_pivot_loc, 2);
        check_size(grp, "pivot_restype", n_pivot_loc);

        traverse_dset<2,int>(grp, "pivot_atom",    [&](size_t np, size_t na, int x) {pivot_loc[np].rama_atom[na] =x;});
        traverse_dset<2,int>(grp, "pivot_range",   [&](size_t np, size_t j,  int x) {pivot_loc[np].pivot_range[j]=x;});
        traverse_dset<1,int>(grp, "pivot_restype", [&](size_t np,            int x) {pivot_loc[np].restype =x;});

        for(auto &p: pivot_loc) {
            for(int na=0; na<5; ++na) {
                if(p.pivot_range[0] <= p.rama_atom[na] && p.rama_atom[na] < p.pivot_range[1])
                    throw std::string("pivot_range cannot contain any atoms in pivot_atom ") + 
                        std::to_string(p.pivot_range[0]) + " <= " + std::to_string(p.rama_atom[na]) + " < " + std::to_string(p.pivot_range[1]);
            }
        }

        traverse_dset<3,float>(grp, "proposal_pot", [&](size_t nl, size_t nx, size_t ny, float x) {proposal_pot.push_back(x);});

        for(int nl=0; nl<n_layer; ++nl) {
            double sum_prob = 0.;
            for(int i=0; i<n_bin*n_bin; ++i) {
                proposal_prob_cdf[nl*n_bin*n_bin + i] = sum_prob;
                sum_prob += exp(-proposal_pot[nl*n_bin*n_bin + i]);
            }

            // normalize both the negative log probability and the cdf
            double inv_sum_prob = 1./sum_prob;
            double lsum_prob = log(sum_prob);
            for(int i=0; i<n_bin*n_bin; ++i) {
                proposal_prob_cdf[nl*n_bin*n_bin + i] *= inv_sum_prob;
                proposal_pot[nl*n_bin*n_bin + i]      += lsum_prob;
            }
            
            proposal_prob_cdf[(nl+1)*n_bin*n_bin-1] = 1.f;  // ensure no rounding error here
        }
    }

    void reset_stats() {
        for(auto &s: pivot_stats) s.reset();
    };

    void execute_random_pivot(float* delta_lprob, 
            uint32_t seed, uint64_t n_round, SysArray pos) const {
        Timer timer(std::string("random_pivot"));
        #pragma omp parallel for schedule(static,1)
        for(int ns=0; ns<n_system; ++ns) {
            RandomGenerator random(seed, PIVOT_MOVE_RANDOM_STREAM, ns, n_round);
            float4 random_values = random.uniform_open_closed();

            // pick a random pivot location
            int loc = n_pivot_loc * (1.f - random_values.z);
            auto p = pivot_loc[loc];

            // select a bin
            float cdf_value = random_values.w;
            
            int pivot_bin = std::lower_bound(
                    proposal_prob_cdf.data()+ p.restype   *n_bin*n_bin, 
                    proposal_prob_cdf.data()+(p.restype+1)*n_bin*n_bin,
                    cdf_value) - (proposal_prob_cdf.data()+p.restype*n_bin*n_bin);
            float new_lprob = proposal_pot[p.restype*n_bin*n_bin + pivot_bin];

            int phi_bin = pivot_bin/n_bin;
            int psi_bin = pivot_bin%n_bin;

            // now pick a random location in that bin
            float2 new_rama = (2.f*M_PI_F/n_bin)*make_float2(phi_bin+random_values.x, psi_bin+random_values.y) - M_PI_F;


            // printf("proposing %f %f from %f\n", new_rama.x*180.f/M_PI_F, new_rama.y*180.f/M_PI_F, cdf_value);

            // find deviation from old rama
            float3 d1,d2,d3,d4;
            float3 prevC = StaticCoord<3>(pos, ns, p.rama_atom[0]).f3();
            float3 N     = StaticCoord<3>(pos, ns, p.rama_atom[1]).f3();
            float3 CA    = StaticCoord<3>(pos, ns, p.rama_atom[2]).f3();
            float3 C     = StaticCoord<3>(pos, ns, p.rama_atom[3]).f3();
            float3 nextN = StaticCoord<3>(pos, ns, p.rama_atom[4]).f3();

            float2 old_rama = make_float2(
                    dihedral_germ(prevC,N,CA,C, d1,d2,d3,d4),
                    dihedral_germ(N,CA,C,nextN, d1,d2,d3,d4));

            int old_phi_bin = (old_rama.x+M_PI_F) * (0.5/M_PI_F) * n_bin;
            int old_psi_bin = (old_rama.y+M_PI_F) * (0.5/M_PI_F) * n_bin;
            float old_lprob = proposal_pot[(p.restype*n_bin + old_phi_bin)*n_bin + old_psi_bin];

            // apply rotations
            float3 phi_origin = CA;
            float3 psi_origin = C;

            float2 delta_rama = new_rama - old_rama;
            float phi_U[9]; axis_angle_to_rot(phi_U, delta_rama.x, normalize3(CA-N ));
            float psi_U[9]; axis_angle_to_rot(psi_U, delta_rama.y, normalize3(C -CA));

            {
                MutableCoord<3> y(pos, ns, p.rama_atom[3]);  // C
                float3 after_psi = psi_origin + apply_rotation(psi_U, y.f3()   -psi_origin); // unnecessary but harmless
                float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
                y.set_value(after_phi);
                y.flush();
            }

            {
                MutableCoord<3> y(pos, ns, p.rama_atom[4]);  // nextN
                float3 after_psi = psi_origin + apply_rotation(psi_U, y.f3()   -psi_origin);
                float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
                y.set_value(after_phi);
                y.flush();
            }

            for(int na=p.pivot_range[0]; na<p.pivot_range[1]; ++na) {
                MutableCoord<3> y(pos, ns, na);
                float3 after_psi = psi_origin + apply_rotation(psi_U, y.f3()   -psi_origin);
                float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
                y.set_value(after_phi);
                y.flush();
            }

            // {
            //     float deg = M_PI_F/180.f;
            //     float3 zprevC = StaticCoord<3>(pos, ns, p.rama_atom[0]).f3();
            //     float3 zN     = StaticCoord<3>(pos, ns, p.rama_atom[1]).f3();
            //     float3 zCA    = StaticCoord<3>(pos, ns, p.rama_atom[2]).f3();
            //     float3 zC     = StaticCoord<3>(pos, ns, p.rama_atom[3]).f3();
            //     float3 znextN = StaticCoord<3>(pos, ns, p.rama_atom[4]).f3();

            //     float2 actual_rama = make_float2(
            //             dihedral_germ(zprevC,zN,zCA,zC, d1,d2,d3,d4),
            //             dihedral_germ(zN,zCA,zC,znextN, d1,d2,d3,d4));

            //     printf("ns %i nr %2i old_rama % 8.3f % 8.3f new_rama % 8.3f % 8.3f actual_new_rama % 8.3f % 8.3f\n",
            //             ns, p.rama_atom[1]/3, 
            //             old_rama.x   /deg, old_rama.y   /deg, 
            //             new_rama.x   /deg, new_rama.y   /deg,
            //             actual_rama.x/deg, actual_rama.y/deg);
            // }

            delta_lprob[ns] = new_lprob - old_lprob;
        }
    }

    void pivot_monte_carlo_step(
            uint32_t seed, 
            uint64_t round,
            const std::vector<float>& temperatures,
            DerivEngine& engine) 
    {
        SysArray pos_sys = engine.pos->coords().value;
        std::vector<float> pos_copy = engine.pos->output;
        std::vector<float> delta_lprob(engine.pos->n_system);

        engine.compute(PotentialAndDerivMode);
        std::vector<float> old_potential = engine.potential;

        execute_random_pivot(delta_lprob.data(), seed, round, engine.pos->coords().value);

        engine.compute(PotentialAndDerivMode);
        std::vector<float> new_potential = engine.potential;

        for(int ns=0; ns<engine.pos->n_system; ++ns) {
            float lboltz_diff = delta_lprob[ns] - (1.f/temperatures[ns]) * (new_potential[ns]-old_potential[ns]);
            RandomGenerator random(seed, PIVOT_MONTE_CARLO_RANDOM_STREAM, ns, round);
            pivot_stats[ns].n_attempt++;

            if(lboltz_diff >= 0.f || expf(lboltz_diff) >= random.uniform_open_closed().x) {
                pivot_stats[ns].n_success++;
            } else {
                // If we reject the pivot, we must reverse it
                std::copy(
                        pos_copy.data()+ ns   *pos_sys.offset,
                        pos_copy.data()+(ns+1)*pos_sys.offset, 
                        pos_sys.x+ns*pos_sys.offset);
            }
        }
    }

};

#endif
