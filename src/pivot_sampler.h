#ifndef PIVOT_SAMPLER_H
#define PIVOT_SAMPLER_H

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
    uint64_t n_success;
    uint64_t n_attempt;
    PivotStats() {reset();}
    void reset() {n_attempt = n_success = 0u;}
};

// FIXME take a set of temperatures and propose at the appropriate temperature
struct PivotSampler {
    int n_layer;
    int n_bin;
    int n_pivot_loc;

    PivotStats    pivot_stats;
    std::vector<PivotLocation> pivot_loc;
    std::vector<float>         proposal_pot;
    std::vector<float>         proposal_prob_cdf;

    PivotSampler(): n_layer(0), n_bin(0), n_pivot_loc(0) {}

    PivotSampler(hid_t grp): 
        n_layer(h5::get_dset_size(3, grp, "proposal_pot")[0]), 
        n_bin  (h5::get_dset_size(3, grp, "proposal_pot")[1]),
        n_pivot_loc(h5::get_dset_size(2, grp, "pivot_atom")[0]),

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
            if(p.restype<0 || p.restype>=n_layer) throw std::string("invalid pivot restype");
            for(int na=0; na<5; ++na) {
                if(p.pivot_range[0] <= p.rama_atom[na] && p.rama_atom[na] < p.pivot_range[1])
                    throw std::string("pivot_range cannot contain any atoms in pivot_atom ") + 
                        std::to_string(p.pivot_range[0]) + " <= " + std::to_string(p.rama_atom[na]) + 
                        " < " + std::to_string(p.pivot_range[1]);
            }
        }

        traverse_dset<3,float>(grp, "proposal_pot", [&](size_t nl, size_t nx, size_t ny, float x) {proposal_pot.push_back(x);});

        for(int nl=0; nl<n_layer; ++nl) {
            double sum_prob = 0.;
            for(int i=0; i<n_bin*n_bin; ++i) {
                sum_prob += exp(-proposal_pot[nl*n_bin*n_bin + i]);
                proposal_prob_cdf[nl*n_bin*n_bin + i] = sum_prob;
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
        pivot_stats.reset();
    };

    void execute_random_pivot(float* delta_lprob, 
            uint32_t seed, uint64_t n_round, VecArray pos) const {
        Timer timer(std::string("random_pivot"));
        RandomGenerator random(seed, PIVOT_MOVE_RANDOM_STREAM, 0, n_round);
        float4 random_values = random.uniform_open_closed();

        // pick a random pivot location
        int loc = int(n_pivot_loc * random_values.z());
        if(loc == n_pivot_loc) loc--;  // this may occur due to rounding
        auto p = pivot_loc[loc];

        // select a bin
        float cdf_value = random_values.w();

        int pivot_bin = std::lower_bound(
                proposal_prob_cdf.data()+ p.restype   *n_bin*n_bin, 
                proposal_prob_cdf.data()+(p.restype+1)*n_bin*n_bin,
                cdf_value) - (proposal_prob_cdf.data()+p.restype*n_bin*n_bin);
        float new_lprob = proposal_pot[p.restype*n_bin*n_bin + pivot_bin];

        int phi_bin = pivot_bin/n_bin;
        int psi_bin = pivot_bin%n_bin;

        // now pick a random location in that bin
        // Note the half-bin shift because we want the bin center of the left-most bin at 0
        float2 new_rama = (2.f*M_PI_F/n_bin)*make_vec2(phi_bin+random_values.x()-0.5f, psi_bin+random_values.y()-0.5f) - M_PI_F;

        // find deviation from old rama
        float3 d1,d2,d3,d4;
        float3 prevC = load_vec<3>(pos, p.rama_atom[0]);
        float3 N     = load_vec<3>(pos, p.rama_atom[1]);
        float3 CA    = load_vec<3>(pos, p.rama_atom[2]);
        float3 C     = load_vec<3>(pos, p.rama_atom[3]);
        float3 nextN = load_vec<3>(pos, p.rama_atom[4]);

        float2 old_rama = make_vec2(
                dihedral_germ(prevC,N,CA,C, d1,d2,d3,d4),
                dihedral_germ(N,CA,C,nextN, d1,d2,d3,d4));

        // reverse the half-bin shift
        int old_phi_bin = (old_rama.x()+M_PI_F) * (0.5f/M_PI_F) * n_bin + 0.5f;
        int old_psi_bin = (old_rama.y()+M_PI_F) * (0.5f/M_PI_F) * n_bin + 0.5f;
        old_phi_bin = old_phi_bin>=n_bin ? 0 : old_phi_bin;  // enforce periodicity
        old_psi_bin = old_psi_bin>=n_bin ? 0 : old_psi_bin;
        float old_lprob = proposal_pot[(p.restype*n_bin + old_phi_bin)*n_bin + old_psi_bin];

        // apply rotations
        float3 phi_origin = CA;
        float3 psi_origin = C;

        float2 delta_rama = new_rama - old_rama;
        float phi_U[9]; axis_angle_to_rot(phi_U, delta_rama.x(), normalized(CA-N ));
        float psi_U[9]; axis_angle_to_rot(psi_U, delta_rama.y(), normalized(C -CA));

        {
            auto y = load_vec<3>(pos, p.rama_atom[3]);  // C
            float3 after_psi = psi_origin + apply_rotation(psi_U, y        -psi_origin); // unnecessary but harmless
            float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
            store_vec(pos, p.rama_atom[3], after_phi);
        }

        {
            auto y = load_vec<3>(pos, p.rama_atom[4]);  // nextN
            float3 after_psi = psi_origin + apply_rotation(psi_U, y        -psi_origin);
            float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
            store_vec(pos, p.rama_atom[4], after_phi);
        }

        for(int na=p.pivot_range[0]; na<p.pivot_range[1]; ++na) {
            auto y = load_vec<3>(pos, na);
            float3 after_psi = psi_origin + apply_rotation(psi_U, y        -psi_origin);
            float3 after_phi = phi_origin + apply_rotation(phi_U, after_psi-phi_origin);
            store_vec(pos, na, after_phi);
        }

        *delta_lprob = new_lprob - old_lprob;
    }

    void pivot_monte_carlo_step(
            uint32_t seed, 
            uint64_t round,
            const float temperature,
            DerivEngine& engine) 
    {
        auto &pos = engine.pos->output;
        VecArrayStorage pos_copy(pos);
        float delta_lprob;

        engine.compute(PotentialAndDerivMode);
        float old_potential = engine.potential;

        execute_random_pivot(&delta_lprob, seed, round, pos);

        engine.compute(PotentialAndDerivMode);
        float new_potential = engine.potential;

        float lboltz_diff = delta_lprob - (1.f/temperature) * (new_potential-old_potential);
        RandomGenerator random(seed, PIVOT_MONTE_CARLO_RANDOM_STREAM, 0, round);
        pivot_stats.n_attempt++;

        if(lboltz_diff >= 0.f || expf(lboltz_diff) >= random.uniform_open_closed().x()) {
            pivot_stats.n_success++;
        } else {
            // If we reject the pivot, we must reverse it
            copy(pos_copy, pos);
        }
    }

};

#endif
