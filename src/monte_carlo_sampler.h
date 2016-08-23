#ifndef MONTE_CARLO_SAMPLER_H
#define MONTE_CARLO_SAMPLER_H

#include "affine.h"
#include "random.h"
#include "h5_support.h"
#include "timing.h"
#include "deriv_engine.h"
#include <algorithm>
#include "state_logger.h"

struct MonteCarloSampler {
    struct MoveStats {
        uint64_t n_success;
        uint64_t n_attempt;
        MoveStats() { reset(); }
        void reset() { n_attempt = n_success = 0u; }
    };

    void reset_stats() { move_stats.reset(); }

    MoveStats move_stats;
    std::string name;
    RandomStreamType stream_id;

    MonteCarloSampler() {}; // Empty default constructor

    MonteCarloSampler(const std::string& name_, RandomStreamType stream_id_, H5Logger& logger):
        name(name_),
        stream_id(stream_id_)
    {
        logger.add_logger<int>((name + "_stats").c_str(), {2}, [&](int* stats_buffer) {
                stats_buffer[0] = move_stats.n_success;
                stats_buffer[1] = move_stats.n_attempt;
                move_stats.reset();
                });
    }

    virtual void propose_random_move(float* delta_lprob, RandomGenerator& random, VecArray pos) const = 0;

    void monte_carlo_step(uint32_t seed, uint64_t round, const float temperature,
            DerivEngine& engine);
};

struct MultipleMonteCarloSampler {
    std::vector<std::unique_ptr<MonteCarloSampler>> samplers;

    MultipleMonteCarloSampler() {}; // Empty default constructor  
    
    MultipleMonteCarloSampler(hid_t sampler_group, H5Logger& logger);
    
    void execute(uint32_t seed, uint64_t round, const float temperature, DerivEngine& engine);
};

#endif /* MONTE_CARLO_SAMPLER_H */
