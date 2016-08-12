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
	virtual void monte_carlo_step(uint32_t seed, uint64_t round, const float temperature, DerivEngine& engine) = 0;
};

struct MultipleMonteCarloSampler {
	std::vector<std::unique_ptr<MonteCarloSampler>> samplers;

	MultipleMonteCarloSampler() {}; // Empty default constructor  
	
	MultipleMonteCarloSampler(hid_t sampler_group, H5Logger& logger);
	
	void execute(uint32_t seed, uint64_t round, const float temperature, DerivEngine& engine);
};

// ===[Derived Class Definitions]===

struct MoveStats {
    uint64_t n_success;
    uint64_t n_attempt;
    MoveStats() {reset();}
    void reset() {n_attempt = n_success = 0u;}
};


#endif /* MONTE_CARLO_SAMPLER_H */
