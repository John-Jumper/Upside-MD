#ifndef RANDOM_H
#define RANDOM_H

#include <cmath>
#include <Random123/threefry.h>

#include "uniform.hpp"
#include "boxmuller.hpp"

// if you want random numbers, you need to add a new entry so that no one else
// overlaps your random stream
enum RandomStreamType {
    THERMOSTAT_RANDOM_STREAM = 0,
    REPLICA_EXCHANGE_RANDOM_STREAM = 1,
    PIVOT_MOVE_RANDOM_STREAM = 2,
    JUMP_MOVE_RANDOM_STREAM = 3
};

struct RandomGenerator
{
    private:
        threefry4x32_key_t k;
        threefry4x32_ctr_t c;

        threefry4x32_ctr_t random_bits() {
            threefry4x32_ctr_t result = threefry4x32(c,k);
            c.v[3]++;
            return result;
        }

    public:
        RandomGenerator(uint32_t seed, uint32_t generator_id, uint32_t atom_number, uint64_t timestep)
        {
            k.v[0] = seed;
            k.v[1] = generator_id;
            k.v[2] = 0u;
            k.v[3] = 0u;

            uint64_t mask = 0xffffffff;
            c.v[0] =  timestep      & mask;
            c.v[1] = (timestep>>32) & mask;
            c.v[2] = atom_number;
            c.v[3] = 0u;
        }

        float4 uniform_open_closed() {
            threefry4x32_ctr_t bits = random_bits();
            return make_vec4(
                    r123::u01<float>(bits.v[0]),
                    r123::u01<float>(bits.v[1]),
                    r123::u01<float>(bits.v[2]),
                    r123::u01<float>(bits.v[3]));
        }

        void many_normal(float* output, int n_gen) {
            // output must have at least one extra space in case we overwrite

            int idx=0;
            while(idx<n_gen) {
                // threefry4x32_ctr_t bits = random_bits();
                // r123::float2 n1 = r123::boxmuller(bits.v[0], bits.v[1]);
                // r123::float2 n2 = r123::boxmuller(bits.v[2], bits.v[3]);

                // output[idx++] = n1.x;
                // if(idx<n_gen) output[idx++] = n1.y;
                // if(idx<n_gen) output[idx++] = n2.x;
                // if(idx<n_gen) output[idx++] = n2.y;
                // uses the polar Marsaglia method
                auto x = uniform_open_closed()*2.f-1.f;

                for(int i=0; (i<2)&(idx<n_gen); ++i) {
                    auto s = sqr(x[i]) + sqr(x[i+2]);
                    if((idx<n_gen) & (s<1.f) & (s>1e-10f)) {
                        float scale = sqrtf(-2.f*logf(s)*rcp(s));
                        output[idx++] = x[i]*scale;
                        if(idx<n_gen)
                            output[idx++] = x[i+2]*scale;
                    }
                }
            }
        }

        Vec<4> normal() {
            Vec<4> ret;
            many_normal(&ret[0], 4);
            return ret;
        }

        float3 normal3 () {
            // just discard the 4th random number
            Vec<3> ret;
            many_normal(&ret[0], 3);
            return ret;
        };
};


#endif
