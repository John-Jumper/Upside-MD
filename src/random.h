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

        float4 normal() {
            threefry4x32_ctr_t bits = random_bits();
            r123::float2 n1 = r123::boxmuller(bits.v[0], bits.v[1]);
            r123::float2 n2 = r123::boxmuller(bits.v[2], bits.v[3]);
            return make_vec4(n1.x, n1.y, n2.x, n2.y);
        }

        float3 normal3 () {
            // just discard the 4th random number
            float4 ret = normal();
            return make_vec3(ret.x(), ret.y(), ret.z());
        };
};

#endif
