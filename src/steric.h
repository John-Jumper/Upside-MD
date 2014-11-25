#ifndef STERIC_H
#define STERIC_H

#include "coord.h"
#include "Float4.h"


struct PointCloud {
    int   n_pts;   // must be divisible by 4
    float *x;  // rotated but not translated into position
    float *y;
    float *z;
    float *weight;
    int*  type;
    float3 translation;  
};

struct StericPoint {
    float3 pos;
    float  weight;
    int    type;
};


struct StericResidue {
    float3 center;
    float  radius;   // all points are within radius of the center
    int    start_point;
    int    n_pts;
};


struct StericParams {
    CoordPair loc;
    int       restype;
};


struct Interaction {
    float   largest_cutoff;
    int     n_types;
    int     n_bin;
    float   inv_dx;

    float*  cutoff2;
    float4* germ_arr;

    Interaction(int n_types_, int n_bin_, float dx_):
        largest_cutoff(0.f),
        n_types(n_types_),
        n_bin(n_bin_),
        inv_dx(1.f/dx_),
        cutoff2 (new float [n_types*n_types]),
        germ_arr(new float4[n_types*n_types*n_bin]) {
            for(int nb=0; nb<n_bin; ++nb) 
                germ_arr[nb] = make_float4(0.f,0.f,0.f,0.f);
        };


    float interaction(float deriv1[6], float deriv2[6], const PointCloud &r1, const PointCloud &r2) {
        Vec34 trans(r1.translation.x - r2.translation.x,
                    r1.translation.y - r2.translation.y,
                    r1.translation.z - r2.translation.z);

        Float4 pot;
        Vec34 c1;
        Vec34 t1;
        Vec34 negation_of_t2;
        Float4 cutoff_radius(largest_cutoff-1e-6f);

        alignas(16) int coord_bin[16];

        for(int i1=0; i1<r1.n_pts; i1+=4) {
            auto x1 = Vec34(r1.x+i1, r1.y+i1, r1.z+i1);
            auto w1 = Float4(r1.weight + i1);
            auto x1trans = x1 + trans;

            for(int i2=0; i2<r2.n_pts; i2+=4) {
                auto x2 = Vec34(r2.x+i2, r2.y+i2, r2.z+i2);
                auto w2 = Float4(r2.weight + i2);

                for(int it=0; it<4; ++it) {
                    auto disp = x1trans-x2;
                    auto r_mag = disp.mag(); r_mag = cutoff_radius.blendv(r_mag, (r_mag<cutoff_radius));
                    auto coord = Float4(inv_dx) * r_mag;

                    // variables are named for what they will contain *after* the transpose
                    // FIXME should use Int4 type here
                    coord.store_int(coord_bin+4*it);
                    Float4 pot1((float*)(germ_arr + (r1.type[i1+0]*n_types + r2.type[i2+(it+0)%4])*n_bin + coord_bin[4*it+0]));
                    Float4 der1((float*)(germ_arr + (r1.type[i1+1]*n_types + r2.type[i2+(it+1)%4])*n_bin + coord_bin[4*it+1]));
                    Float4 pot2((float*)(germ_arr + (r1.type[i1+2]*n_types + r2.type[i2+(it+2)%4])*n_bin + coord_bin[4*it+2]));
                    Float4 der2((float*)(germ_arr + (r1.type[i1+3]*n_types + r2.type[i2+(it+3)%4])*n_bin + coord_bin[4*it+3]));

                    auto r_excess = coord - coord.round<_MM_FROUND_TO_ZERO>();
                    auto l_excess = Float4(1.f) - r_excess;

                    // excess is multiplied by the weights as linear coefficients
                    auto w = w1*w2;
                    r_excess *= w;
                    l_excess *= w;

                    transpose4(pot1, der1, pot2, der2);

                    pot += l_excess*pot1 + r_excess*pot2;
                    auto deriv = (l_excess*der1 + r_excess*der2) * disp;

                    c1             +=       deriv;
                    t1             += cross(deriv, x1);
                    negation_of_t2 += cross(deriv, x2);

                    x2.left_rotate();
                }
            }
        }

        Float4 c1_sum = c1.sum();
        deriv1[0] += c1_sum.x();  deriv2[0] -= c1_sum.x();
        deriv1[1] += c1_sum.y();  deriv2[1] -= c1_sum.y();
        deriv1[2] += c1_sum.z();  deriv2[2] -= c1_sum.z();

        Float4 t1_sum = t1.sum();
        Float4 t2_sum = -negation_of_t2.sum();

        deriv1[3] += t1_sum.x();  deriv2[3] -= t2_sum.x();
        deriv1[4] += t1_sum.y();  deriv2[4] -= t2_sum.y();
        deriv1[5] += t1_sum.z();  deriv2[5] -= t2_sum.z();

        return pot.sum();
    }


    
    float2 germ(int loc, float r_mag) const {
        float coord = inv_dx*r_mag;
        int   coord_bin = int(coord);

        float4 vals = germ_arr[loc*n_bin + coord_bin]; 
        float r_excess = coord - coord_bin;
        float l_excess = 1.f-r_excess;

        return make_float2(l_excess * vals.x + r_excess * vals.y,
                           l_excess * vals.z + r_excess * vals.w);
    }

    ~Interaction() {
        delete [] cutoff2;
        delete [] germ_arr;
    }
};

void steric_pairs(
        const CoordArray     rigid_body,
        const StericParams*  residues,  //  size (n_res,)
        const StericResidue* ref_res,
        const StericPoint*   ref_point,
        const Interaction &interaction,
        const int*           point_starts,  // size (n_res+1,)
        int n_res, int n_system);

#endif
