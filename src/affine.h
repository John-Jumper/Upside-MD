#ifndef AFFINE_H
#define AFFINE_H

#include "vector_math.h"

namespace {

inline float3 apply_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x() = U[0]*r.x() + U[1]*r.y() + U[2]*r.z();
    ret.y() = U[3]*r.x() + U[4]*r.y() + U[5]*r.z();
    ret.z() = U[6]*r.x() + U[7]*r.y() + U[8]*r.z();
    return ret;
}

inline float3 apply_inverse_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x() = U[0]*r.x() + U[3]*r.y() + U[6]*r.z();
    ret.y() = U[1]*r.x() + U[4]*r.y() + U[7]*r.z();
    ret.z() = U[2]*r.x() + U[5]*r.y() + U[8]*r.z();
    return ret;
}

inline float3 apply_affine(const float* restrict U, const float3& t, const float3& r) {
    float3 ret;
    ret.x() = U[0]*r.x() + U[1]*r.y() + U[2]*r.z() + t[0];
    ret.y() = U[3]*r.x() + U[4]*r.y() + U[5]*r.z() + t[1];
    ret.z() = U[6]*r.x() + U[7]*r.y() + U[8]*r.z() + t[2];
    return ret;
}


inline void axis_angle_to_rot(
        float* U,
        float angle,
        float3 axis) { // must be normalized
    float x = axis.x();
    float y = axis.y();
    float z = axis.z();

    float c = std::cos(angle);
    float s = std::sin(angle);
    float C = 1.f-c;

    U[0*3+0] = x*x*C+c;   U[0*3+1] = x*y*C-z*s; U[0*3+2] = x*z*C+y*s;
    U[1*3+0] = y*x*C+z*s; U[1*3+1] = y*y*C+c;   U[1*3+2] = y*z*C-x*s;
    U[2*3+0] = z*x*C-y*s; U[2*3+1] = z*y*C+x*s; U[2*3+2] = z*z*C+c;
}



inline void relative_rotation(
        float* restrict V,
        const float* U1,
        const float* U2) // V = transpose(U1) * U2
{

    V[0] = U1[0]*U2[0] + U1[3]*U2[3] + U1[6]*U2[6];
    V[1] = U1[0]*U2[1] + U1[3]*U2[4] + U1[6]*U2[7];
    V[2] = U1[0]*U2[2] + U1[3]*U2[5] + U1[6]*U2[8];
    V[3] = U1[1]*U2[0] + U1[4]*U2[3] + U1[7]*U2[6];
    V[4] = U1[1]*U2[1] + U1[4]*U2[4] + U1[7]*U2[7];
    V[5] = U1[1]*U2[2] + U1[4]*U2[5] + U1[7]*U2[8];
    V[6] = U1[2]*U2[0] + U1[5]*U2[3] + U1[8]*U2[6];
    V[7] = U1[2]*U2[1] + U1[5]*U2[4] + U1[8]*U2[7];
    V[8] = U1[2]*U2[2] + U1[5]*U2[5] + U1[8]*U2[8];
}

inline void 
quat_to_rot(
        float*       restrict U,    // length 9 (row-major order)
        const float* restrict quat) // length 4 (must be normalized)
{
    float a=quat[0], b=quat[1], c=quat[2], d=quat[3];

    U[0*3+0] = a*a+b*b-c*c-d*d; U[0*3+1] = 2.f*b*c-2.f*a*d; U[0*3+2] = 2.f*b*d+2.f*a*c;
    U[1*3+0] = 2.f*b*c+2.f*a*d; U[1*3+1] = a*a-b*b+c*c-d*d; U[1*3+2] = 2.f*c*d-2.f*a*b;
    U[2*3+0] = 2.f*b*d-2.f*a*c; U[2*3+1] = 2.f*c*d+2.f*a*b; U[2*3+2] = a*a-b*b-c*c+d*d;
}
}


namespace { 



inline void
relative_quat(float* restrict ret, const float* q1, const float* q2)
// ret = conjugate(p) * q  -- it takes you from the right reference frame to the left
{
    ret[0] =  q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
    ret[1] = -q1[1]*q2[0] + q1[0]*q2[1] + q1[3]*q2[2] - q1[2]*q2[3];
    ret[2] = -q1[2]*q2[0] - q1[3]*q2[1] + q1[0]*q2[2] + q1[1]*q2[3];
    ret[3] = -q1[3]*q2[0] + q1[2]*q2[1] - q1[1]*q2[2] + q1[0]*q2[3];
}

}
#endif
