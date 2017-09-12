#ifndef AFFINE_H
#define AFFINE_H

#include "vector_math.h"


//! \brief Apply a 3x3 rotation matrix U to 3d vector r
inline float3 apply_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x() = U[0]*r.x() + U[1]*r.y() + U[2]*r.z();
    ret.y() = U[3]*r.x() + U[4]*r.y() + U[5]*r.z();
    ret.z() = U[6]*r.x() + U[7]*r.y() + U[8]*r.z();
    return ret;
}

//! \brief Apply a 3x3 rotation matrix U^{-1} to 3d vector r
//!
//! Given a rotation matrix U, this applies the transpose/inverse of U
//! so that it "undoes" a rotation by U
inline float3 apply_inverse_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x() = U[0]*r.x() + U[3]*r.y() + U[6]*r.z();
    ret.y() = U[1]*r.x() + U[4]*r.y() + U[7]*r.z();
    ret.z() = U[2]*r.x() + U[5]*r.y() + U[8]*r.z();
    return ret;
}

//! \brief Apply an affine transformation to r
//!
//! A affine transformation is the combination of rotation and translation so 
//! that output = U*r + t.
inline float3 apply_affine(const float* restrict U, const float3& t, const float3& r) {
    float3 ret;
    ret.x() = U[0]*r.x() + U[1]*r.y() + U[2]*r.z() + t[0];
    ret.y() = U[3]*r.x() + U[4]*r.y() + U[5]*r.z() + t[1];
    ret.z() = U[6]*r.x() + U[7]*r.y() + U[8]*r.z() + t[2];
    return ret;
}


//! \brief Convert a rotation from axis-angle format to a rotation matrix
//!
//! Rotations may be expressed in a couple of formats.  Every rotation in 
//! 3d can be expressed as a rotation by a specified angle about an single axis.
//! This function converts rotations specified by a axis and an angle into
//! a 3x3 rotation matrix. The axis must be a unit-vector.
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



//! \brief Compute the relative rotation between two reference frames (U1^{-1}*U2)
//!
//! The relative rotation of two rotations with respect to a common reference frame
//! is the result of undoing rotation U1 then apply rotation U2, hence V=U1^{-1}*U2.
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

//! \brief Convert rotation from unit quaternion representation to 3x3 matrix
//!
//! The unit quaternion representation of rotations is a 4-dimensional unit
//! vector q, where ! q and -q map to the same rotation (and choosing a random
//! normalized ! 4-vector chooses a uniformly random rotation).  This format 
//! is convenient for constructing and combining rotations but annoying for
//! applying rotations to specific vectors.  Use this function to convert 
//! to an appropriate format for applying the rotations (see Wikipedia for
//! details).
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

//! \brief Compute the relative rotation between two reference frames for quaternions
//!
//! The relative rotation of two rotations with respect to a common reference frame
//! is the result of undoing rotation q1 then apply rotation q2, hence ret=q1^{-1}*q2.
//! The inverse rotation for a quaternion is its quaternion complex-conjugate.
inline void
relative_quat(float* restrict ret, const float* q1, const float* q2)
// ret = conjugate(p) * q  -- it takes you from the right reference frame to the left
{
    ret[0] =  q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
    ret[1] = -q1[1]*q2[0] + q1[0]*q2[1] + q1[3]*q2[2] - q1[2]*q2[3];
    ret[2] = -q1[2]*q2[0] - q1[3]*q2[1] + q1[0]*q2[2] + q1[1]*q2[3];
    ret[3] = -q1[3]*q2[0] + q1[2]*q2[1] - q1[1]*q2[2] + q1[0]*q2[3];
}

#endif
