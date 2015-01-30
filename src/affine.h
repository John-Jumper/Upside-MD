#ifndef AFFINE_H
#define AFFINE_H
#include "coord.h"

namespace {

inline float3 apply_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x = U[0]*r.x + U[1]*r.y + U[2]*r.z;
    ret.y = U[3]*r.x + U[4]*r.y + U[5]*r.z;
    ret.z = U[6]*r.x + U[7]*r.y + U[8]*r.z;
    return ret;
}

inline float3 apply_inverse_rotation(const float* restrict U, const float3 &r)
{
    float3 ret;
    ret.x = U[0]*r.x + U[3]*r.y + U[6]*r.z;
    ret.y = U[1]*r.x + U[4]*r.y + U[7]*r.z;
    ret.z = U[2]*r.x + U[5]*r.y + U[8]*r.z;
    return ret;
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

//! Coordinate representing a rigid body rotation and its derivative/torque
template <int NDIM_OUTPUT=1>
struct AffineCoord
{
    float t[3];    //!< translation vector
    float U[9];    //!< rotation matrix
    float d[NDIM_OUTPUT][6]; //!< first three components are translation deriv, second 3 are torque deriv

    float* deriv_arr;

    AffineCoord() {};

    //! Initialize from coordinate array, system index, and an index into the coordinate array
    AffineCoord(const CoordArray arr, int system, const CoordPair &c):
        deriv_arr(arr.deriv.x + system*arr.deriv.offset + c.slot*6)
    {
        float q[4];
        for(int nd=0; nd<3; ++nd) t[nd] = arr.value.x[system*arr.value.offset + c.index*7 + nd    ];
        for(int nd=0; nd<4; ++nd) q[nd] = arr.value.x[system*arr.value.offset + c.index*7 + nd + 3];

        // FIXME remove normalization
        float norm_factor = 1.f/sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
        for(int d=0; d<4; ++d) q[d] *= norm_factor;
        quat_to_rot(U,q);

        for(int ndo=0; ndo<NDIM_OUTPUT; ++ndo) for(int nd=0; nd<6; ++nd) d[ndo][nd] = 0.f;
    }

    //! Apply the rigid body transformation to a point in the reference frame

    //! Result is the transformed point in the simulation residue frame.
    //! This function is used to convert atomic positions from the reference frame
    //! to the laboratory frame of the simulation.
    float3 apply(const float3& r) {
        float3 ret;
        ret.x = U[0]*r.x + U[1]*r.y + U[2]*r.z + t[0];
        ret.y = U[3]*r.x + U[4]*r.y + U[5]*r.z + t[1];
        ret.z = U[6]*r.x + U[7]*r.y + U[8]*r.z + t[2];
        return ret;
    }

    //! Apply the rigid body transformation to a point in the laboratory frame

    //! Result is the transformed point in the reference frame.
    //! This function is used to convert atomic positions from the laboratory frame
    //! of the simulation to the reference frame for the residue.
    float3 apply_inverse(const float3& r) {
        float3 s = r-tf3();

        float3 ret;
        ret.x = U[0]*s.x + U[3]*s.y + U[6]*s.z;
        ret.y = U[1]*s.x + U[4]*s.y + U[7]*s.z;
        ret.z = U[2]*s.x + U[5]*s.y + U[8]*s.z;
        return ret;
    }

    //! Apply only the rotational part of the rigid body transformation
    
    //! This function should be used for transforming directions (vectors like
    //! the CA-CB bond vector) from the reference frame to the laboratory frame 
    //! of the simulation.
    float3 apply_rotation(const float3& r) {
        float3 ret;
        ret.x = U[0]*r.x + U[1]*r.y + U[2]*r.z;
        ret.y = U[3]*r.x + U[4]*r.y + U[5]*r.z;
        ret.z = U[6]*r.x + U[7]*r.y + U[8]*r.z;
        return ret;
    }

    //! Same as apply() above, but acts on float* instead of float3
    float3 apply(const float* r) { return apply(make_float3(r[0], r[1], r[2])); }

    //! Return translation vector as a float3
    float3 tf3() const {return make_float3(t[0], t[1], t[2]);}

    //! Add a derivative and torque for a force applied at a specified point for CoordNode

    //! This function is similar to add_deriv_at_location(r_lab_frame, r_deriv), but you must
    //! use this version if you have multiple output values (only applies to CoordNode's).
    void add_deriv_at_location(int ndo, const float3& r_lab_frame, const float3& r_deriv) {
        // add translation derivs
        d[ndo][0] += r_deriv.x;
        d[ndo][1] += r_deriv.y;
        d[ndo][2] += r_deriv.z;

        // work relative to the center of the rigid body
        float3 r = r_lab_frame - tf3();

        // add torque derivs
        d[ndo][3] += r.y*r_deriv.z - r.z*r_deriv.y;
        d[ndo][4] += r.z*r_deriv.x - r.x*r_deriv.z;
        d[ndo][5] += r.x*r_deriv.y - r.y*r_deriv.x;
    }
    //! Add a derivative and torque for a force applied at a specified point for CoordNode

    //! This function is used when you are applying a potential based on a specific
    //! location in the laboratory frame.  A simple example might be a potential based on the 
    //! the distance between CB-atoms, where each CB location was computed by applying the affine
    //! transformation to a point in the residue reference frame.
    //!
    //! For code of the form float3 x = rigid_body.apply(ref_pos); potential = f(x);, you should
    //! call rigid_body.add_deriv_at_location(x, df_dx).  This function will automatically 
    //! compute and apply the torque cross(x, df_dx).
    void add_deriv_at_location(const float3& r_lab_frame, const float3& r_deriv) {
        add_deriv_at_location(0, r_lab_frame, r_deriv);
    }

    //! Add a derivative and torque for CoordNode

    //! Only useful when writing CoordNode, otherwise see add_deriv_and_torque(r_deriv, torque).
    void add_deriv_and_torque(int ndo, const float3& r_deriv, const float3& torque) {
        d[ndo][0] += r_deriv.x;
        d[ndo][1] += r_deriv.y;
        d[ndo][2] += r_deriv.z;

        d[ndo][3] += torque.x;
        d[ndo][4] += torque.y;
        d[ndo][5] += torque.z;
    }
    //! Add a derivative and torque

    //! Add a derivative and torque value for this rigid body.  Most of the 
    //! time it is more convenient to use add_deriv_at_location instead of this
    //! function.
    void add_deriv_and_torque(const float3& r_deriv, const float3& torque) {
        add_deriv_and_torque(0, r_deriv, torque);
    }

    //! Write the derivative to the CoordArray
    void flush() const {
        for(int ndo=0; ndo<NDIM_OUTPUT; ++ndo) 
            for(int nd=0; nd<6; ++nd) 
                deriv_arr[ndo*6+nd] = d[ndo][nd];
    }
};

namespace { 

// struct AffineTransformOrthog {
//     float tx, ty, tz;
//     float xx,xy,xz, yx,yy,yz, zx,zy,zz;
// 
//     AffineTransformOrthog& zero() {
//         tx=0.f; ty=0.f; tz=0.f;
//         xx=0.f; xy=0.f; xz=0.f;
//         yx=0.f; yy=0.f; yz=0.f;
//         zx=0.f; zy=0.f; zz=0.f;
//         return *this;
//     }
// 
//     float3 apply(const float3& r) {
//         float3 ret;
//         ret.x = xx*r.x + xy*r.y + xz*r.z + tx;
//         ret.y = yx*r.x + yy*r.y + yz*r.z + ty;
//         ret.z = zx*r.x + zy*r.y + zz*r.z + tz;
//         return ret;
//     }
// }


// struct AffineTransform {
//     float tx,ty,tz;
//     float a,b,c,d;
// 
//     AffineTransform& zero() {
//         tx=0.f; ty=0.f; tz=0.f;
//         a =0.f; b =0.f; c =0.f; d =0.f;
//         return *this;
//     }
// 
//     AffineTransform 
//     pullback_orthog_deriv(const AffineTransformOrthog& dU) const {
//         AffineTransform dq;
// 
//         dq.tx = dU.tx;
//         dq.ty = dU.ty;
//         dq.tz = dU.tz;
// 
//         dq.a = 2.f*((dU.xx+dU.yy+dU.zz)*q.a + (     -dU.yz+dU.zy)*q.b + (       dU.xz-dU.zx)*q.c + (      -dU.xy+dU.yx)*q.d);
//         dq.b = 2.f*((     -dU.yz+dU.zy)*q.a + (dU.xx-dU.yy-dU.zz)*q.b + (       dU.xy+dU.yx)*q.c + (       dU.xz+dU.zx)*q.d);
//         dq.c = 2.f*((      dU.xz-dU.zx)*q.a + (      dU.xy+dU.yx)*q.b + (-dU.xx+dU.yy-dU.zz)*q.c + (       dU.yz+dU.zy)*q.d);
//         dq.d = 2.f*((     -dU.xy+dU.yx)*q.a + (      dU.xz+dU.zx)*q.b + (       dU.yz+dU.zy)*q.c + (-dU.xx-dU.yy+dU.zz)*q.d);
// 
//         return dq;
//     }
// 
//     AffineTransformOrthog
//     convert_to_orthog() const {
//         AffineTransformOrthog U;
// 
//         U.tx = tx;              U.ty = ty;              U.tz = tz;
//         U.xx = a*a+b*b-c*c-d*d; U.xy = 2.f*(b*c-a*d);   U.xz = 2.f*(b*d+a*c);
//         U.yx = 2.f*(b*c+a*d);   U.yy = a*a-b*b+c*c-d*d; U.yz = 2.f*(c*d-a*b);
//         U.zx = 2.f*(b*d-a*c);   U.zy = 2.f*(c*d+a*b);   U.zz = a*a-b*b-c*c+d*d;
// 
//         return U;
//     }
// };
// 
// 


inline void
relative_quat(float* restrict ret, const float* q1, const float* q2)
// ret = conjugate(p) * q  -- it takes you from the right reference frame to the left
{
    ret[0] =  q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
    ret[1] = -q1[1]*q2[0] + q1[0]*q2[1] + q1[3]*q2[2] - q1[2]*q2[3];
    ret[2] = -q1[2]*q2[0] - q1[3]*q2[1] + q1[0]*q2[2] + q1[1]*q2[3];
    ret[3] = -q1[3]*q2[0] + q1[2]*q2[1] - q1[1]*q2[2] + q1[0]*q2[3];
}

// inline void
// apply_affine(float* restrict ret, const float* restrict t, const float* restrict U, const float* restrict x)
// {
//     ret[0] = U[0]*x[0] + U[1]*x[1] + U[2]*x[2] + t[0];
//     ret[1] = U[3]*x[0] + U[4]*x[1] + U[5]*x[2] + t[1];
//     ret[2] = U[6]*x[0] + U[7]*x[1] + U[8]*x[2] + t[2];
// }



// inline void convert_rot_deriv_to_quat_deriv(
//         float dq[4], const float q[4], const float dU[9])
// {
//     dq[0] = 2.f*(( dU[0]+dU[4]+dU[8])*q[0] + (-dU[5]+dU[7])*q[1] + ( dU[2]-dU[6])*q[2] + (-dU[1]+dU[3])*q[3]);
//     dq[1] = 2.f*((-dU[5]+dU[7])*q[0] + ( dU[0]-dU[4]-dU[8])*q[1] + ( dU[1]+dU[3])*q[2] + ( dU[2]+dU[6])*q[3]);
//     dq[2] = 2.f*(( dU[2]-dU[6])*q[0] + ( dU[1]+dU[3])*q[1] + (-dU[0]+dU[4]-dU[8])*q[2] + ( dU[5]+dU[7])*q[3]);
//     dq[3] = 2.f*((-dU[1]+dU[3])*q[0] + ( dU[2]+dU[6])*q[1] + ( dU[5]+dU[7])*q[2] + (-dU[0]-dU[4]+dU[8])*q[3]);
// }

}
#endif
