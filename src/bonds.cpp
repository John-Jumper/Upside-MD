#include "md.h"
#include "coord.h"
#include "md_export.h"

template <typename CoordT>
inline void pos_spring_body(
        CoordT &x1,
        const PosSpringParams &p)
{
    float3 disp = x1.f3() - make_float3(p.x,p.y,p.z);
    x1.set_deriv(p.spring_constant * disp);
}


void pos_spring(
        const CoordArray pos,
        const PosSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            pos_spring_body(x1, params[nt]);
            x1.flush();
        }
    }
}


template <typename CoordT>
inline void dist_spring_body(
        CoordT &x1,
        CoordT &x2,
        const DistSpringParams &p)
{
    float3 disp = x1.f3() - x2.f3();
    float3 deriv = p.spring_constant * (1.f - p.equil_dist*inv_mag(disp)) * disp;
    // V(x1,x2) = spring_const * (|x1-x2| - equil_dist)^2

    x1.set_deriv( deriv);
    x2.set_deriv(-deriv);
}


void dist_spring(
        const CoordArray pos,
        const DistSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            dist_spring_body(x1,x2, params[nt]);
            x1.flush();
            x2.flush();
        }
    }
}


void z_flat_bottom_spring(
        const CoordArray pos,
        const ZFlatBottomParams* params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            ZFlatBottomParams p = params[nt];
            Coord<3> atom_pos(pos, ns, p.atom);
            
            float z = atom_pos.f3().z;
            float3 deriv(0.f, 0.f, 0.f);
            if(z-p.z0 >  p.radius) deriv.z = p.spring_constant * (z-p.z0 - p.radius);
            if(z-p.z0 < -p.radius) deriv.z = p.spring_constant * (z-p.z0 + p.radius);
            atom_pos.set_deriv(deriv);
            atom_pos.flush();
        }
    }
}


template <typename CoordT>
inline void angle_spring_body(
        CoordT &atom1,
        CoordT &atom2,
        CoordT &atom3,   // middle atom
        const AngleSpringParams &p)
{
    float3 x1 = atom1.f3() - atom3.f3(); float inv_d1 = inv_mag(x1); float3 x1h = x1*inv_d1;
    float3 x2 = atom2.f3() - atom3.f3(); float inv_d2 = inv_mag(x2); float3 x2h = x2*inv_d2;

    float dp = dot(x1h, x2h);
    float force_prefactor = p.spring_constant * (dp - p.equil_dp);

    atom1.set_deriv(force_prefactor * (x2h - x1h*dp) * inv_d1);
    atom2.set_deriv(force_prefactor * (x1h - x2h*dp) * inv_d2);
    atom3.set_deriv(-atom1.df3(0)-atom2.df3(0));  // computed by the condition of zero net force
}


void angle_spring(
        const CoordArray pos,
        const AngleSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            angle_spring_body(x1,x2,x3, params[nt]);
            x1.flush();
            x2.flush();
            x3.flush();
        }
    }
}



template <typename CoordT>
inline void dihedral_spring_body(
        CoordT &x1,
        CoordT &x2,
        CoordT &x3,
        CoordT &x4,
        const DihedralSpringParams &p)
{
    float3 d1,d2,d3,d4;
    float dihedral = dihedral_germ(x1.f3(),x2.f3(),x3.f3(),x4.f3(), d1,d2,d3,d4);

    // determine minimum periodic image (can be off by at most 2pi)
    float displacement = dihedral - p.equil_dihedral;
    displacement = (displacement> M_PI_F) ? displacement-2.f*M_PI_F : displacement;
    displacement = (displacement<-M_PI_F) ? displacement+2.f*M_PI_F : displacement;

    float c = p.spring_constant * displacement;
    x1.set_deriv(c*d1);
    x2.set_deriv(c*d2);
    x3.set_deriv(c*d3);
    x4.set_deriv(c*d4);
}


void dynamic_dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* restrict params,
        int params_offset,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            Coord<3> x4(pos, ns, params[nt].atom[3]);
            dihedral_spring_body(x1,x2,x3,x4, params[ns*params_offset + nt]);
            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}


void dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            Coord<3> x4(pos, ns, params[nt].atom[3]);
            dihedral_spring_body(x1,x2,x3,x4, params[nt]);
            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}


void dihedral_angle_range(
        const CoordArray   pos,
        const DihedralRangeParams* params,
        int n_terms, 
        int n_system)
{
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            DihedralRangeParams p = params[nt];

            Coord<3> x1(pos, ns, p.atom[0]);
            Coord<3> x2(pos, ns, p.atom[1]);
            Coord<3> x3(pos, ns, p.atom[2]);
            Coord<3> x4(pos, ns, p.atom[3]);

            // potential function is 
            //   V(theta) = 

            float3 d1, d2, d3, d4;
            float theta = dihedral_germ(x1.f3(),x2.f3(),x3.f3(),x4.f3(), d1,d2,d3,d4);
            // d* contains the derivative of the dihedral with respect to the position
            //    of atom *

            // Compute correct image of angle to handle periodicity
            float center_angle = 0.5f * (p.angle_range[0] + p.angle_range[1]);
            float disp = theta - center_angle;   // disp is in range [0,2pi) 
            while(disp >  M_PI_F) disp -= 2.f*M_PI_F;
            while(disp < -M_PI_F) disp += 2.f*M_PI_F;
            theta = center_angle + disp;  // guarantees |theta-center_angle| <= pi

            float zl = expf(p.scale*(p.angle_range[0]-theta));
            float zu = expf(p.scale*(theta-p.angle_range[1]));
            float wl = 1.f/(1.f+zl);
            float wu = 1.f/(1.f+zu);
            
            float dV_dtheta = p.energy * p.scale * wl*wu * (wl*zl - zu*wu);

            x1.set_deriv(dV_dtheta * d1);
            x2.set_deriv(dV_dtheta * d2);
            x3.set_deriv(dV_dtheta * d3);
            x4.set_deriv(dV_dtheta * d4);

            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}
