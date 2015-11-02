#include "coord.h"
#include "deriv_engine.h"
#include "random.h"
#include "md_export.h"
#include <cmath>

void
integration_stage_body(
        MutableCoord<3> &mom,
        MutableCoord<3> &pos,
        Vec<3>  &deriv,
        float vel_factor,
        float pos_factor, 
        float max_force)
{
    // assumes unit mass for all particles
    if(max_force) {
        float f_mag = mag(deriv)+1e-6f;  // ensure no NaN when mag(deriv)==0.
        float scale_factor = atan(f_mag * ((0.5f*M_PI_F) / max_force)) * (max_force/f_mag * (2.f/M_PI_F));
        deriv *= scale_factor;
    }

    mom.set_value(mom.f3() - vel_factor*deriv);
    pos.set_value(pos.f3() + pos_factor*mom.f3());
}

void
integration_stage(
        VecArray mom,
        VecArray pos,
        const VecArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom)
{
    for(int na=0; na<n_atom; ++na) {
        MutableCoord<3> p(mom,   na);
        MutableCoord<3> x(pos,   na);
        auto d = load_vec<3>(deriv, na);
        integration_stage_body(p, x, d, vel_factor, pos_factor, max_force);
        x.flush();
        p.flush();
    }
}


void deriv_accumulation(
        VecArray deriv, 
        const VecArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom)
{
    std::vector<MutableCoord<3>> coords;
    coords.reserve(n_atom);
    for(int na=0; na<n_atom; ++na) coords.emplace_back(deriv, na, MutableCoord<3>::Zero);

    for(int nt=0; nt<n_tape; ++nt) {
        auto tape_elem = tape[nt];
        for(int rec=0; rec<int(tape_elem.output_width); ++rec) {
            coords[tape_elem.atom] += load_vec<3>(accum_buffer, tape_elem.loc + rec);
        }
    }

    for(int na=0; na<n_atom; ++na) coords[na].flush();
}

void
recenter(VecArray pos, bool xy_recenter_only, int n_atom)
{
    float3 center = make_vec3(0.f, 0.f, 0.f);
    for(int na=0; na<n_atom; ++na) center += load_vec<3>(pos,na);
    center /= float(n_atom);

    if(xy_recenter_only) center.z() = 0.f;

    for(int na=0; na<n_atom; ++na) {
        MutableCoord<3> x(pos,na);
        x.set_value(x.f3() - center);
        x.flush();
    }
}
