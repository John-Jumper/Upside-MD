#include "coord.h"
#include "deriv_engine.h"
#include "random.h"
#include "md_export.h"
#include <cmath>

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
        // assumes unit mass for all particles

        auto d = load_vec<3>(deriv, na);
        if(max_force) {
            float f_mag = mag(d)+1e-6f;  // ensure no NaN when mag(deriv)==0.
            float scale_factor = atan(f_mag * ((0.5f*M_PI_F) / max_force)) * (max_force/f_mag * (2.f/M_PI_F));
            d *= scale_factor;
        }

        auto p = load_vec<3>(mom, na) - vel_factor*d;
        store_vec (mom, na, p);
        update_vec(pos, na, pos_factor*p);
    }
}


void deriv_accumulation(
        VecArray deriv, 
        const VecArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom)
{
    std::vector<Vec<3>> coords(n_atom);
    for(int na=0; na<n_atom; ++na) coords[na] = make_zero<3>();

    for(int nt=0; nt<n_tape; ++nt) {
        auto tape_elem = tape[nt];
        for(int rec=0; rec<int(tape_elem.output_width); ++rec) {
            coords[tape_elem.atom] += load_vec<3>(accum_buffer, tape_elem.loc + rec);
        }
    }

    for(int na=0; na<n_atom; ++na) store_vec(deriv, na, coords[na]);
}

void
recenter(VecArray pos, bool xy_recenter_only, int n_atom)
{
    float3 center = make_vec3(0.f, 0.f, 0.f);
    for(int na=0; na<n_atom; ++na) center += load_vec<3>(pos,na);
    center /= float(n_atom);

    if(xy_recenter_only) center.z() = 0.f;

    for(int na=0; na<n_atom; ++na)
        update_vec(pos, na, -center);
}
