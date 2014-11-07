#include "md.h"
#include "coord.h"
#include "random.h"
#include "md_export.h"
#include <cmath>


template <typename MutableCoordT>
void ornstein_uhlenbeck_thermostat_body(
        RandomGenerator& random,
        MutableCoordT &mom,
        const float mom_scale,
        const float noise_scale)
{
    mom.set_value(mom_scale*mom.f3() + noise_scale*random.normal3());
}


void
ornstein_uhlenbeck_thermostat(
        SysArray mom,
        const float mom_scale,
        const float noise_scale,
        int seed,  // FIXME uint32_t
        int n_atoms,
        int n_round, // FIXME uint64_t
        int n_system)
{
#pragma omp parallel for
    for(uint32_t ns=0; ns<(uint32_t)n_system; ++ns) {
        for(int na=0; na<n_atoms; ++na) {
            RandomGenerator random(seed, 0u, na*n_system + ns, n_round);
            MutableCoord<3> p(mom, ns, na);
            ornstein_uhlenbeck_thermostat_body(random, p, mom_scale, noise_scale);
            p.flush();
        }
    }
}


template <typename CoordT>
float3 group_com_body(
        CoordT* restrict coords,
        const GroupCOMParams* restrict p)
{
    int n_atom_in_group = p->n_atom_in_group;
    float inv_n_atom = 1.f / n_atom_in_group;

    float dcol0[3] = {inv_n_atom,        0.f,        0.f};
    float dcol1[3] = {0.f,        inv_n_atom,        0.f};
    float dcol2[3] = {0.f,               0.f, inv_n_atom};

    float3 my_com_sum = {0.f, 0.f, 0.f};
    for(int at=0; at<n_atom_in_group; ++at) {
        my_com_sum += static_cast<float3>(coords[at]);
        coords[at].set_deriv(0, dcol0);
        coords[at].set_deriv(1, dcol1);
        coords[at].set_deriv(2, dcol2);
    }
    return my_com_sum*inv_n_atom;
}


template <typename MutableCoordT, typename StaticCoordT>
void
integration_stage_body(
        MutableCoordT &mom,
        MutableCoordT &pos,
        StaticCoordT  &deriv,
        float vel_factor,
        float pos_factor, 
        float max_force)
{
    // assumes unit mass for all particles
    float3 f = deriv.f3();
    if(max_force) {
        float f_mag = mag(f)+1e-6f;  // ensure no NaN when mag(f)==0.
        float scale_factor = atan(f_mag * ((0.5f*M_PI_F) / max_force)) * (max_force/f_mag * (2.f/M_PI_F));
        f *= scale_factor;
    }

    mom.set_value(mom.f3() - vel_factor*f);
    pos.set_value(pos.f3() + pos_factor*mom.f3());
}

void
integration_stage(
        SysArray mom,
        SysArray pos,
        const SysArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom,
        int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int na=0; na<n_atom; ++na) {
            MutableCoord<3> p(mom,   ns, na);
            MutableCoord<3> x(pos,   ns, na);
            StaticCoord <3> d(deriv, ns, na);
            integration_stage_body(p, x, d, vel_factor, pos_factor, max_force);
            x.flush();
            p.flush();
        }
    }
}


void deriv_accumulation(
        SysArray deriv, 
        const SysArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom,
        int n_system)
{
#pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        std::vector<MutableCoord<3>> coords;
        coords.reserve(n_atom);
        for(int na=0; na<n_atom; ++na) coords.emplace_back(deriv, ns, na, MutableCoord<3>::Zero);

        for(int nt=0; nt<n_tape; ++nt) {
            auto tape_elem = tape[nt];
            for(int rec=0; rec<tape_elem.output_width; ++rec) 
                coords[tape_elem.atom] += StaticCoord<3>(accum_buffer, ns, tape_elem.loc + rec).f3();
        }

        for(int na=0; na<n_atom; ++na) coords[na].flush();
    }
}

void
recenter(
        SysArray pos,
        int n_atom, int n_system
        )
{
    for(int ns=0; ns<n_system; ++ns) {
        float3 center = {0.f, 0.f, 0.f};
        for(int na=0; na<n_atom; ++na) center += StaticCoord<3>(pos,ns,na).f3();
        center /= n_atom;

        for(int na=0; na<n_atom; ++na) {
            MutableCoord<3> x(pos,ns,na);
            x.set_value(x.f3() - center);
            x.flush();
        }
    }
}
