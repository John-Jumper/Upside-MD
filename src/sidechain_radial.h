#ifndef ATTRACTION_H
#define ATTRACTION_H

#include "coord.h"

struct SidechainRadialParams {
    CoordPair loc;
    int       restype;
    float3    sc_ref_pos;
};


struct SidechainRadialInteraction {
    float r0_squared;
    float scale;
    float energy;
};


struct ContactPair {
    CoordPair loc[2];
    float3    sc_ref_pos[2];
    float     r0;
    float     scale;
    float     energy;
};


void radial_pairs(
        const CoordArray             rigid_body,
        const SidechainRadialParams*      residue_param,
        const SidechainRadialInteraction* interaction_params,
        int n_types, float cutoff,
        int n_res, int n_system);

void contact_energy(
        const CoordArray   rigid_body,
        const ContactPair* contact_param,
        int n_contacts, float cutoff, int n_system);
#endif
