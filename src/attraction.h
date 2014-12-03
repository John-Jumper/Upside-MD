#ifndef ATTRACTION_H
#define ATTRACTION_H

#include "coord.h"

struct AttractionParams {
    CoordPair loc;
    int       restype;
    float3    sc_ref_pos;
};


struct AttractionInteraction {
    float r0_squared;
    float scale;
    float energy;
};

void attraction_pairs(
        const CoordArray             rigid_body,
        const AttractionParams*      residue_param,
        const AttractionInteraction* interaction_params,
        int n_types, float cutoff,
        int n_res, int n_system);
#endif
