#ifndef MD_EXPORT_H
#define MD_EXPORT_H

#include "coord.h"

void
integration_stage(
        VecArray mom,
        VecArray pos,
        const VecArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom);

void 
deriv_accumulation(
        VecArray deriv, 
        const VecArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom);

void
recenter(VecArray pos, bool xy_recenter_only, int n_atom);
#endif
