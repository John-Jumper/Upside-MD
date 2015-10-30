#ifndef MD_EXPORT_H
#define MD_EXPORT_H

#include "coord.h"

void
integration_stage(
        SysArray mom,
        SysArray pos,
        const SysArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom);

void 
deriv_accumulation(
        SysArray deriv, 
        const SysArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom);

void
recenter(SysArray pos, bool xy_recenter_only, int n_atom);
#endif
