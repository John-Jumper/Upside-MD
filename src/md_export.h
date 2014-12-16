#ifndef MD_EXPORT_H
#define MD_EXPORT_H

#include "md.h"
#include "coord.h"

#define N_STATE 5

struct RamaMapGerm {
    float val [N_STATE];
    float dphi[N_STATE];
    float dpsi[N_STATE];
};

struct PackedRefPos {
    int32_t n_atoms;
    uint32_t pos[4];
};

void dist_spring(
        const CoordArray pos,
        const DistSpringParams* params,
        int n_terms, int n_system);
void angle_spring(
        const CoordArray pos,
        const AngleSpringParams* params,
        int n_terms, int n_system);
void dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* params,
        int n_terms, int n_system);
void dynamic_dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* restrict params,
        int params_offset,
        int n_terms, int n_system);
void dihedral_angle_range(
        const CoordArray   pos,
        const DihedralRangeParams* params,
        int n_terms, 
        int n_system);
void affine_alignment(
        SysArray rigid_body,
        CoordArray pos,
        const AffineAlignmentParams* restrict params,
        int n_res,
        int n_system);
void backbone_pairs(
        const CoordArray rigid_body,
        const PackedRefPos* ref_pos,
        const AffineParams* params,
        float energy_scale,
        float dist_cutoff2,
        int n_res, int n_system);
void pos_spring(
        const CoordArray pos,
        const PosSpringParams* restrict params,
        int n_terms, int n_system);
void affine_reverse_autodiff(
        const SysArray affine,
        const SysArray affine_accum,
        SysArray pos_deriv,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_atom, int n_system);

uint32_t pack_atom(const float x[3]);

void
ornstein_uhlenbeck_thermostat(
        SysArray mom,
        const float mom_scale,
        const float noise_scale,
        int seed,
        int n_atoms,
        int n_round, int n_system);

void
integration_stage(
        SysArray mom,
        SysArray pos,
        const SysArray deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom, int n_system);

void 
deriv_accumulation(
        SysArray deriv, 
        const SysArray accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom, int n_system);

void
recenter(
        SysArray pos,
        int n_atom, int n_system
        );

void hmm(
        const CoordArray pos,
        const HMMParams* params,
        const float* trans_matrices,
        int n_bin,
        const RamaMapGerm* rama_map_data,
        int n_residue, int n_system);

void z_flat_bottom_spring(
        const CoordArray pos,
        const ZFlatBottomParams* params,
        int n_terms, int n_system);

#endif
