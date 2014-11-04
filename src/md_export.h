#ifndef MD_EXPORT_H
#define MD_EXPORT_H

#include "md.h"

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
        const float* pos,
        float* pos_deriv,
        const DistSpringParams* params,
        int n_terms);
void angle_spring(
        const float* pos,
        float* pos_deriv,
        const AngleSpringParams* params,
        int n_terms);
void dihedral_spring(
        const float* pos,
        float* pos_deriv,
        const DihedralSpringParams* params,
        int n_terms);
void affine_alignment(
        float* rigid_body,
        float* pos,
        float* pos_deriv,
        const AffineAlignmentParams* params,
        int n_res);
void affine_pairs(
        const float* rigid_body,
        float*       rigid_body_deriv,
        const PackedRefPos* ref_pos,
        const AffineParams* params,
        float energy_scale,
        float dist_cutoff2,
        int n_res);
void pos_spring(
        const float* restrict pos,
        float* restrict pos_deriv,
        const PosSpringParams* restrict params,
        int n_terms);
void affine_reverse_autodiff(
        const float*    affine,
        const float*    affine_accum,
        float* restrict pos_deriv,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_atom);

uint32_t pack_atom(const float x[3]);

void
ornstein_uhlenbeck_thermostat(
        float* mom,
        const float mom_scale,
        const float noise_scale,
        int seed,
        int n_atoms,
        int n_round);

void
integration_stage(
        float* mom,
        float* pos,
        const float* deriv,
        float vel_factor,
        float pos_factor,
        float max_force,
        int n_atom);

void 
deriv_accumulation(
        float* restrict deriv, 
        const float* restrict accum_buffer, 
        const DerivRecord* restrict tape,
        int n_tape,
        int n_atom);

void
recenter(
        float* restrict pos,
        int n_atom
        );

void hmm(
        const float* pos,
        float* pos_deriv,
        const HMMParams* params,
        const float* trans_matrices,
        int n_bin,
        const RamaMapGerm* rama_map_data,
        int n_residue);

#endif
