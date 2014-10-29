#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cstdio>
#include <iostream>
#include "hbond.h"
#include <algorithm>

using namespace Eigen;
extern "C" {
    void rdc_prediction_simple(
            float* pred_rdc,
            int n_frame,
            int n_atom,
            const float *pos);
    void helicity(
            char* helicity,
            int n_frame,
            int n_res,
            const float *pos);

    void detailed_helicity(
            float* helicity,
            int* is_proline,
            int n_frame,
            int n_res,
            const float *pos);

    void compressed_rdc_prediction(
            double* rdc_table,   // dims (n_heights, n_res)
            double* rdc_count,   // dims (n_heights, )
            float d_height,
            int n_heights,

            int n_quats,
            const float* rotation_quats,  // size (n_heights,4)

            int n_frame,
            int n_atom,
            const float *pos);
}

inline Vector3f infer_x_dir(
        const Vector3f &prev,
        const Vector3f &curr,
        const Vector3f &next)
{
    return ((curr-prev).normalized() + (curr-next).normalized()).normalized();
}


void rdc_prediction_simple(
        float* pred_rdc,
        int n_frame,
        int n_atom,
        const float *pos) 
{
    int n_res = n_atom/3;
    assert(n_res*3 == n_atom);
    for(int nr=0; nr<n_res; ++nr) pred_rdc[nr-1] = 0.f;

    {
        ArrayXd pred_rdc_mat = VectorXd::Zero(n_res-1);

        for(int nf=0; nf<n_frame; ++nf) {
            Map<const Matrix<float,Dynamic,3,RowMajor>> config_map(pos+nf*n_atom*3, n_atom, 3);

            MatrixX3f config = config_map.rowwise() - config_map.colwise().mean();
            float diag_weight = config.rowwise().squaredNorm().sum();

            Matrix3f inertial_tensor = diag_weight*Matrix3f::Identity() - config.transpose() * config;
            SelfAdjointEigenSolver<Matrix3f> inertial_system(inertial_tensor);

            Vector3f alignment_direction = inertial_system.eigenvectors().col(2);
            MatrixX3f NH(n_res-1, 3);

            for(int nr=1; nr<n_res; ++nr) {
                NH.row(nr-1) = infer_x_dir(config.row(3*nr-1), config.row(3*nr), config.row(3*nr+1));
            }

            pred_rdc_mat += (1.5f*(NH * alignment_direction).array().square()-0.5f).cast<double>();
        }

        for(int nr=1; nr<n_res; ++nr) pred_rdc[nr] += pred_rdc_mat[nr-1] / n_frame; // exclude first residue
    }
}


void compressed_rdc_prediction(
        double* rdc_table,   // dims (n_heights, n_res)
        double* rdc_count,   // dims (n_heights, )
        float d_height,
        int n_heights,

        int n_quats,
        const float* rotation_quats,  // size (n_heights,4)

        int n_frame,
        int n_atom,
        const float *pos) 
{
    int n_res = n_atom/3;
    assert(n_res*3 == n_atom);

    float inv_d_height = 1.f/d_height;

    Map<Array<double,Dynamic,Dynamic,RowMajor>> rdc_table_mat(rdc_table, n_heights, n_res);
    Map<Array<double,Dynamic,1>>                rdc_count_mat(rdc_count, n_heights);

    rdc_table_mat.setZero();
    rdc_count_mat.setZero();

    for(int nf=0; nf<n_frame; ++nf) {
        Map<const Matrix<float,Dynamic,3,RowMajor>> config_map(pos+nf*n_atom*3, n_atom, 3);
        MatrixX3f config = config_map.rowwise() - config_map.colwise().mean();

        MatrixX3f NH = MatrixX3f(n_res, 3);
        NH.row(0) = Vector3f::Zero();
        for(int nr=1; nr<n_res; ++nr)
            NH.row(nr) = infer_x_dir(config.row(3*nr-1), config.row(3*nr), config.row(3*nr+1));

        for(int nq=0; nq<n_quats; ++nq) {
            Matrix3f rot = Map<const Quaternionf>(rotation_quats + 4*nq).normalized().toRotationMatrix();

            VectorXf rot_config_z = (config*rot).col(2);
            const int bin_number = (int)(inv_d_height * (rot_config_z.maxCoeff() - rot_config_z.minCoeff()));

            if(bin_number < n_heights) {
                Vector3f alignment_direction  = rot.col(2);   // rot * (0,0,1) = rot.col(2)
                rdc_count_mat[bin_number]     += 1.;
                rdc_table_mat.row(bin_number) += (1.5f*(NH * alignment_direction).array().square()-0.5f).cast<double>();
            }
        }
    }

    // blank the 0th entry because there is no H on the N-terminus
    rdc_table_mat.col(0) *= 0.;
}


void helicity(
        char* is_helix, // size (n_frame, n_res)
        int n_frame,
        int n_res,
        const float *pos) 
{
    int n_atom = n_res*3;

    for(int nf=0; nf<n_frame; ++nf) {
        Map<const Matrix<float,Dynamic,3,RowMajor>> config_map(pos+nf*n_atom*3, n_atom, 3);
        MatrixX3f config = config_map;

        ArrayXi helicity_cand = ArrayXi::Zero(n_res);
        for(int nr=0; nr<n_res-4; ++nr) {
            // look for i<-i+4 H-bonds
            Vector3f C = config.row(3*nr+2);
            Vector3f COdir = infer_x_dir(config.row(3*nr+1),C,config.row(3*nr+3));
            Vector3f O = C + 1.24 * COdir;

            Vector3f N = config.row(3*(nr+4));
            Vector3f NHdir = infer_x_dir(config.row(3*(nr+4)-1),N,config.row(3*(nr+4)+1));
            Vector3f H = N + 0.88 * NHdir;

            Vector3f OH = O-H;
            Vector3f OHdir = OH.normalized();

            float dist = OH.norm();
            float cos_angle_NHO =  NHdir.dot(OHdir);  // cutoff at 70 degrees off linear
            float cos_angle_COH = -COdir.dot(OHdir);  // cutoff at 70 degrees off linear

            if(dist < 3.2f && cos_angle_NHO > 0.342f && cos_angle_COH > 0.342f) {
                helicity_cand[nr  ] = 1;
                helicity_cand[nr+4] = 1;
            }
        }

        for(int nr=0; nr<n_res; ++nr) is_helix[nf*n_res+nr] = 0;

        for(int nr=0; nr<n_res-4; ++nr) {
            bool h = true;
            for(int i=0; i<=4; ++i) h &= bool(helicity_cand[nr+i]);
            if(h) for(int i=0; i<=4; ++i) is_helix[nf*n_res + nr+i] = 1;
        }
    }
}


void detailed_helicity(
        float* helicity,  // size (n_frame, n_residue/3)
        int* is_proline,
        int n_frame,
        int n_res,
        const float *pos) 
{
    int n_atom = n_res*3;

    auto dummy_deriv = vector<float>(4*3*9);  
    auto HN_OC = vector<float>(n_res*12);

    auto infer_params = vector<VirtualParams>();
    auto donor_params = vector<VirtualHBondParams>();
    auto accep_params = vector<VirtualHBondParams>();

    for(int nr=0; nr<n_res; ++nr) {
        if(nr<n_res-1) { // acceptor
            VirtualParams ip = {{CoordPair(3*nr+1,0), CoordPair(3*nr+2,0), CoordPair(3*nr+3,0)}, 1.24f};
            infer_params.push_back(ip);
            VirtualHBondParams ap = {CoordPair(infer_params.size()-1,0), cl_ushort(nr), 0.f};
            accep_params.push_back(ap);
        }

        if(nr>0 && !is_proline[nr]) { // donor
            VirtualParams ip = {{CoordPair(3*nr-1,0), CoordPair(3*nr+0,0), CoordPair(3*nr+1,0)}, 0.88f};
            infer_params.push_back(ip);
            VirtualHBondParams dp = {CoordPair(infer_params.size()-1,0), cl_ushort(nr), 0.f};
            donor_params.push_back(dp);
        }
    }

    for(int nf=0; nf<n_frame; ++nf) {
        // compute virtual atom positions
        infer_HN_OC_pos_and_dir(HN_OC.data(), pos+nf*n_atom*3,dummy_deriv.data(), 
                infer_params.data(), infer_params.size());

        // calculate hbonds
        helical_probabilities(n_res, helicity + nf*n_res, HN_OC.data(), 
                donor_params.size(), donor_params.data(),
                accep_params.size(), accep_params.data());
    }
}
