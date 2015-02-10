#ifndef SPLINE_H
#define  SPLINE_H

#include <algorithm>
#include <random>
#include <cstring>
#include "vector_math.h"

void solve_periodic_1d_spline(
        int n, 
        double* coefficients, // length 4*n
        const double* data,   // length n
        double* temp_storage);// length 8*n


void solve_clamped_1d_spline(
        int n, 
        double* coefficients, // length 4*(n-1)
        const double* data,   // length n
        double* temp_storage);// length 4*n


void solve_periodic_2d_spline(
        int nx, int ny,
        double* coefficients, // length (nx,ny,4,4) row-major
        const double* data,   // length (nx,ny), row-major
        double* temp_storage);// length (nx+8)*(ny+8)*4


template<typename T>
inline void spline_value_and_deriv(T deriv[3], const T* c, T fx, T fy) {
    T fx2 = fx*fx; 
    T fx3 = fx*fx2;

    T fy2 = fy*fy;

    T vx0 = c[ 0] + fy*(c[ 1] + fy*(c[ 2] + fy*c[ 3]));
    T vx1 = c[ 4] + fy*(c[ 5] + fy*(c[ 6] + fy*c[ 7]));
    T vx2 = c[ 8] + fy*(c[ 9] + fy*(c[10] + fy*c[11]));
    T vx3 = c[12] + fy*(c[13] + fy*(c[14] + fy*c[15]));

    // T vy0 = c[ 0] + fx*(c[ 4] + fx*(c[ 8] + fx*c[12]));
    T vy1 = c[ 1] + fx*(c[ 5] + fx*(c[ 9] + fx*c[13]));
    T vy2 = c[ 2] + fx*(c[ 6] + fx*(c[10] + fx*c[14]));
    T vy3 = c[ 3] + fx*(c[ 7] + fx*(c[11] + fx*c[15]));

    // dx,dy,value is the order
    deriv[0] = vx1 + T(2.f)*fx*vx2 + T(3.f)*fx2*vx3;
    deriv[1] = vy1 + T(2.f)*fy*vy2 + T(3.f)*fy2*vy3;
    deriv[2] = vx0 + fx*vx1 + fx2*vx2 + fx3*vx3;
}

template<typename T>
inline void spline_value_and_deriv(T result[2], const T* c, T fx) {
    T fx2 = fx*fx; 
    T fx3 = fx*fx2;
    
    result[0] =           c[1] + T(2.f)*fx *c[2] + T(3.f)*fx2*c[3]; // deriv
    result[1] = c[0] + fx*c[1] +        fx2*c[2] +        fx3*c[3]; // value
}


template<int NDIM_VALUE>
struct LayeredPeriodicSpline2D {
    const int n_layer;
    const int nx;
    const int ny;
    std::vector<float> coefficients;

    LayeredPeriodicSpline2D(int n_layer_, int nx_, int ny_):
        n_layer(n_layer_), nx(nx_), ny(ny_), coefficients(n_layer*nx*ny*NDIM_VALUE*16)
    {}

    void fit_spline(const double* data) // size (n_layer, nx, ny, NDIM_VALUE)
    {
        // store values in float, but solve system in double
        std::vector<double> coeff_tmp(nx*ny*16);
        std::vector<double> data_tmp(nx*ny);
        std::vector<double> temp_storage((nx+8)*(ny+8)*4);

        for(int il=0; il<n_layer; ++il) {
            for(int id=0; id<NDIM_VALUE; ++id) {

                // copy data to buffer to solve spline
                for(int ix=0; ix<nx; ++ix) 
                    for(int iy=0; iy<ny; ++iy) 
                        data_tmp[ix*ny+iy] = data[((il*nx + ix)*ny + iy)*NDIM_VALUE + id];

                solve_periodic_2d_spline(nx,ny, coeff_tmp.data(), data_tmp.data(), temp_storage.data());

                // copy spline coefficients to coefficient array
                for(int ix=0; ix<nx; ++ix) 
                    for(int iy=0; iy<ny; ++iy) 
                        for(int ic=0; ic<16; ++ic) 
                            coefficients[(((il*nx+ix)*ny+iy)*NDIM_VALUE+id)*16+ic] = coeff_tmp[(ix*ny+iy)*16+ic];
            }
        }
    }


    void evaluate_value_and_deriv(float* restrict result, int layer, float x, float y) const {
        // order of answer is (dx1,dy1,value1, dx2,dy2,value2, ...)
        int x_bin = int(x);
        int y_bin = int(y);

        float fx = x - x_bin;
        float fy = y - y_bin;

        const float* c = coefficients.data() + (layer*nx*ny + x_bin*ny + y_bin)*16*NDIM_VALUE;

        for(int id=0; id<NDIM_VALUE; ++id) 
            spline_value_and_deriv(result+id*3, c+id*16, fx, fy);
    }
};


template<int NDIM_VALUE>
struct LayeredClampedSpline1D {
    const int n_layer;
    const int nx;
    std::vector<float> coefficients;
    std::vector<float> left_clamped_value;
    std::vector<float> right_clamped_value;

    LayeredClampedSpline1D(int n_layer_, int nx_):
        n_layer(n_layer_), nx(nx_), coefficients(n_layer*nx*NDIM_VALUE*4),
        left_clamped_value (n_layer*NDIM_VALUE),
        right_clamped_value(n_layer*NDIM_VALUE)
    {}

    void fit_spline(const double* data)  // size (n_layer, nx, NDIM_VALUE)
    {
        // store values in float, but solve system in double
        std::vector<double> coeff_tmp((nx-1)*4);
        std::vector<double> data_tmp(nx);
        std::vector<double> temp_storage(4*nx);

        for(int il=0; il<n_layer; ++il) {
            for(int id=0; id<NDIM_VALUE; ++id) {
                left_clamped_value [il*NDIM_VALUE+id] = data[(il*nx + 0     )*NDIM_VALUE + id];
                right_clamped_value[il*NDIM_VALUE+id] = data[(il*nx + (nx-1))*NDIM_VALUE + id];

                // copy data to buffer to solve spline
                for(int ix=0; ix<nx; ++ix) 
                    data_tmp[ix] = data[(il*nx + ix)*NDIM_VALUE + id];

                solve_clamped_1d_spline(nx, coeff_tmp.data(), data_tmp.data(), temp_storage.data());

                // copy spline coefficients to coefficient array
                for(int ix=0; ix<nx-1; ++ix)   // nx-1 splines in clamped spline
                    for(int ic=0; ic<4; ++ic) 
                        coefficients[((il*(nx-1)+ix)*NDIM_VALUE+id)*4+ic] = coeff_tmp[ix*4+ic];
            }
        }
    }
        
    // result contains (dx1, value1, dx2, value2, ...)
    void evaluate_value_and_deriv(float* restrict result, int layer, float x) const {
        if(x>=nx-1) {
            for(int id=0; id<NDIM_VALUE; ++id) {
                result[id*2+0] = 0.f;
                result[id*2+1] = right_clamped_value[layer*NDIM_VALUE+id]; 
            }
        } else if(x<=0) {
            for(int id=0; id<NDIM_VALUE; ++id) {
                result[id*2+0] = 0.f;
                result[id*2+1] = left_clamped_value[layer*NDIM_VALUE+id]; 
            }
        } else {
            int x_bin = int(x);
            float fx = x - x_bin;

            const float* c = coefficients.data() + (layer*(nx-1) + x_bin)*4*NDIM_VALUE;
            for(int id=0; id<NDIM_VALUE; ++id) 
                spline_value_and_deriv(result+id*2, c+id*4, fx);
            // printf("%i %f  %i %f  %f %f\n", layer,x, x_bin,fx, result[0], result[1]);
        }
    }
};
#endif
