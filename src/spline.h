#ifndef SPLINE_H
#define  SPLINE_H

#include <algorithm>
#include <random>
#include <cstring>
#include "vector_math.h"

using namespace std;

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


template<int NDIM_VALUE>
struct LayeredPeriodicSpline2D {
    const int n_layer;
    const int nx;
    const int ny;
    vector<float> coefficients;

    LayeredPeriodicSpline2D(int n_layer_, int nx_, int ny_):
        n_layer(n_layer_), nx(nx_), ny(ny_), coefficients(n_layer*nx*ny*NDIM_VALUE*16)
    {}

    void fit_spline(const double* data) // size (n_layer, nx, ny, NDIM_VALUE)
    {
        // store values in float, but solve system in double
        vector<double> coeff_tmp(nx*ny*16);
        vector<double> data_tmp(nx*ny);
        vector<double> temp_storage((nx+8)*(ny+8)*4);

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
#endif
