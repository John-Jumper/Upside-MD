#ifndef SPLINE_H
#define  SPLINE_H

#include <algorithm>
#include <random>
#include <cstring>
#include "vector_math.h"
#include "Float4.h"

//! \brief Compute polynomial coefficients from periodic data
//!
//! Computes data in polynomial representation, where
//! each segment of the spline is represented by coefficients
//! for ax^3+bx^2+cx+d
void solve_periodic_1d_spline(
        int n,                //!< number of data points
        double* coefficients, //!< [out] coefficients, length 4*n
        const double* data,   //!< [in] data points to fit with spline, length n
        double* temp_storage  //!< [out] scratch storage for the fitting algorithm, length 8*n
        );

//! \brief Compute polynomial coefficients from data for clamped spline
//!
//! Computes data in polynomial representation, where
//! each segment of the spline is represented by coefficients
//! for ax^3+bx^2+cx+d.  Clamped spline means that the derivative
//! of the spline is zero on both ends.
void solve_clamped_1d_spline(
        int n, 
        double* coefficients, // length 4*(n-1)
        const double* data,   // length n
        double* temp_storage);// length 4*n

//! \brief Compute bspline coefficients from data for clamped spline
//!
//! Computes data in bspline coefficient representation, where
//! each of the values represents the scale factor for a single
//! B-spline centered at that knot.  Clamped spline means that the derivative
//! of the spline is zero on both ends.
void solve_clamped_1d_spline_for_bsplines(
        int n_coeff, 
        double* coefficients, // length n_coeff
        const double* data,   // length n_coeff-2
        double* temp_storage);// length 3*n_coeff


//! \brief Compute polynomial coefficients for 2d periodic data
//!
//! Computes data in polynomial representation, where
//! each segment of the spline is represented by the 16 coefficients
//! necessary for a 3d bicubic polynomial.
void solve_periodic_2d_spline(
        int nx, int ny,
        double* coefficients, // length (nx,ny,4,4) row-major
        const double* data,   // length (nx,ny), row-major
        double* temp_storage);// length (nx+8)*(ny+8)*4


//! \brief Evaluate 2d polynomial in polynomial representation to get value and deriv
template<typename T>
inline void spline_value_and_deriv(T &value, T &dx, T &dy, const T* c, T fx, T fy) {
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

    dx    = vx1 + T(2.f)*fx*vx2 + T(3.f)*fx2*vx3;
    dy    = vy1 + T(2.f)*fy*vy2 + T(3.f)*fy2*vy3;
    value = vx0 + fx*vx1 + fx2*vx2 + fx3*vx3;
}

//! \brief Evaluate 1d polynomial in polynomial representation to get value and deriv
template<typename T>
inline void spline_value_and_deriv(T result[2], const T* c, T fx) {
    T fx2 = fx*fx; 
    T fx3 = fx*fx2;
    
    result[0] =           c[1] + T(2.f)*fx *c[2] + T(3.f)*fx2*c[3]; // deriv
    result[1] = c[0] + fx*c[1] +        fx2*c[2] +        fx3*c[3]; // value
}

//! \brief Evaluate 1d polynomial in B-spline basis to get value and deriv
//!
//! Uses the de Boor algorithm for splines in the B-spline representation.  The 
//! value is returned in the zeroth component and the derivative is returned in the
//! first component.  This function computes a single spline evaluation at a time.
inline Vec<2> deBoor_value_and_deriv(const float* bspline_coeff, const float x) {
    // this function assumes that endpoint conditions (say x<=1.) have already been taken care of
    // the first spline is centered at -1
    // return is value then derivative

    int x_bin = int(x); // must be at least 1
    Float4 y = Float4(x-x_bin);

    Float4 c0(bspline_coeff+(x_bin-1), Alignment::unaligned);
    // let's do value first

    // leftmost value of u is unused
    alignas(16) const float u[4] = {0.f, -2.f, -1.f, 0.f};
    Float4 yu = y - Float4(u);
    Float4 one = Float4(1.f);

    Float4 alpha1 = Float4(1.f/3.f) * yu;
    Float4 c1 = (one-alpha1)*c0.right_rotate() + alpha1*c0;

    Float4 alpha2 = Float4(1.f/2.f) * yu;
    Float4 c2 = (one-alpha2)*c1.right_rotate() + alpha2*c1;

    Float4 alpha3 =                   yu;
    Float4 c3 = (one-alpha3)*c2.right_rotate() + alpha3*c2;

    // now let's do derivatives
    Float4 d1 = c0 - c0.right_rotate();  // coeffs for spline of order 2
    Float4 d2 = (one-alpha2)*d1.right_rotate() + alpha2*d1;
    Float4 d3 = (one-alpha3)*d2.right_rotate() + alpha3*d2;

    return make_vec2(c3.w(), d3.w());
}


//! \brief Evaluate 1d polynomial in B-spline basis to get value and deriv (Float4)
//!
//! Uses the de Boor algorithm for splines in the B-spline representation.  The 
//! value is returned in the zeroth component and the derivative is returned in the
//! first component.  This function computes four spline evaluation at a time.
inline Vec<2,Float4> uniform_deBoor_algorithm(
    const Float4& c00, const Float4& c01, const Float4& c02, const Float4& c03,
    const Float4& excess) 
    // excess should be on [0,1]
{
    auto one = Float4(1.f);

    auto yu1 = excess + Float4(2.f);
    auto yu2 = excess + one;
    auto yu3 = excess;

    auto frac13 = Float4(1.f/3.f);
    auto alpha11 = frac13*yu1;
    auto alpha12 = frac13*yu2;
    auto alpha13 = frac13*yu3;

    auto c11 = fmadd(one-alpha11,c00, alpha11*c01);
    auto d11 = c01 - c00;  // deriv coeffs for spline of order 2
    auto c12 = fmadd(one-alpha12,c01, alpha12*c02);
    auto d12 = c02 - c01;
    auto c13 = fmadd(one-alpha13,c02, alpha13*c03);
    auto d13 = c03 - c02;

    auto frac12 = Float4(1.f/2.f);
    auto alpha22 = frac12*yu2;
    auto alpha23 = frac12*yu3;

    auto c22 = fmadd(one-alpha22,c11, alpha22*c12);
    auto d22 = fmadd(one-alpha22,d11, alpha22*d12);
    auto c23 = fmadd(one-alpha23,c12, alpha23*c13);
    auto d23 = fmadd(one-alpha23,d12, alpha23*d13);

    auto alpha33 = yu3;

    auto c33 = fmadd(one-alpha33,c22, alpha33*c23);
    auto d33 = fmadd(one-alpha33,d22, alpha33*d23);

    return make_vec2(c33, d33);
}


//! \brief Evaluate 2d polynomial in B-spline basis to get value and deriv (Float4)
//!
//! Uses the de Boor algorithm for splines in the B-spline representation.  The 
//! value is returned in the zeroth component and the derivative is returned in the
//! first and second components.  This function computes four spline evaluation at a time.
inline Vec<3,Float4> deBoor2d_value_and_deriv(const int ny, const float* bspline_coeff[4],
        const Float4& x, const Float4& y) {
    // return array contains (value, dvalue/dx, dvalue/dy)

    // First compute the x bins so I know where to pick out rows for the y deBoor
    alignas(16) int x_bin[4]; x.truncate_to_int().store(x_bin);  // must be at least 1
    alignas(16) int y_bin[4]; y.truncate_to_int().store(y_bin);  // must be at least 1
    alignas(16) float x_excess[4]; (x - x.round<_MM_FROUND_TO_ZERO>()).store(x_excess);

    Float4 val_coeff[4];
    Float4  dx_coeff[4];

    for(int i=0; i<4; ++i) {
        const float* bspline_base = bspline_coeff[i] + (x_bin[i]-1)*ny + y_bin[i]-1;  // lower left corner of coeffs
        Float4 c00(bspline_base + 0*ny, Alignment::unaligned);
        Float4 c01(bspline_base + 1*ny, Alignment::unaligned);
        Float4 c02(bspline_base + 2*ny, Alignment::unaligned);
        Float4 c03(bspline_base + 3*ny, Alignment::unaligned);

        // execute deBoor for all y columns simultaneously
        auto v = uniform_deBoor_algorithm(c00,c01,c02,c03, Float4(x_excess[i]));
        val_coeff[i] = v[0];
        dx_coeff [i] = v[1];
    }

    // prepare for second round
    transpose4(val_coeff[0],val_coeff[1],val_coeff[2],val_coeff[3]);
    transpose4( dx_coeff[0], dx_coeff[1], dx_coeff[2], dx_coeff[3]);

    Float4 y_excess = y - y.round<_MM_FROUND_TO_ZERO>();

    auto val = uniform_deBoor_algorithm(val_coeff[0],val_coeff[1],val_coeff[2],val_coeff[3], y_excess);
    auto dx  = uniform_deBoor_algorithm( dx_coeff[0], dx_coeff[1], dx_coeff[2], dx_coeff[3], y_excess);

    return make_vec3(val[0],dx[0],val[1]);
}


//! \brief Evaluate 1d polynomial in B-spline basis to get value and deriv (Float4)
//!
//! Uses the de Boor algorithm for splines in the B-spline representation.  The 
//! value is returned in the zeroth component and the derivative is returned in the
//! first component.  This function computes four spline evaluation at a time.
//!
//! This function assumes that endpoint conditions (say x<=1.) have already been taken care of.
//! The first spline is centered at -1.
inline Vec<2,Float4> deBoor_value_and_deriv(const float* bspline_coeff[4], const Float4& x) {

    Int4 x_bin = x.truncate_to_int();  // must be at least 1
    Float4 y = x - x.round<_MM_FROUND_TO_ZERO>();

    // Read the coefficients for each slot then transpose to prepare
    //   for vertical simd
    Float4 c00(bspline_coeff[0]-1 + x_bin.x(), Alignment::unaligned);
    Float4 c01(bspline_coeff[1]-1 + x_bin.y(), Alignment::unaligned);
    Float4 c02(bspline_coeff[2]-1 + x_bin.z(), Alignment::unaligned);
    Float4 c03(bspline_coeff[3]-1 + x_bin.w(), Alignment::unaligned);
    transpose4(c00,c01,c02,c03);

    return uniform_deBoor_algorithm(c00,c01,c02,c03, y);
}

//! \brief Left side clamping value for clamped spline
//!
//! For clamping with derivative continuity, you should really have
//! bspline_coeff[0] == bspline_coeff[2]
inline float clamped_spline_left(const float* bspline_coeff, int n_coeff) {
    return (1.f/6.f)*bspline_coeff[0] + 
           (2.f/3.f)*bspline_coeff[1] + 
           (1.f/6.f)*bspline_coeff[2];
}

//! \brief Right side clamping value for clamped spline
//!
//! For clamping with derivative continuity, you should really have
//! bspline_coeff[n_coeff-3] == bspline_coeff[n_coeff-1]
inline float clamped_spline_right(const float* bspline_coeff, int n_coeff) {
    return (1.f/6.f)*bspline_coeff[n_coeff-3] + 
           (2.f/3.f)*bspline_coeff[n_coeff-2] + 
           (1.f/6.f)*bspline_coeff[n_coeff-1];
}

//! \brief Evaluate clamped spline that has n_knots
//!
//! For clamping with derivative continuity, you should really have
//! bspline_coeff[0] == bspline_coeff[2] and  bspline_coeff[n_knot-3] == bspline_coeff[n_knot-1]
inline Vec<2> clamped_deBoor_value_and_deriv(const float* bspline_coeff, const float x, int n_knot) {
    if(x<=1.f)      return make_vec2(clamped_spline_left (bspline_coeff, n_knot), 0.f);
    if(x>=n_knot-2) return make_vec2(clamped_spline_right(bspline_coeff, n_knot), 0.f);
    return deBoor_value_and_deriv(bspline_coeff, x);
}

//! \brief Float4 implementation of clamped_deBoor_value_and_deriv
inline Vec<2,Float4> clamped_deBoor_value_and_deriv(const float* bspline_coeff[4],
        const Float4& x, int n_knot) {
    auto one = Float4(1.f);
    auto too_small = x<one;
    auto too_big   = Float4(n_knot-2)<=x;
    auto clamp_mask = too_small | too_big;

    auto x_clamp = ternary(clamp_mask, one, x); // make all values in bounds to proceed with spline eval
    auto value = deBoor_value_and_deriv(bspline_coeff, x_clamp);

    if(clamp_mask.any()) { // hopefully rare
        value.y() = ternary(clamp_mask, Float4(), value.y());  // both cases set derivative to 0

        auto outer = Float4(1.f/6.f);
        auto inner = Float4(2.f/3.f);
        
        Float4 c0(bspline_coeff[0], Alignment::unaligned);
        Float4 c1(bspline_coeff[1], Alignment::unaligned);
        Float4 c2(bspline_coeff[2], Alignment::unaligned);
        Float4 c3(bspline_coeff[3], Alignment::unaligned);
        transpose4(c0,c1,c2,c3);
        auto left_clamp = outer*c0 + inner*c1 + outer*c2;

        value.x() = ternary(too_small, left_clamp, value.x());

        Float4 cn4(bspline_coeff[0]+n_knot-4, Alignment::unaligned);
        Float4 cn3(bspline_coeff[1]+n_knot-4, Alignment::unaligned);
        Float4 cn2(bspline_coeff[2]+n_knot-4, Alignment::unaligned);
        Float4 cn1(bspline_coeff[3]+n_knot-4, Alignment::unaligned);
        transpose4(cn4,cn3,cn2,cn1);
        auto right_clamp = outer*cn3 + inner*cn2 + outer*cn1;

        value.x() = ternary(too_big,  right_clamp, value.x());
    }
    return value;
}

//! \brief Coefficient derivative for splines in the B-spline representation
//!
//! The derivative of a spline has 4 non-zero coefficients in the B-spline basis,
//! which are the 4 coefficients that contribute to that value.  This function
//! returns the value of those coefficients in result and the first coefficient with
//! non-zero derivative as starting_bin.
inline void deBoor_coeff_deriv(int* starting_bin, float result[4], const float x) {
    // this function is not intended to be especially efficient

    int x_bin = int(x); // must be at least 1
    *starting_bin = x_bin-1;
    float y = x-x_bin + 1.f; // ensure y in [1.,2)

    float dc[4];

    // derivative with respect to coefficient
    // is the value of spline with zeros for every coefficient
    // except that one

    for(int i=0; i<4; ++i) {
        dc[0]=dc[1]=dc[2]=dc[3]=0.f;
        dc[i]=1.f;
        result[i] = deBoor_value_and_deriv(dc, y).x();
    }
}

//! \brief Coefficient derivative for 2d splines in the B-spline representation
//! 
//! The derivative of a spline has 4x4 non-zero coefficients in the B-spline basis,
//! which are the 4x4 coefficients that contribute to that value.  This function
//! returns the value of those coefficients in result and the first coefficient with
//! non-zero derivative as starting_bin.
inline void deBoor2d_coeff_deriv(int* starting_bin, float result[4][4], int ny, const float x, const float y) {
    // this function is not intended to be especially efficient

    int x_bin = int(x); // must be at least 1
    int y_bin = int(y); // must be at least 1
    *starting_bin = (x_bin-1)*ny + (y_bin-1);  // lower left corner of coeffs

    // Derivative with respect to coefficient
    // is the value of spline with zeros for every coefficient
    // except that one since the coefficients are linearly coupled
    // to the basis functions
    
    // Since the 2d spline basis functions are just products of 1d spline basis functions,
    // we can just take the outer product of the 1d coeff_deriv values.
    
    int unused;
    float x_deriv[4]; deBoor_coeff_deriv(&unused, x_deriv, x);
    float y_deriv[4]; deBoor_coeff_deriv(&unused, y_deriv, y);

    for(int i=0; i<4; ++i)
        for(int j=0; j<4; ++j)
            result[i][j] = x_deriv[i]*y_deriv[j];
}


//! \brief Coefficient derivative for clamped splines in the B-spline representation
//!
//! The derivative of a spline has 4 non-zero coefficients in the B-spline basis,
//! which are the 4 coefficients that contribute to that value.  This function
//! returns the value of those coefficients in result and the first coefficient with
//! non-zero derivative as starting_bin.
inline void clamped_deBoor_coeff_deriv(int* starting_bin,
        float result[4], const float x, int n_knot) {
    if(x<=1.f) {
        *starting_bin = 0;
        result[0] = 1.f/6.f;
        result[1] = 2.f/3.f;
        result[2] = 1.f/6.f;
        result[3] = 0.f;
    } else if(x>=n_knot-2) {
        *starting_bin = n_knot-4;
        result[0] = 0.f;
        result[1] = 1.f/6.f;
        result[2] = 2.f/3.f;
        result[3] = 1.f/6.f;
    } else {
        deBoor_coeff_deriv(starting_bin, result, x);
    }
}


//! \brief Old-style spline object -- not intended for new code
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


    void evaluate_value_and_deriv(
            float* restrict value, 
            float* restrict dx, 
            float* restrict dy, 
            int layer, float x, float y) const {
        // order of answer is (dx1,dy1,value1, dx2,dy2,value2, ...)
        int x_bin = int(x);
        int y_bin = int(y);

        float fx = x - x_bin;
        float fy = y - y_bin;

        const float* c = coefficients.data() + (layer*nx*ny + x_bin*ny + y_bin)*16*NDIM_VALUE;

        for(int id=0; id<NDIM_VALUE; ++id) 
            spline_value_and_deriv(value[id], dx[id], dy[id], c+id*16, fx, fy);
    }
};


//! \brief Old-style spline object -- not intended for new code
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
        }
    }
};
#endif
