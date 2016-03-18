#include <algorithm>
#include "vector_math.h"
#include "spline.h"

using namespace std;

void solve_tridiagonal_system(
        int n, 
        double* d,   // right-hand-side (length n), after function finishes, solution is stored in d
        double* a,   // sub-diagonal (length n-1)
        double* b,   // main-diagonal (length n)
        double* c)   // super-diagonal (length n-1)
{
    // this function uses Thomas's algorithm to solve a tridiagonal system
    // original coefficients will be destroyed in the process
    // solution was verified

    // Forward phase
    for(int k=1; k<n; ++k) {
        double m = a[k-1]/b[k-1];
        b[k] -= m*c[k-1];
        d[k] -= m*d[k-1];
    }

    // Backward phase
    d[n-1] = d[n-1]/b[n-1];
    for(int k=n-2; k>=0; --k) {
        d[k] = (d[k] - c[k]*d[k+1])/b[k];
    }
}


void solve_periodic_tridiagonal_system(
        int n, 
        double* solution, // length n
        double* d,   // right-hand-side (length n), after function finishes, solution is stored in d
        double* a,   // sub-diagonal (length n)
        double* b,   // main-diagonal (length n)
        double* c,   // super-diagonal (length n)
        double* temp_storage)  // (length 3*n)
{
    // verified solution 
    double b1 = b[0];
    double cn = c[n-1];
    double ratio = a[0]/b1;

    // modify the coefficients to use the Sherman-Morrison formula
    b[0]   += b1;
    b[n-1] += ratio*cn;

    copy(a, a+n, temp_storage+0*n);
    copy(b, b+n, temp_storage+1*n);
    copy(c, c+n, temp_storage+2*n);

    // get ready to solve in vector solution
    solution[0] = -b1;
    for(int i=1; i<n-1; ++i) solution[i] = 0.;
    solution[n-1] = cn;

    // contains vector u before solving and q after solving
    double* q = solution;
    solve_tridiagonal_system(n, q, a+1, b, c);

    // restore coefficients
    copy(temp_storage+0*n, temp_storage+1*n, a);
    copy(temp_storage+1*n, temp_storage+2*n, b);
    copy(temp_storage+2*n, temp_storage+3*n, c);

    // contains vector d before solving and y after solving
    double* y = d;
    solve_tridiagonal_system(n, d, a+1, b, c);

    double q_prefactor = (y[0]-y[n-1]*ratio)/(1. + q[0] - q[n-1]*ratio);

    for(int i=0; i<n; ++i) solution[i] = y[i] - q_prefactor*q[i];
}


void evaluate_1d_spline(
        double* y, // length nx
        int n_knot, 
        const double* coeff,  // length 4*n_knot
        int nx,
        const double* x)  // length nx
{
    // verified
    for(int ix=0; ix<nx; ++ix) {
        // if(x[ix]<0 || x[ix]>=nx) throw "impossible";
        int x_bin = int(x[ix]);
        double f = x[ix] - x_bin;
        const double* c = coeff + 4*x_bin;
        y[ix] = c[0] + c[1]*f + c[2]*(f*f) + (c[3]*f)*(f*f);
    }
}

void evaluate_1d_spline_first_derivative(
        double* dy, // length nx
        int n_knot, 
        const double* coeff,  // length 4*n_knot
        int nx,
        const double* x)  // length nx
{
    // verified
    for(int ix=0; ix<nx; ++ix) {
        // if(x[ix]<0 || x[ix]>=nx) throw "impossible";
        int x_bin = int(x[ix]);
        double f = x[ix] - x_bin;
        const double* c = coeff + 4*x_bin;
        dy[ix] = c[1] + 2.*c[2]*f + 3.*c[3]*(f*f);
    }
}

// coefficients are valid only the range (0,1) reflecting the distance past the knot on the right
constexpr const double bspline_coeffs[4][4] = {
    { 0./6., 0./6., 0./6., 1./6.},
    { 1./6., 3./6., 3./6.,-3./6.},
    { 4./6., 0./6.,-6./6., 3./6.},
    { 1./6.,-3./6., 3./6.,-1./6.}};


void solve_periodic_1d_spline(
        int n, 
        double* coefficients, // length 4*n
        const double* data,   // length n
        double* temp_storage) // length 8*n
{
    double* a = temp_storage+0*n;
    double* b = temp_storage+1*n;
    double* c = temp_storage+2*n;
    double* d = temp_storage+3*n;
    double* solution = temp_storage + 4*n;
    double* temp_for_later = temp_storage+5*n;

    // b-spline coefficients for the natural cubic on integer grid
    for(int i=0; i<n; ++i) a[i] = 1./6.;
    for(int i=0; i<n; ++i) b[i] = 2./3.;
    for(int i=0; i<n; ++i) c[i] = 1./6.;
    for(int i=0; i<n; ++i) d[i] = data[i];

    solve_periodic_tridiagonal_system(n, solution, d, a, b, c, temp_for_later);

    // now we need to convert the b-spline coefficients to coefficients for {1,x,x^2,x^3}
    for(int i=0; i<4*n; ++i) coefficients[i] = 0.;

    for(int i=0; i<n; ++i) {
        double my_val = solution[i];
        for(int increment=0; increment<4; ++increment) {
            int idx = i+increment-2;
            if(idx< 0) idx += n; 
            if(idx>=n) idx -= n;

            for(int c=0; c<4; ++c)
                coefficients[idx*4+c] += my_val*bspline_coeffs[increment][c];
        }
    }
}

void solve_clamped_1d_spline_for_bsplines(
        int n_coeff, 
        double* coefficients, // length n_coeff
        const double* data,   // length n_coeff-2
        double* temp_storage) // length 3*n_coeff
{
    int n = n_coeff-2;
    double* a = temp_storage+0*n_coeff;
    double* b = temp_storage+1*n_coeff;
    double* c = temp_storage+2*n_coeff;

    // b-spline coefficients for the natural cubic on integer grid
    for(int i=0; i<n; ++i) a[i] = 1./6.;
    for(int i=0; i<n; ++i) b[i] = 2./3.;
    for(int i=0; i<n; ++i) c[i] = 1./6.;
    for(int i=0; i<n; ++i) coefficients[i+1] = data[i];

    // this is a homogenous boundary condition and quite convenient for cartesian potentials
    // for a zero-clamped spline, we must have coeff[-1] == coeff[1] and coeff[N-2] == coeff[N]
    // this conditions break the tridiagonal condition if added naively.
    // Instead, I can fold the condition into the matrix by doubling the (0,1) and (N-2,N-1)
    // elements of the tridiagonal matrix.  These would be elements c[0] and a[n-1].
    // Then, I need to remember to add the special basis functions that I absorbed
    // when computing the coefficients below.

    a[n-1] *= 2.;
    c[0]   *= 2.;
    solve_tridiagonal_system(n_coeff-2, coefficients+1, a+1, b, c);

    coefficients[0] = coefficients[2];
    coefficients[n_coeff-1] = coefficients[n_coeff-3];
}


void solve_clamped_1d_spline(
        int n, 
        double* coefficients, // length 4*(n-1)
        const double* data,   // length n
        double* temp_storage) // length 4*n
{
    double* a = temp_storage+0*n;
    double* b = temp_storage+1*n;
    double* c = temp_storage+2*n;
    double* solution = temp_storage + 3*n;
    // double* d = temp_storage+3*n;  // unneed since solution overwrites data
    // double* temp_for_later = temp_storage+5*n;

    // b-spline coefficients for the natural cubic on integer grid
    for(int i=0; i<n; ++i) a[i] = 1./6.;
    for(int i=0; i<n; ++i) b[i] = 2./3.;
    for(int i=0; i<n; ++i) c[i] = 1./6.;
    for(int i=0; i<n; ++i) solution[i] = data[i];

    // FIXME also implement zero-derivative clamped splines
    // this is a homogenous boundary condition and quite convenient for cartesian potentials
    // for a zero-clamped spline, we must have coeff[-1] == coeff[1] and coeff[N-2] == coeff[N]
    // this conditions break the tridiagonal condition if added naively.
    // Instead, I can fold the condition into the matrix by doubling the (0,1) and (N-2,N-1)
    // elements of the tridiagonal matrix.  These would be elements c[0] and a[n-1].
    // Then, I need to remember to add the special basis functions that I absorbed
    // when computing the coefficients below.

    a[n-1] *= 2.;
    c[0]   *= 2.;
    solve_tridiagonal_system(n, solution, a+1, b, c);

    // now we need to convert the b-spline coefficients to coefficients for {1,x,x^2,x^3}
    for(int i=0; i<4*(n-1); ++i) coefficients[i] = 0.;

    // There is one fewer interval for clamped splines since they have nothing to the right
    //    of their last point;
    for(int i=0; i<n; ++i) {
        double my_val = solution[i];
        for(int increment=0; increment<4; ++increment) {
            int idx = i+increment-2;
            if(idx< 0)   continue;
            if(idx>=n-1) continue;

            for(int c=0; c<4; ++c)
                coefficients[idx*4+c] += my_val*bspline_coeffs[increment][c];
        }
    }
    
    {
        // left wing B-spline
        double my_val = solution[1];
        int increment = 3;
        int idx = 0;

        for(int c=0; c<4; ++c)
            coefficients[idx*4+c] += my_val*bspline_coeffs[increment][c];
    }
    {
        // right wing B-spline
        double my_val = solution[n-2];
        int increment = 0;
        int idx = n-2;

        for(int c=0; c<4; ++c)
            coefficients[idx*4+c] += my_val*bspline_coeffs[increment][c];
    }
}


void solve_periodic_2d_spline(
        int nx, int ny,
        double* coefficients, // length (nx,ny,4,4) row-major
        const double* data,  // length (nx,ny), row-major
        double* temp_storage) // length (nx+8)*(ny+8)*4
{
    // verified
    int sum_dim = nx+ny;
    double* splines_1d         = temp_storage;                   // size nx*ny*4
    double* splines_1d_scratch = splines_1d         + nx*ny*4;   // size sum_dim*8 for safety
    double* values_temp        = splines_1d_scratch + sum_dim*8; // size sum_dim*4
    double* coeffs_temp        = values_temp        + sum_dim*4; // size sum_dim*16


    // compute splines across x-axis
    for(int ix=0; ix<nx; ++ix) 
        solve_periodic_1d_spline(ny, splines_1d+ix*ny*4, data+ix*ny, splines_1d_scratch);

    // now transpose to compute splines across the coefficients of the previous step
    // the data probably fits in L2 or L3, so I won't worry about an efficient transpose
    for(int iy=0; iy<ny; ++iy) {
        for(int power_y=0; power_y<4; ++power_y) {
            for(int ix=0; ix<nx; ++ix) values_temp[ix] = splines_1d[ix*ny*4 + iy*4 + power_y];
            solve_periodic_1d_spline(nx, coeffs_temp, values_temp, splines_1d_scratch);

            for(int ix=0; ix<nx; ++ix) 
                for(int power_x=0; power_x<4; ++power_x)
                    coefficients[ix*ny*16 + iy*16 + power_x*4 + power_y] = coeffs_temp[ix*4+power_x];
        }
    }
}




void evaluate_2d_spline(
        double* z, // length nx
        int n_knot_x, int n_knot_y, 
        const double* coeff,  // length 4*n_knot
        int n_point,
        const double* points)  // length nx*2
{
    // verified
    for(int ip=0; ip<n_point; ++ip) {
        float x = points[ip*2+0];
        float y = points[ip*2+1];

        // if(x<0. || x>=n_knot_x || y<0. || y>=n_knot_y) throw "impossible";
        int x_bin = int(x);
        int y_bin = int(y);

        double fx = x - x_bin;
        double fy = y - y_bin;

        const double* c = coeff + x_bin*n_knot_y*16 + y_bin*16;
        double fx2 = fx*fx; 
        double fx3 = fx2*fx;

        z[ip] = 1.f * (c[ 0] + fy*(c[ 1] + fy*(c[ 2] + fy*c[ 3]))) +
                fx  * (c[ 4] + fy*(c[ 5] + fy*(c[ 6] + fy*c[ 7]))) +
                fx2 * (c[ 8] + fy*(c[ 9] + fy*(c[10] + fy*c[11]))) +
                fx3 * (c[12] + fy*(c[13] + fy*(c[14] + fy*c[15])));
    }
}
