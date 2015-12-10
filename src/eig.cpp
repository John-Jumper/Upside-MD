// Author: John Jumper
//
// RMSD alignment algorithm based on Coutsias, Seok, and Dill, Journal of
// Computational Chemistry, Vol. 25, No. 15
//
// Linear algebra based on algorithms in Golub and van Loan, Matrix
// Computations, chapters 5 and 8

#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"

using namespace h5;
using namespace std;

typedef Float4 S;

namespace {

// S house_accurate(
//         int n,
//         S* x)  // size (n,), replaced by v, v[0] = |x|
// {
//     S beta,mu,s;
//     S sigma = zero<S>();
// 
//     for(int i=1; i<n; ++i) sigma += x[i]*x[i];
//     
//     mu = sqrtf(x[0]*x[0] + sigma);   // |x|
//     if(sigma == zero<S>()) {  // FIXME more robust condition needed here
//         beta = (x[0]>=zero<S>()) ? zero<S>() : two<S>();
//         s = one<S>();
//     } else {
//         s = (x[0]>zero<S>()) ? -sigma/(x[0]+mu) : (x[0]-mu);
//         beta = two<S>()*s*s / (sigma+s*s);
//     }
// 
//     x[0] = mu;
//     for(int i=1; i<n; ++i) x[i] *= 1./s;
//     return beta;
// }
// 
// void givens_accurate(S *c, S *s, S a, S b)
// {
//     if(b==zero<S>()) {
//         *c=one<S>(); *s=zero<S>();
//     } else {
//         if(fabsf(b) > fabsf(a)) {
//             S tau = -a/b; *s = one<S>()/sqrtf(one<S>()+tau*tau); *c = *s * tau;
//         } else {
//             S tau = -b/a; *c = one<S>()/sqrtf(one<S>()+tau*tau); *s = *c * tau;
//         }
//     }
// }

S house(
        int n,
        S* x)  // size (n,), replaced by v, v[0] = |x|
{
    S beta,mu,s;
    S sigma2 = 1e-20f;

    for(int i=1; i<n; ++i) sigma2 += x[i]*x[i];
    
    mu = sqrtf(x[0]*x[0] + sigma2);   // |x|
    s = ternary(zero<S>()<x[0], -sigma2*rcp(x[0]+mu), x[0]-mu);
    beta = two<S>()*s*s * rcp(sigma2+s*s);

    x[0] = mu;
    for(int i=1; i<n; ++i) x[i] *= rcp(s);

    return beta;
}

void givens(S *c, S *s, S a, S b)
{
    S inv_r = rsqrt(a*a+b*b);

    auto trivial = b==zero<S>();
    *c = ternary(trivial, one <S>(),  a*inv_r);
    *s = ternary(trivial, zero<S>(), -b*inv_r);
}



int r(int i, int j) {
    int ii = (i<j) ? i : j;
    int jj = (i<j) ? j : i;

    switch(ii) {
        case 0:
            return (0-0) + jj;
        case 1:
            return (4-1) + jj;
        case 2:
            return (7-2) + jj;
        case 3:
            return (9-3) + jj;
        default:
            return 1000; // error
    }
}


void
symmetric_tridiagonalize_4x4(
        S* restrict beta, // length 2 beta sequence of Householder sequence
        S* restrict A)   // upper triangle of 4x4 matrix, 10 elements
{
    S p[3];
    S w[3];

    for(int k=0; k<4-2; ++k) {
        int m = 4-(k+1);
        beta[k] = house(m, A+r(k,k+1));   // overwrite with householder

        #define v(j) ((j)==0 ? one<S>() : A[r(k,k+1+(j))])
        for(int i=0; i<m; ++i) {
            p[i] = zero<S>();
            for(int j=0; j<m; ++j)
                p[i] += A[r(i+k+1,j+k+1)] * v(j);
            p[i] *= beta[k];
        }

        S p_dot_v = zero<S>();
        for(int i=0; i<m; ++i) p_dot_v += p[i]*v(i);
        for(int i=0; i<m; ++i) w[i] = p[i] - (half<S>()*beta[k]*p_dot_v) * v(i);

        for(int i=0; i<m; ++i)
            for(int j=i; j<m; ++j)
                A[r(i+k+1,j+k+1)] -= v(i)*w[j] + v(j)*w[i];
        #undef v
    }
}


void
unpack_tridiagonalize_4x4(
        S * restrict d,  // length 4
        S * restrict u,  // length 3
        S * restrict rot_, // size 4x4 = 16, nonsymmetric
        S * restrict beta, // length 2
        S * restrict A)    // symmetric_4x4, length 10
{
    #define rot(i,j) (rot_[(i)*4+(j)])
    for(int i=0; i<4; ++i) d[i] = A[r(i,i)];
    for(int i=0; i<3; ++i) u[i] = A[r(i,i+1)];

    // first update with the symmetric part
    for(int i=0; i<4; ++i) for(int j=i; j<4; ++j) rot(i,j) = (i==j);   // identity matrix to start

    // unpack Householder reflection vectors
    #define v0(j) ((j)==0 ? one<S>() : A[r(0,1+(j))])
    #define v1(j) ((j)==0 ? one<S>() : A[r(1,2+(j))])

    // first Householder reflection operates on lower left 3x3 block
    for(int i=1; i<4; ++i)
        for(int j=i; j<4; ++j)
            rot(i,j) -= beta[0] * (v0(i-1)*v0(j-1));

    // second Householder reflection operates on the lowest 2x2 block
    for(int i=2; i<4; ++i)
        for(int j=i; j<4; ++j)
            rot(i,j) -= beta[1] * (v1(i-2)*v1(j-2));

    // fill in using symmetry
    for(int i=0; i<4; ++i) for(int j=i+1; j<4; ++j) rot(j,i) = rot(i,j);

    // now fill in non-symmetric Householder interaction (note leading zero for v1)
    // fills in lower left 2x3 block
    S coeff = beta[0]*beta[1] * (v0(1)*v1(0) + v0(2)*v1(1));
    for(int i=2; i<4; ++i)
        for(int j=1; j<4; ++j)
            rot(i,j) += coeff * v1(i-2)*v0(j-1);
    #undef rot
    #undef v0
    #undef v1
}


#define rot(i,j) (rot_[(i)*4 + (j)])

void
implicit_symm_QR_step_4x4(
        int n,   // length of diagonal
        S * restrict d,    // diagonal
        S * restrict u,   // upper superdiagonal
        S * restrict rot_) // rotation to accumulate Givens rotations
                     // must be at least nx4-sized
{
    S dval = half<S>()*(d[n-2] - d[n-1]) + S(1e-20f);  // paranoia against NaN
    S mu = d[n-1] - u[n-2]*u[n-2]*rcp(dval+copysignf(sqrtf(dval*dval + u[n-2]*u[n-2]),dval));

    S x = d[0] - mu;
    S z = u[0];
    
    for(int k=0; k<n-1; ++k){
        S c,s;
        givens(&c,&s,x,z);

        /* accumulate Givens rotation */
        for(int j=0; j<4; ++j) {
            S t1 = rot(k  ,j);
            S t2 = rot(k+1,j);
            rot(k  ,j) = c*t1 - s*t2;
            rot(k+1,j) = s*t1 + c*t2;
        }

        /* compute the T(k-1,:) row */
        if(k > 0) u[k-1] = c*x - s*z;

        /* mix the T(k,k),T(k,k+1),T(k+1,k+1) block */
        S T00 = d[k  ];
        S T11 = d[k+1];
        S T01 = u[k  ];
        
        d[k  ]  = T00*c*c - T01*two<S>()*c*s + T11*s*s;
        d[k+1]  = T00*s*s + T01*two<S>()*c*s + T11*c*c;
        u[k  ]  = (T00-T11)*c*s + T01*(c*c-s*s);
        x = u[k];

        /* compute the T(k+2,:) row */
        if(k<n-2) {
            z = -u[k+1]*s;
            u[k+1] *= c;
        }
    } 
}
#undef rot


int  __attribute__ ((noinline))
symm_QR_4x4(
        S* restrict d, // will contain eigenvalues, length 4
        S* restrict rot, // will contain eigenvectors in rows, size 4x4
        S* restrict A, // symmetric 4x4 matrix, 10 elements, overwritten
        S tol,  // relative tolerance for off-diagonal entries (suggest something like 1e-5)
        int max_iter) 
{
    const int n = 4;

    S beta[2];
    symmetric_tridiagonalize_4x4(beta, A);

    S u[3];
    unpack_tridiagonalize_4x4(d, u, rot, beta, A);

    for(int k=0; k<max_iter; ++k) {
        // exactly zero off-diagonal elements that are nearly zero
        for(int i=0; i<n-1; ++i) {
            auto small_u = fabsf(u[i]) <= tol * (fabsf(d[i]) + fabsf(d[i+1]));
            u[i] = ternary(small_u, zero<S>(), u[i]);
        }

        // find largest diagonal lower left subblock
        int q;
        if      (any(u[2] != zero<S>())) q = 0;
        else if (any(u[1] != zero<S>())) q = 1;
        else if (any(u[0] != zero<S>())) q = 2;
        else return k;   // q=3 and the matrix is diagonal and we are done

        // p is start of largest unreduced triangular submatrix adjacent to the q region
        // unreduced means no entries in the upper triangle are zero
        int p;
        for(p=n-q-1; p>0; --p) {
            if(none(u[p-1] != zero<S>())) break;
        }

        // perform QR iteration and accumulate into rot
        implicit_symm_QR_step_4x4(n-q-p, d+p, u+p, rot+4*p);
    }
    return -1;  // non-convergence
}
}


struct AffineAlignment : public CoordNode
{
    struct Params {
        alignas(16) int32_t atom_offsets[3][4];
        alignas(16) float   ref_geom[3][3][4];
    };

    int n_group;
    CoordNode& pos;
    vector<Params> params;
    unique_ptr<Float4[]> evals_storage;
    unique_ptr<Float4[]> evecs_storage;

    AffineAlignment(hid_t grp, CoordNode& pos_):
        CoordNode(get_dset_size(2, grp, "atoms")[0], 7),
        n_group(round_up(n_elem,4)/4),
        
        pos(pos_), params(n_group),
        evals_storage(new_aligned<Float4>(n_group* 4)),
        evecs_storage(new_aligned<Float4>(n_group*16))
    {
        check_size(grp, "atoms",    n_elem, 3);
        check_size(grp, "ref_geom", n_elem, 3,3);  // (residue, atom, xyz)

        traverse_dset<2,int  >(grp,"atoms",   [&](size_t i,size_t j,          int   x){
                params[i/4].atom_offsets[j][i%4] = x*pos.output.row_width;});
        traverse_dset<3,float>(grp,"ref_geom",[&](size_t i,size_t na,size_t d,float x){
                params[i/4].ref_geom[na][d][i%4]=x;});

        // handle non-multiple of 4's by padding (needs testing)
        for(int i=n_elem; i<round_up(n_elem,4); ++i) {
            for(int j: range(3))
                params[i/4].atom_offsets[j][i%4] = params[i/4].atom_offsets[j][0];

            for(int na: range(3))
                for(int d: range(3))
                    params[i/4].ref_geom[na][d][i%4] = params[i/4].ref_geom[na][d][0];
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("affine_alignment"));

        VecArray rigid_body = output;
        float* posc = pos.output.x.get();

        for(int ng=0; ng<n_group; ++ng) {
            const auto& p = params[ng];

            auto atom1 = aligned_gather_vec<3>(posc, Int4(p.atom_offsets[0]));
            auto atom2 = aligned_gather_vec<3>(posc, Int4(p.atom_offsets[1]));
            auto atom3 = aligned_gather_vec<3>(posc, Int4(p.atom_offsets[2]));

            auto center = S(1.f/3.f)*(atom1+atom2+atom3);
            atom1 -= center;
            atom2 -= center;
            atom3 -= center;

            auto ref_geom1 = make_vec3(Float4(p.ref_geom[0][0]), Float4(p.ref_geom[0][1]), Float4(p.ref_geom[0][2]));
            auto ref_geom2 = make_vec3(Float4(p.ref_geom[1][0]), Float4(p.ref_geom[1][1]), Float4(p.ref_geom[1][2]));
            auto ref_geom3 = make_vec3(Float4(p.ref_geom[2][0]), Float4(p.ref_geom[2][1]), Float4(p.ref_geom[2][2]));

            S R_[3][3];
            #define R(i,j) (R_[i][j])
            for(int i=0; i<3; ++i)
                for(int j=0; j<3; ++j)
                    R(i,j) = atom1[j] * ref_geom1[i]
                           + atom2[j] * ref_geom2[i]
                           + atom3[j] * ref_geom3[i];

            S F[10] = {R(0,0)+R(1,1)+R(2,2), R(1,2)-R(2,1),         R(2,0)-R(0,2),         R(0,1)-R(1,0),
                                             R(0,0)-R(1,1)-R(2,2),  R(0,1)+R(1,0),         R(0,2)+R(2,0),
                                                                   -R(0,0)+R(1,1)-R(2,2),  R(1,2)+R(2,1),
                                                                                          -R(0,0)-R(1,1)+R(2,2)};
            #undef R

            // S evals[4], evecs[16];
            Float4* restrict evals = evals_storage.get() + ng* 4;
            Float4* restrict evecs = evecs_storage.get() + ng*16;

            symm_QR_4x4(evals, evecs, F, 1e-5f, 100);

            // swap largest eigenvalue into location 0
            for(int i=1; i<4; ++i) {
                auto do_flip = evals[0] < evals[i];

                auto eval0 = ternary(do_flip, evals[i], evals[0]);
                auto evali = ternary(do_flip, evals[0], evals[i]);
                evals[0] = eval0;
                evals[i] = evali;

                for(int d=0; d<4; ++d) {
                    auto evec0 = ternary(do_flip, evecs[i*4+d], evecs[0*4+d]);
                    auto eveci = ternary(do_flip, evecs[0*4+d], evecs[i*4+d]);
                    evecs[0*4+d] = evec0;
                    evecs[i*4+d] = eveci;
                }
            }

            Vec<8,S> body; // really 7 components but I need the eight for the transpose
            for(int j=0; j<3; ++j) body[j] = center[j];
            for(int j=0; j<4; ++j) body[3+j] = evecs[0*4+j];

            transpose4(body[0],body[1],body[2],body[3]);
            for(int i=0; i<4; ++i) body[i].store(&rigid_body(0,4*ng+i));

            transpose4(body[4],body[5],body[6],body[7]);
            for(int i=0; i<4; ++i) body[4+i].store(&rigid_body(4,4*ng+i));
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("affine_alignment_deriv"));
        float* pos_sens = pos.sens.x.get();

        for(int ng=0; ng<n_group; ++ng) {
            const auto& p = params[ng];

            const Float4* restrict evals = evals_storage.get() + ng* 4;
            const Float4* restrict evecs = evecs_storage.get() + ng*16;

            // compute inverse eigenvalue differences
            S inv_evals[4];
            for(int j=1; j<4; ++j) inv_evals[j] = rcp(evals[0]-evals[j]);

            // push back torque to affine derivatives (multiply by quaternion)
            // the torque is in the tangent space of the rotated frame
            // to act on a tangent in the affine space, I need to push that tangent into the rotated space
            // this means a right multiply by the quaternion itself

            // evecs[0] is the rotation quaternion
            S sens3  [3] = {Float4(&sens(0,4*ng+0)), Float4(&sens(0,4*ng+1)),Float4(&sens(0,4*ng+2))};
            S torque [3] = {Float4(&sens(0,4*ng+3)), Float4(&sens(4,4*ng+0)),Float4(&sens(4,4*ng+1))};
            S padding[2] = {Float4(&sens(4,4*ng+2)), Float4(&sens(4,4*ng+3))};

            transpose4(sens3 [0],sens3 [1],sens3  [2],torque [0]);
            transpose4(torque[1],torque[2],padding[0],padding[1]);

            S quat_sens[4] = {
                two<S>()*(-torque[0]*evecs[1] - torque[1]*evecs[2] - torque[2]*evecs[3]),
                two<S>()*( torque[0]*evecs[0] + torque[1]*evecs[3] - torque[2]*evecs[2]),
                two<S>()*( torque[1]*evecs[0] + torque[2]*evecs[1] - torque[0]*evecs[3]),
                two<S>()*( torque[2]*evecs[0] + torque[0]*evecs[2] - torque[1]*evecs[1])};

            S quat_sens_diag_basis[4] = {zero<S>(),zero<S>(),zero<S>(),zero<S>()};
            for(int d=0; d<4; ++d) for(int i=0; i<4; ++i) quat_sens_diag_basis[d] += quat_sens[i]*evecs[d*4+i];

            // The derivative of an eigenvector is given by perturbation theory as 
            // a linear combination of the other eigenvectors weighted by interaction
            // matrix elements and the difference of eigenvalues

            // I will assume that the largest eigenvalue is nondegenerate, as would be expected
            // for a reasonable alignment to a nearly rigid structure

            // unsafe macro for perturbation term
            #define t(i,j) (f[r(i,j)] * ((i==j) \
                                            ? evecs[k*4+i]*evecs[0*4+j] \
                                            : evecs[k*4+i]*evecs[0*4+j]+evecs[k*4+j]*evecs[0*4+i]))
            #define perturb(ret, f00,f01,f02,f03,f11,f12,f13,f22,f23,f33) do { \
                const S f[10] = {f00,f01,f02,f03,f11,f12,f13,f22,f23,f33}; \
                for(int k=1; k<4; ++k) { \
                    S c = inv_evals[k]*(t(0,0)+t(0,1)+t(0,2)+t(0,3) \
                                              +t(1,1)+t(1,2)+t(1,3) \
                                                     +t(2,2)+t(2,3) \
                                                            +t(3,3)); \
                    (ret) += c*quat_sens_diag_basis[k]; \
                } \
            } while(0)


            for(int na=0; na<3; ++na) {
                Vec<3,S> deriv = S(1.f/3.f)*make_vec3(sens3[0],sens3[1],sens3[2]);
                auto g = make_vec3(Float4(p.ref_geom[na][0]), Float4(p.ref_geom[na][1]), Float4(p.ref_geom[na][2]));

                // quaternion rotation derivative
                perturb(deriv[0], g[0],  zero<S>(), g[2],-g[1], 
                                         g[0],      g[1], g[2], 
                                                   -g[0],  zero<S>(), 
                                                          -g[0]);

                perturb(deriv[1], g[1],-g[2],  zero<S>(), g[0], 
                                       -g[1],  g[0],      zero<S>(), 
                                               g[1],      g[2], 
                                                         -g[1]);

                perturb(deriv[2], g[2], g[1],  -g[0],      zero<S>(), 
                                       -g[2],   zero<S>(), g[0], 
                                               -g[2],      g[1], 
                                                           g[2]);

                aligned_scatter_update_vec_destructive(pos_sens, Int4(p.atom_offsets[na]), deriv);
            }
        }
    }
};

static RegisterNodeType<AffineAlignment,1>_8("affine_alignment");
