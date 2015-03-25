// Author: John Jumper
//
// RMSD alignment algorithm based on Coutsias, Seok, and Dill, Journal of
// Computational Chemistry, Vol. 25, No. 15
//
// Linear algebra based on algorithms in Golub and van Loan, Matrix
// Computations, chapters 5 and 8

#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"
#include "affine.h"

using namespace h5;
using namespace std;

#if 0
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#endif

struct AffineAlignmentParams {
    CoordPair atom[3];
    float     ref_geom[9];
} ;

namespace {
float house(
        int n,
        float* x)  // size (n,), replaced by v, v[0] = |x|
{
    float beta,mu,s;
    float sigma = 0.f;

    for(int i=1; i<n; ++i) sigma += x[i]*x[i];
    
    mu = sqrtf(x[0]*x[0] + sigma);   // |x|
    if(sigma == 0.f) {  // FIXME more robust condition needed here
        beta = (x[0]>=0.f) ? 0.f : 2.f;
        s = 1.f;
    } else {
        s = (x[0]>0.f) ? -sigma/(x[0]+mu) : (x[0]-mu);
        beta = 2.f*s*s / (sigma+s*s);
    }

    x[0] = mu;
    for(int i=1; i<n; ++i) x[i] *= 1./s;
    return beta;
}

void givens_accurate(float *c, float *s, float a, float b)
{
    if(b==0.f) {
        *c=1.f; *s=0.f;
    } else {
        if(fabsf(b) > fabsf(a)) {
            float tau = -a/b; *s = 1.f/sqrtf(1.f+tau*tau); *c = *s * tau;
        } else {
            float tau = -b/a; *c = 1.f/sqrtf(1.f+tau*tau); *s = *c * tau;
        }
    }
}

void givens(float *c, float *s, float a, float b)
{
    if(b==0.f) {
        *c=1.f; *s=0.f;
    } else {
        float inv_r = rsqrt(a*a+b*b);
        *c =  a*inv_r;
        *s = -b*inv_r;
    }
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
        float* restrict beta, // length 2 beta sequence of Householder sequence
        float* restrict A)   // upper triangle of 4x4 matrix, 10 elements
{
    float p[3];
    float w[3];

    for(int k=0; k<4-2; ++k) {
        int m = 4-(k+1);
        beta[k] = house(m, A+r(k,k+1));   // overwrite with householder

        #define v(j) ((j)==0 ? 1.f : A[r(k,k+1+(j))])
        for(int i=0; i<m; ++i) {
            p[i] = 0.f;
            for(int j=0; j<m; ++j)
                p[i] += A[r(i+k+1,j+k+1)] * v(j);
            p[i] *= beta[k];
        }

        float p_dot_v = 0.f;
        for(int i=0; i<m; ++i) p_dot_v += p[i]*v(i);
        for(int i=0; i<m; ++i) w[i] = p[i] - (0.5f*beta[k]*p_dot_v) * v(i);

        for(int i=0; i<m; ++i)
            for(int j=i; j<m; ++j)
                A[r(i+k+1,j+k+1)] -= v(i)*w[j] + v(j)*w[i];
        #undef v
    }
}


void
unpack_tridiagonalize_4x4(
        float * restrict d,  // length 4
        float * restrict u,  // length 3
        float * restrict rot_, // size 4x4 = 16, nonsymmetric
        float * restrict beta, // length 2
        float * restrict A)    // symmetric_4x4, length 10
{
    #define rot(i,j) (rot_[(i)*4+(j)])
    for(int i=0; i<4; ++i) d[i] = A[r(i,i)];
    for(int i=0; i<3; ++i) u[i] = A[r(i,i+1)];

    // first update with the symmetric part
    for(int i=0; i<4; ++i) for(int j=i; j<4; ++j) rot(i,j) = (i==j);   // identity matrix to start

    // unpack Householder reflection vectors
    #define v0(j) ((j)==0 ? 1.f : A[r(0,1+(j))])
    #define v1(j) ((j)==0 ? 1.f : A[r(1,2+(j))])

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
    float coeff = beta[0]*beta[1] * (v0(1)*v1(0) + v0(2)*v1(1));
    for(int i=2; i<4; ++i)
        for(int j=1; j<4; ++j)
            rot(i,j) += coeff * v1(i-2)*v0(j-1);
    #undef rot
    #undef v0
    #undef v1
}


#define rot(i,j) (rot_[(i)*4 + (j)])

// general implementation
void
implicit_symm_QR_step_4x4(
        int n,   // length of diagonal
        float * restrict d,    // diagonal
        float * restrict u,   // upper superdiagonal
        float * restrict rot_) // rotation to accumulate Givens rotations
                     // must be at least nx4-sized
{
    float dval = 0.5*(d[n-2] - d[n-1]);
    float mu = d[n-1] - u[n-2]*u[n-2]/(dval+copysignf(sqrtf(dval*dval + u[n-2]*u[n-2]),dval));
    float x = d[0] - mu;
    float z = u[0];
    
    for(int k=0; k<n-1; ++k){
        float c,s;
        givens(&c,&s,x,z);

        /* accumulate Givens rotation */
        for(int j=0; j<4; ++j) {
            float t1 = rot(k  ,j);
            float t2 = rot(k+1,j);
            rot(k  ,j) = c*t1 - s*t2;
            rot(k+1,j) = s*t1 + c*t2;
        }

        /* compute the T(k-1,:) row */
        if(k > 0) u[k-1] = c*x - s*z;

        /* mix the T(k,k),T(k,k+1),T(k+1,k+1) block */
        float T00 = d[k  ];
        float T11 = d[k+1];
        float T01 = u[k  ];
        
        d[k  ]  = T00*c*c - T01*2.f*c*s + T11*s*s;
        d[k+1]  = T00*s*s + T01*2.f*c*s + T11*c*c;
        u[k  ]  = (T00-T11)*c*s + T01*(c*c-s*s);
        x = u[k];

        /* compute the T(k+2,:) row */
        if(k<n-2) {
            z = -u[k+1]*s;
            u[k+1] *= c;
        }
    } 
}

// // specializations for the length of the array
// void implicit_symm_QR_step_4x4_n4(float *d, float *u, float *rot_) {implicit_symm_QR_step_4x4(4,d,u,rot);}
// void implicit_symm_QR_step_4x4_n3(float *d, float *u, float *rot_) {implicit_symm_QR_step_4x4(3,d,u,rot);}
// void implicit_symm_QR_step_4x4_n2(float *d, float *u, float *rot_) {implicit_symm_QR_step_4x4(2,d,u,rot);}
#undef rot



int  __attribute__ ((noinline))
symm_QR_4x4(
        float* restrict d, // will contain eigenvalues, length 4
        float* restrict rot, // will contain eigenvectors in rows, size 4x4
        float* restrict A, // symmetric 4x4 matrix, 10 elements, overwritten
        float tol,  // relative tolerance for off-diagonal entries (suggest something like 1e-5)
        int max_iter) 
{
#if 1
    const int n = 4;

    float beta[2];
    symmetric_tridiagonalize_4x4(beta, A);

    float u[3];
    unpack_tridiagonalize_4x4(d, u, rot, beta, A);

    for(int k=0; k<max_iter; ++k) {
        // exactly zero off-diagonal elements that are nearly zero
        for(int i=0; i<n-1; ++i) {
            if(fabsf(u[i]) <= tol * (fabsf(d[i]) + fabsf(d[i+1]))) u[i] = 0.f;
        }

        // find largest diagonal lower left subblock
        int q;
        if      (u[2] != 0.f) q = 0;
        else if (u[1] != 0.f) q = 1;
        else if (u[0] != 0.f) q = 2;
        else return k;   // q=3 and the matrix is diagonal and we are done

        // p is start of largest unreduced triangular submatrix adjacent to the q region
        // unreduced means no entries in the upper triangle are zero
        int p;
        for(p=n-q-1; p>0; --p) {
            if(u[p-1] == 0.f) break;
        }

        // perform QR iteration and accumulate into rot
        // if     (n-q-p == 4) implicit_symm_QR_step_4x4_n4(d+p, u+p, rot+4*p);
        // else if(n-q-p == 3) implicit_symm_QR_step_4x4_n3(d+p, u+p, rot+4*p);
        // else if(n-q-p == 2) implicit_symm_QR_step_4x4_n2(d+p, u+p, rot+4*p);
        implicit_symm_QR_step_4x4(n-q-p, d+p, u+p, rot+4*p);
    }
    return -1;  // non-convergence
#else 
    using namespace Eigen;
    Matrix4f Amat; // only the lower triangle will be used
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) Amat(i,j) = A[r(i,j)];
    SelfAdjointEigenSolver<Matrix4f> solver(Amat);
    auto &eigvals = solver.eigenvalues();
    auto &eigvecs = solver.eigenvectors();
    for(int i=0; i<4; ++i) {
        d[i] = eigvals[i];
        for(int j=0; j<4; ++j) rot[i*4+j] = eigvecs(j,i);
    }
    return 1;
#endif
}
}


void 
three_atom_alignment(
        float * restrict rigid_body, //length 7 = 3 translation + 4 quaternion rotation
        float * restrict deriv,    // length 3x3x7
        const float * restrict atom1, const float*  restrict atom2, const float*  restrict atom3,  // each length 3
        const float*  restrict ref_geom)  // length 9
{
    for(int j=0; j<3; ++j) rigid_body[j] = (1.f/3.f)*(atom1[j]+atom2[j]+atom3[j]);
    
    float R_[9];
    #define R(i,j) (R_[(i)*3+(j)])
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            R(i,j) = (atom1[j]-rigid_body[j])*ref_geom[0+i] 
                   + (atom2[j]-rigid_body[j])*ref_geom[3+i] 
                   + (atom3[j]-rigid_body[j])*ref_geom[6+i];

    float F[10] = {R(0,0)+R(1,1)+R(2,2), R(1,2)-R(2,1),         R(2,0)-R(0,2),         R(0,1)-R(1,0),
                                         R(0,0)-R(1,1)-R(2,2),  R(0,1)+R(1,0),         R(0,2)+R(2,0),
                                                               -R(0,0)+R(1,1)-R(2,2),  R(1,2)+R(2,1),
                                                                                      -R(0,0)-R(1,1)+R(2,2)};
    #undef R

    float evals[4], evecs[16];

    symm_QR_4x4(evals, evecs, F, 1e-5, 100);

    int largest_loc=0;
    for(int i=1; i<4; ++i) if(evals[largest_loc] < evals[i]) largest_loc = i;

    // swap largest eval into location 0
    float tmp = evals[0];
    evals[0] = evals[largest_loc];
    evals[largest_loc] = tmp;
    
    for(int j=0; j<4; ++j) {
        tmp = evecs[0*4+j];
        evecs[0*4+j] = evecs[largest_loc*4+j];
        evecs[largest_loc*4+j] = tmp;
    }

    for(int j=0; j<4; ++j) rigid_body[3+j] = evecs[0*4+j];

    // overwrite evals with the inverse eigenvalue differences
    for(int j=1; j<4; ++j) evals[j] = 1.f / (evals[0]-evals[j]);

    // The derivative of an eigenvector is given by perturbation theory as 
    // a linear combination of the other eigenvectors weighted by interaction
    // matrix elements and the difference of eigenvalues

    // I will assume that the largest eigenvalue is nondegenerate, as would be expected
    // for a reasonable alignment to a nearly rigid structure

    // unsafe macro for perturbation term
    #define t(i,j) (f[r(i,j)] * ((i==j) ? evecs[k*4+i]*evecs[0*4+j] : evecs[k*4+i]*evecs[0*4+j]+evecs[k*4+j]*evecs[0*4+i]))
    #define perturb(ret, f00,f01,f02,f03,f11,f12,f13,f22,f23,f33) do { \
        const float f[10] = {f00,f01,f02,f03,f11,f12,f13,f22,f23,f33}; \
        (ret)[3] = (ret)[4] = (ret)[5] = (ret)[6] = 0.f; \
        for(int k=1; k<4; ++k) { \
            float c = evals[k]*(t(0,0)+t(0,1)+t(0,2)+t(0,3) \
                                      +t(1,1)+t(1,2)+t(1,3) \
                                             +t(2,2)+t(2,3) \
                                                    +t(3,3)); \
            for(int j=0; j<4; ++j) (ret)[3+j] += c*evecs[k*4+j]; \
        } \
    } while(0)

    #define g(j) ref_geom[3*i+(j)]
    for(int i=0; i<3; ++i) {
        // translation vector derivative is (1/N) * identity_matrix(3)
        deriv[(3*i+0)*7 + 0] = 1.f/3.f;  deriv[(3*i+0)*7 + 1] =     0.f;  deriv[(3*i+0)*7 + 2] =     0.f;
        deriv[(3*i+1)*7 + 0] =     0.f;  deriv[(3*i+1)*7 + 1] = 1.f/3.f;  deriv[(3*i+1)*7 + 2] =     0.f;
        deriv[(3*i+2)*7 + 0] =     0.f;  deriv[(3*i+2)*7 + 1] =     0.f;  deriv[(3*i+2)*7 + 2] = 1.f/3.f;

        // quaternion rotation derivative
        perturb(deriv+(3*i+0)*7, g(0),  0.f, g(2),-g(1), 
                                       g(0), g(1), g(2), 
                                            -g(0),  0.f, 
                                                  -g(0));

        perturb(deriv+(3*i+1)*7, g(1),-g(2),  0.f, g(0), 
                                      -g(1), g(0),  0.f, 
                                             g(1), g(2), 
                                                  -g(1));

        perturb(deriv+(3*i+2)*7, g(2), g(1),-g(0),  0.f, 
                                      -g(2),  0.f, g(0), 
                                            -g(2), g(1), 
                                                   g(2));
    }
}

namespace {

template <typename CoordT, typename MutableCoordT>
void affine_alignment_body(
        MutableCoordT &rigid_body,
        CoordT &x1,
        CoordT &x2,
        CoordT &x3,
        const AffineAlignmentParams &p)
{
    float my_deriv[3*3*7];

    three_atom_alignment(rigid_body.v,my_deriv, x1.v,x2.v,x3.v, p.ref_geom);

    for(int quat_dim=0; quat_dim<7; ++quat_dim) {
        for(int atom_dim=0; atom_dim<3; ++atom_dim) {
            x1.d[quat_dim][atom_dim] = my_deriv[0*21 + atom_dim*7 + quat_dim];
            x2.d[quat_dim][atom_dim] = my_deriv[1*21 + atom_dim*7 + quat_dim];
            x3.d[quat_dim][atom_dim] = my_deriv[2*21 + atom_dim*7 + quat_dim];
        }
    }
}
}

void affine_alignment(
        SysArray rigid_body,
        CoordArray pos,
        const AffineAlignmentParams* restrict params,
        int n_res,
        int n_system)
{
#pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nr=0; nr<n_res; ++nr) {
            MutableCoord<7> rigid_body_coord(rigid_body, ns, nr);

            Coord<3,7> x1(pos, ns, params[nr].atom[0]);
            Coord<3,7> x2(pos, ns, params[nr].atom[1]);
            Coord<3,7> x3(pos, ns, params[nr].atom[2]);

            affine_alignment_body(rigid_body_coord, x1,x2,x3, params[nr]);

            rigid_body_coord.flush();
            x1.flush();
            x2.flush();
            x3.flush();
        }
    }
}



void
affine_reverse_autodiff(
        const SysArray  affine,
        const SysArray  affine_accum,
        SysArray         pos_deriv,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_res, 
        int n_system)
{
#pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        std::vector<TempCoord<6>> torque_sens(n_res);

        for(int nt=0; nt<n_tape; ++nt) {
            auto tape_elem = tape[nt];
            for(int rec=0; rec<int(tape_elem.output_width); ++rec) {
                auto val = StaticCoord<6>(affine_accum, ns, tape_elem.loc + rec);
                for(int d=0; d<6; ++d)
                    torque_sens[tape_elem.atom].v[d] += val.v[d];
            }
        }

        for(int na=0; na<n_res; ++na) {
            float sens[7]; for(int d=0; d<3; ++d) sens[d] = torque_sens[na].v[d];

            float q[4]; for(int d=0; d<4; ++d) q[d] = affine[ns](d+3,na);

            // push back torque to affine derivatives (multiply by quaternion)
            // the torque is in the tangent space of the rotated frame
            // to act on a tangent in the affine space, I need to push that tangent into the rotated space
            // this means a right multiply by the quaternion itself

            float *torque = torque_sens[na].v+3;
            sens[3] = 2.f*(-torque[0]*q[1] - torque[1]*q[2] - torque[2]*q[3]);
            sens[4] = 2.f*( torque[0]*q[0] + torque[1]*q[3] - torque[2]*q[2]);
            sens[5] = 2.f*( torque[1]*q[0] + torque[2]*q[1] - torque[0]*q[3]);
            sens[6] = 2.f*( torque[2]*q[0] + torque[0]*q[2] - torque[1]*q[1]);

            for(int nsl=0; nsl<p[na].n_slots1; ++nsl) {
                for(int sens_dim=0; sens_dim<7; ++sens_dim) {
                    MutableCoord<3> c(pos_deriv, ns, p[na].slots1[nsl]+sens_dim);
                    for(int d=0; d<3; ++d) c.v[d] *= sens[sens_dim];
                    c.flush();
                }
            }
        }
    }
}


struct AffineAlignment : public CoordNode
{
    CoordNode& pos;
    int n_system;
    vector<AffineAlignmentParams> params;
    vector<AutoDiffParams> autodiff_params;

    AffineAlignment(hid_t grp, CoordNode& pos_):
        CoordNode(pos_.n_system, get_dset_size(2, grp, "atoms")[0], 7),
        pos(pos_), params(n_elem)
    {
        int n_dep = 3;
        check_size(grp, "atoms",    n_elem, n_dep);
        check_size(grp, "ref_geom", n_elem, n_dep,3);

        traverse_dset<2,int  >(grp,"atoms",   [&](size_t i,size_t j,          int   x){params[i].atom[j].index=x;});
        traverse_dset<3,float>(grp,"ref_geom",[&](size_t i,size_t na,size_t d,float x){params[i].ref_geom[na*n_dep+d]=x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<params.size(); ++i) pos.slot_machine.add_request(7, params[i].atom[j]);
        for(auto &p: params) autodiff_params.push_back(AutoDiffParams({p.atom[0].slot, p.atom[1].slot, p.atom[2].slot}));
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("affine_alignment"));
        affine_alignment(coords().value, pos.coords(), params.data(), 
                n_elem, pos.n_system);}

    virtual void propagate_deriv() {
        Timer timer(string("affine_alignment_deriv"));
        affine_reverse_autodiff(
                coords().value, coords().deriv, pos.slot_machine.accum_array(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_elem, pos.n_system);
    }

    virtual double test_value_deriv_agreement() {
        return compute_relative_deviation_for_node<3>(*this, pos, extract_pairs(params, potential_term), BODY_VALUE);
    }
};
static RegisterNodeType<AffineAlignment,1>_8("affine_alignment");
