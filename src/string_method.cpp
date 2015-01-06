#include "force.h"
#include "string_method.h"
#include "timing.h"
#include <Eigen/Sparse>
#include <algorithm>

using namespace Eigen;
using namespace std;
typedef SparseMatrix<float,RowMajor> ColSparse;

DihedralStringSystem::DihedralStringSystem(const std::vector<int> &id_, 
        int n_string_coords_, int n_system_per_string_coord_,
        float string_mass_, float spring_constant_):
    id(id_), n_dihe(id.size()/4), atom_space(*std::max_element(begin(id), end(id))+1),
    n_string_coords(n_string_coords_), n_system_per_string_coord(n_system_per_string_coord_),
    string_mass(string_mass_), spring_constant(spring_constant_)
{
    if(id.size() != unsigned(n_dihe*4)) throw std::string("invalid id size");
}


void dihedral_string_deriv(
        const double* string_pos,
        float* string_deriv,
        float spring_constant,
        const int* id,
        const SysArray pos, 
        int n_string_coords,
        int n_dihe,
        int max_atom,   // must be at least 1 + max(id)
        int n_system_per_string_coord) {

    for(int nsc=0; nsc<n_string_coords; ++nsc) {
        ColSparse   M[2]         = {ColSparse(n_dihe,max_atom), ColSparse(n_dihe,max_atom)};
        RowVectorXf theta_dev[2] = {RowVectorXf::Zero(n_dihe),  RowVectorXf::Zero(n_dihe)};

        for(int i=0; i<2; ++i) M[i].reserve(n_dihe*4*3);

        for(int nd=0; nd<n_dihe; ++nd) {
            for(int ns=0; ns<n_system_per_string_coord; ++ns) {
                StaticCoord<3> a0(pos, nsc*n_system_per_string_coord + ns, id[nd*4+0]);
                StaticCoord<3> a1(pos, nsc*n_system_per_string_coord + ns, id[nd*4+1]);
                StaticCoord<3> a2(pos, nsc*n_system_per_string_coord + ns, id[nd*4+2]);
                StaticCoord<3> a3(pos, nsc*n_system_per_string_coord + ns, id[nd*4+3]);

                float3 dd[4];
                float theta_disp = string_pos[nsc*n_dihe+nd] - dihedral_germ(
                        a0.f3(), a1.f3(), a2.f3(), a3.f3(),
                        dd[0],   dd[1],   dd[2],   dd[3]);

                // ensure smallest possible deviation for theta_disp
                if(theta_disp < M_PI_F) theta_disp += 2*M_PI_F;
                if(theta_disp > M_PI_F) theta_disp -= 2*M_PI_F;
                theta_dev[ns%2][nd] += theta_disp;

                for(int i=0; i<4; ++i) {
                    int loc = id[nd*4+i];
                    if(ns<2) {
                        M[ns%2].insert  (nd,loc*3+0)  = dd[i].x;
                        M[ns%2].insert  (nd,loc*3+1)  = dd[i].y;
                        M[ns%2].insert  (nd,loc*3+2)  = dd[i].z;
                    } else {  // all positions have been inserted
                        M[ns%2].coeffRef(nd,loc*3+0) += dd[i].x;
                        M[ns%2].coeffRef(nd,loc*3+1) += dd[i].y;
                        M[ns%2].coeffRef(nd,loc*3+2) += dd[i].z;
                    }
                }
            }

            Map<VectorXf> string_deriv_coord(string_deriv+nsc*n_dihe, n_dihe);
            string_deriv_coord = M[0]*(theta_dev[1]*M[0]).transpose() + M[1]*(theta_dev[0]*M[1]).transpose();
        }
    }

    float norm = (2.f/n_system_per_string_coord);
    for(int i=0; i<n_string_coords*n_dihe; ++i) string_deriv[i] *= 0.5f*spring_constant*norm*norm;
}


void DihedralStringSystem::string_deriv(float* deriv, SysArray pos) 
{
    Timer timer(std::string("string_deriv")); 
    dihedral_string_deriv(string_pos.data(), deriv, spring_constant, id.data(), pos, 
            n_string_coords, n_dihe, atom_space, n_system_per_string_coord);
}


void string_update(
        double* string_pos, const float* deriv, float dt,
        float inv_string_mass, int n_string_coords, int n_dim) 
{
    for(int i=0; i<n_string_coords*n_dim; ++i)
        string_pos[i] -= inv_string_mass*deriv[i];

    // reparametrize the string
    // this action is independent for each dimension

    MatrixXd coords = Map<MatrixXd,RowMajor>(string_pos,n_string_coords,n_dim) + 
        ((dt/inv_string_mass) * Map<const MatrixXf,RowMajor>(deriv,n_string_coords,n_dim)).cast<double>();

    VectorXd string_length = (coords.bottomRows(n_string_coords-1) - 
                              coords.topRows   (n_string_coords-1)).rowwise().norm();
    string_length *= 1./string_length.sum();

    double prev_arclen = 0.;
    int    current_left = 0;
    auto   new_coords = Map<MatrixXd,RowMajor>(string_pos, n_string_coords, n_dim);

    for(int nsc=0; nsc<n_string_coords; ++nsc) {
        // 1e-10 makes sure that final string coordinate is < 1.
        double s = nsc * ((1.-1e-10)/(n_string_coords-1)) - prev_arclen;

        // find correct image
        while(s>string_length[current_left]) {
            prev_arclen += string_length[current_left];
            s           -= string_length[current_left];
            current_left++;
            if(current_left == n_string_coords-1) throw std::string("impossible happened in string");
        }
        // compute fraction of distance between points
        double f = s / string_length[current_left];

        new_coords.row(nsc) = (1.-f)*coords.row(current_left) + f*coords.row(current_left+1);
    }
}


void DihedralStringSystem::update(const float* deriv, float dt) 
{
    Timer timer(std::string("string_update")); 
    string_update(string_pos.data(), deriv, dt,
            1.f/string_mass, n_string_coords, n_dihe) ;
}
