#include "generate_from_rotamer.h"
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <string>
#include "h5_support.h"
#include <map>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace Eigen;
using namespace std;
using namespace h5;

constexpr float M_PI_F = 4.f*atanf(1.f);

//static const float M_PI_F = 4.*atan(1.);
static const float deg = 4.*atan(1.)/180.;

static float dihedral(
        Vector3f  r1, Vector3f  r2, Vector3f  r3, Vector3f  r4)
    // Formulas and notation taken from Blondel and Karplus, 1995
{
    Vector3f F = r1-r2;
    Vector3f G = r2-r3;
    Vector3f H = r4-r3;

    Vector3f A = F.cross(G);
    Vector3f B = H.cross(G);
    Vector3f C = B.cross(A);

    return atan2(C.dot(G), A.dot(B) * G.norm());
}

// static float dihedral(
//         RowVector3f  r1, RowVector3f  r2, RowVector3f  r3, RowVector3f  r4) {
//     return dihedral(r1.transpose(), r2.transpose(), r3.transpose(), r4.transpose());
// }
// 
static float dihedral(MatrixX3f r) {
    if(r.rows() != 4) throw string("impossible");
    return dihedral(r.row(0), r.row(1), r.row(2), r.row(3));
}

void get_rotamer(const string &name, const MatrixX3f &pos, array<float,4> &chi) {
    if(name == "ALA") {
    }

    if(name == "ARG") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
        chi[2] = dihedral(pos.row(4), pos.row(5), pos.row(6), pos.row(7));
        chi[3] = dihedral(pos.row(5), pos.row(6), pos.row(7), pos.row(8));
    }

    if(name == "ASN") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "ASP") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "CYS") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
    }

    if(name == "GLN") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
        chi[2] = dihedral(pos.row(4), pos.row(5), pos.row(6), pos.row(7));
    }

    if(name == "GLU") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
        chi[2] = dihedral(pos.row(4), pos.row(5), pos.row(6), pos.row(7));
    }

    if(name == "GLY") {
    }

    if(name == "HIS") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "ILE") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(7));
    }

    if(name == "LEU") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "LYS") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
        chi[2] = dihedral(pos.row(4), pos.row(5), pos.row(6), pos.row(7));
        chi[3] = dihedral(pos.row(5), pos.row(6), pos.row(7), pos.row(8));
    }

    if(name == "MET") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
        chi[2] = dihedral(pos.row(4), pos.row(5), pos.row(6), pos.row(7));
    }

    if(name == "PHE") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "PRO") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "SER") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
    }

    if(name == "THR") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
    }

    if(name == "TRP") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "TYR") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
        chi[1] = dihedral(pos.row(1), pos.row(4), pos.row(5), pos.row(6));
    }

    if(name == "VAL") {
        chi[0] = dihedral(pos.row(0), pos.row(1), pos.row(4), pos.row(5));
    }
}

string to_string(RowVector3f x) {
    return string("[") + to_string(x[0]) + " " + to_string(x[1]) + " " + to_string(x[2]) + "]";
}
string to_string(Matrix3f x) {
    return string("\n[[") + to_string(x(0,0)) + ", " + to_string(x(0,1)) + ", " + to_string(x(0,2)) +"],\n" +
           string(" [") + to_string(x(1,0)) + ", " + to_string(x(1,1)) + ", " + to_string(x(1,2)) +"],\n"  +
           string(" [") + to_string(x(2,0)) + ", " + to_string(x(2,1)) + ", " + to_string(x(2,2)) +"]]\n";
}

string np_string(const MatrixX3f &x) {
    // WARNING not efficient for large matrices because of the string concatenation
    string s = "";
    for(int nr=0; nr<x.rows(); ++nr) {
        string ln = nr ? "          [" : "np.array([[";
        for(int nc=0; nc<x.cols(); ++nc)
            ln += to_string(x(nr,nc)) + ", ";
        ln += (nr<x.rows()-1) ? "],\n" : "]])\n";
        s += ln;
    }
    return s;
}

MatrixXf dmat(MatrixX3f& x) {
    int n=x.rows();
    MatrixXf ret(n, n);
    for(int i=0; i<n; ++i) for(int j=0; j<n; ++j) ret(i,j) = (x.row(i)-x.row(j)).norm();
    return ret;
}


float angle_wrap(float x) {
    while(x> M_PI_F) x-=2.f*M_PI_F;
    while(x<-M_PI_F) x+=2.f*M_PI_F;
    return x;
}


float rmsd(const MatrixX3f& x) {
    return sqrtf(x.squaredNorm()/x.rows());
}

Vector4d quat2vec(const Quaterniond &q) {
    Vector4d ret;
    ret[0] = q.x();
    ret[1] = q.y();
    ret[2] = q.z();
    ret[3] = q.w();
    return ret;
}


Affine3f rigid_alignment(
        const MatrixX3f &model,
        const MatrixX3f &ref) {
    Matrix4f transform = umeyama(ref.transpose(), model.transpose(), false);
    // cout << "transform: " << transform <<"\n\n";
    return Affine3f(Matrix4f(transform));
}

// int main_CoM(int argc, char** argv) try {
//     auto config = h5_obj(H5Fclose, H5Fopen("/home/jumper/Dropbox/code/upside-parameters/Dunbrack-Rotamer-Simple1-5.h5", H5F_ACC_RDONLY, H5P_DEFAULT));
// 
//     auto output = h5_obj(H5Fclose, H5Fcreate("backbone-dependent-CoM.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
// 
//     const int n_bin = 37;
//     const float bin_size = (2*M_PI_F) / (n_bin-1);
// 
//     for(auto &rf: res_func_map()) {
//         string rname   = rf.first;
//         auto &res_func = *rf.second;
//         printf("%s\n", rname.c_str());
// 
//         std::vector<float> centers;
//         auto out_grp = ensure_group(output.get(), rname.c_str());
// 
//         if(rname == "ALA" || rname == "GLY") {
//             for(int i1=0; i1<n_bin; ++i1) {
//                 for(int i2=0; i2<n_bin; ++i2) {
//                     float psi = i2*bin_size - M_PI_F;
//                     MatrixX3f pos;
//                     array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
//                     res_func(pos, psi, chi);
//                     RowVector3f curr_center = pos.row((rname == "GLY") ?  1 : 4);
//                     Vector3d center = curr_center.transpose().cast<double>();
//                     centers.push_back(center[0]);
//                     centers.push_back(center[1]);
//                     centers.push_back(center[2]);
//                     // cout << rname << " " << center.transpose() << "\n";
//                 }
//             }
//         } else {
//             auto grp     = open_group  (config.get(), rname.c_str());
// 
//             // int n_bin = get_dset_size(3, grp.get(), "rama_index_range")[0];
//             int n_res_total = get_dset_size(2, grp.get(), "chi")[0];
// 
//             check_size(grp.get(), "rama_index_range", n_bin, n_bin, 2);
//             check_size(grp.get(), "chi",  n_res_total, 4);
//             check_size(grp.get(), "prob", n_res_total);
// 
//             MatrixXi starts(n_bin,n_bin);
//             MatrixXi stops (n_bin,n_bin);
//             traverse_dset<3,int>(grp.get(), "rama_index_range", [&](size_t nb1, size_t nb2, size_t ss, int x) {
//                     if(ss==0) starts(nb1,nb2) = x;
//                     else      stops (nb1,nb2) = x; });
// 
//             MatrixXf chis(n_res_total,4);
//             traverse_dset<2,float>(grp.get(), "chi", [&](size_t nr, size_t nchi, float x) { chis(nr,nchi)=x; });
// 
//             VectorXf prob(n_res_total);
//             traverse_dset<1,float>(grp.get(), "prob", [&](size_t nr, float x) { prob[nr]=x; });
// 
//             for(int i1=0; i1<n_bin; ++i1) {
//                 for(int i2=0; i2<n_bin; ++i2) {
//                     Vector3d center = Vector3d::Zero();
//                     MatrixX3f pos;
// 
//                     float psi = i2*bin_size - M_PI_F;
// 
//                     double total_prob = 0.;
//                     for(int nr=starts(i1,i2); nr<stops(i1,i2); ++nr) {
//                         array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
//                         for(int nchi=0; nchi<4; ++nchi) chi[nchi] = chis(nr,nchi);
//                         res_func(pos, psi, chi);
// 
//                         RowVector3f curr_center = pos.bottomRows(pos.rows()-5).colwise().mean();
//                         center += (prob[nr]*curr_center).transpose().cast<double>();
//                         total_prob += prob[nr];
//                     }
//                     center *= 1./total_prob;
//                     centers.push_back(center[0]);
//                     centers.push_back(center[1]);
//                     centers.push_back(center[2]);
//                     // cout << rname << " " << center.transpose() << "\n";
//                 }
//             }
//         }
// 
//         auto center_array = create_earray(out_grp.get(), "center", H5T_NATIVE_FLOAT,
//                 {-1,n_bin,3}, {1,n_bin,3});
//         append_to_dset(center_array.get(), centers, 0);
// 
//         vector<float> bin_vals(n_bin);
//         for(int nb=0; nb<n_bin; ++nb) bin_vals[nb] = nb*bin_size - M_PI_F;
//         auto bin_array = create_earray(out_grp.get(), "bin_values", H5T_NATIVE_FLOAT,
//                 {-1}, {n_bin});
//         append_to_dset(bin_array.get(), bin_vals, 0);
//     }
// 
//     return 0;
// } catch (const string &e) {
//     fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
// }

void push_vector(vector<float>& a, const RowVector3f& v) {
    a.push_back(v[0]); a.push_back(v[1]); a.push_back(v[2]);
}

int get_beads(const std::string& rname, vector<float>& centers, vector<float>& direcs, MatrixX3f &pos) {
    int n_bead = 0;
    auto pc = [&](const RowVector3f& v) {push_vector(centers, v); n_bead++;};
    auto pd = [&](const RowVector3f& v) {push_vector(direcs,  v.normalized());};

    if(rname == "GLY") {
        pc(pos.row(1));
        pd(pos.row(1)  - 0.5f*(pos.row(0) + pos.row(2)));
    } else if(rname == "ALA"){
        pc(pos.row(4));
        pd(pos.row(4) - pos.row(1));
    } else if(rname == "GLU") {
        pc(pos.bottomRows(pos.rows()-6).colwise().mean());
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
    } else if(rname == "LYS") {
        pc(pos.row(8)); // NZ atom
        pc(pos.row(6)); // CD atom
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
    } else if(rname == "ARG") {
        pc((1.f/3.f)*(pos.row(7)+pos.row(9)+pos.row(10))); // center of nitrogens
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
    } else if(rname == "ASP") {
        pc(pos.row(5)); // CG atom
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
    } else if(rname == "THR") {
        pc(pos.row(5)); // atom 5 is OG1 
        pc(pos.row(6)); // atom 6 is CG1
        pd(pos.row(5)-pos.row(4)); // CB->OG1 bond vector
        pd(pos.row(6)-pos.row(4)); // CB->CG1 bond vector
    } else if(rname == "TRP") {
        pc((1.f/6.f)*(pos.row(7)+pos.row(9)+pos.row(10)+pos.row(11)+pos.row(12)+pos.row(13))); // 2nd aromatic ring
        pc(pos.row(8)); // atom 8 is NE1
        pd(pos.row(5)-pos.row(4)); // CB->CG bond vector
        pd(pos.row(8)-pos.row(4)); // CB->NE1 bond vector
    } else if(rname == "TYR") {
        pc((1.f/6.f)*(pos.row(5)+pos.row(6)+pos.row(7)+pos.row(8)+pos.row(9)+pos.row(10))); // aromatic ring
        pc(pos.row(11)); // atom 11 is OH
        pd(pos.row( 5)-pos.row(4)); // CB->CG bond vector
        pd(pos.row(11)-pos.row(4)); // CB->OH bond vector
    } else {
        RowVector3f rest = pos.bottomRows(pos.rows()-5).colwise().mean();
        pc(rest);
        pd(rest-pos.row(4)); // CB->rest bond vector
    }
    return n_bead;
}

int main(int argc, char** argv) try {
    if(argc!=2) throw string("wrong number of arguments");
    auto config = h5_obj(H5Fclose, 
            H5Fopen(argv[1],
                H5F_ACC_RDWR, H5P_DEFAULT));

    const int n_bin = 37;
    const float bin_size = (2*M_PI_F) / (n_bin-1);
    H5Obj out_grp;

    for(auto &rf: res_func_map()) {
        string rname   = rf.first;
        auto &res_func = *rf.second;
        printf("%s\n", rname.c_str());
        int n_bead = -1;

        std::vector<float> centers;
        std::vector<float> direcs;
        auto grp = ensure_group(config.get(), rname.c_str());

        if(rname == "ALA" || rname == "GLY") {
            auto rir_array = create_earray(grp.get(), "rama_index_range", H5T_NATIVE_INT,
                    {-1,n_bin,2}, {n_bin,n_bin,2});
            std::vector<float> rir;
            for(int i=0; i<n_bin*n_bin; ++i) {
                rir.push_back(i);
                rir.push_back(i+1);
            }
            append_to_dset(rir_array.get(), rir, 0);

            auto prob_array = create_earray(grp.get(), "prob", H5T_NATIVE_INT, {-1}, {n_bin*n_bin});
            std::vector<float> prob(n_bin*n_bin,1.f);
            append_to_dset(prob_array.get(), prob, 0);

            for(int i1=0; i1<n_bin; ++i1) {
                for(int i2=0; i2<n_bin; ++i2) {
                    MatrixX3f pos;
                    float psi = i2*bin_size - M_PI_F;
                    array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
                    res_func(pos, psi, chi);
                    n_bead = get_beads(rname, centers, direcs, pos);
                }
            }
        } else {
            int n_res_total = get_dset_size(2, grp.get(), "chi")[0];
            check_size(grp.get(), "chi",  n_res_total, 4);

            MatrixXf chis(n_res_total,4);
            traverse_dset<2,float>(grp.get(), "chi", [&](size_t nr, size_t nchi, float x) { chis(nr,nchi)=x; });

            MatrixXi starts(n_bin,n_bin);
            MatrixXi stops (n_bin,n_bin);
            traverse_dset<3,int>(grp.get(), "rama_index_range", [&](size_t nb1, size_t nb2, size_t ss, int x) {
                    if(ss==0) starts(nb1,nb2) = x;
                    else      stops (nb1,nb2) = x; });

            for(int i1=0; i1<n_bin; ++i1) {
                for(int i2=0; i2<n_bin; ++i2) {
                    MatrixX3f pos;
                    float psi = i2*bin_size - M_PI_F;

                    for(int nr=starts(i1,i2); nr<stops(i1,i2); ++nr) {
                        array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
                        for(int nchi=0; nchi<4; ++nchi) chi[nchi] = chis(nr,nchi);
                        res_func(pos, psi, chi);
                        n_bead = get_beads(rname, centers, direcs, pos);
                    }
                }
            }
        }

        auto center_array = create_earray(grp.get(), "center", H5T_NATIVE_FLOAT, {-1,n_bead,3}, {8,n_bead,3});
        append_to_dset(center_array.get(), centers, 0);

        auto direc_array = create_earray(grp.get(), "direc", H5T_NATIVE_FLOAT, {-1,n_bead,3}, {8,n_bead,3});
        append_to_dset(direc_array.get(), direcs, 0);

        vector<float> bin_vals(n_bin);
        for(int nb=0; nb<n_bin; ++nb) bin_vals[nb] = nb*bin_size - M_PI_F;
        auto bin_array = create_earray(grp.get(), "bin_values", H5T_NATIVE_FLOAT, {-1}, {n_bin});
        append_to_dset(bin_array.get(), bin_vals, 0);
    }

    return 0;
} catch (const string &e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
}


// void align_to_ref(
//         Vector3f &trans,
//         Matrix3f &rot,
//         const RowVector3f atom1,
//         const RowVector3f atom2,
//         const RowVector3f atom3,
//         const VectorXf ref) { // only first 9 components will be used
// 
//     auto ref1 = Vector3f(ref.segment<3>(0));
//     auto ref2 = Vector3f(ref.segment<3>(3));
//     auto ref3 = Vector3f(ref.segment<3>(6));
// 
//     trans = (1.f/3.f) * (atom1+atom2+atom3).transpose();
//     Matrix3f R = ref1 * (atom1-trans.transpose()) + 
//                  ref2 * (atom2-trans.transpose()) + 
//                  ref3 * (atom3-trans.transpose());
//     JacobiSVD<Matrix3f> svd(R, ComputeFullU | ComputeFullV);
//     Matrix3f U = svd.matrixU();
//     Matrix3f V = svd.matrixV();
// 
//     if(U.determinant()*V.determinant()<0.f) V.col(2) *= -1.f;  // fix improper rotation
//     rot = V * U.transpose();
// }



//     map<string,RefGeom> ref = compute_reference_geometry("/rust/work/residues.h5");
//     if(argc!=3) throw string("require 2 arguments but ") + to_string(argc-1) + " given";
//     string file_path = argv[1];
//     string inference_alg = argv[2];
// 
//     auto prot = h5_obj(H5Fclose, H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
// 
//     int n_res = get_dset_size(1, prot.get(), "fasta")[0];
//     printf("found %i residues\n", n_res);
//     vector<string> fasta(n_res);
//     MatrixXf       atom_pos(3*n_res,3);
//     vector<size_t> rotamer(n_res);
// 
//     check_size(prot.get(), "pos",  3*n_res, 3);
//     check_size(prot.get(), "fasta",  n_res);
//     check_size(prot.get(), "chi1",   n_res);
// 
//     traverse_string_dset<1>(prot.get(),"fasta", [&](size_t i, string &s) {fasta[i] = s;});
//     traverse_dset<2,float>(prot.get(),"pos", [&](size_t i, size_t j, float x){atom_pos(i,j)=x;});
//     traverse_dset<1,float>(prot.get(),"chi1", [&](size_t i, float x){
//         rotamer[i] = (x < -M_PI_F/3.f) ? 0 : ((x < M_PI_F/3.f) ? 1 :2);;});
// 
//     //for(auto &a: dai::builtinInfAlgNames()) printf("%s\n", a.c_str());
//     printf("building\n");
//     vector<Matrix3d> imats = interaction_matrices(atom_pos, fasta, ref);
//    
//     {
//         auto output = h5_obj(H5Fclose, H5Fcreate("/rust/work/imats.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
//         auto imat_tbl = create_earray(output.get(), "interaction_matrices", H5T_NATIVE_FLOAT,
//                 {-1,n_res,3,3}, {1,n_res,3,3});
//         auto imat_buffer = vector<float>(n_res*n_res*3*3);
//         int loc=0;
//         for(int nr1=0; nr1<n_res; ++nr1)
//             for(int nr2=0; nr2<n_res; ++nr2) {
//                 for(int r1=0; r1<3; ++r1)
//                     for(int r2=0; r2<3; ++r2)
//                         imat_buffer[loc++] = imats[nr1*n_res+nr2](r1,r2);
//             }
//         append_to_dset(imat_tbl.get(), imat_buffer, 0);
//     }
// 
//     for(auto &m: imats) 
//         if(m.maxCoeff() == 0.f) 
//             m = Matrix3d::Ones();
// 
//     printf("\nbuilding2\n");
//     auto fg = build_residue_factor_graph(n_res, imats);
//     printf("solving\n");
//     auto marginals = solve_factor_graph(fg, inference_alg);
// 
//     int n_hit = 0;
//     int n_poss = 0;
//     for(int nr=0; nr<n_res; ++nr) {
//         size_t prediction = 10;
//         marginals.row(nr).maxCoeff(&prediction);
// 
//         printf("%4i %s %lu %lu %4s    %.2f %.2f %.2f\n", nr, fasta[nr].c_str(),
//                 rotamer[nr], prediction, 
//                 (rotamer[nr]==prediction ? "HIT" : "MISS"),
//                 marginals(nr,0), marginals(nr,1), marginals(nr,2));
//         if(fasta[nr] != "GLY" && fasta[nr] != "ALA") {
//             n_poss++;
//             n_hit += rotamer[nr]==prediction;
//         }
//     }
// 
// 
//     printf("\ntotal_score: %i/%i (%.1f%%)\n", n_hit, n_poss, n_hit*100./n_poss);
// 
// 
//     
//     return 0;
// } catch (const string &e) {
//     fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
// }

// map<string, RefGeom> 
// compute_reference_geometry(const string &file_path) {
//     map<string, RefGeom> ret;
// 
//     auto config = h5_obj(H5Fclose, H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
//     auto residues = open_group(config.get(), "residues");
//     
//     vector<string> residue_names = node_names_in_group(residues.get(), ".");
//     for(auto &nm: residue_names) {
//         auto pos_shape = get_dset_size(3, residues.get(), (nm+"/pos").c_str());
//         int n_atom = pos_shape[1] - 1; // don't count oxygen atom
// 
//         MatrixXf atom_pos = MatrixXf::Zero(pos_shape[0], n_atom*3);
//         traverse_dset<3,float>(residues.get(), (nm+"/pos").c_str(), [&](size_t nr, size_t na, size_t d, float v) {
//                 // when computing location, ignore the oxygen (index=3)
//                 // if(nr==0) printf("idx %lu %lu %lu (%i)\n", nr,na,d,n_atom);
//                 if(na==3) return;
//                 int loc = (na<3 ? na : na-1)*3 + d;
//                 atom_pos(nr, loc) = v; 
//            });
// 
// 
//         // now we need to compute the chi1 dihedral
//         VectorXi chi1_rotamer = VectorXi::Zero(atom_pos.rows());
//         int rotamer_counts[3] = {0,0,0};
//         for(int nr=0; nr<atom_pos.rows(); ++nr) {
//             float3 d1,d2,d3,d4;
// 
//             auto read_pos = [&](int nr, int na) {
//                 return float3(atom_pos(nr,3*na+0), atom_pos(nr,3*na+1), atom_pos(nr,3*na+2));};
// 
//             // atom order is N,CA,C,CB,CG since O is excluded
//             float chi1 = (nm=="PRO" || nm=="ALA" || nm=="GLY") 
//                 ? (nr%3 - 1)* 2.f*M_PI_F/3.f  // fake value for GLY and ALA
//                 : dihedral_germ(
//                         read_pos(nr,0),  // N
//                         read_pos(nr,1),  // CA
//                         read_pos(nr,3),  // CB
//                         read_pos(nr,4),  // CG
//                         d1,d2,d3,d4);
//             // printf("%.2f %.2f %.2f\n", mag(read_pos(nr,0)-read_pos(nr,1)), mag(read_pos(nr,1)-read_pos(nr,2)), mag(read_pos(nr,0)-read_pos(nr,2)));
// 
//             chi1_rotamer[nr] = (chi1 < -M_PI_F/3.f) ? 0 : ((chi1 < M_PI_F/3.f) ? 1 :2);
//             rotamer_counts[chi1_rotamer[nr]]++;
//         }
// 
//         RefGeom item;
//         for(int rot=0; rot<3; ++rot) {
//             MatrixXd atom_pos_with_rotamer(rotamer_counts[rot], n_atom*3);
// 
//             // populate pos array
//             for(int nr=0, sc_idx=0; nr<atom_pos.rows(); ++nr)
//                 if(chi1_rotamer[nr] == rot) 
//                     atom_pos_with_rotamer.row(sc_idx++) = atom_pos.row(nr).cast<double>();
// 
//             RowVectorXd mean_pos = atom_pos_with_rotamer.colwise().mean();
//             atom_pos_with_rotamer.rowwise() -= mean_pos;
//             MatrixXd covariances = (1./atom_pos_with_rotamer.rows()) * 
//                 (atom_pos_with_rotamer.transpose() * atom_pos_with_rotamer);
//             // add small diagonal bias to avoid exact zero determinant
//             covariances += VectorXd::Constant(n_atom*3, 0.01*0.01).asDiagonal();
// 
//             // shift mean_pos so that the origin is at the center of the backbone positions 
//             Vector3d old_center = (1./3.) * 
//                 (mean_pos.segment<3>(3*0) + mean_pos.segment<3>(3*1) + mean_pos.segment<3>(3*2));
//             for(int i=0; i<mean_pos.size(); ++i) mean_pos[i] -= old_center[i%3];
// 
//             item.mean_pos   [rot] = mean_pos   .cast<float>();
//             item.covariances[rot] = covariances.cast<float>();
//             // printf("%s mean_pos_shape %li covariance log10(det)/3n %f\n", nm.c_str(),
//             //         mean_pos.cols(), covariances.householderQr().logAbsDeterminant()/(3*n_atom)/log(10.));
//         }
//         ret[nm] = item;
//     }
// 
//     return ret;
// }
