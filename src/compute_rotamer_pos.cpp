#include "vector_math.h"
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

static float dihedral(MatrixX3f r) {
    if(r.rows() != 4) throw string("impossible");
    return dihedral(r.row(0), r.row(1), r.row(2), r.row(3));
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

void push_vector(vector<float>& a, const RowVector3f& v) {
    a.push_back(v[0]); a.push_back(v[1]); a.push_back(v[2]);
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
        if(rname=="ALA" || rname=="GLY") continue;
        auto &res_func = *rf.second;
        printf("%s\n", rname.c_str());
        auto grp = ensure_group(config.get(), rname.c_str());

        int n_res_total = get_dset_size(2, grp.get(), "chi")[0];
        check_size(grp.get(), "chi",  n_res_total, 4);

        MatrixXf chis(n_res_total,4);
        traverse_dset<2,float>(grp.get(), "chi", [&](size_t nr, size_t nchi, float x) { chis(nr,nchi)=x; });

        VectorXf probs(n_res_total);
        traverse_dset<1,float>(grp.get(), "prob", [&](size_t nr, float x) { probs[nr]=x; });

        MatrixXi starts(n_bin,n_bin);
        MatrixXi stops (n_bin,n_bin);
        traverse_dset<3,int>(grp.get(), "rama_index_range", [&](size_t nb1, size_t nb2, size_t ss, int x) {
                if(ss==0) starts(nb1,nb2) = x;
                else      stops (nb1,nb2) = x;});

        MatrixXi rotamer(get_dset_size(2, grp.get(), "rotamer")[0], get_dset_size(2, grp.get(), "rotamer")[1]);
        traverse_dset<2,float>(grp.get(), "rotamer", [&](size_t nr, size_t nchi, float x) {
                rotamer(nr,nchi)=(x==0 ? 0 : x-1);});
        VectorXi n_rot = rotamer.colwise().maxCoeff(); 
        for(int d: range(n_rot.rows())) n_rot[d]+=1;
        int n_conf = 1; for(int r: range(4)) n_conf *= n_rot[r];
        printf("n_conf %i\n", n_conf);

        array<float,4> chi_zero = {{0.f, 0.f, 0.f, 0.f}};
        MatrixX3f pos_zero; res_func(pos_zero, 0.f, chi_zero);
        int n_atom = pos_zero.rows();

        // avoid redundancy due to periodicity
        vector<float> chi_data ((n_bin-1)*(n_bin-1) * n_conf * 4);           fill(begin(chi_data), end(chi_data),   -1e-8f);
        vector<float> prob_data((n_bin-1)*(n_bin-1) * n_conf);               fill(begin(prob_data), end(prob_data), 0.f);
        vector<float> pos_data ((n_bin-1)*(n_bin-1) * n_conf * n_atom * 3);  fill(begin(pos_data), end(pos_data),   -1e-8f);

        auto f = [](int i) {return i==0 ? 1 : i;};
        auto offset = [&](size_t nb1, size_t nb2, size_t chi1, size_t chi2, size_t chi3, size_t chi4) {
            return ((((nb1  *size_t(n_bin-1u)
                     + nb2 )*size_t(f(n_rot[0])) 
                     + chi1)*size_t(f(n_rot[1]))
                     + chi2)*size_t(f(n_rot[2]))
                     + chi3)*size_t(f(n_rot[3]))
                     + chi4;
        };

        auto chi_val  = [&](size_t nb1, size_t nb2, const RowVectorXi& chi, size_t chi_index)->float& {
            return chi_data [offset(nb1,nb2, chi[0],chi[1],chi[2],chi[3])*4 + chi_index];};
        auto prob_val = [&](size_t nb1, size_t nb2, const RowVectorXi& chi)->float& {
            return prob_data[offset(nb1,nb2, chi[0],chi[1],chi[2],chi[3])];};
        auto pos_val = [&](size_t nb1, size_t nb2, const RowVectorXi& chi, size_t na, size_t d)->float& {
            return pos_data [offset(nb1,nb2, chi[0],chi[1],chi[2],chi[3])*n_atom*3 + na*3 + d];};

        for(int nb1: range(n_bin-1)) {
            for(int nb2: range(n_bin-1)) {
                MatrixX3f pos;
                float psi = nb2*bin_size - M_PI_F;

                for(int nr=starts(nb1,nb2); nr<stops(nb1,nb2); ++nr) {
                    RowVectorXi rot = rotamer.row(nr);
                    array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
                    for(int nchi=0; nchi<4; ++nchi) {
                        chi_val(nb1,nb2,rot, nchi) = chis(nr,nchi);
                        chi[nchi] = chis(nr,nchi);
                    }
                    res_func(pos, psi, chi);
                    for(int na: range(n_atom)) for(int d: range(3)) pos_val(nb1,nb2,rot, na,d) = pos(na,d);
                    prob_val(nb1,nb2,rot) = probs[nr];
                }
            }
        }

        printf("%i %i  %i %i %i %i\n", n_bin-1,n_bin-1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3]));
        auto chi_array  = create_earray(grp.get(),  "chi_tensor", H5T_NATIVE_FLOAT,
                {     -1,n_bin-1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3]), 4},
                {      1,      1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3]), 4});
        auto prob_array = create_earray(grp.get(), "prob_tensor", H5T_NATIVE_FLOAT,
                {     -1,n_bin-1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3])},
                {      1,      1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3])});
        auto pos_array  = create_earray(grp.get(),  "pos_tensor", H5T_NATIVE_FLOAT,
                {     -1,n_bin-1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3]), n_atom,3},
                {      1,      1, f(n_rot[0]), f(n_rot[1]), f(n_rot[2]), f(n_rot[3]), n_atom,3});

        append_to_dset( chi_array.get(),  chi_data, 0);
        append_to_dset(prob_array.get(), prob_data, 0);
        append_to_dset( pos_array.get(),  pos_data, 0);

        vector<float> bin_vals(n_bin-1);
        for(int nb=0; nb<n_bin-1; ++nb) bin_vals[nb] = nb*bin_size - M_PI_F;
        auto bin_array = create_earray(grp.get(), "bin_values", H5T_NATIVE_FLOAT, {-1}, {n_bin-1});
        append_to_dset(bin_array.get(), bin_vals, 0);
    }

        // if(rname == "ALA" || rname == "GLY") {
        //     auto rir_array = create_earray(grp.get(), "rama_index_range", H5T_NATIVE_INT,
        //             {-1,n_bin,2}, {n_bin,n_bin,2});
        //     std::vector<float> rir;
        //     for(int i=0; i<n_bin*n_bin; ++i) {
        //         rir.push_back(i);
        //         rir.push_back(i+1);
        //     }
        //     append_to_dset(rir_array.get(), rir, 0);

        //     auto prob_array = create_earray(grp.get(), "prob", H5T_NATIVE_INT, {-1}, {n_bin*n_bin});
        //     std::vector<float> prob(n_bin*n_bin,1.f);
        //     append_to_dset(prob_array.get(), prob, 0);

        //     for(int i1=0; i1<n_bin; ++i1) {
        //         for(int i2=0; i2<n_bin; ++i2) {
        //             MatrixX3f pos;
        //             float psi = i2*bin_size - M_PI_F;
        //             array<float,4> chi = {{0.f, 0.f, 0.f, 0.f}};
        //             res_func(pos, psi, chi);
        //             n_bead = get_beads(rname, centers, direcs, pos);
        //         }
        //     }
        //  }

    return 0;
} catch (const string &e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
}
