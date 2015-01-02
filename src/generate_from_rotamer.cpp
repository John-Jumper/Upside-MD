#include <Eigen/Dense>
#include <string>
#include <array>
#include <map>
#include <cmath>
#include <Eigen/Geometry>

#include <iostream>

using namespace std;
using namespace Eigen;

static const float deg = 4.*atan(1.) / 180.;

inline Affine3f make_tab(float phi, float theta, float bond) 
{
    Affine3f out;
    float cp(cos(phi)),   sp(sin(phi));
    float ct(cos(theta)), st(sin(theta));
    float l(bond);

    out(0,0)=   -ct; out(0,1)=    -st; out(0,2)=   0; out(0,3)=   -l*ct;
    out(1,0)= cp*st; out(1,1)= -cp*ct; out(1,2)= -sp; out(1,3)= l*cp*st;
    out(2,0)= sp*st; out(2,1)= -sp*ct; out(2,2)=  cp; out(2,3)= l*sp*st;
    //  out(3,0)=     0; out(3,1)=      0; out(3,2)=   0; out(3,3)=       1;

    return out;
}

Affine3f place_bb(MatrixX3f& ret, float psi, bool include_CB=true) {
    Affine3f a = Affine3f::Identity();
    a(0,0)=0.8191292; a(0,1)=-0.3103239; a(0,2)= 0.4824173; a(0,3)=-1.2079210;
    a(1,0)=0.5736088; a(1,1)= 0.4423396; a(1,2)=-0.6894263; a(1,3)=-0.2636016;
    a(2,0)=0.0005532; a(2,1)= 0.8414480; a(2,2)= 0.5403378; a(2,3)=-0.0009170;

    Affine3f N  = a *make_tab(          0.f,        0.f,   0.f);  ret.row(0) = N .translation().transpose();
    Affine3f CA = N *make_tab(          0.f,        0.f, 1.45f);  ret.row(1) = CA.translation().transpose();
    Affine3f C  = CA*make_tab(   122.7f*deg, 110.3f*deg, 1.53f);  ret.row(2) = C .translation().transpose();
    Affine3f O  = C *make_tab(psi+180.f*deg, 120.5f*deg, 1.23f);  ret.row(3) = O .translation().transpose();
    Affine3f CB = CA*make_tab(          0.f, 110.6f*deg, 1.53f);  if(include_CB) ret.row(4) = CB.translation().transpose();

    return CB;
}

static void ALA_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(5,3);
    place_bb(ret, psi);
}

static void ARG_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(11,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.9f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 71.2 1.9 0.02
    Affine3f CD  = CG *make_tab(           chi[1], 111.7f*deg, 1.52f); ret.row( 6) = CD .translation().transpose(); // dev 51.0 2.1 0.02
    Affine3f NE  = CD *make_tab(           chi[2], 111.7f*deg, 1.46f); ret.row( 7) = NE .translation().transpose(); // dev 83.5 2.2 0.01
    Affine3f CZ  = NE *make_tab(           chi[3], 124.7f*deg, 1.33f); ret.row( 8) = CZ .translation().transpose(); // dev 69.8 1.7 0.01
    Affine3f NH1 = CZ *make_tab(         0.0f*deg, 120.7f*deg, 1.33f); ret.row( 9) = NH1.translation().transpose(); // dev  3.0 1.2 0.03
    Affine3f NH2 = CZ *make_tab(      -180.0f*deg, 119.6f*deg, 1.33f); ret.row(10) = NH2.translation().transpose(); // dev  2.8 1.2 0.04
}

static void ASN_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(8,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 112.7f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 73.9 1.1 0.02
    Affine3f OD1 = CG *make_tab(           chi[1], 120.9f*deg, 1.23f); ret.row( 6) = OD1.translation().transpose(); // dev 72.1 1.0 0.01
    Affine3f ND2 = CG *make_tab(chi[1]+180.0f*deg, 116.5f*deg, 1.33f); ret.row( 7) = ND2.translation().transpose(); // dev  2.0 1.0 0.01
}

static void ASP_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(8,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.0f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 78.9 1.1 0.02
    Affine3f OD1 = CG *make_tab(           chi[1], 119.2f*deg, 1.25f); ret.row( 6) = OD1.translation().transpose(); // dev 60.3 1.5 0.01
    Affine3f OD2 = CG *make_tab(chi[1]-179.9f*deg, 118.2f*deg, 1.25f); ret.row( 7) = OD2.translation().transpose(); // dev  1.9 1.7 0.01
}

static void CYS_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(6,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f SG  = CB *make_tab(           chi[0], 113.8f*deg, 1.81f); ret.row( 5) = SG .translation().transpose(); // dev 80.2 2.0 0.02
}

static void GLN_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(9,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.9f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 67.5 1.9 0.02
    Affine3f CD  = CG *make_tab(           chi[1], 112.8f*deg, 1.52f); ret.row( 6) = CD .translation().transpose(); // dev 69.5 1.6 0.02
    Affine3f OE1 = CD *make_tab(           chi[2], 120.9f*deg, 1.23f); ret.row( 7) = OE1.translation().transpose(); // dev 76.6 1.0 0.01
    Affine3f NE2 = CD *make_tab(chi[2]-180.0f*deg, 116.5f*deg, 1.33f); ret.row( 8) = NE2.translation().transpose(); // dev  2.0 1.0 0.01
}

static void GLU_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(9,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.9f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 70.1 1.9 0.02
    Affine3f CD  = CG *make_tab(           chi[1], 113.2f*deg, 1.52f); ret.row( 6) = CD .translation().transpose(); // dev 65.6 1.7 0.02
    Affine3f OE1 = CD *make_tab(           chi[2], 119.0f*deg, 1.25f); ret.row( 7) = OE1.translation().transpose(); // dev 70.6 1.5 0.01
    Affine3f OE2 = CD *make_tab(chi[2]-180.0f*deg, 118.1f*deg, 1.25f); ret.row( 8) = OE2.translation().transpose(); // dev  2.0 1.4 0.02
}

static void GLY_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(4,3);
    place_bb(ret, psi, false);
}

static void HIS_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(10,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.6f*deg, 1.50f); ret.row( 5) = CG .translation().transpose(); // dev 76.2 1.2 0.01
    Affine3f ND1 = CG *make_tab(           chi[1], 122.7f*deg, 1.38f); ret.row( 6) = ND1.translation().transpose(); // dev 101.2 0.9 0.01
    Affine3f CD2 = CG *make_tab(chi[1]+179.9f*deg, 131.0f*deg, 1.36f); ret.row( 7) = CD2.translation().transpose(); // dev  3.2 0.9 0.01
    Affine3f CE1 = ND1*make_tab(       179.9f*deg, 109.2f*deg, 1.32f); ret.row( 8) = CE1.translation().transpose(); // dev  2.0 0.8 0.01
    Affine3f NE2 = CD2*make_tab(      -179.9f*deg, 107.2f*deg, 1.37f); ret.row( 9) = NE2.translation().transpose(); // dev  2.2 0.5 0.01
}

static void ILE_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(8,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG1 = CB *make_tab(           chi[0], 110.4f*deg, 1.53f); ret.row( 5) = CG1.translation().transpose(); // dev 55.7 1.3 0.01
    Affine3f CG2 = CB *make_tab(chi[0]-123.2f*deg, 110.7f*deg, 1.53f); ret.row( 6) = CG2.translation().transpose(); // dev  2.4 1.1 0.02
    Affine3f CD1 = CG1*make_tab(           chi[1], 114.0f*deg, 1.52f); ret.row( 7) = CD1.translation().transpose(); // dev 56.4 1.4 0.03
}

static void LEU_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(8,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 116.4f*deg, 1.53f); ret.row( 5) = CG .translation().transpose(); // dev 55.7 2.7 0.01
    Affine3f CD1 = CG *make_tab(           chi[1], 110.4f*deg, 1.53f); ret.row( 6) = CD1.translation().transpose(); // dev 58.1 1.8 0.02
    Affine3f CD2 = CG *make_tab(chi[1]+122.9f*deg, 110.6f*deg, 1.53f); ret.row( 7) = CD2.translation().transpose(); // dev  2.7 1.8 0.02
}

static void LYS_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(9,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 114.0f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 68.6 1.8 0.02
    Affine3f CD  = CG *make_tab(           chi[1], 111.5f*deg, 1.52f); ret.row( 6) = CD .translation().transpose(); // dev 54.0 1.9 0.02
    Affine3f CE  = CD *make_tab(           chi[2], 111.6f*deg, 1.52f); ret.row( 7) = CE .translation().transpose(); // dev 55.5 1.7 0.02
    Affine3f NZ  = CE *make_tab(           chi[3], 111.8f*deg, 1.49f); ret.row( 8) = NZ .translation().transpose(); // dev 68.2 2.3 0.02
}

static void MET_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(8,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.9f*deg, 1.52f); ret.row( 5) = CG .translation().transpose(); // dev 66.5 1.9 0.02
    Affine3f SD  = CG *make_tab(           chi[1], 112.7f*deg, 1.81f); ret.row( 6) = SD .translation().transpose(); // dev 69.9 2.7 0.02
    Affine3f CE  = SD *make_tab(           chi[2], 100.7f*deg, 1.79f); ret.row( 7) = CE .translation().transpose(); // dev 97.8 2.3 0.04
}

static void PHE_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(11,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.8f*deg, 1.50f); ret.row( 5) = CG .translation().transpose(); // dev 76.1 1.2 0.01
    Affine3f CD1 = CG *make_tab(           chi[1], 120.7f*deg, 1.39f); ret.row( 6) = CD1.translation().transpose(); // dev 80.5 0.7 0.01
    Affine3f CD2 = CG *make_tab(chi[1]-180.0f*deg, 120.5f*deg, 1.39f); ret.row( 7) = CD2.translation().transpose(); // dev  2.6 0.7 0.01
    Affine3f CE1 = CD1*make_tab(      -180.0f*deg, 120.8f*deg, 1.39f); ret.row( 8) = CE1.translation().transpose(); // dev  1.9 0.6 0.02
    Affine3f CE2 = CD2*make_tab(       180.0f*deg, 120.8f*deg, 1.39f); ret.row( 9) = CE2.translation().transpose(); // dev  2.0 0.6 0.02
    Affine3f CZ  = CE1*make_tab(        -0.0f*deg, 119.9f*deg, 1.39f); ret.row(10) = CZ .translation().transpose(); // dev  1.4 0.6 0.02
}

static void PRO_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(7,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 104.2f*deg, 1.50f); ret.row( 5) = CG .translation().transpose(); // dev 26.5 1.5 0.03
    Affine3f CD  = CG *make_tab(           chi[1], 104.9f*deg, 1.51f); ret.row( 6) = CD .translation().transpose(); // dev 35.5 2.4 0.02
}

static void SER_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(6,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f OG  = CB *make_tab(           chi[0], 110.8f*deg, 1.42f); ret.row( 5) = OG .translation().transpose(); // dev 89.6 1.5 0.01
}

static void THR_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(7,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f OG1 = CB *make_tab(           chi[0], 109.2f*deg, 1.43f); ret.row( 5) = OG1.translation().transpose(); // dev 76.3 1.1 0.01
    Affine3f CG2 = CB *make_tab(chi[0]-120.4f*deg, 111.1f*deg, 1.53f); ret.row( 6) = CG2.translation().transpose(); // dev  3.0 1.1 0.02
}

static void TRP_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(14,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.9f*deg, 1.50f); ret.row( 5) = CG .translation().transpose(); // dev 81.6 2.3 0.02
    Affine3f CD1 = CG *make_tab(           chi[1], 127.1f*deg, 1.37f); ret.row( 6) = CD1.translation().transpose(); // dev 90.1 0.8 0.01
    Affine3f CD2 = CG *make_tab(chi[1]-179.7f*deg, 126.6f*deg, 1.43f); ret.row( 7) = CD2.translation().transpose(); // dev  3.2 0.8 0.01
    Affine3f NE1 = CD1*make_tab(      -179.8f*deg, 110.1f*deg, 1.38f); ret.row( 8) = NE1.translation().transpose(); // dev  2.2 0.6 0.01
    Affine3f CE2 = CD2*make_tab(       179.8f*deg, 107.2f*deg, 1.41f); ret.row( 9) = CE2.translation().transpose(); // dev  2.3 0.5 0.01
    Affine3f CE3 = CD2*make_tab(        -0.2f*deg, 133.9f*deg, 1.40f); ret.row(10) = CE3.translation().transpose(); // dev  2.7 0.5 0.01
    Affine3f CZ2 = CE2*make_tab(       180.0f*deg, 122.4f*deg, 1.40f); ret.row(11) = CZ2.translation().transpose(); // dev  1.0 0.4 0.01
    Affine3f CZ3 = CE3*make_tab(      -180.0f*deg, 118.7f*deg, 1.39f); ret.row(12) = CZ3.translation().transpose(); // dev  1.2 0.5 0.02
    Affine3f CH2 = CZ2*make_tab(        -0.0f*deg, 117.5f*deg, 1.37f); ret.row(13) = CH2.translation().transpose(); // dev  1.3 0.5 0.01
}

static void TYR_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(12,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG  = CB *make_tab(           chi[0], 113.7f*deg, 1.51f); ret.row( 5) = CG .translation().transpose(); // dev 77.2 2.2 0.01
    Affine3f CD1 = CG *make_tab(           chi[1], 120.9f*deg, 1.39f); ret.row( 6) = CD1.translation().transpose(); // dev 80.5 0.7 0.01
    Affine3f CD2 = CG *make_tab(chi[1]-179.9f*deg, 120.8f*deg, 1.39f); ret.row( 7) = CD2.translation().transpose(); // dev  2.6 0.7 0.01
    Affine3f CE1 = CD1*make_tab(      -179.9f*deg, 121.1f*deg, 1.39f); ret.row( 8) = CE1.translation().transpose(); // dev  2.1 0.6 0.02
    Affine3f CE2 = CD2*make_tab(       179.9f*deg, 121.1f*deg, 1.39f); ret.row( 9) = CE2.translation().transpose(); // dev  2.1 0.6 0.02
    Affine3f CZ  = CE1*make_tab(        -0.0f*deg, 119.5f*deg, 1.38f); ret.row(10) = CZ .translation().transpose(); // dev  1.4 0.6 0.01
    Affine3f OH  = CZ *make_tab(       180.0f*deg, 119.8f*deg, 1.38f); ret.row(11) = OH .translation().transpose(); // dev  1.3 1.3 0.01
}

static void VAL_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {
    ret.resize(7,3);
    Affine3f CB  = place_bb(ret, psi);
    Affine3f CG1 = CB *make_tab(           chi[0], 110.7f*deg, 1.53f); ret.row( 5) = CG1.translation().transpose(); // dev 60.3 1.1 0.02
    Affine3f CG2 = CB *make_tab(chi[0]+122.9f*deg, 110.4f*deg, 1.53f); ret.row( 6) = CG2.translation().transpose(); // dev  2.4 1.1 0.02
}


typedef void (*ResFuncPtr)(MatrixX3f& ret, float psi, const array<float,4> &chi);

map<string,ResFuncPtr>& res_func_map() {
    static map<string,ResFuncPtr> m;
    if(!m.size()) {
        m["ALA"] = &ALA_res_func;
        m["ARG"] = &ARG_res_func;
        m["ASN"] = &ASN_res_func;
        m["ASP"] = &ASP_res_func;
        m["CYS"] = &CYS_res_func;
        m["GLN"] = &GLN_res_func;
        m["GLU"] = &GLU_res_func;
        m["GLY"] = &GLY_res_func;
        m["HIS"] = &HIS_res_func;
        m["ILE"] = &ILE_res_func;
        m["LEU"] = &LEU_res_func;
        m["LYS"] = &LYS_res_func;
        m["MET"] = &MET_res_func;
        m["PHE"] = &PHE_res_func;
        m["PRO"] = &PRO_res_func;
        m["SER"] = &SER_res_func;
        m["THR"] = &THR_res_func;
        m["TRP"] = &TRP_res_func;
        m["TYR"] = &TYR_res_func;
        m["VAL"] = &VAL_res_func;
    }

    return m;
}

