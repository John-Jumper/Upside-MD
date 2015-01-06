#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"

using namespace h5;
using namespace std;

struct DihedralSpringParams {
    CoordPair atom[4];
    float equil_dihedral;
    float spring_constant;
};

struct AngleSpringParams {
    CoordPair atom[3];
    float equil_dp;
    float spring_constant;
};

struct DistSpringParams {
    CoordPair atom[2];
    float equil_dist;
    float spring_constant;
};

struct ZFlatBottomParams {
    CoordPair atom;
    float     z0;
    float     radius;
    float     spring_constant;
};

struct RamaCoordParams {
    CoordPair atom[5];
};

struct PosSpringParams {
    CoordPair atom[1];
    float x,y,z;
    float spring_constant;
};

template <typename CoordT>
inline void pos_spring_body(
        CoordT &x1,
        const PosSpringParams &p)
{
    float3 disp = x1.f3() - make_float3(p.x,p.y,p.z);
    x1.set_deriv(p.spring_constant * disp);
}


void pos_spring(
        const CoordArray pos,
        const PosSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            pos_spring_body(x1, params[nt]);
            x1.flush();
        }
    }
}

struct PosSpring : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    vector<PosSpringParams> params;

    PosSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<1>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 2;  // number of atoms that each term depends on 
        check_size(grp, "id", n_elem, n_dep);
        for(auto& nm: {"x","y","z","spring_const"}) check_size(grp, nm, n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "x",            [&](size_t i,           float x) { p[i].x = x;});
        traverse_dset<1,float>(grp, "y",            [&](size_t i,           float x) { p[i].y = x;});
        traverse_dset<1,float>(grp, "z",            [&](size_t i,           float x) { p[i].z = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_value() {
        Timer timer(string("pos_spring")); 
        pos_spring(pos.coords(), params.data(), n_elem, pos.n_system);}
};
static RegisterNodeType<PosSpring,1> pos_spring_node("atom_pos_spring");


void rama_coord(
        const SysArray output,
        const CoordArray pos,
        const RamaCoordParams* params,
        const int n_term, const int n_system) 
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_term; ++nt) {
            MutableCoord<2> rama_pos(output, ns, nt);

            bool has_prev = params[nt].atom[0].index != index_t(-1);
            bool has_next = params[nt].atom[4].index != index_t(-1);

            Coord<3,2> prev_C(pos, ns, has_prev ? params[nt].atom[0] : CoordPair(0,0));
            Coord<3,2> N     (pos, ns, params[nt].atom[1]);
            Coord<3,2> CA    (pos, ns, params[nt].atom[2]);
            Coord<3,2> C     (pos, ns, params[nt].atom[3]);
            Coord<3,2> next_N(pos, ns, has_next ? params[nt].atom[4] : CoordPair(0,0));

            {
                float3 d1,d2,d3,d4;
                if(has_prev) {
                    rama_pos.v[0] = dihedral_germ(
                            prev_C.f3(), N.f3(), CA.f3(), C.f3(),
                            d1, d2, d3, d4);
                } else {
                    rama_pos.v[0] = -1.3963f;  // -80 degrees
                    d1 = d2 = d3 = d4 = make_float3(0.f,0.f,0.f);
                }

                prev_C.set_deriv(0,d1);
                N     .set_deriv(0,d2);
                CA    .set_deriv(0,d3);
                C     .set_deriv(0,d4);
                next_N.set_deriv(0,make_float3(0.f,0.f,0.f));
            }

            {
                float3 d2,d3,d4,d5;
                if(has_next) {
                    rama_pos.v[1] = dihedral_germ(
                            N.f3(), CA.f3(), C.f3(), next_N.f3(),
                            d2, d3, d4, d5);
                } else {
                    rama_pos.v[1] = -1.3963f;  // -80 degrees
                    d2 = d3 = d4 = d5 = make_float3(0.f,0.f,0.f);
                }

                prev_C.set_deriv(1,make_float3(0.f,0.f,0.f));
                N     .set_deriv(1,d2);
                CA    .set_deriv(1,d3);
                C     .set_deriv(1,d4);
                next_N.set_deriv(1,d5);
            }

            if(has_prev) prev_C.flush();
            N .flush();
            CA.flush();
            C .flush();
            if(has_next) next_N.flush();
        }
    }
}


struct RamaCoord : public CoordNode
{
    CoordNode& pos;
    vector<RamaCoordParams> params;
    vector<AutoDiffParams> autodiff_params;

    RamaCoord(hid_t grp, CoordNode& pos_):
        CoordNode(pos_.n_system, get_dset_size<2>(grp, "id")[0], 2),
        pos(pos_),
        params(n_elem)
    {
        int n_dep = 5;
        check_size(grp, "id", n_elem, 5);

        traverse_dset<2,int>(grp, "id", [&](size_t nr, size_t na, int x) {
                params[nr].atom[na].index = x;});

        typedef decltype(params[0].atom[0].index) index_t;
        typedef decltype(params[0].atom[0].slot)  slot_t;

        // an index of -1 is required for the fact the Rama coords are undefined for the termini
        for(int j=0; j<n_dep; ++j) {
            for(size_t i=0; i<params.size(); ++i) {
                if(params[i].atom[j].index != index_t(-1))
                    pos.slot_machine.add_request(2, params[i].atom[j]);
                else
                    params[i].atom[j].slot = slot_t(-1);
            }
        }

        for(auto &p: params) autodiff_params.push_back(
                AutoDiffParams({p.atom[0].slot, p.atom[1].slot, p.atom[2].slot, p.atom[3].slot, p.atom[4].slot}));
    }

    virtual void compute_value() {
        Timer timer(string("rama_coord"));
        rama_coord(
                coords().value,
                pos.coords(),
                params.data(),
                n_elem, pos.n_system);
    }

    virtual void propagate_deriv() {
        Timer timer(string("rama_coord_deriv"));
        reverse_autodiff<2,3,0>(
                slot_machine.accum_array(), 
                pos.slot_machine.accum_array(), SysArray(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_elem, pos.n_system);
    }
};
static RegisterNodeType<RamaCoord,1> rama_coord_node("rama_coord");


template <typename CoordT>
inline void dist_spring_body(
        CoordT &x1,
        CoordT &x2,
        const DistSpringParams &p)
{
    float3 disp = x1.f3() - x2.f3();
    float3 deriv = p.spring_constant * (1.f - p.equil_dist*inv_mag(disp)) * disp;
    // V(x1,x2) = spring_const * (|x1-x2| - equil_dist)^2

    x1.set_deriv( deriv);
    x2.set_deriv(-deriv);
}


void dist_spring(
        const CoordArray pos,
        const DistSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            dist_spring_body(x1,x2, params[nt]);
            x1.flush();
            x2.flush();
        }
    }
}


struct DistSpring : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    vector<DistSpringParams> params;

    DistSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 2;  // number of atoms that each term depends on 
        check_size(grp, "id",              n_elem, n_dep);
        check_size(grp, "equil_dist",      n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) { p[i].equil_dist = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_value() {
        Timer timer(string("dist_spring"));
        dist_spring(pos.coords(), params.data(), n_elem, pos.n_system);}
};
static RegisterNodeType<DistSpring,1> dist_spring_node("dist_spring");


void z_flat_bottom_spring(
        const CoordArray pos,
        const ZFlatBottomParams* params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            ZFlatBottomParams p = params[nt];
            Coord<3> atom_pos(pos, ns, p.atom);
            
            float z = atom_pos.f3().z;
            float3 deriv(0.f, 0.f, 0.f);
            if(z-p.z0 >  p.radius) deriv.z = p.spring_constant * (z-p.z0 - p.radius);
            if(z-p.z0 < -p.radius) deriv.z = p.spring_constant * (z-p.z0 + p.radius);
            atom_pos.set_deriv(deriv);
            atom_pos.flush();
        }
    }
}


struct ZFlatBottom : public PotentialNode
{
    int n_term;
    CoordNode& pos;
    vector<ZFlatBottomParams> params;

    ZFlatBottom(hid_t hdf_group, CoordNode& pos_):
        pos(pos_), params( get_dset_size<1>(hdf_group, "atom")[0] )
    {
        n_term = params.size();
        check_size(hdf_group, "atom",            n_term);
        check_size(hdf_group, "z0",              n_term);
        check_size(hdf_group, "radius",          n_term);
        check_size(hdf_group, "spring_constant", n_term);

        traverse_dset<1,int  >(hdf_group, "atom", [&](size_t nt, int i) {params[nt].atom.index=i;});
        traverse_dset<1,float>(hdf_group, "z0", [&](size_t nt, float x) {params[nt].z0 = x;});
        traverse_dset<1,float>(hdf_group, "radius", [&](size_t nt, float x) {params[nt].radius = x;});
        traverse_dset<1,float>(hdf_group, "spring_constant", [&](size_t nt, float x) {
                params[nt].spring_constant = x;});

        for(int nt=0; nt<n_term; ++nt) pos.slot_machine.add_request(1, params[nt].atom);
    }

    virtual void compute_value() {
        Timer timer(string("z_flat_bottom"));
        z_flat_bottom_spring(pos.coords(), params.data(), n_term, pos.n_system);
    }
};
static RegisterNodeType<ZFlatBottom,1> z_flat_bottom_node("z_flat_bottom");


template <typename CoordT>
inline void angle_spring_body(
        CoordT &atom1,
        CoordT &atom2,
        CoordT &atom3,   // middle atom
        const AngleSpringParams &p)
{
    float3 x1 = atom1.f3() - atom3.f3(); float inv_d1 = inv_mag(x1); float3 x1h = x1*inv_d1;
    float3 x2 = atom2.f3() - atom3.f3(); float inv_d2 = inv_mag(x2); float3 x2h = x2*inv_d2;

    float dp = dot(x1h, x2h);
    float force_prefactor = p.spring_constant * (dp - p.equil_dp);

    atom1.set_deriv(force_prefactor * (x2h - x1h*dp) * inv_d1);
    atom2.set_deriv(force_prefactor * (x1h - x2h*dp) * inv_d2);
    atom3.set_deriv(-atom1.df3(0)-atom2.df3(0));  // computed by the condition of zero net force
}


void angle_spring(
        const CoordArray pos,
        const AngleSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            angle_spring_body(x1,x2,x3, params[nt]);
            x1.flush();
            x2.flush();
            x3.flush();
        }
    }
}


struct AngleSpring : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    vector<AngleSpringParams> params;

    AngleSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 3;  // number of atoms that each term depends on 
        check_size(grp, "id",              n_elem, n_dep);
        check_size(grp, "equil_dist",        n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) { p[i].equil_dp = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_value() {
        Timer timer(string("angle_spring"));
        angle_spring(pos.coords(), params.data(), n_elem, pos.n_system);}
};
static RegisterNodeType<AngleSpring,1> angle_spring_node("angle_spring");


template <typename CoordT>
inline void dihedral_spring_body(
        CoordT &x1,
        CoordT &x2,
        CoordT &x3,
        CoordT &x4,
        const DihedralSpringParams &p)
{
    float3 d1,d2,d3,d4;
    float dihedral = dihedral_germ(x1.f3(),x2.f3(),x3.f3(),x4.f3(), d1,d2,d3,d4);

    // determine minimum periodic image (can be off by at most 2pi)
    float displacement = dihedral - p.equil_dihedral;
    displacement = (displacement> M_PI_F) ? displacement-2.f*M_PI_F : displacement;
    displacement = (displacement<-M_PI_F) ? displacement+2.f*M_PI_F : displacement;

    float c = p.spring_constant * displacement;
    x1.set_deriv(c*d1);
    x2.set_deriv(c*d2);
    x3.set_deriv(c*d3);
    x4.set_deriv(c*d4);
}


void dynamic_dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* restrict params,
        int params_offset,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            Coord<3> x4(pos, ns, params[nt].atom[3]);
            dihedral_spring_body(x1,x2,x3,x4, params[ns*params_offset + nt]);
            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}


struct DynamicDihedralSpring : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    int params_offset;
    vector<DihedralSpringParams> params;  // separate params for each system, id's must be the same

    DynamicDihedralSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params_offset(n_elem), params(pos.n_system*params_offset)
    {
        int n_dep = 4;  // number of atoms that each term depends on 
        check_size(grp, "id", n_elem, n_dep, pos.n_system);  // only id is required for dynamic spring

        auto& p = params;
        traverse_dset<2,int>  (grp, "id", [&](size_t nt, size_t na, int x) {
                for(int ns=0; ns<pos.n_system; ++ns) 
                    p[ns*n_elem+nt].atom[na].index = x;});

        for(auto& p: params) {
            p.equil_dihedral  = 0.f;
            p.spring_constant = 0.f;
        }

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_value() {
        Timer timer(string("dynamic_dihedral_spring"));
        dynamic_dihedral_spring(pos.coords(), params.data(), params_offset, n_elem, pos.n_system);
    }
};


void dihedral_spring(
        const CoordArray pos,
        const DihedralSpringParams* restrict params,
        int n_terms, int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            Coord<3> x1(pos, ns, params[nt].atom[0]);
            Coord<3> x2(pos, ns, params[nt].atom[1]);
            Coord<3> x3(pos, ns, params[nt].atom[2]);
            Coord<3> x4(pos, ns, params[nt].atom[3]);
            dihedral_spring_body(x1,x2,x3,x4, params[nt]);
            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}

struct DihedralSpring : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    vector<DihedralSpringParams> params;

    DihedralSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 4;  // number of atoms that each term depends on 
        check_size(grp, "id",           n_elem, n_dep);
        check_size(grp, "equil_dist",   n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) {p[i].atom[j].index  =x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) {p[i].equil_dihedral =x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) {p[i].spring_constant=x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_value() {
        Timer timer(string("dihedral_spring"));
        dihedral_spring(pos.coords(), params.data(), n_elem, pos.n_system);
    }
};
static RegisterNodeType<DihedralSpring,1> dihedral_spring_node("dihedral_spring");
