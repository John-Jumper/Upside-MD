#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"

using namespace h5;
using namespace std;


struct PosSpring : public PotentialNode
{
    struct Params {
        index_t atom;
        Vec<3> x0;
        float spring_constant;
    };

    int n_elem;
    CoordNode& pos;
    vector<Params> params;

    PosSpring(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        n_elem(get_dset_size(1, grp, "id")[0]), pos(pos_), params(n_elem)
    {
        check_size(grp, "id", n_elem);
        check_size(grp, "x0", n_elem, 3);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<1,int>  (grp, "id",           [&](size_t i,           int   x) { p[i].atom = x;});
        traverse_dset<2,float>(grp, "x0",           [&](size_t i, size_t d, float x) { p[i].x0[d] = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("pos_spring")); 
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        VecArray posc = pos.output;
        VecArray pos_sens = pos.sens;

        if(pot) *pot = 0.f;
        for(int nt=0; nt<n_elem; ++nt) {
            auto &p = params[nt];
            float3 disp = load_vec<3>(posc, p.atom) - p.x0;
            if(pot) *pot += 0.5f * p.spring_constant * mag2(disp);
            update_vec(pos_sens, p.atom, p.spring_constant * disp);
        }
    }
};
static RegisterNodeType<PosSpring,1> pos_spring_node("atom_pos_spring");


struct TensionPotential : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    struct Param {
        index_t atom;
        Vec<3>  tension_coeff;
    };
    vector<Param> params;

    TensionPotential(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        n_elem(get_dset_size(1, grp, "atom")[0]), pos(pos_), params(n_elem)
    {
        check_size(grp, "atom", n_elem);
        check_size(grp, "tension_coeff", n_elem, 3);

        auto &p = params;
        traverse_dset<1,int>  (grp,"atom",         [&](size_t i,         int   x){p[i].atom = x;});
        traverse_dset<2,float>(grp,"tension_coeff",[&](size_t i,size_t d,float x){p[i].tension_coeff[d] = x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("tension"));

        VecArray pos_c = pos.output;
        VecArray pos_sens = pos.sens;

        float pot = 0.f;
        for(auto &p: params) {
            auto x = load_vec<3>(pos_c, p.atom);
            pot -= dot(x, p.tension_coeff);
            update_vec(pos_sens, p.atom, -p.tension_coeff);
        }
        potential = pot;
    }
};
static RegisterNodeType<TensionPotential,1> tension_node("tension");


struct AFMPotential : public PotentialNode
{
    float time_initial;
    float time_step;      // WARNING: this should be the same as the global time step !
    float time_estimate;  // WARNING: changes frequently
    int round_num;        // WARNING: changes frequently
    int n_elem;
    CoordNode& pos;
    struct Param {
        index_t atom;
        float spring_const;
        Vec<3> starting_tip_pos;
        Vec<3> pulling_vel;
        };
    vector<Param> params;
    
    AFMPotential(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        time_initial(read_attribute<float>(grp, "pulling_vel", "time_initial")),
        time_step(read_attribute<float>(grp, "pulling_vel", "time_step")),
        time_estimate(time_initial),
        round_num(0),
        n_elem(get_dset_size(1, grp, "atom")[0]),
        pos(pos_),
        params(n_elem)
    {
        check_size(grp, "atom",          n_elem);
        check_size(grp, "spring_const",  n_elem);
        check_size(grp, "starting_tip_pos", n_elem, 3);
        check_size(grp, "pulling_vel",      n_elem, 3);
        
        auto &p = params;
        traverse_dset<1,int>  (grp,"atom",             [&](size_t i,         int   x){p[i].atom                = x;});
        traverse_dset<1,float>(grp,"spring_const",     [&](size_t i,         float x){p[i].spring_const        = x;});
        traverse_dset<2,float>(grp,"starting_tip_pos", [&](size_t i,size_t d,float x){p[i].starting_tip_pos[d] = x;});
        traverse_dset<2,float>(grp,"pulling_vel",      [&](size_t i,size_t d,float x){p[i].pulling_vel[d]      = x;});
        
        if(logging(LOG_BASIC)) {
            default_logger->add_logger<float>("tip_pos", {n_elem, 3}, [&](float* buffer) {
                round_num -= 2;
                time_estimate = time_initial + float(time_step)*round_num;
                
                for(int nt=0; nt<n_elem; ++nt) {
                    auto p = params[nt];
                    auto tip_pos = p.starting_tip_pos + p.pulling_vel * time_estimate;
                    for(int d=0; d<3; ++d) buffer[nt*3+d] = tip_pos[d];
                }
            });
           
            default_logger->add_logger<float>("time_estimate", {1}, [&](float* buffer) {
                buffer[0] = time_estimate;
            });
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("AFM"));
        
        round_num += 1;
        time_estimate = time_initial + float(time_step)*round_num;
        
        VecArray pos_c    = pos.output;
        VecArray pos_sens = pos.sens;
        
        float pot = 0.f;
        for(auto &p: params) {
            auto x = load_vec<3>(pos_c, p.atom);
            auto tip_pos = p.starting_tip_pos + p.pulling_vel * time_estimate;
            auto x_diff = x - tip_pos;
            pot += 0.5*p.spring_const*mag2(x_diff);
            update_vec(pos_sens, p.atom, p.spring_const*x_diff);
        }
        potential = pot;
    }
};
static RegisterNodeType<AFMPotential,1> AFM_node("AFM");


struct RamaCoord : public CoordNode
{
    struct alignas(16) Jac {float j[2][5][4];}; // padding for vector load/store
    struct Params {bool dummy_angle[2]; index_t atom[5];};

    CoordNode& pos;
    vector<Params> params;
    unique_ptr<Jac[]> jac;

    RamaCoord(hid_t grp, CoordNode& pos_):
        CoordNode(get_dset_size(2, grp, "id")[0], 2),
        pos(pos_),
        params(n_elem),
        jac(new_aligned<Jac>(n_elem,1))
    {
        check_size(grp, "id", n_elem, 5);
        traverse_dset<2,int>(grp, "id", [&](size_t nr, size_t na, int x) {
                params[nr].atom[na] = x;});

        for(auto& p: params) {
            // handle dummy angles uniformly (N-terminal phi and C-terminal psi)
            p.dummy_angle[0] = p.atom[0] == index_t(-1);
            p.dummy_angle[1] = p.atom[4] == index_t(-1);

            p.atom[0] = p.dummy_angle[0] ? 0 : p.atom[0];
            p.atom[4] = p.dummy_angle[1] ? 0 : p.atom[4];
        }

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("rama", {n_elem,2}, [&](float* buffer) {
                    copy_vec_array_to_buffer(output, n_elem,2, buffer);});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("rama_coord"));

        VecArray rama_pos = output;
        float*   posv     = pos.output.x.get();

        for(int nt=0; nt<n_elem; ++nt) {
            const auto& p = params[nt];
            Float4 x[5];
            for(int na: range(5)) x[na] = Float4(posv + 4*p.atom[na]);

            for(int phipsi: range(2)) {  // phi then psi
                Float4 d[5];

                rama_pos(phipsi,nt) = p.dummy_angle[phipsi]
                    ? -1.3963f   // -80 degrees if dummy angle
                    : dihedral_germ(x[0+phipsi],x[1+phipsi],x[2+phipsi],x[3+phipsi], // shift by 1 for psi
                                    d[0+phipsi],d[1+phipsi],d[2+phipsi],d[3+phipsi]).x();

                for(int na: range(5)) d[na].store(jac[nt].j[phipsi][na]);
            }
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("rama_coord_deriv"));
        float* pos_sens = pos.sens.x.get();

        for(int nt=0; nt<n_elem; ++nt) {
            const auto& p = params[nt];
            Float4 s[2] = {Float4(sens(0,nt)), Float4(sens(1,nt))};

            Float4 ps[5];
            for(int na: range(5))
                ps[na] =
                    fmadd(s[0], Float4(jac[nt].j[0][na]),
                            fmadd(s[1], Float4(jac[nt].j[1][na]),
                                Float4(pos_sens + 4*p.atom[na])));

            for(int na: range(5))
                ps[na].store(pos_sens + 4*p.atom[na]);  // value was added above
        }
    }
};
static RegisterNodeType<RamaCoord,1> rama_coord_node("rama_coord");


struct DistSpring : public PotentialNode
{
    struct Params {
        index_t atom[2];
        float equil_dist;
        float spring_constant;
    };

    int n_elem;
    CoordNode& pos;
    vector<Params> params;
    vector<int> bonded_atoms;

    DistSpring(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        n_elem(get_dset_size(2, grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 2;  // number of atoms that each term depends on 
        check_size(grp, "id",           n_elem, n_dep);
        check_size(grp, "equil_dist",   n_elem);
        check_size(grp, "spring_const", n_elem);
        check_size(grp, "bonded_atoms", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) {p[i].atom[j] = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) {p[i].equil_dist = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) {p[i].spring_constant = x;});
        traverse_dset<1,int>  (grp, "bonded_atoms", [&](size_t i,           int   x) {bonded_atoms.push_back(x);});

        if(logging(LOG_DETAILED))
            default_logger->add_logger<float>("nonbonded_spring_energy", {1}, [&](float* buffer) {
                    float pot = 0.f;
                    VecArray x = pos.output;

                    for(int nt=0; nt<n_elem; ++nt) {
                    if(bonded_atoms[nt]) continue;  // don't count bonded spring energy

                    auto p = params[nt];
                    float dmag = mag(load_vec<3>(x, p.atom[0]) - load_vec<3>(x, p.atom[1]));
                    pot += 0.5f * p.spring_constant * sqr(dmag - p.equil_dist);
                    }
                    buffer[0] = pot;
                    });
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("dist_spring"));

        VecArray posc = pos.output;
        VecArray pos_sens = pos.sens;
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        if(pot) *pot = 0.f;

        for(int nt=0; nt<n_elem; ++nt) {
            auto& p = params[nt];

            auto x1 = load_vec<3>(posc, p.atom[0]);
            auto x2 = load_vec<3>(posc, p.atom[1]);

            auto disp = x1 - x2;
            auto deriv = p.spring_constant * (1.f - p.equil_dist*inv_mag(disp)) * disp;
            if(pot) *pot += 0.5f * p.spring_constant * sqr(mag(disp) - p.equil_dist);

            update_vec(pos_sens, p.atom[0],  deriv);
            update_vec(pos_sens, p.atom[1], -deriv);
        }
    }
};
static RegisterNodeType<DistSpring,1> dist_spring_node("dist_spring");


struct CavityRadial : public PotentialNode
{
    struct Params {
        index_t id;
        float     radius;
        float     spring_constant;
    };

    int n_term;
    CoordNode& pos;
    vector<Params> params;

    CavityRadial(hid_t hdf_group, CoordNode& pos_):
        PotentialNode(),
        pos(pos_), params(get_dset_size(1, hdf_group, "id")[0] )
    {
        n_term = params.size();
        check_size(hdf_group, "id",              n_term);
        check_size(hdf_group, "radius",          n_term);
        check_size(hdf_group, "spring_constant", n_term);

        traverse_dset<1,int  >(hdf_group, "id", [&](size_t nt, int i) {params[nt].id=i;});
        traverse_dset<1,float>(hdf_group, "radius", [&](size_t nt, float x) {params[nt].radius = x;});
        traverse_dset<1,float>(hdf_group, "spring_constant", [&](size_t nt, float x) {
                params[nt].spring_constant = x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("cavity_radial"));

        VecArray posc = pos.output;
        VecArray pos_sens = pos.sens;
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        if(pot) *pot = 0.f;

        for(int nt=0; nt<n_term; ++nt) {
            auto &p = params[nt];
            auto x = load_vec<3>(posc, p.id);

            float r2 = mag2(x);
            if(r2>sqr(p.radius)) {
                float inv_r = rsqrt(r2);
                float r = r2*inv_r;
                float excess = r-p.radius;

                if(pot) *pot += 0.5f * p.spring_constant * sqr(excess);
                update_vec(pos_sens, p.id, (p.spring_constant*excess*inv_r)*x);
            }
        }
    }
};
static RegisterNodeType<CavityRadial,1> cavity_radial_node("cavity_radial");


struct ZFlatBottom : public PotentialNode
{
    struct Params {
        index_t id;
        float     z0;
        float     radius;
        float     spring_constant;
    };

    int n_term;
    CoordNode& pos;
    vector<Params> params;

    ZFlatBottom(hid_t hdf_group, CoordNode& pos_):
        PotentialNode(),
        pos(pos_), params( get_dset_size(1, hdf_group, "atom")[0] )
    {
        n_term = params.size();
        check_size(hdf_group, "atom",            n_term);
        check_size(hdf_group, "z0",              n_term);
        check_size(hdf_group, "radius",          n_term);
        check_size(hdf_group, "spring_constant", n_term);

        traverse_dset<1,int  >(hdf_group, "atom", [&](size_t nt, int i) {params[nt].id=i;});
        traverse_dset<1,float>(hdf_group, "z0", [&](size_t nt, float x) {params[nt].z0 = x;});
        traverse_dset<1,float>(hdf_group, "radius", [&](size_t nt, float x) {params[nt].radius = x;});
        traverse_dset<1,float>(hdf_group, "spring_constant", [&](size_t nt, float x) {
                params[nt].spring_constant = x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("z_flat_bottom"));

        VecArray posc = pos.output;
        VecArray pos_sens = pos.sens;
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        if(pot) *pot = 0.f;

        for(int nt=0; nt<n_term; ++nt) {
            auto& p = params[nt];
            float z = posc(2,p.id);

            float excess = z-p.z0 >  p.radius ? z-p.z0 - p.radius
                :         (z-p.z0 < -p.radius ? z-p.z0 + p.radius : 0.f);
            pos_sens(2,p.id) += p.spring_constant * excess;

            if(pot) *pot += 0.5f*p.spring_constant * sqr(excess);
        }
    }
};
static RegisterNodeType<ZFlatBottom,1> z_flat_bottom_node("z_flat_bottom");


struct AngleSpring : public PotentialNode
{
    struct Params {
        index_t atom[3];
        float equil_dp;
        float spring_constant;
    };

    int n_elem;
    CoordNode& pos;
    vector<Params> params;

    AngleSpring(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        n_elem(get_dset_size(2, grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 3;  // number of atoms that each term depends on 
        check_size(grp, "id",              n_elem, n_dep);
        check_size(grp, "equil_dist",        n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j] = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) { p[i].equil_dp = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("angle_spring"));

        float* posc = pos.output.x.get();
        float* pos_sens = pos.sens.x.get();
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        if(pot) *pot = 0.f;

        for(int nt=0; nt<n_elem; ++nt) {
            auto& p = params[nt];
            auto atom1 = Float4(posc + 4*p.atom[0]);
            auto atom2 = Float4(posc + 4*p.atom[1]);
            auto atom3 = Float4(posc + 4*p.atom[2]);

            auto x1 = atom1 - atom3; auto inv_d1 = inv_mag(x1); auto x1h = x1*inv_d1;
            auto x2 = atom2 - atom3; auto inv_d2 = inv_mag(x2); auto x2h = x2*inv_d2;

            auto dp = dot(x1h, x2h);
            auto force_prefactor = Float4(p.spring_constant) * (dp - Float4(p.equil_dp));

            auto d1 = force_prefactor * (x2h - x1h*dp) * inv_d1;
            auto d2 = force_prefactor * (x1h - x2h*dp) * inv_d2;
            auto d3 = -d1-d2;

            d1.update(pos_sens + 4*p.atom[0]);
            d2.update(pos_sens + 4*p.atom[1]);
            d3.update(pos_sens + 4*p.atom[2]);

            if(pot) *pot += 0.5f * p.spring_constant * sqr(dp.x()-p.equil_dp);
        }
    }
};
static RegisterNodeType<AngleSpring,1> angle_spring_node("angle_spring");


struct DihedralSpring : public PotentialNode
{
    struct Params {
        index_t atom[4];
        float equil_dihedral;
        float spring_constant;
    };

    int n_elem;
    CoordNode& pos;
    vector<Params> params;

    DihedralSpring(hid_t grp, CoordNode& pos_):
        PotentialNode(),
        n_elem(get_dset_size(2, grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 4;  // number of atoms that each term depends on 
        check_size(grp, "id",           n_elem, n_dep);
        check_size(grp, "equil_dist",   n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params;
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) {p[i].atom[j]  =x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) {p[i].equil_dihedral =x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) {p[i].spring_constant=x;});
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("dihedral_spring"));

        float* posc = pos.output.x.get();
        float* pos_sens = pos.sens.x.get();
        float* pot = mode==PotentialAndDerivMode ? &potential : nullptr;
        if(pot) *pot = 0.f;

        for(int nt=0; nt<n_elem; ++nt) {
            const auto& p = params[nt];
            Float4 x[4];
            for(int na: range(4)) x[na] = Float4(posc + 4*params[nt].atom[na]);

            Float4 d[4];
            float dihedral = dihedral_germ(x[0],x[1],x[2],x[3], d[0],d[1],d[2],d[3]).x();

            // determine minimum periodic image (can be off by at most 2pi)
            float displacement = dihedral - p.equil_dihedral;
            displacement = (displacement> M_PI_F) ? displacement-2.f*M_PI_F : displacement;
            displacement = (displacement<-M_PI_F) ? displacement+2.f*M_PI_F : displacement;

            auto s = Float4(p.spring_constant * displacement);
            for(int na: range(4)) d[na].scale_update(s, pos_sens + 4*params[nt].atom[na]);

            if(pot) *pot += 0.5f * p.spring_constant * sqr(displacement);
        }
    }
};
static RegisterNodeType<DihedralSpring,1> dihedral_spring_node("dihedral_spring");


struct ConstantCoord : public CoordNode
{
    VecArrayStorage value;

    ConstantCoord(hid_t grp):
        CoordNode(get_dset_size(2, grp, "value")[0], 
                  get_dset_size(2, grp, "value")[1]),
        value(elem_width, round_up(n_elem,4))
    {
        traverse_dset<2,float>(grp, "value", [&](size_t ne, size_t nd, float x) {
                value(nd,ne) = x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        copy(value, output);
    }

    virtual void propagate_deriv() override {}

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const override {
        vector<float> p; p.reserve(n_elem*elem_width);
        for(int ne: range(n_elem))
            for(int d: range(elem_width))
                p.push_back(value(d,ne));
        return p;
    }

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() != size_t(n_elem*elem_width)) throw string("invalid size to set_param");
        for(int ne: range(n_elem))
            for(int d: range(elem_width))
                value(d,ne) = new_param[ne*elem_width + d];
    }
#endif

};
static RegisterNodeType<ConstantCoord,0> constant_coord_node("constant");

struct Slice : public CoordNode
{
    int n_atom;
    vector<int> id;
    CoordNode& pos;

    Slice(hid_t grp, CoordNode& pos_):
        CoordNode(get_dset_size(1, grp, "id")[0], pos_.elem_width),
        n_atom(n_elem),
        id(n_atom),
        pos(pos_)
    {
        check_size(grp, "id", n_atom);
        traverse_dset<1,int> (grp, "id", [&](size_t i, int x) {id[i] = x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        for (int na = 0; na < n_atom; na++) {
            for (int d = 0; d < elem_width; d++) {
                output(d, na) = pos.output(d, id[na]);
            }
        }
    }

    virtual void propagate_deriv() override {
        for (int na = 0; na < n_atom; na++) {
            for (int d = 0; d < elem_width; d++) {
                pos.sens(d, id[na]) += sens(d, na);
            }
        }
    }
 };
static RegisterNodeType<Slice,1> slice_node("slice");

struct Concat : public CoordNode
{
    vector<CoordNode*> coord_nodes;

    static int sum_n_elem(const vector<CoordNode*>& coord_nodes_) {
        int ne = 0;
        for(auto& cn: coord_nodes_) ne += cn->n_elem;
        return ne;
    }

    Concat(hid_t grp, const std::vector<CoordNode*> &coord_nodes_):
        CoordNode(sum_n_elem(coord_nodes_), coord_nodes[0]->elem_width),
        coord_nodes(coord_nodes_)
    {
        for(auto cn: coord_nodes)
            if(cn->n_elem != coord_nodes[0]->n_elem)
                throw string("Coord node n_elem mismatch");
    }

    virtual void compute_value(ComputeMode mode) override {
        int loc = 0;
        for(auto cn: coord_nodes) {
            int n_elem_cn = cn->n_elem;
            VecArray cn_output = cn->output;
            
            for(int ne=0; ne<n_elem_cn; ++ne){
                for(int nw=0; nw<elem_width; ++nw)
                    output(nw,loc) = cn_output(nw,ne);
                loc++;
            }
        }
        assert(loc==n_elem);
    } 

    virtual void propagate_deriv() override {
        int loc = 0;
        for(auto cn: coord_nodes) {
            int n_elem_cn = cn->n_elem;
            VecArray cn_sens = cn->sens;
            
            for(int ne=0; ne<n_elem_cn; ++ne){
                for(int nw=0; nw<elem_width; ++nw)
                    cn_sens(nw,loc) += sens(nw,ne);
                loc++;
            }
        }
        assert(loc==n_elem);
    }  
}; 
static RegisterNodeType<Concat,-1> concat_node("concat");
