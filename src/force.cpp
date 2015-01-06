#include "backbone_dependent_sidechain.h"
#include "sidechain_radial.h"
#include "hbond.h"
#include "force.h"
#include "md_export.h"
#include "coord.h"
#include "timing.h"
#include "sidechain.h"
#include <map>
#include "gpu.h"
#include "steric.h"

using namespace h5;

using namespace std;

void Pos::propagate_deriv() {
    Timer timer(string("pos_deriv"));
    deriv_accumulation(SysArray(deriv.data(),n_atom*3), 
            slot_machine.accum_array(), slot_machine.deriv_tape.data(), 
            slot_machine.deriv_tape.size(), n_atom, n_system);
}

void DerivEngine::add_node(
        const string& name, 
        unique_ptr<DerivComputation>&& fcn, 
        vector<string> argument_names) 
{
    if(any_of(nodes.begin(), nodes.end(), [&](const Node& n) {return n.name==name;})) 
        throw string("name conflict in DerivEngine");

    nodes.emplace_back(name, move(fcn));
    auto& node = nodes.back();

    for(auto& nm: argument_names) { 
        int parent_idx = get_idx(nm);
        node.parents.push_back(parent_idx);
        nodes[parent_idx].children.push_back(nodes.size()-1);
    }
}

DerivEngine::Node& DerivEngine::get(const string& name) {
    auto loc = find_if(begin(nodes), end(nodes), [&](const Node& n) {return n.name==name;});
    if(loc == nodes.end()) throw string("name not found");
    return *loc;
}

int DerivEngine::get_idx(const string& name, bool must_exist) {
    auto loc = find_if(begin(nodes), end(nodes), [&](const Node& n) {return n.name==name;});
    if(must_exist && loc == nodes.end()) throw string("name not found");
    return loc != nodes.end() ? loc-begin(nodes) : -1;
}

void DerivEngine::compute() {
    // FIXME add CUDA streams support
    // for each BFS depth, each computation gets the stream ID of its parent
    // if the parents have multiple streams or another (sibling) computation has already taken the stream id,
    //   the computation gets a new stream ID and a synchronization wait event
    // if the node has multiple children, it must record a compute_germ event for synchronization of the children
    // the compute_deriv must run in the same stream as compute_germ
    // if a node has multiple parents, it must record an event for compute_deriv
    // pos_node is special since its compute_germ is empty

    for(auto& n: nodes) n.germ_exec_level = n.deriv_exec_level = -1;

    // BFS traversal
    for(int lvl=0, not_finished=1; ; ++lvl, not_finished=0) {
        for(auto& n: nodes) {
            if(n.germ_exec_level == -1) {
                not_finished = 1;
                bool all_parents = all_of(begin(n.parents), end(n.parents), [&] (int ip) {
                        int exec_lvl = nodes[ip].germ_exec_level;
                        return exec_lvl!=-1 && exec_lvl!=lvl; // do not execute at same level as your parents
                        });
                if(all_parents) {
                    n.computation->compute_germ();
                    n.germ_exec_level = lvl;
                }
            }

            if(n.deriv_exec_level == -1 && n.germ_exec_level != -1) {
                not_finished = 1;
                bool all_children = all_of(begin(n.children), end(n.children), [&] (int ip) {
                        int exec_lvl = nodes[ip].deriv_exec_level;
                        return exec_lvl!=-1 && exec_lvl!=lvl; // do not execute at same level as your children
                        });
                if(all_children) {
                    n.computation->propagate_deriv();
                    n.deriv_exec_level = lvl;
                }
            }
        }
        if(!not_finished) break;
    }
}


void DerivEngine::integration_cycle(float* mom, float dt, float max_force, IntegratorType type) {
    // integrator from Predescu et al., 2012
    // http://dx.doi.org/10.1080/00268976.2012.681311

    float a = (type==Predescu) ? 0.108991425403425322 : 1./6.;
    float b = (type==Predescu) ? 0.290485609075128726 : 1./3.;

    float mom_update[] = {1.5f-3.f*a, 1.5f-3.f*a, 6.f*a};
    float pos_update[] = {     3.f*b, 3.0f-6.f*b, 3.f*b};

    for(int stage=0; stage<3; ++stage) {
        compute();   // compute derivatives
        Timer timer(string("integration"));
        integration_stage( 
                SysArray(mom,                pos->n_atom*3), 
                SysArray(pos->output.data(), pos->n_atom*3), 
                SysArray(pos->deriv.data(),  pos->n_atom*3),
                dt*mom_update[stage], dt*pos_update[stage], max_force, 
                pos->n_atom, pos->n_system);
    }
}


struct PosSpring : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    shared_vector<PosSpringParams> params;

    PosSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<1>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 2;  // number of atoms that each term depends on 
        check_size(grp, "id", n_elem, n_dep);
        for(auto& nm: {"x","y","z","spring_const"}) check_size(grp, nm, n_elem);

        auto& p = params.host();
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "x",            [&](size_t i,           float x) { p[i].x = x;});
        traverse_dset<1,float>(grp, "y",            [&](size_t i,           float x) { p[i].y = x;});
        traverse_dset<1,float>(grp, "z",            [&](size_t i,           float x) { p[i].z = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("pos_spring")); 
        pos_spring(pos.coords(), params.host().data(), n_elem, pos.n_system);}
};


struct ZFlatBottom : public DerivComputation
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

    virtual void compute_germ() {
        Timer timer(string("z_flat_bottom"));
        z_flat_bottom_spring(pos.coords(), params.data(), n_term, pos.n_system);
    }
};


struct DistSpring : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    shared_vector<DistSpringParams> params;

    DistSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 2;  // number of atoms that each term depends on 
        check_size(grp, "id",              n_elem, n_dep);
        check_size(grp, "equil_dist",      n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params.host();
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) { p[i].equil_dist = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("dist_spring"));
        dist_spring(pos.coords(), params.host().data(), n_elem, pos.n_system);}
};


struct AngleSpring : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    shared_vector<AngleSpringParams> params;

    AngleSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 3;  // number of atoms that each term depends on 
        check_size(grp, "id",              n_elem, n_dep);
        check_size(grp, "equil_dist",        n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params.host();
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) { p[i].atom[j].index = x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) { p[i].equil_dp = x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) { p[i].spring_constant = x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("angle_spring"));
        angle_spring(pos.coords(), params.host().data(), n_elem, pos.n_system);}
};

struct DihedralSpring : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    shared_vector<DihedralSpringParams> params;

    DihedralSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_elem)
    {
        int n_dep = 4;  // number of atoms that each term depends on 
        check_size(grp, "id",           n_elem, n_dep);
        check_size(grp, "equil_dist",   n_elem);
        check_size(grp, "spring_const", n_elem);

        auto& p = params.host();
        traverse_dset<2,int>  (grp, "id",           [&](size_t i, size_t j, int   x) {p[i].atom[j].index  =x;});
        traverse_dset<1,float>(grp, "equil_dist",   [&](size_t i,           float x) {p[i].equil_dihedral =x;});
        traverse_dset<1,float>(grp, "spring_const", [&](size_t i,           float x) {p[i].spring_constant=x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("dihedral_spring"));
        dihedral_spring(pos.coords(), params.host().data(), n_elem, pos.n_system);
    }
};


struct DynamicDihedralSpring : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    int params_offset;
    shared_vector<DihedralSpringParams> params;  // separate params for each system, id's must be the same

    DynamicDihedralSpring(hid_t grp, CoordNode& pos_):
    n_elem(get_dset_size<2>(grp, "id")[0]), pos(pos_), params_offset(n_elem), params(pos.n_system*params_offset)
    {
        int n_dep = 4;  // number of atoms that each term depends on 
        check_size(grp, "id", n_elem, n_dep, pos.n_system);  // only id is required for dynamic spring

        auto& p = params.host();
        traverse_dset<2,int>  (grp, "id", [&](size_t nt, size_t na, int x) {
                for(int ns=0; ns<pos.n_system; ++ns) 
                    p[ns*n_elem+nt].atom[na].index = x;});

        for(auto& p: params.host()) {
            p.equil_dihedral  = 0.f;
            p.spring_constant = 0.f;
        }

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<p.size(); ++i) pos.slot_machine.add_request(1, p[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("dynamic_dihedral_spring"));
        dynamic_dihedral_spring(pos.coords(), params.host().data(), params_offset, n_elem, pos.n_system);
    }
};

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

    virtual void compute_germ() {
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


struct HMMPot : public DerivComputation
{
    int n_residue;
    CoordNode& pos;
    shared_vector<HMMParams> params;
    int n_bin;
    shared_vector<float>       trans_matrices;
    shared_vector<RamaMapGerm> rama_maps;

    HMMPot(hid_t grp, CoordNode& pos_):
    n_residue(get_dset_size<2>(grp, "id")[0]), pos(pos_), params(n_residue)
    {
        int n_dep = 5;  // number of atoms that each term depends on 
        n_bin     = get_dset_size<5>(grp, "rama_deriv")[2];
        rama_maps.host().resize(n_residue*N_STATE*n_bin*n_bin);

        check_size(grp, "id",             n_residue,   n_dep);
        check_size(grp, "trans_matrices", n_residue-1, N_STATE, N_STATE);
        check_size(grp, "rama_deriv",     n_residue,   N_STATE, n_bin, n_bin, 3);

        traverse_dset<2,int>  (grp, "id",             [&](size_t i, size_t j, int   x) { params.host()[i].atom[j].index = x;});
        traverse_dset<3,float>(grp, "trans_matrices", [&](size_t i, size_t j, size_t k, float x) {trans_matrices.host().push_back(x);});
        traverse_dset<5,float>(grp, "rama_deriv",     [&](size_t i, size_t ns, size_t k, size_t l, size_t m, float x) {
                if(m==0) rama_maps.host()[i*n_bin*n_bin + k*n_bin + l].val [ns] = x;
                if(m==1) rama_maps.host()[i*n_bin*n_bin + k*n_bin + l].dphi[ns] = x;
                if(m==2) rama_maps.host()[i*n_bin*n_bin + k*n_bin + l].dpsi[ns] = x;
                });

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<params.size(); ++i) pos.slot_machine.add_request(1, params.host()[i].atom[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("rama_hmm"));
        hmm(pos.coords(), params.host().data(),
                trans_matrices.host().data(), n_bin, rama_maps.host().data(), 
                n_residue, pos.n_system);
    }
};


struct AffineAlignment : public CoordNode
{
    CoordNode& pos;
    int n_system;
    shared_vector<AffineAlignmentParams> params;
    shared_vector<AutoDiffParams> autodiff_params;

    AffineAlignment(hid_t grp, CoordNode& pos_):
        CoordNode(pos_.n_system, get_dset_size<2>(grp, "atoms")[0], 7),
        pos(pos_), params(n_elem)
    {
        int n_dep = 3;
        check_size(grp, "atoms",    n_elem, n_dep);
        check_size(grp, "ref_geom", n_elem, n_dep,3);

        traverse_dset<2,int  >(grp,"atoms",   [&](size_t i,size_t j,          int   x){params.host()[i].atom[j].index=x;});
        traverse_dset<3,float>(grp,"ref_geom",[&](size_t i,size_t na,size_t d,float x){params.host()[i].ref_geom[na*n_dep+d]=x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<params.size(); ++i) pos.slot_machine.add_request(7, params.host()[i].atom[j]);
        for(auto &p: params.host()) autodiff_params.host().push_back(AutoDiffParams({p.atom[0].slot, p.atom[1].slot, p.atom[2].slot}));
    }

    virtual void compute_germ() {
        Timer timer(string("affine_alignment"));
        affine_alignment(coords().value, pos.coords(), params.host().data(), 
                n_elem, pos.n_system);}

    virtual void propagate_deriv() {
        Timer timer(string("affine_alignment_deriv"));
        affine_reverse_autodiff(
                coords().value, coords().deriv, pos.slot_machine.accum_array(), 
                slot_machine.deriv_tape.data(), autodiff_params.host().data(), 
                slot_machine.deriv_tape.size(), 
                n_elem, pos.n_system);}
};


struct BackboneDependentPoint : public CoordNode
{
    CoordNode& rama;
    CoordNode& alignment;

    int n_restype;
    int n_bin;
    vector<float> backbone_point_map_data;

    vector<BackboneSCParam> params;
    vector<AutoDiffParams> autodiff_params;

    BackboneDependentPoint(hid_t grp, CoordNode& rama_, CoordNode& alignment_):
        CoordNode(rama_.n_system, get_dset_size<1>(grp, "restype")[0], 3),
        rama(rama_), alignment(alignment_),
        n_restype(get_dset_size<5>(grp, "backbone_point_map")[0]),
        n_bin    (get_dset_size<5>(grp, "backbone_point_map")[1]),
        backbone_point_map_data(n_restype*n_bin*n_bin*3*3, 0.f),
        params(n_elem)
    {
        check_elem_width(rama,     2);
        check_elem_width(alignment, 7);

        check_size(grp, "rama_residue",       n_elem);
        check_size(grp, "alignment_residue",  n_elem);
        check_size(grp, "restype",            n_elem);
        check_size(grp, "backbone_point_map", n_restype, n_bin, n_bin, 3, 3);


        traverse_dset<1,int>(grp, "rama_residue",      [&](size_t nr, int x) {params[nr].rama_residue.index = x;});
        traverse_dset<1,int>(grp, "alignment_residue", [&](size_t nr, int x) {params[nr].alignment_residue.index = x;});
        traverse_dset<1,int>(grp, "restype",           [&](size_t nr, int x) {params[nr].restype = x;});

        traverse_dset<5,float>(grp, "backbone_point_map", 
                [&](size_t rt, size_t nb1, size_t nb2, size_t val_or_dphi_or_dpsi, size_t d, float x) {
                backbone_point_map_data[(((rt*n_bin + nb1)*n_bin + nb2)*3 + val_or_dphi_or_dpsi)*3 + d] = x;});

        for(size_t i=0; i<params.size(); ++i) rama     .slot_machine.add_request(3, params[i].rama_residue);
        for(size_t i=0; i<params.size(); ++i) alignment.slot_machine.add_request(3, params[i].alignment_residue);

        for(auto &p: params) autodiff_params.push_back(
                AutoDiffParams({p.rama_residue.slot}, {p.alignment_residue.slot}));
    }

    virtual void compute_germ() {
        Timer timer(string("backbone_point"));
        BackbonePointMap bb_map;
        bb_map.n_bin = n_bin;
        bb_map.germ = backbone_point_map_data.data();

        backbone_dependent_point(
                coords().value, rama.coords(), alignment.coords(),
                params.data(), bb_map, n_elem, alignment.n_system);
    }

    virtual void propagate_deriv() {
        Timer timer(string("backbone_point_deriv"));
        reverse_autodiff<3,2,6>(
                slot_machine.accum_array(), 
                rama.slot_machine.accum_array(), alignment.slot_machine.accum_array(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_elem, alignment.n_system);
    }
};

template <int ndim>
struct ComputeMyDeriv {
    const float* accum;
    vector<StaticCoord<ndim>> deriv;

    ComputeMyDeriv(const std::vector<float> &accum_, int n_atom):
        accum(accum_.data()), deriv(n_atom)
    {zero();}

    void add_pair(const CoordPair &cp) { deriv.at(cp.index) += StaticCoord<ndim>(accum, cp.slot); }
    void zero() { for(auto &a: deriv) for(int i=0; i<ndim; ++i) a.v[i] = 0.f; }
    void print(const char* nm) { 
        for(int nr=0; nr<(int)deriv.size(); ++nr) {
            printf("%-12s %2i", nm,nr);
            for(int i=0; i<ndim; ++i) {
                if(i%3==0) printf(" ");
                printf(" % 6.2f", deriv[nr].v[i]);
            }
            printf("\n");
        }
        printf("\n");
    }
};


struct BackbonePairs : public DerivComputation
{
    int n_residue;
    CoordNode& alignment;
    vector<AffineParams> params;
    vector<PackedRefPos> ref_pos;
    float energy_scale;
    float dist_cutoff;

    BackbonePairs(hid_t grp, CoordNode& alignment_):
        n_residue(get_dset_size<1>(grp, "id")[0]), alignment(alignment_), 
        params(n_residue), ref_pos(n_residue),
        energy_scale(read_attribute<float>(grp, ".", "energy_scale")),
        dist_cutoff (read_attribute<float>(grp, ".", "dist_cutoff"))
    {
        check_elem_width(alignment, 7);

        check_size(grp, "id",      n_residue);
        check_size(grp, "ref_pos", n_residue, 4, 3);

        traverse_dset<1,int  >(grp, "id", [&](size_t nr, int x) {params[nr].residue.index = x;});

        // read and pack reference positions
        float tmp[3];
        traverse_dset<3,float>(grp, "ref_pos", [&](size_t nr, size_t na, size_t d, float x) {
                tmp[d] = x;
                if(d==2) ref_pos[nr].pos[na] = pack_atom(tmp);
                });
        for(auto &rp: ref_pos) 
            rp.n_atoms = count_if(rp.pos, rp.pos+4, [](uint32_t i) {return i != uint32_t(-1);});

        for(size_t nr=0; nr<params.size(); ++nr) alignment.slot_machine.add_request(1, params[nr].residue);
    }

    virtual void compute_germ() {
        Timer timer(string("backbone_pairs"));
        backbone_pairs(
                alignment.coords(), 
                ref_pos.data(), params.data(), energy_scale, dist_cutoff, n_residue, 
                alignment.n_system);
    }
};


struct DihedralRange : public DerivComputation
{
    int n_elem;
    CoordNode& pos;
    vector<DihedralRangeParams> params; // in sidechain_radial.h 
    DihedralRange(hid_t grp, CoordNode& pos_):
        n_elem(get_dset_size<2>(grp, "id")[0]),
        pos(pos_),
        params(n_elem)
    {
        check_size(grp, "id",          n_elem, 4);
        check_size(grp, "angle_range", n_elem, 2);
        check_size(grp, "scale",       n_elem);
        check_size(grp, "energy",      n_elem);

        traverse_dset<2,int  >(grp, "id",          [&](size_t nda, size_t i, int   x) {
            if(x<0 || x>=pos.n_elem) throw string("illegal atom number ") + to_string(x);
            params[nda].atom[i].index = x;});
        traverse_dset<2,float>(grp, "angle_range", [&](size_t nda, size_t i, float x) {params[nda].angle_range[i]= x;});
        traverse_dset<1,float>(grp, "scale",       [&](size_t nda, float x) {params[nda].scale  = x;});
        traverse_dset<1,float>(grp, "energy",      [&](size_t nda, float x) {params[nda].energy = x;});

        for(int j=0; j<4; ++j)
            for(size_t i=0; i<params.size(); ++i)
                pos.slot_machine.add_request(1, params[i].atom[j]);

    }
    virtual void compute_germ() {
            Timer timer(string("dihedral_range"));
            dihedral_angle_range(pos.coords(), params.data(),
                           n_elem, pos.n_system);
    }

};

struct ContactEnergy : public DerivComputation
{
    int n_contact;
    CoordNode& alignment;
    vector<ContactPair> params;
    float cutoff;

    ContactEnergy(hid_t grp, CoordNode& alignment_):
        n_contact(get_dset_size<2>(grp, "id")[0]),
        alignment(alignment_), 
        params(n_contact),
        cutoff(read_attribute<float>(grp, ".", "cutoff"))
    {
        check_elem_width(alignment, 7);

        check_size(grp, "id",         n_contact, 2);
        check_size(grp, "sc_ref_pos", n_contact, 2, 3);
        check_size(grp, "r0",         n_contact);
        check_size(grp, "scale",      n_contact);
        check_size(grp, "energy",     n_contact);

        traverse_dset<2,int  >(grp, "id",         [&](size_t nc, size_t i, int x) {params[nc].loc[i].index = x;});
        traverse_dset<3,float>(grp, "sc_ref_pos", [&](size_t nc, size_t i, size_t d, float x) {
                component(params[nc].sc_ref_pos[i], d) = x;});

        traverse_dset<1,float>(grp, "r0",     [&](size_t nc, float x) {params[nc].r0     = x;});
        traverse_dset<1,float>(grp, "scale",  [&](size_t nc, float x) {params[nc].scale  = x;});
        traverse_dset<1,float>(grp, "energy", [&](size_t nc, float x) {params[nc].energy = x;});

        for(int j=0; j<2; ++j) 
            for(size_t i=0; i<params.size(); ++i) 
                alignment.slot_machine.add_request(1, params[i].loc[j]);
    }

    virtual void compute_germ() {
        Timer timer(string("contact_energy"));
        contact_energy(
                alignment.coords(), params.data(), 
                n_contact, cutoff, alignment.n_system);
    }
};


struct SidechainRadialPairs : public DerivComputation
{
    map<string,int> name_map;
    int n_residue;
    int n_type;
    CoordNode& bb_point;

    vector<SidechainRadialParams> params;
    vector<SidechainRadialInteraction> interaction_params;
    float cutoff;

    SidechainRadialPairs(hid_t grp, CoordNode& bb_point_):
        n_residue(get_dset_size<1>(grp, "id"   )[0]), 
        n_type   (get_dset_size<1>(grp, "data/names")[0]),
        bb_point(bb_point_), 

        params(n_residue), interaction_params(n_type*n_type),
        cutoff(read_attribute<float>(grp, "data", "cutoff"))
    {
        check_elem_width(bb_point, 3);

        check_size(grp, "id",      n_residue);
        check_size(grp, "restype", n_residue);

        check_size(grp, "data/names",      n_type);
        check_size(grp, "data/energy",     n_type, n_type);
        check_size(grp, "data/scale",      n_type, n_type);
        check_size(grp, "data/r0_squared", n_type, n_type);

        int i=0; 
        traverse_string_dset<1>(grp, "data/names", [&](size_t nt, std::string &s) {name_map[s]=i++;});
        if(i!=n_type) throw std::string("internal error");

        traverse_dset<2,float>(grp, "data/energy", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].energy = x;});
        traverse_dset<2,float>(grp, "data/scale", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].scale = x;});
        traverse_dset<2,float>(grp, "data/r0_squared", [&](size_t rt1, size_t rt2, float x) {
                interaction_params[rt1*n_type+rt2].r0_squared = x;});

        traverse_dset<1,int   >(grp, "id",      [&](size_t nr, int x) {params[nr].loc.index = x;});
        traverse_string_dset<1>(grp, "restype", [&](size_t nr, std::string &s) {
                if(name_map.find(s) == end(name_map)) std::string("restype contains name not found in data/");
                params[nr].restype    = name_map[s];
            });

        for(size_t nr=0; nr<params.size(); ++nr) bb_point.slot_machine.add_request(1, params[nr].loc);
    }

    virtual void compute_germ() {
        Timer timer(string("radial_pairs"));
        radial_pairs(
                bb_point.coords(), 
                params.data(), interaction_params.data(), n_type, cutoff, 
                n_residue, bb_point.n_system);
    }
};

struct Infer_H_O : public CoordNode
{
    CoordNode& pos;
    int n_donor, n_acceptor, n_virtual;
    vector<VirtualParams> params;
    vector<AutoDiffParams> autodiff_params;

    Infer_H_O(hid_t grp, CoordNode& pos_):
        CoordNode(pos_.n_system, 
                get_dset_size<2>(grp, "donors/id")[0]+get_dset_size<2>(grp, "acceptors/id")[0], 6),
        pos(pos_), n_donor(get_dset_size<2>(grp, "donors/id")[0]), n_acceptor(get_dset_size<2>(grp, "acceptors/id")[0]),
        n_virtual(n_donor+n_acceptor), params(n_virtual)
    {
        int n_dep = 3;
        auto don = h5_obj(H5Gclose, H5Gopen2(grp, "donors",    H5P_DEFAULT));
        auto acc = h5_obj(H5Gclose, H5Gopen2(grp, "acceptors", H5P_DEFAULT));
        
        check_size(don.get(), "id",          n_donor,    n_dep);
        check_size(don.get(), "bond_length", n_donor);
        check_size(acc.get(), "id",          n_acceptor, n_dep);
        check_size(acc.get(), "bond_length", n_acceptor);

        traverse_dset<2,int  >(don.get(),"id",          [&](size_t i,size_t j, int   x){params[        i].atom[j].index=x;});
        traverse_dset<1,float>(don.get(),"bond_length", [&](size_t i,          float x){params[        i].bond_length  =x;});
        traverse_dset<2,int  >(acc.get(),"id",          [&](size_t i,size_t j, int   x){params[n_donor+i].atom[j].index=x;});
        traverse_dset<1,float>(acc.get(),"bond_length", [&](size_t i,          float x){params[n_donor+i].bond_length  =x;});

        for(int j=0; j<n_dep; ++j) for(size_t i=0; i<params.size(); ++i) pos.slot_machine.add_request(6, params[i].atom[j]);
        for(auto &p: params) autodiff_params.push_back(AutoDiffParams({p.atom[0].slot, p.atom[1].slot, p.atom[2].slot}));
    }

    CoordArray coords() {
        return CoordArray(SysArray(output.data(), n_virtual*6), slot_machine.accum_array());
    }

    virtual void compute_germ() {
        Timer timer(string("infer_H_O"));
        infer_HN_OC_pos_and_dir(
                SysArray(output.data(),n_virtual*6), pos.coords(), 
                params.data(), n_virtual, pos.n_system);
    }

    virtual void propagate_deriv() {
        Timer timer(string("infer_H_O_deriv"));
        reverse_autodiff<6,3,0>(
                slot_machine.accum_array(), 
                pos.slot_machine.accum_array(), SysArray(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_virtual, pos.n_system);}
};


struct HBondEnergy : public DerivComputation
{
    int n_donor, n_acceptor;
    CoordNode& infer;
    vector<VirtualHBondParams> don_params;
    vector<VirtualHBondParams> acc_params;
    float hbond_energy;
    float n_hbond;

    HBondEnergy(hid_t grp, CoordNode& infer_):
        n_donor   (get_dset_size<1>(grp, "donors/residue_id")[0]), 
        n_acceptor(get_dset_size<1>(grp, "acceptors/residue_id")[0]),
        infer(infer_), 
        don_params(n_donor), acc_params(n_acceptor), 
        hbond_energy(        read_attribute<float>(grp, ".", "hbond_energy")), 
        n_hbond(-1.f)
    {
        check_elem_width(infer, 6);

        auto don = h5_obj(H5Gclose, H5Gopen2(grp, "donors",    H5P_DEFAULT));
        auto acc = h5_obj(H5Gclose, H5Gopen2(grp, "acceptors", H5P_DEFAULT));
        
        check_size(don.get(), "residue_id",     n_donor);
        check_size(don.get(), "helix_energy_bonus", n_donor);

        check_size(acc.get(), "residue_id",     n_acceptor);
        check_size(acc.get(), "helix_energy_bonus", n_acceptor);

        traverse_dset<1,float>(don.get(),"residue_id",        [&](size_t i, float x){don_params[i].residue_id        =x;});
        traverse_dset<1,float>(don.get(),"helix_energy_bonus",[&](size_t i, float x){don_params[i].helix_energy_bonus=x;});

        traverse_dset<1,float>(acc.get(),"residue_id",        [&](size_t i, float x){acc_params[i].residue_id        =x;});
        traverse_dset<1,float>(acc.get(),"helix_energy_bonus",[&](size_t i, float x){acc_params[i].helix_energy_bonus=x;});

        for(int nd=0; nd<n_donor;    ++nd) don_params[nd].id.index = nd;
        for(int na=0; na<n_acceptor; ++na) acc_params[na].id.index = na + n_donor;

        for(auto &p: don_params) infer.slot_machine.add_request(1, p.id);
        for(auto &p: acc_params) infer.slot_machine.add_request(1, p.id);
    }

    virtual void compute_germ() {
        Timer timer(string("hbond_energy"));
        n_hbond = (1.f/hbond_energy) * count_hbond(
                infer.coords(), 
                n_donor, don_params.data(), n_acceptor, acc_params.data(),
                hbond_energy, infer.n_system);
    }
};


struct StericInteraction : public DerivComputation
{
    int n_res;
    CoordNode&  alignment;
    map<string,int>   name_map;

    vector<StericParams>  params;
    vector<StericResidue> ref_res;
    vector<StericPoint>   ref_point;
    Interaction           pot;
    vector<int>           point_starts;

    void pushback_residue(hid_t grp) {
        ref_res.push_back(StericResidue());  auto& r = ref_res.back();
        r.start_point = ref_point.size();
        r.n_pts = get_dset_size<1>(grp, "weight")[0];

        for(int np=0; np<r.n_pts; ++np) ref_point.push_back(StericPoint());

        check_size(grp, "point",  r.n_pts, 3);
        check_size(grp, "weight", r.n_pts);
        check_size(grp, "type",   r.n_pts);

        traverse_dset<2,float>(grp, "point",  [&](size_t np, size_t dim, float v) {
                switch(dim) {
                    case 0: ref_point[r.start_point+np].pos.x = v; break;
                    case 1: ref_point[r.start_point+np].pos.y = v; break;
                    case 2: ref_point[r.start_point+np].pos.z = v; break;
                }});

        traverse_dset<1,float>(grp, "weight", [&](size_t np, float v) {
                ref_point[r.start_point+np].weight = v;});

        traverse_dset<1,int>  (grp, "type", [&](size_t np, int v) {
                ref_point[r.start_point+np].type = v;});

        float3 center = make_float3(0.f,0.f,0.f);
        for(int np=0; np<r.n_pts; ++np) center += ref_point[r.start_point+np].pos;
        center *= 1.f/r.n_pts;

        float  radius = 0.f;
        for(int np=0; np<r.n_pts; ++np) radius += mag2(ref_point[r.start_point+np].pos - center);
        radius = sqrtf(radius/r.n_pts);

        r.center = center;
        r.radius = radius;
    }

    StericInteraction(hid_t grp, CoordNode& alignment_):
        n_res(h5::get_dset_size<1>(grp, "restype")[0]), alignment(alignment_),
        params(n_res),
        pot(get_dset_size<2>     (grp, "atom_interaction/cutoff")[0], 
            get_dset_size<3>     (grp, "atom_interaction/potential")[2], 
            read_attribute<float>(grp, "atom_interaction", "dx")) {
            check_elem_width(alignment, 7);

            traverse_string_dset<1>(grp, "restype", [&](size_t nr, std::string &s) {
                if(name_map.find(s) == end(name_map)) {
                   pushback_residue(
                       h5_obj(H5Gclose, 
                           H5Gopen2(grp, (string("residue_data/")+s).c_str(), H5P_DEFAULT)).get());
                   name_map[s] = ref_res.size()-1;
                }
                params[nr].loc.index  = nr;
                params[nr].restype = name_map[s];
                });

            // Parse potential
            check_size(grp, "atom_interaction/cutoff",      pot.n_types, pot.n_types);
            check_size(grp, "atom_interaction/potential",   pot.n_types, pot.n_types, pot.n_bin);
            check_size(grp, "atom_interaction/deriv_over_r",pot.n_types, pot.n_types, pot.n_bin);

            traverse_dset<2,float>(grp,"atom_interaction/cutoff", [&](size_t tp1, size_t tp2, float c) {
                    pot.cutoff2[tp1*pot.n_types+tp2]=c*c;});
            traverse_dset<3,float>(grp,"atom_interaction/potential", [&](size_t rt1,size_t rt2,size_t nb,float x) {
                    int loc = rt1*pot.n_types*pot.n_bin + rt2*pot.n_bin + nb;
                    pot.germ_arr[loc].x = x;
                    if(nb>0) pot.germ_arr[loc-1].y = x;
                    });
            traverse_dset<3,float>(grp,"atom_interaction/deriv_over_r", [&](size_t rt1,size_t rt2,size_t nb,float x){
                    int loc = rt1*pot.n_types*pot.n_bin + rt2*pot.n_bin + nb;
                    pot.germ_arr[loc].z = x;
                    if(nb>0) pot.germ_arr[loc-1].w = x;
                    });

            pot.largest_cutoff = 0.f;
            for(int rt=0; rt<pot.n_types; ++rt) 
                pot.largest_cutoff = max(pot.largest_cutoff, sqrtf(pot.cutoff2[rt]));

            // determine location to write each residue's points
            point_starts.push_back(0);  // first residue starts at 0
            for(auto& p: params)
                point_starts.push_back(point_starts.back()+ref_res.at(p.restype).n_pts);

            if(n_res!= alignment.n_elem) throw string("invalid restype array");
            for(int nr=0; nr<n_res; ++nr) alignment.slot_machine.add_request(1,params[nr].loc);
        }

    virtual void compute_germ() {
        Timer timer(string("steric"));
        steric_pairs(
                alignment.coords(),
                params.data(),
                ref_res.data(),
                ref_point.data(),
                pot,
                point_starts.data(),
                n_res, alignment.n_system);
    }

};


struct SidechainInteraction : public DerivComputation 
{
    int n_residue;
    CoordNode&  alignment;
    vector<Sidechain> sidechain_params;
    float             dist_cutoff;
    float             energy_cutoff;
    map<string,int>   name_map;
    vector<float>     density_data;
    vector<float4>    center_data;
    vector<SidechainParams> params;

    SidechainInteraction(hid_t grp, CoordNode& alignment_):
        n_residue(h5::get_dset_size<1>(grp, "restype")[0]), alignment(alignment_),
        energy_cutoff(h5::read_attribute<float>(grp, "./sidechain_data", "energy_cutoff")),
        params(n_residue)
    {
        traverse_string_dset<1>(grp, "restype", [&](size_t nr, std::string &s) {
                if(name_map.find(s) == end(name_map)) {
                    sidechain_params.push_back(
                        parse_sidechain(energy_cutoff,
                            h5_obj(H5Gclose, 
                                H5Gopen2(grp, (string("sidechain_data/")+s).c_str(), H5P_DEFAULT)).get()));
                    name_map[s] = sidechain_params.size()-1;
                }
                params[nr].res.index  = nr;
                params[nr].restype = name_map[s];
            });

        // find longest possible interaction
        float max_interaction_radius = 0.f;
        float max_density_radius = 0.f;
        for(auto &sc: sidechain_params) {
            // FIXME is the sidechain data in the correct reference frame
            float interaction_maxdist = sc.interaction_radius+mag(sc.interaction_center);
            float density_maxdist = sc.density_radius+mag(sc.density_center);
            max_interaction_radius = max(max_interaction_radius, interaction_maxdist);
            max_density_radius = max(max_density_radius, density_maxdist);
            printf("%4.1f %4.1f %4.1f %4.1f\n", 
                    mag(sc.interaction_center), sc.interaction_radius, 
                    mag(sc.density_center), sc.density_radius);
        }
        dist_cutoff = max_interaction_radius + max_density_radius;
        printf("total_cutoff: %4.1f\n", dist_cutoff);

        if(n_residue != alignment.n_elem) throw string("invalid restype array");
        for(int nr=0; nr<n_residue; ++nr) alignment.slot_machine.add_request(1,params[nr].res);
    }

    static Sidechain parse_sidechain(float energy_cutoff, hid_t grp) 
    {
        using namespace h5;
        int nkernel = get_dset_size<2>(grp, "kernels")[0];
        check_size(grp, "kernels", nkernel, 4);

        auto kernels = vector<float4>(nkernel);

        traverse_dset<2,float>(grp, "kernels", [&](size_t nk, size_t dim, float v) {
                switch(dim) {
                case 0: kernels[nk].x = v; break;
                case 1: kernels[nk].y = v; break;
                case 2: kernels[nk].z = v; break;
                case 3: kernels[nk].w = v; break;
                }});

        auto dims = get_dset_size<4>(grp, "interaction");
        check_size(grp, "interaction", dims[0], dims[1], dims[2], 4);

        auto data = vector<float4>(dims[0]*dims[1]*dims[2]);
        traverse_dset<4,float>(grp, "interaction", [&](size_t i, size_t j, size_t k, size_t dim, float v) {
                int idx = i*dims[1]*dims[2] + j*dims[2] + k;
                switch(dim) {
                case 0: data[idx].x = v; break;
                case 1: data[idx].y = v; break;
                case 2: data[idx].z = v; break;
                case 3: data[idx].w = v; break;
                }});

        check_size(grp, "corner_location", 3);
        float3 corner;
        traverse_dset<1,float>(grp, "corner_location", [&](size_t dim, float v) {
                switch(dim) {
                case 0: corner.x = v; break;
                case 1: corner.y = v; break;
                case 2: corner.z = v; break;
                }});

        auto bin_side_length = read_attribute<float>(grp, "interaction", "bin_side_length");

        return Sidechain(
                kernels, 
                Density3D(corner, 1.f/bin_side_length, dims[0],dims[1],dims[2], data),
                energy_cutoff);
    }

    virtual void compute_germ() {
        Timer timer(string("sidechain_pairs"));
        sidechain_pairs(
                alignment.coords(), 
                sidechain_params.data(), params.data(), 
                dist_cutoff, n_residue, alignment.n_system);
    }
};



template <typename ComputationT, typename ArgumentT> 
void attempt_add_node(
        DerivEngine& engine,
        hid_t force_group,
        const string  name,
        const string  argument_name,
        const string* table_name_if_different = nullptr) 
{
    auto table_name = table_name_if_different ? *table_name_if_different : name;
    if(!h5_noerr(H5LTpath_valid(force_group, table_name.c_str(), 1))) return;
    auto grp = h5_obj(H5Gclose, H5Gopen2(force_group, table_name.c_str(), H5P_DEFAULT));
    printf("initializing %s\n", name.c_str());

    // note that the unique_ptr holds the object by a superclass pointer
    try {
        auto& argument = dynamic_cast<ArgumentT&>(*engine.get(argument_name).computation);
        auto computation = unique_ptr<DerivComputation>(new ComputationT(grp.get(), argument));
        engine.add_node(name, move(computation), {argument_name});
    } catch(const string &e) {
        throw "while adding '" + name + "', " + e;
    }
}

template <typename ComputationT, typename Argument1T, typename Argument2T> 
void attempt_add_node(
        DerivEngine& engine,
        hid_t force_group,
        const string  name,
        const string  argument_name1,
        const string  argument_name2,
        const string* table_name_if_different = nullptr) 
{
    auto table_name = table_name_if_different ? *table_name_if_different : name;
    if(!h5_noerr(H5LTpath_valid(force_group, table_name.c_str(), 1))) return;
    auto grp = h5_obj(H5Gclose, H5Gopen2(force_group, table_name.c_str(), H5P_DEFAULT));
    printf("initializing %s\n", name.c_str());

    // note that the unique_ptr holds the object by a superclass pointer
    try {
        auto& argument1 = dynamic_cast<Argument1T&>(*engine.get(argument_name1).computation);
        auto& argument2 = dynamic_cast<Argument2T&>(*engine.get(argument_name2).computation);
        auto computation = unique_ptr<DerivComputation>(new ComputationT(grp.get(), argument1, argument2));

        engine.add_node(name, move(computation), {argument_name1, argument_name2});
    } catch(const string &e) {
        throw "while adding '" + name + "', " + e;
    }
}


DerivEngine initialize_engine_from_hdf5(int n_atom, int n_system, hid_t force_group)
{
    auto engine = DerivEngine(n_atom, n_system);
    auto& m = node_creation_map();

    map<string, vector<string>> dep_graph;
    dep_graph["pos"] = vector<string>();
    for(const auto &name : node_names_in_group(force_group, "."))
        dep_graph[name] = read_attribute<vector<string>>(force_group, name.c_str(), "arguments");

    for(auto &kv : dep_graph) {
        for(auto& dep_name : kv.second) {
            if(dep_graph.find(dep_name) == end(dep_graph)) 
                throw string("Node ") + kv.first + " takes " + dep_name + 
                    " as an argument, but no node of that name can be found.";
        }
    }

    vector<string> topo_order;
    auto in_topo = [&](const string &name) {
        return find(begin(topo_order), end(topo_order), name) != end(topo_order);};

    int graph_size = dep_graph.size();
    for(int round_num=0; round_num<graph_size; ++round_num) {
        for(auto it=begin(dep_graph); it!=end(dep_graph); ++it) {
            if(all_of(begin(it->second), end(it->second), in_topo)) {
                topo_order.push_back(it->first);
                dep_graph.erase(it);
            }
        }
    }
    if(dep_graph.size()) throw string("Unsatisfiable dependency in force computation");

    // using topo_order here ensures that a node is only parsed after all its arguments
    for(auto &nm : topo_order) {
        if(nm=="pos") continue;  // pos node is added specially
        // some name in the node_creation_map must be a prefix of this name
        string node_type_name = "";
        for(auto &kv : m) {
            if(is_prefix(kv.first, nm))
                node_type_name = kv.first;
        }
        if(node_type_name == "") throw string("No node type found for name '") + nm + "'";
        NodeCreationFunction& node_func = m[node_type_name];

        auto argument_names = read_attribute<vector<string>>(force_group, nm.c_str(), "arguments");
        ArgList arguments;
        for(const auto& arg_name : argument_names)  {
            // if the node is not a CoordNode, a null pointer will be returned from dynamic_cast
            arguments.push_back(dynamic_cast<CoordNode*>(engine.get(arg_name).computation.get()));
            if(!arguments.back()) 
                throw arg_name + " is not an intermediate value, but it is an argument of " + nm;
        }

        auto grp = open_group(force_group,nm.c_str());
        auto computation = unique_ptr<DerivComputation>(node_func(grp.get(),arguments));
        engine.add_node(nm, move(computation), argument_names);
    }

    return engine;
}

static RegisterNodeType<PosSpring,1>      _1("atom_pos_spring");
static RegisterNodeType<DistSpring,1>     _2("dist_spring");
static RegisterNodeType<AngleSpring,1>    _3("angle_spring");
static RegisterNodeType<DihedralSpring,1> _4("dihedral_spring");
static RegisterNodeType<DihedralRange,1>  _5("dihedral_range");
static RegisterNodeType<ZFlatBottom,1>    _6("z_flat_bottom");
static RegisterNodeType<HMMPot,1>         _7("rama_hmm_pot");
static RegisterNodeType<AffineAlignment,1>_8("affine_alignment");

static RegisterNodeType<BackbonePairs,1>         _9 ("backbone_pairs");
static RegisterNodeType<SidechainInteraction,1>  _10("sidechain");
static RegisterNodeType<StericInteraction,1>     _11("steric");
static RegisterNodeType<ContactEnergy,1>         _12("contact");
static RegisterNodeType<RamaCoord,1>             _13("rama_coord");
static RegisterNodeType<BackboneDependentPoint,2>_14("backbone_dependent_point");
static RegisterNodeType<SidechainRadialPairs,1>  _15("radial");

static RegisterNodeType<Infer_H_O,1>  _16("infer_H_O");
static RegisterNodeType<HBondEnergy,1>_17("hbond_energy");

NodeCreationMap& node_creation_map() 
{
    static NodeCreationMap m;
    if(!m.size()) {
        m[string("pos")] = NodeCreationFunction([](hid_t grp, const ArgList& arguments) {
                throw string("Cannot create pos-type node");
                return nullptr; });
    }
    return m;
}


double get_n_hbond(DerivEngine &engine) {
    HBondEnergy* hbond_comp = engine.get_idx("hbond_energy",false)==-1 ? 
        nullptr : 
        &engine.get_computation<HBondEnergy>("hbond_energy");
    return hbond_comp?hbond_comp->n_hbond:-1.f;
}
