#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include <math.h>
#include "md_export.h"
#include <algorithm>
#include "state_logger.h"
#include "interaction_graph.h"

using namespace h5;
using namespace std;

struct VirtualParams {
    CoordPair atom[3];
    float bond_length;
};

struct NH_CO_Params {
    float H_bond_length, N_bond_length;
    unsigned int Cprev, N, CA, C, Nnext;
    unsigned int H_slots[9];
    unsigned int O_slots[9];
};

namespace {

inline void
hat_deriv(
        float3 v_hat, float v_invmag, 
        float3 &col0, float3 &col1, float3 &col2) // matrix is symmetric, so these are rows or cols
{
    float s = v_invmag;
    col0 = make_vec3(s*(1.f-v_hat.x()*v_hat.x()), s*    -v_hat.y()*v_hat.x() , s*    -v_hat.z()*v_hat.x() );
    col1 = make_vec3(s*    -v_hat.x()*v_hat.y() , s*(1.f-v_hat.y()*v_hat.y()), s*    -v_hat.z()*v_hat.y() );
    col2 = make_vec3(s*    -v_hat.x()*v_hat.z() , s*    -v_hat.y()*v_hat.z() , s*(1.f-v_hat.z()*v_hat.z()));
}

template <typename MutableCoordT, typename CoordT>
void infer_x_body(
        MutableCoordT &hbond_pos,
        CoordT &prev_c,
        CoordT &curr_c,
        CoordT &next_c,
        float bond_length)
{
    // For output, first three components are position of H/O and second three are HN/OC bond direction
    // Bond direction is a unit vector
    // curr is the N or C atom
    // This algorithm assumes perfect 120 degree bond angles
    
    float3 prev = prev_c.f3() - curr_c.f3(); float prev_invmag = inv_mag(prev); prev *= prev_invmag;
    float3 next = next_c.f3() - curr_c.f3(); float next_invmag = inv_mag(next); next *= next_invmag;
    float3 disp = prev+next;                 float disp_invmag = inv_mag(disp); disp *= disp_invmag;

    float3 pcol0, pcol1, pcol2; hat_deriv(prev, prev_invmag, pcol0, pcol1, pcol2);
    float3 ncol0, ncol1, ncol2; hat_deriv(next, next_invmag, ncol0, ncol1, ncol2);
    float3 drow0, drow1, drow2; hat_deriv(disp, disp_invmag, drow0, drow1, drow2);

    // prev derivatives
    prev_c.set_deriv(3, -make_vec3(dot(drow0,pcol0), dot(drow0,pcol1), dot(drow0,pcol2)));
    prev_c.set_deriv(4, -make_vec3(dot(drow1,pcol0), dot(drow1,pcol1), dot(drow1,pcol2)));
    prev_c.set_deriv(5, -make_vec3(dot(drow2,pcol0), dot(drow2,pcol1), dot(drow2,pcol2)));

    // position derivative is direction derivative times bond length
    for(int no=0; no<3; ++no) for(int nc=0; nc<3; ++nc) prev_c.d[no][nc] = bond_length*prev_c.d[no+3][nc];

    // next derivatives
    next_c.set_deriv(3, -make_vec3(dot(drow0,ncol0), dot(drow0,ncol1), dot(drow0,ncol2)));
    next_c.set_deriv(4, -make_vec3(dot(drow1,ncol0), dot(drow1,ncol1), dot(drow1,ncol2)));
    next_c.set_deriv(5, -make_vec3(dot(drow2,ncol0), dot(drow2,ncol1), dot(drow2,ncol2)));

    // position derivative is direction derivative times bond length
    for(int no=0; no<3; ++no) for(int nc=0; nc<3; ++nc) next_c.d[no][nc] = bond_length*next_c.d[no+3][nc];

    // curr direction derivatives (set by condition of zero next force / translation invariance)
    for(int no=0; no<3; ++no) {
        for(int nc=0; nc<3; ++nc) {
            curr_c.d[no+3][nc] = -prev_c.d[no+3][nc]-next_c.d[no+3][nc];
            curr_c.d[no  ][nc] = (no==nc) + bond_length*curr_c.d[no+3][nc];
        }
    }

    // set values
    hbond_pos.v[0] = curr_c.v[0] - bond_length*disp.x();
    hbond_pos.v[1] = curr_c.v[1] - bond_length*disp.y();
    hbond_pos.v[2] = curr_c.v[2] - bond_length*disp.z();
    hbond_pos.v[3] = -disp.x();
    hbond_pos.v[4] = -disp.y();
    hbond_pos.v[5] = -disp.z();
}

}

void infer_HN_OC_pos_and_dir(
        const SysArray   HN_OC,
        const CoordArray pos,
        const VirtualParams* params,
        int n_term, int n_system)
{
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_term; ++nt) {
            MutableCoord<6> hbond_pos(HN_OC, ns, nt);
            Coord<3,6>      prev_c(pos, ns, params[nt].atom[0]);
            Coord<3,6>      curr_c(pos, ns, params[nt].atom[1]);
            Coord<3,6>      next_c(pos, ns, params[nt].atom[2]);

            infer_x_body(hbond_pos, prev_c,curr_c,next_c, params[nt].bond_length);

            hbond_pos.flush();
            prev_c.flush();
            curr_c.flush();
            next_c.flush();
        }
    }
}


struct Infer_H_O : public CoordNode
{
    CoordNode& pos;
    int n_donor, n_acceptor, n_virtual;
    vector<VirtualParams> params;
    vector<AutoDiffParams> autodiff_params;

    Infer_H_O(hid_t grp, CoordNode& pos_):
        CoordNode(pos_.n_system, 
                get_dset_size(2, grp, "donors/id")[0]+get_dset_size(2, grp, "acceptors/id")[0], 6),
        pos(pos_), n_donor(get_dset_size(2, grp, "donors/id")[0]), n_acceptor(get_dset_size(2, grp, "acceptors/id")[0]),
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

        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("virtual", {n_system, n_elem, 3}, [&](float* buffer) {
                    for(int ns=0; ns<n_system; ++ns) {
                        for(int nv=0; nv<n_virtual; ++nv) {
                            StaticCoord<6> x(coords().value, ns, nv);
                            for(int d=0; d<3; ++d) buffer[ns*n_elem*3 + nv*3 + d] = x.v[d];
                        }
                    }
                });
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("infer_H_O"));
        infer_HN_OC_pos_and_dir(
                coords().value, pos.coords(), 
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

    virtual double test_value_deriv_agreement() {
        return compute_relative_deviation_for_node<3>(*this, pos, extract_pairs(params, potential_term));
    }
};
static RegisterNodeType<Infer_H_O,1>  infer_node("infer_H_O");


#define radial_cutoff2 (3.5f*3.5f)
// angular cutoff at 90 degrees
#define angular_cutoff (0.f)    

float2 hbond_radial_potential(float input) 
{
    const float outer_barrier    = 2.5f;
    const float inner_barrier    = 1.4f;   // barrier to prevent H-O overlap
    const float inv_outer_width  = 1.f/0.125f;
    const float inv_inner_width  = 1.f/0.10f;

    float2 outer_sigmoid = sigmoid((outer_barrier-input)*inv_outer_width);
    float2 inner_sigmoid = sigmoid((input-inner_barrier)*inv_inner_width);

    return make_vec2( outer_sigmoid.x() * inner_sigmoid.x(),
            - inv_outer_width * outer_sigmoid.y() * inner_sigmoid.x()
            + inv_inner_width * inner_sigmoid.y() * outer_sigmoid.x());
}


float2 hbond_angular_potential(float dotp) 
{
    const float wall_dp = 0.682f;  // half-height is at 47 degrees
    const float inv_dp_width = 1.f/0.05f;

    float2 v = sigmoid((dotp-wall_dp)*inv_dp_width);
    return make_vec2(v.x(), inv_dp_width*v.y());
}


float hbond_score(
        float3   H, float3   O, float3   rHN, float3   rOC,
        float3& dH, float3& dO, float3& drHN, float3& drOC)
{
    float3 HO = H-O;
    
    float magHO2 = mag2(HO) + 1e-6; // a bit of paranoia to avoid division by zero later
    float invHOmag = rsqrt(magHO2);
    float magHO    = magHO2 * invHOmag;  // avoid a sqrtf later

    float3 rHO = HO*invHOmag;

    float dotHOC =  dot(rHO,rOC);
    float dotOHN = -dot(rHO,rHN);

    if(!((dotHOC > angular_cutoff) & (dotOHN > angular_cutoff))) {
        dH=dO=drHN=drOC=make_zero<3>();
        return 0.f;
    }

    float2 radial   = hbond_radial_potential (magHO );  // x has val, y has deriv
    float2 angular1 = hbond_angular_potential(dotHOC);
    float2 angular2 = hbond_angular_potential(dotOHN);

    float val =  radial.x() * angular1.x() * angular2.x();
    float c0  =  radial.y() * angular1.x() * angular2.x();
    float c1  =  radial.x() * angular1.y() * angular2.x();
    float c2  = -radial.x() * angular1.x() * angular2.y();

    drOC = c1*rHO;
    drHN = c2*rHO;

    dH = c0*rHO + (c1*invHOmag)*(rOC-dotHOC*rHO) + (c2*invHOmag)*(rHN+dotOHN*rHO);
    dO = -dH;

    return val;
}

// displacement is sc-H
inline float coverage_score(
        float3 displace, float3 rHN, float3& d_displace, float3& drHN, float radius, float scale,
        float cover_angular_cutoff, float cover_angular_scale)
{
    float  dist2 = mag2(displace);
    float  inv_dist = rsqrt(dist2);
    float  dist = dist2*inv_dist;
    float3 displace_unitvec = inv_dist*displace;
    float  cos_coverage_angle = dot(rHN,displace_unitvec);

    float2 radial_cover  = compact_sigmoid(dist-radius, scale);
    float2 angular_cover = compact_sigmoid(cover_angular_cutoff-cos_coverage_angle, cover_angular_scale);

    float3 col0, col1, col2;
    hat_deriv(displace_unitvec, inv_dist, col0, col1, col2);
    float3 deriv_dir = make_vec3(dot(col0,rHN), dot(col1,rHN), dot(col2,rHN));

    d_displace = (angular_cover.x()*radial_cover.y()) * displace_unitvec + (-radial_cover.x()*angular_cover.y()) * deriv_dir;
    drHN = (-radial_cover.x()*angular_cover.y()) * displace_unitvec;

    return radial_cover.x() * angular_cover.x();
}


namespace {
    struct ProteinHBondInteraction {
        // inner_barrier, inner_scale, outer_barrier, outer_scale, wall_dp, inv_dp_width
        // first group is donors; second group is acceptors
        constexpr static const int n_param=6, n_dim1=6, n_dim2=6, n_deriv=12;

        static float cutoff(const Vec<n_param> &p) {
            return sqrtf(radial_cutoff2); // FIXME make parameter dependent
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return false; // no exclusions
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
            float3 dH,dO,drHN,drOC;

            float hb = hbond_score(extract<0,3>(x1), extract<0,3>(x2), extract<3,6>(x1), extract<3,6>(x2),
                                   dH,               dO,               drHN,             drOC);
            float hb_log = hb>=1.f ? -1e10f : -logf(1.f-hb);  // work in multiplicative space

            float deriv_prefactor = min(1.f/(1.f-hb),1e5f); // FIXME this is a mess
            store<0, 3>(d_base, dH   * deriv_prefactor);
            store<3, 6>(d_base, drHN * deriv_prefactor);
            store<6, 9>(d_base, dO   * deriv_prefactor);
            store<9,12>(d_base, drOC * deriv_prefactor);

            return hb_log;
        }

        static void expand_deriv(Vec<n_dim1> &d1, Vec<n_dim2> &d2, const Vec<n_deriv> &d_base) {
            d1 = extract<0, 6>(d_base);
            d2 = extract<6,12>(d_base);
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
            throw "not implemented";
        }
    };


    struct HBondCoverageInteraction {
        // radius scale angular_width angular_scale
        // first group is donors; second group is acceptors
        constexpr static const int n_param=5, n_dim1=7, n_dim2=3, n_deriv=7;

        static float cutoff(const Vec<n_param> &p) {
            return p[0] + 1.f/p[1];
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return false; // no exclusions
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            float3 d_sc, d_rHN;
            float coverage = coverage_score(sc_pos-extract<0,3>(hb_pos), extract<3,6>(hb_pos), 
                    d_sc, d_rHN, p[0], p[1], p[2], p[3]);
            float prefactor = p[4] * sqr(1.f-hb_pos[6]);

            store<0,3>(d_base, prefactor * d_sc);
            store<3,6>(d_base, prefactor * d_rHN);
            d_base[6] = -p[4]*coverage * (1.f-hb_pos[6])*2.f;

            return prefactor * coverage;
        }

        static void expand_deriv(Vec<n_dim1> &d_hb, Vec<n_dim2> &d_sc, const Vec<n_deriv> &d_base) {
            store<0,3>(d_hb, -extract<0,3>(d_base));  // opposite of d_sc by Newton's third
            store<3,6>(d_hb,  extract<3,6>(d_base));
            d_hb[6] = d_base[6];
            store<0,3>(d_sc,  extract<0,3>(d_base));
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            float3 displace = sc_pos-extract<0,3>(hb_pos);
            float3 rHN = extract<3,6>(hb_pos);

            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist = dist2*inv_dist;
            float3 displace_unitvec = inv_dist*displace;
            float  cos_coverage_angle = dot(rHN,displace_unitvec);

            float2 radial_cover  = compact_sigmoid(dist-p[0], p[1]);
            float2 angular_cover = compact_sigmoid(p[2]-cos_coverage_angle, p[3]);

            float radial_cover_s =(dist-p[0])*compact_sigmoid((dist-p[0])*p[1],1.f).y();
            float angular_cover_s=(p[2]-cos_coverage_angle)*compact_sigmoid((p[2]-cos_coverage_angle)*p[3],1.f).y();

            float prefactor = p[4] * sqr(1.f-hb_pos[6]);

            d_param[0] = prefactor * -radial_cover.y() * angular_cover.x();
            d_param[1] = prefactor *  radial_cover_s   * angular_cover.x();
            d_param[2] = prefactor *  radial_cover.x() * angular_cover.y();
            d_param[3] = prefactor *  radial_cover.x() * angular_cover_s;
            d_param[4] = sqr(1.f-hb_pos[6]) * radial_cover.x() * angular_cover.x();
        }
    };
}


struct ProteinHBond : public CoordNode 
{
    CoordNode& infer;
    BetweenInteractionGraph<ProteinHBondInteraction> igraph;
    int n_donor, n_acceptor, n_virtual;

    ProteinHBond(hid_t grp, CoordNode& infer_):
        CoordNode(infer_.n_system, get_dset_size(1,grp,"index1")[0]+get_dset_size(1,grp,"index2")[0], 7),
        infer(infer_),
        igraph(grp, infer, infer) ,
        n_donor   (igraph.n_elem1),
        n_acceptor(igraph.n_elem2),
        n_virtual (n_donor+n_acceptor)
    {
        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("hbond", {n_system,n_donor+n_acceptor}, [&](float* buffer) {
                    for(int ns: range(n_system))
                       for(int nv: range(n_donor+n_acceptor))
                           buffer[ns*(n_donor+n_acceptor) + nv] = coords().value[ns](6,nv);});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("protein_hbond"));

        for(int ns=0; ns<n_system; ++ns) {
            int n_virtual = n_donor + n_acceptor;
            VecArray vs = coords().value[ns];
            VecArray ho = infer.coords().value[ns];

            for(int nv: range(n_virtual)) {
                for(int d: range(6))
                    vs(d,nv) = ho(d,nv);
                vs(6,nv) = 0.f;
            }

            // Compute protein hbonding score and its derivative
            igraph.compute_edges(ns, [&](int ne, float hb_log,
                        int index1, unsigned type1, unsigned id1,
                        int index2, unsigned type2, unsigned id2) {
                    vs(6,index1) += hb_log;
                    vs(6,index2) += hb_log;});

            for(int nv: range(n_virtual)) vs(6,nv) = 1.f-expf(-vs(6,nv));
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("protein_hbond_deriv"));

        for(int ns=0; ns<n_system; ++ns) {
            vector<Vec<7>> sens(n_virtual, make_zero<7>());
            VecArray accum = slot_machine.accum_array()[ns];
            VecArray hb    = coords().value[ns];

           for(auto tape_elem: slot_machine.deriv_tape)
               for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                   sens[tape_elem.atom] += load_vec<7>(accum, tape_elem.loc+rec);

           for(int nv: range(n_virtual))
               sens[nv][6] *= 1.f-hb(6,nv);

           // Push protein HBond derivatives
           for(int ned: range(igraph.n_edge[ns])) {
               int don_idx = igraph.param1[igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 0]].index;
               int acc_idx = igraph.param2[igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 1]].index;
               igraph.use_derivative(ns, ned,  sens[don_idx][6] + sens[acc_idx][6]);
           }

           // pass through derivatives on all other components
           VecArray pd1 = igraph.pos_deriv1[ns];
           for(int nd: range(n_donor)) {
               // the last component is taken care of by the edge loop
               update_vec(pd1, nd, extract<0,6>(sens[nd]));
           }
           VecArray pd2 = igraph.pos_deriv2[ns];
           for(int na: range(n_acceptor)) {  // acceptor loop
               // the last component is taken care of by the edge loop
               update_vec(pd2, na, extract<0,6>(sens[na+n_donor]));
           }

           igraph.propagate_derivatives(ns);
        }
    }
};
static RegisterNodeType<ProteinHBond,1> hbond_node("protein_hbond");


struct HBondCoverage : public CoordNode {
    BetweenInteractionGraph<HBondCoverageInteraction> igraph;
    int n_sc;

    HBondCoverage(hid_t grp, CoordNode& infer_, CoordNode& sidechains_):
        CoordNode(infer_.n_system, get_dset_size(1,grp,"index2")[0], 1),
        igraph(grp, infer_, sidechains_),
        n_sc(igraph.n_elem2) {}

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_coverage"));
        #pragma omp parallel for
        for(int ns=0; ns<n_system; ++ns) {
            VecArray cov = coords().value[ns];
            for(int nc: range(n_sc)) cov(0,nc) = 0.f;

            // Compute coverage and its derivative
            igraph.compute_edges(ns, [&](int ne, float en,
                        int index1, unsigned type1, unsigned id1,
                        int index2, unsigned type2, unsigned id2) {
                    cov(0,index2) += en;});
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("hbond_coverage_deriv"));
        #pragma omp parallel for
        for(int ns=0; ns<n_system; ++ns) {
            vector<float> sens(n_sc, 0.f);
            VecArray accum = slot_machine.accum_array()[ns];

           for(auto tape_elem: slot_machine.deriv_tape)
               for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                   sens[tape_elem.atom] += accum(0, tape_elem.loc+rec);

           // Push coverage derivatives
           for(int ned: range(igraph.n_edge[ns])) {
               int sc_num = igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 1];
               int sc_idx = igraph.param2[sc_num].index;
               igraph.use_derivative(ns, ned, sens[sc_idx]);
           }
           igraph.propagate_derivatives(ns);
        }
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const {return igraph.get_param();}
    virtual std::vector<float> get_param_deriv() const {return igraph.get_param_deriv();}
    virtual void set_param(const std::vector<float>& new_param) {igraph.set_param(new_param);}
#endif
};
static RegisterNodeType<HBondCoverage,2> coverage_node("hbond_coverage");


struct HBondEnergy : public HBondCounter
{
    CoordNode& protein_hbond;
    int n_virtual;
    float E_protein;
    vector<slot_t> slots;

    HBondEnergy(hid_t grp, CoordNode& protein_hbond_):
        HBondCounter(protein_hbond_.n_system),
        protein_hbond(protein_hbond_),
        n_virtual(protein_hbond.n_elem),
        E_protein(read_attribute<float>(grp, ".", "protein_hbond_energy"))
    {
        for(int nv: range(n_virtual)) {
            CoordPair cp;
            cp.index = nv;
            protein_hbond.slot_machine.add_request(1,cp);
            slots.push_back(cp.slot);
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_energy"));
        for(int ns=0; ns<n_system; ++ns) {
            float pot = 0.f;
            VecArray pp   = protein_hbond .coords().value[ns];
            VecArray d_pp = protein_hbond .coords().deriv[ns];

            // Compute probabilities of P and S states
            for(int nv=0; nv<n_virtual; ++nv) {
                pot += pp(6,nv)*E_protein;
                auto d = make_zero<7>(); d[6] = E_protein;
                store_vec(d_pp, slots[nv], d);
            }
            potential[ns] = pot;
            if(!ns) n_hbond = pot/E_protein; // only report for system 0
        }
    }

//     virtual double test_value_deriv_agreement() {
//         vector<vector<CoordPair>> coord_pairs_pp (1);
//         for(int nv: range(n_virtual)) coord_pairs_pp .back().push_back(CoordPair(nv, slots[nv]));
//         return compute_relative_deviation_for_node<7>(*this, protein_hbond, coord_pairs_pp);
//     }
};
static RegisterNodeType<HBondEnergy,1> hbond_energy_node("hbond_energy");
