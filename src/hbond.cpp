#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include <math.h>
#include "md_export.h"
#include <algorithm>
#include "state_logger.h"
#include "interaction_graph.h"
#include "spline.h"

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

template <typename CoordT>
void infer_x_body(
        Vec<6> &hbond_pos,
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
    hbond_pos[0] = curr_c.v[0] - bond_length*disp.x();
    hbond_pos[1] = curr_c.v[1] - bond_length*disp.y();
    hbond_pos[2] = curr_c.v[2] - bond_length*disp.z();
    hbond_pos[3] = -disp.x();
    hbond_pos[4] = -disp.y();
    hbond_pos[5] = -disp.z();
}

}

struct Infer_H_O : public CoordNode
{
    CoordNode& pos;
    int n_donor, n_acceptor, n_virtual;
    vector<VirtualParams> params;
    vector<AutoDiffParams> autodiff_params;

    Infer_H_O(hid_t grp, CoordNode& pos_):
        CoordNode(
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
            default_logger->add_logger<float>("virtual", {n_elem, 3}, [&](float* buffer) {
                    for(int nv=0; nv<n_virtual; ++nv) {
                        auto x = load_vec<6>(coords().value, nv);
                        for(int d=0; d<3; ++d) buffer[nv*3 + d] = x[d];
                    }
                });
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("infer_H_O"));

        auto HN_OC = coords().value;
        auto posc  = pos.coords();
        for(int nt=0; nt<n_virtual; ++nt) {
            Vec<6>     hbond_pos;
            Coord<3,6> prev_c(posc, params[nt].atom[0]);
            Coord<3,6> curr_c(posc, params[nt].atom[1]);
            Coord<3,6> next_c(posc, params[nt].atom[2]);

            infer_x_body(hbond_pos, prev_c,curr_c,next_c, params[nt].bond_length);

            store_vec(HN_OC, nt, hbond_pos);
            prev_c.flush();
            curr_c.flush();
            next_c.flush();
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("infer_H_O_deriv"));
        reverse_autodiff<6,3,0>(
                slot_machine.accum_array(), 
                pos.slot_machine.accum_array(), VecArray(), 
                slot_machine.deriv_tape.data(), autodiff_params.data(), 
                slot_machine.deriv_tape.size(), 
                n_virtual);}

    virtual double test_value_deriv_agreement() {
        return -1.; // compute_relative_deviation_for_node<3>(*this, pos, extract_pairs(params, potential_term));
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
            for(int np: range(n_param)) d_param[np] = -1.f;
        }
    };


    struct HBondCoverageInteraction {
        // radius scale angular_width angular_scale
        // first group is donors; second group is acceptors
        constexpr static const int n_knot = 18, n_knot_angular=15;
        // FIXME ensure that the spline argument is correctly bounded
        constexpr static const float inv_dx = 1.f/0.5f, inv_dtheta = (n_knot_angular-2)/2.f;
        constexpr static const int n_param=2*n_knot_angular+2*n_knot, n_dim1=7, n_dim2=6, n_deriv=10;

        static float cutoff(const Vec<n_param> &p) {
            return (n_knot-2-1e-6)/inv_dx;  // 1e-6 insulates from roundoff
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return false; // no exclusions
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            float3 displace = extract<0,3>(sc_pos)-extract<0,3>(hb_pos);
            float3 rHN = extract<3,6>(hb_pos);
            float3 rSC = extract<3,6>(sc_pos);

            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist_coord = dist2*(inv_dist*inv_dx);
            float3 displace_unitvec = inv_dist*displace;

            float  cos_coverage_angle1 = dot(rHN, displace_unitvec);
            float  cos_coverage_angle2 = dot(rSC,-displace_unitvec);

            float2 wide_cover   = clamped_deBoor_value_and_deriv(p.v+2*n_knot_angular,        dist_coord, n_knot);
            float2 narrow_cover = clamped_deBoor_value_and_deriv(p.v+2*n_knot_angular+n_knot, dist_coord, n_knot);

            float2 angular_sigmoid1 = deBoor_value_and_deriv(p.v,                (cos_coverage_angle1+1.f)*inv_dtheta+1.f);
            float2 angular_sigmoid2 = deBoor_value_and_deriv(p.v+n_knot_angular, (cos_coverage_angle2+1.f)*inv_dtheta+1.f);
            float  angular_weight   = angular_sigmoid1.x() * angular_sigmoid2.x();

            // negative sign on angular_deriv is due to -cos_coverage_angle
            float radial_deriv   = inv_dx*(wide_cover.y() + angular_weight*narrow_cover.y());
            float angular_deriv1 = angular_sigmoid1.y()*angular_sigmoid2.x()*narrow_cover.x();
            float angular_deriv2 = angular_sigmoid1.x()*angular_sigmoid2.y()*narrow_cover.x();

            float3 col0, col1, col2;
            hat_deriv(displace_unitvec, inv_dist, col0, col1, col2);
            float3 rXX = angular_deriv1*rHN - angular_deriv2*rSC;
            float3 deriv_dir = make_vec3(dot(col0,rXX), dot(col1,rXX), dot(col2,rXX));

            float3 d_displace = radial_deriv  * displace_unitvec + deriv_dir;
            float3 d_rHN      =  angular_deriv1 * displace_unitvec;
            float3 d_rSC      = -angular_deriv2 * displace_unitvec;

            float coverage = wide_cover.x() + angular_weight*narrow_cover.x();
            float prefactor = sqr(1.f-hb_pos[6]);

            store<0,3>(d_base, prefactor * d_displace);
            store<3,6>(d_base, prefactor * d_rHN);
            d_base[6] = -coverage * (1.f-hb_pos[6])*2.f;
            store<7,10>(d_base, prefactor * d_rSC);

            return prefactor * coverage;
        }

        static void expand_deriv(Vec<n_dim1> &d_hb, Vec<n_dim2> &d_sc, const Vec<n_deriv> &d_base) {
            store<0,3>(d_hb, -extract<0, 3>(d_base));  // opposite of d_sc by Newton's third
            store<3,6>(d_hb,  extract<3, 6>(d_base));
            d_hb[6] = d_base[6];
            store<0,3>(d_sc,  extract<0, 3>(d_base));
            store<3,6>(d_sc,  extract<7,10>(d_base));
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            d_param = make_zero<n_param>();

            float3 displace = extract<0,3>(sc_pos)-extract<0,3>(hb_pos);
            float3 rHN = extract<3,6>(hb_pos);
            float3 rSC = extract<3,6>(sc_pos);

            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist_coord = dist2*(inv_dist*inv_dx);
            float3 displace_unitvec = inv_dist*displace;

            float  cos_coverage_angle1 = dot(rHN, displace_unitvec);
            float  cos_coverage_angle2 = dot(rSC,-displace_unitvec);

            float2 angular_sigmoid1 = deBoor_value_and_deriv(p.v,                (cos_coverage_angle1+1.f)*inv_dtheta+1.f);
            float2 angular_sigmoid2 = deBoor_value_and_deriv(p.v+n_knot_angular, (cos_coverage_angle2+1.f)*inv_dtheta+1.f);

            // wide_cover derivative
            int starting_bin;
            float result[4];
            clamped_deBoor_coeff_deriv(&starting_bin, result, p.v+2*n_knot_angular, dist_coord, n_knot);
            for(int i: range(4)) d_param[2*n_knot_angular+starting_bin+i] = result[i];

            // narrow_cover derivative
            clamped_deBoor_coeff_deriv(&starting_bin, result, p.v+2*n_knot_angular+n_knot, dist_coord, n_knot);
            for(int i: range(4)) d_param[2*n_knot_angular+n_knot+starting_bin+i] = angular_sigmoid1.x()*angular_sigmoid2.x()*result[i];

            // angular_sigmoid derivatives
            float2 narrow_cover = clamped_deBoor_value_and_deriv(p.v+2*n_knot_angular+n_knot, dist_coord, n_knot);

            deBoor_coeff_deriv(&starting_bin, result, p.v+n_knot_angular, (cos_coverage_angle1+1.f)*inv_dtheta+1.f);
            for(int i: range(4)) d_param[starting_bin+i] = angular_sigmoid2.x()*narrow_cover.x()*result[i];

            deBoor_coeff_deriv(&starting_bin, result, p.v,                (cos_coverage_angle1+1.f)*inv_dtheta+1.f);
            for(int i: range(4)) d_param[n_knot_angular+starting_bin+i] = angular_sigmoid1.x()*narrow_cover.x()*result[i];

            float prefactor = sqr(1.f-hb_pos[6]);
            d_param *= prefactor;
        }
    };
}


struct ProteinHBond : public CoordNode 
{
    CoordNode& infer;
    BetweenInteractionGraph<ProteinHBondInteraction> igraph;
    int n_donor, n_acceptor, n_virtual;

    ProteinHBond(hid_t grp, CoordNode& infer_):
        CoordNode(get_dset_size(1,grp,"index1")[0]+get_dset_size(1,grp,"index2")[0], 7),
        infer(infer_),
        igraph(grp, infer, infer) ,
        n_donor   (igraph.n_elem1),
        n_acceptor(igraph.n_elem2),
        n_virtual (n_donor+n_acceptor)
    {
        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("hbond", {n_donor+n_acceptor}, [&](float* buffer) {
                   for(int nv: range(n_donor+n_acceptor))
                       buffer[nv] = coords().value(6,nv);});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("protein_hbond"));

        int n_virtual = n_donor + n_acceptor;
        VecArray vs = coords().value;
        VecArray ho = infer.coords().value;

        for(int nv: range(n_virtual)) {
            for(int d: range(6))
                vs(d,nv) = ho(d,nv);
            vs(6,nv) = 0.f;
        }

        // Compute protein hbonding score and its derivative
        igraph.compute_edges([&](int ne, float hb_log,
                    int index1, unsigned type1, unsigned id1,
                    int index2, unsigned type2, unsigned id2) {
                vs(6,index1) += hb_log;
                vs(6,index2) += hb_log;});

        for(int nv: range(n_virtual)) vs(6,nv) = 1.f-expf(-vs(6,nv));
    }

    virtual void propagate_deriv() {
        Timer timer(string("protein_hbond_deriv"));

        vector<Vec<7>> sens(n_virtual, make_zero<7>());
        VecArray accum = slot_machine.accum_array();
        VecArray hb    = coords().value;

        for(auto tape_elem: slot_machine.deriv_tape)
            for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                sens[tape_elem.atom] += load_vec<7>(accum, tape_elem.loc+rec);

        for(int nv: range(n_virtual))
            sens[nv][6] *= 1.f-hb(6,nv);

        // Push protein HBond derivatives
        for(int ned: range(igraph.n_edge)) {
            int don_idx = igraph.param1[igraph.edge_indices[2*ned + 0]].index;
            int acc_idx = igraph.param2[igraph.edge_indices[2*ned + 1]].index;
            igraph.use_derivative(ned,  sens[don_idx][6] + sens[acc_idx][6]);
        }

        // pass through derivatives on all other components
        VecArray pd1 = igraph.pos_deriv1;
        for(int nd: range(n_donor)) {
            // the last component is taken care of by the edge loop
            update_vec(pd1, nd, extract<0,6>(sens[nd]));
        }
        VecArray pd2 = igraph.pos_deriv2;
        for(int na: range(n_acceptor)) {  // acceptor loop
            // the last component is taken care of by the edge loop
            update_vec(pd2, na, extract<0,6>(sens[na+n_donor]));
        }

        igraph.propagate_derivatives();
    }
};
static RegisterNodeType<ProteinHBond,1> hbond_node("protein_hbond");


struct HBondCoverage : public CoordNode {
    BetweenInteractionGraph<HBondCoverageInteraction> igraph;
    int n_sc;

    HBondCoverage(hid_t grp, CoordNode& infer_, CoordNode& sidechains_):
        CoordNode(get_dset_size(1,grp,"index2")[0], 1),
        igraph(grp, infer_, sidechains_),
        n_sc(igraph.n_elem2) {}

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_coverage"));
        VecArray cov = coords().value;
        for(int nc: range(n_sc)) cov(0,nc) = 0.f;

        // Compute coverage and its derivative
        igraph.compute_edges([&](int ne, float en,
                    int index1, unsigned type1, unsigned id1,
                    int index2, unsigned type2, unsigned id2) {
                cov(0,index2) += en;});
    }

    virtual void propagate_deriv() {
        Timer timer(string("hbond_coverage_deriv"));
        vector<float> sens(n_sc, 0.f);
        VecArray accum = slot_machine.accum_array();

        for(auto tape_elem: slot_machine.deriv_tape)
            for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                sens[tape_elem.atom] += accum(0, tape_elem.loc+rec);

        // Push coverage derivatives
        for(int ned: range(igraph.n_edge)) {
            int sc_num = igraph.edge_indices[2*ned + 1];
            int sc_idx = igraph.param2[sc_num].index;
            igraph.use_derivative(ned, sens[sc_idx]);
        }
        igraph.propagate_derivatives();
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
        HBondCounter(),
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
        float pot = 0.f;
        VecArray pp   = protein_hbond .coords().value;
        VecArray d_pp = protein_hbond .coords().deriv;

        // Compute probabilities of P and S states
        for(int nv=0; nv<n_virtual; ++nv) {
            pot += pp(6,nv)*E_protein;
            auto d = make_zero<7>(); d[6] = E_protein;
            store_vec(d_pp, slots[nv], d);
        }
        potential = pot;
        n_hbond = pot/E_protein;
    }
};
static RegisterNodeType<HBondEnergy,1> hbond_energy_node("hbond_energy");
