#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include <math.h>
#include <algorithm>
#include "state_logger.h"
#include "interaction_graph.h"
#include "spline.h"
#include "bead_interaction.h"

using namespace h5;
using namespace std;

struct Infer_H_O : public CoordNode
{
    struct Params {
        index_t atom[3];
        float bond_length;
    };

    CoordNode& pos;
    int n_donor, n_acceptor, n_virtual;
    vector<Params> params;
    unique_ptr<float[]> data_for_deriv;

    Infer_H_O(hid_t grp, CoordNode& pos_):
        CoordNode(
                get_dset_size(2, grp, "donors/id")[0]+get_dset_size(2, grp, "acceptors/id")[0], 6),
        pos(pos_),
        n_donor(get_dset_size(2, grp, "donors/id")[0]),
        n_acceptor(get_dset_size(2, grp, "acceptors/id")[0]),
        n_virtual(n_donor+n_acceptor), params(n_virtual),
        data_for_deriv(new_aligned<float>(n_virtual*3*4, 4))
    {
        auto don = open_group(grp, "donors");
        auto acc = open_group(grp, "acceptors");

        check_size(don.get(), "id",          n_donor,    3);
        check_size(don.get(), "bond_length", n_donor);
        check_size(acc.get(), "id",          n_acceptor, 3);
        check_size(acc.get(), "bond_length", n_acceptor);

        traverse_dset<2,int  >(don.get(),"id",          [&](size_t i,size_t j, int   x){params[        i].atom[j]=x;});
        traverse_dset<1,float>(don.get(),"bond_length", [&](size_t i,          float x){params[        i].bond_length  =x;});
        traverse_dset<2,int  >(acc.get(),"id",          [&](size_t i,size_t j, int   x){params[n_donor+i].atom[j]=x;});
        traverse_dset<1,float>(acc.get(),"bond_length", [&](size_t i,          float x){params[n_donor+i].bond_length  =x;});

        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("virtual", {n_elem, 3}, [&](float* buffer) {
                    for(int nv=0; nv<n_virtual; ++nv) {
                        auto x = load_vec<6>(output, nv);
                        for(int d=0; d<3; ++d) buffer[nv*3 + d] = x[d];
                    }
                });
        }
    }


    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("infer_H_O"));

        VecArray posc  = pos.output;
        for(int nv=0; nv<n_virtual; ++nv) {
            // For output, first three components are position of H/O and second three are HN/OC bond direction
            // Bond direction is a unit vector
            // curr is the N or C atom
            // This algorithm assumes perfect 120 degree bond angles

            auto& p = params[nv];

            auto prev_c = Float4(&posc(0,p.atom[0]));
            auto curr_c = Float4(&posc(0,p.atom[1]));
            auto next_c = Float4(&posc(0,p.atom[2]));

            auto prev = prev_c - curr_c; auto prev_invmag = inv_mag(prev); prev *= prev_invmag;
            auto next = next_c - curr_c; auto next_invmag = inv_mag(next); next *= next_invmag;
            auto disp = prev   + next  ; auto disp_invmag = inv_mag(disp); disp *= disp_invmag;

            auto hbond_dir = -disp;
            auto hbond_pos = fmadd(Float4(p.bond_length),hbond_dir, curr_c);

            // store derived values for derivatives later
            prev.blend<0,0,0,1>(prev_invmag).store(data_for_deriv + nv*3*4 + 0);
            next.blend<0,0,0,1>(next_invmag).store(data_for_deriv + nv*3*4 + 4);
            disp.blend<0,0,0,1>(disp_invmag).store(data_for_deriv + nv*3*4 + 8);

            // write pos
            hbond_pos.store(&output(0,nv));
            hbond_dir.store(&output(3,nv), Alignment::unaligned);
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("infer_H_O_deriv"));
        VecArray pos_sens = pos.sens;

        for(int nv=0; nv<n_virtual; ++nv) {
            const auto& p = params[nv];

            auto sens_pos = Float4(&sens(0,nv)).zero_entries<0,0,0,1>(); // last entry should be zero
            auto sens_dir = Float4(&sens(3,nv), Alignment::unaligned);

            auto sens_neg_unitdisp = sens_dir + Float4(p.bond_length)*sens_pos;

            // loading: first 3 entries are unitvec and last is inv_mag
            auto prev4 = Float4(data_for_deriv + nv*3*4 + 0);  auto prev_invmag = prev4.broadcast<3>();
            auto next4 = Float4(data_for_deriv + nv*3*4 + 4);  auto next_invmag = next4.broadcast<3>();
            auto disp4 = Float4(data_for_deriv + nv*3*4 + 8);  auto disp_invmag = disp4.broadcast<3>();

            // use dot3 here so we don't have to zero the last component of disp4, etc
            auto sens_nonunit_disp =   disp_invmag *fmsub(dot3(disp4,sens_neg_unitdisp),disp4, sens_neg_unitdisp);
            auto sens_nonunit_prev = (-prev_invmag)*fmsub(dot3(prev4,sens_nonunit_disp),prev4, sens_nonunit_disp);
            auto sens_nonunit_next = (-next_invmag)*fmsub(dot3(next4,sens_nonunit_disp),next4, sens_nonunit_disp);

            sens_nonunit_prev                                 .update(&pos_sens(0,p.atom[0]));
            (sens_pos - sens_nonunit_prev - sens_nonunit_next).update(&pos_sens(0,p.atom[1]));
            sens_nonunit_next                                 .update(&pos_sens(0,p.atom[2]));
        }
    }
};
static RegisterNodeType<Infer_H_O,1> infer_node("infer_H_O");


#define radial_cutoff2 (3.5f*3.5f)
// angular cutoff at 90 degrees
#define angular_cutoff (0.f)

template <typename S>
Vec<2,S> hbond_radial_potential(S input)
{
    const S outer_barrier    = S(2.5f);
    const S inner_barrier    = S(1.4f);   // barrier to prevent H-O overlap
    const S inv_outer_width  = S(1.f/0.125f);
    const S inv_inner_width  = S(1.f/0.10f);

    Vec<2,S> outer_sigmoid = sigmoid((outer_barrier-input)*inv_outer_width);
    Vec<2,S> inner_sigmoid = sigmoid((input-inner_barrier)*inv_inner_width);

    return make_vec2( outer_sigmoid.x() * inner_sigmoid.x(),
            - inv_outer_width * outer_sigmoid.y() * inner_sigmoid.x()
            + inv_inner_width * inner_sigmoid.y() * outer_sigmoid.x());
}


template <typename S>
Vec<2,S> hbond_angular_potential(S dotp)
{
    const S wall_dp = S(0.682f);  // half-height is at 47 degrees
    const S inv_dp_width = S(1.f/0.05f);

    Vec<2,S> v = sigmoid((dotp-wall_dp)*inv_dp_width);
    return make_vec2(v.x(), inv_dp_width*v.y());
}


template <typename S>
S hbond_score(
        Vec<3,S>   H, Vec<3,S>   O, Vec<3,S>   rHN, Vec<3,S>   rOC,
        Vec<3,S>& dH, Vec<3,S>& dO, Vec<3,S>& drHN, Vec<3,S>& drOC)
{
    auto HO = H-O;

    auto magHO2 = mag2(HO) + S(1e-6f); // a bit of paranoia to avoid division by zero later
    auto invHOmag = rsqrt(magHO2);
    auto magHO    = magHO2 * invHOmag;  // avoid a sqrtf later

    auto rHO = HO*invHOmag;

    auto dotHOC =  dot(rHO,rOC);
    auto dotOHN = -dot(rHO,rHN);

    auto within_angular_cutoff = (S(angular_cutoff) < dotHOC) & (S(angular_cutoff) < dotOHN);
    if(none(within_angular_cutoff)) {
        dH=dO=drHN=drOC=make_zero<3,S>();
        return zero<S>();
    }

    auto radial   = hbond_radial_potential (magHO );  // x has val, y has deriv
    auto angular1 = hbond_angular_potential(dotHOC);
    auto angular2 = hbond_angular_potential(dotOHN);

    auto val =  radial.x() * angular1.x() * angular2.x();
    auto c0  =  radial.y() * angular1.x() * angular2.x();
    auto c1  =  radial.x() * angular1.y() * angular2.x();
    auto c2  = -radial.x() * angular1.x() * angular2.y();

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
        constexpr static const bool symmetric=false;
        constexpr static const int n_param=6, n_dim1=6, n_dim2=6, simd_width=1;

        static float cutoff(const float* p) {
            return sqrtf(radial_cutoff2); // FIXME make parameter dependent
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            return Int4() == Int4();  // No exclusions (all true)
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &x1, const Vec<n_dim2,Float4> &x2) {
            auto one = Float4(1.f);
            Vec<3,Float4> dH,dO,drHN,drOC;

            auto hb = hbond_score(extract<0,3>(x1), extract<0,3>(x2), extract<3,6>(x1), extract<3,6>(x2),
                                   dH,               dO,               drHN,             drOC);
            auto hb_log = ternary(one<=hb, Float4(100.f), -logf(one-hb));  // work in multiplicative space

            auto deriv_prefactor = min(rcp(one-hb),Float4(1e5f)); // FIXME this is a mess
            store<0,3>(d1, dH   * deriv_prefactor);
            store<3,6>(d1, drHN * deriv_prefactor);
            store<0,3>(d2, dO   * deriv_prefactor);
            store<3,6>(d2, drOC * deriv_prefactor);

            return hb_log;
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p,
                const Vec<n_dim1> &x1, const Vec<n_dim2> &x2) {
            for(int np: range(n_param)) d_param[np] = -1.f;
        }

        static bool is_compatible(const float* p1, const float* p2) {return true;};
    };


    struct HBondCoverageInteraction {
        // radius scale angular_width angular_scale
        // first group is hb; second group is sc

        constexpr static bool  symmetric = false;
        constexpr static int   n_knot = 12, n_knot_angular=15;
        constexpr static int   n_param=2*n_knot_angular+2*n_knot, n_dim1=7, n_dim2=6, simd_width=1;
        constexpr static float inv_dx = 1.f/0.75f, inv_dtheta = (n_knot_angular-3)/2.f;

        static float cutoff(const float* p) {
            return (n_knot-2-1e-6)/inv_dx;  // 1e-6 insulates from roundoff
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            // return Int4() == Int4();  // No exclusions (all true)
            return id1 != id2; // exclude interactions on the same residue
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &hb_pos, const Vec<n_dim2,Float4> &sc_pos) {
            Float4 one(1.f);

            auto coverage = quadspline<n_knot_angular, n_knot>(d1,d2, inv_dtheta,inv_dx,p, hb_pos,sc_pos);

            auto prefactor = sqr(one-hb_pos[6]);
            d1 *= prefactor;
            d2 *= prefactor;
            d1[6] = -coverage * (one-hb_pos[6])*Float4(2.f);

            return prefactor * coverage;
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p,
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            quadspline_param_deriv<n_knot_angular, n_knot>(d_param, inv_dtheta,inv_dx,p, hb_pos,sc_pos);
            auto prefactor = sqr(1.f-hb_pos[6]);
            d_param *= prefactor;
        }

        static bool is_compatible(const float* p1, const float* p2) {return true;};
    };
}


struct ProteinHBond : public CoordNode
{
    CoordNode& infer;
    InteractionGraph<ProteinHBondInteraction> igraph;
    int n_donor, n_acceptor, n_virtual;
    unique_ptr<float[]> sens_scaled;

    ProteinHBond(hid_t grp, CoordNode& infer_):
        CoordNode(get_dset_size(1,grp,"index1")[0]+get_dset_size(1,grp,"index2")[0], 7),
        infer(infer_),
        igraph(grp, &infer, &infer) ,
        n_donor   (igraph.n_elem1),
        n_acceptor(igraph.n_elem2),
        n_virtual (n_donor+n_acceptor),
        sens_scaled(new_aligned<float>(n_virtual,4))
    {
        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("hbond", {n_donor+n_acceptor}, [&](float* buffer) {
                   for(int nv: range(n_donor+n_acceptor))
                       buffer[nv] = output(6,nv);});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("protein_hbond"));

        int n_virtual = n_donor + n_acceptor;
        VecArray vs = output;
        VecArray ho = infer.output;

        for(int nv: range(n_virtual)) {
            Float4(&ho(0,nv)).store(&vs(0,nv));
            Float4(&ho(4,nv)).store(&vs(4,nv)); // result is already zero padded
        }

        // Compute protein hbonding score and its derivative
        igraph.compute_edges();
        for(int ne=0; ne<igraph.n_edge; ++ne) {
            int nd = igraph.edge_indices1[ne];
            int na = igraph.edge_indices2[ne];
            float hb_log = igraph.edge_value[ne];
            vs(6,nd)         += hb_log;
            vs(6,na+n_donor) += hb_log;
        }

        for(int nv: range(n_virtual)) vs(6,nv) = 1.f-expf(-vs(6,nv));
    }

    virtual void propagate_deriv() {
        Timer timer(string("protein_hbond_deriv"));

        // we accumulated derivatives for z = 1-exp(-log(no_hb))
        // so we need to convert back with z_sens*(1.f-hb)
        for(int nv=0; nv<n_virtual; ++nv)
            sens_scaled[nv] = sens(6,nv) * (1.f-output(6,nv));

        // Push protein HBond derivatives
        for(int ne: range(igraph.n_edge)) {
            auto don_sens = sens_scaled[igraph.edge_indices1[ne]];
            auto acc_sens = sens_scaled[igraph.edge_indices2[ne]+n_donor];

            igraph.edge_sensitivity[ne] = don_sens + acc_sens;
        }
        igraph.propagate_derivatives();

        // pass through derivatives on all other components
        VecArray pd1 = igraph.pos_node1->sens;
        for(int nd=0; nd<n_donor; ++nd) {
            // the last component is taken care of by the edge loop
            update_vec(pd1, igraph.loc1[nd], load_vec<6>(sens, nd));
        }
        VecArray pd2 = igraph.pos_node2->sens;
        for(int na=0; na<n_acceptor; ++na) {  // acceptor loop
            // the last component is taken care of by the edge loop
            update_vec(pd2, igraph.loc2[na], load_vec<6>(sens, na+n_donor));
        }
    }
};
static RegisterNodeType<ProteinHBond,1> hbond_node("protein_hbond");


struct HBondCoverage : public CoordNode {
    InteractionGraph<HBondCoverageInteraction> igraph;
    int n_sc;

    HBondCoverage(hid_t grp, CoordNode& infer_, CoordNode& sidechains_):
        CoordNode(get_dset_size(1,grp,"index2")[0], 1),
        igraph(grp, &infer_, &sidechains_),
        n_sc(igraph.n_elem2) {}

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_coverage"));

        // Compute coverage and its derivative
        igraph.compute_edges();

        fill(output, 0.f);
        for(int ne=0; ne<igraph.n_edge; ++ne) {
            output(0, igraph.edge_indices2[ne]) += igraph.edge_value[ne];
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("hbond_coverage_deriv"));

        for(int ne: range(igraph.n_edge))
            igraph.edge_sensitivity[ne] = sens(0,igraph.edge_indices2[ne]);
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

    HBondEnergy(hid_t grp, CoordNode& protein_hbond_):
        HBondCounter(),
        protein_hbond(protein_hbond_),
        n_virtual(protein_hbond.n_elem),
        E_protein(read_attribute<float>(grp, ".", "protein_hbond_energy"))
    {}

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_energy"));
        float tot_hb = 0.f;
        VecArray pp      = protein_hbond.output;
        VecArray pp_sens = protein_hbond.sens;
        float Ep = E_protein;

        // Compute probabilities of P and S states
        for(int nv=0; nv<n_virtual; ++nv) {
            tot_hb        += pp(6,nv);
            pp_sens(6,nv) += Ep;
        }
        potential = tot_hb*Ep;
        n_hbond = tot_hb;
    }
};
static RegisterNodeType<HBondEnergy,1> hbond_energy_node("hbond_energy");
