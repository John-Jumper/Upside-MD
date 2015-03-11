#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include <math.h>
#include "md_export.h"
#include <algorithm>
#include "state_logger.h"

using namespace h5;
using namespace std;

struct VirtualParams {
    CoordPair atom[3];
    float bond_length;
};

struct VirtualHBondParams {
    CoordPair id;
    unsigned int residue_id;
    float  helix_energy_bonus;
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

        /*
        if(default_logger) {
            default_logger->add_logger<float>("virtual", {n_system, n_elem, 3}, [&](float* buffer) {
                    for(int ns=0; ns<n_system; ++ns) {
                        for(int nv=0; nv<n_virtual; ++nv) {
                            StaticCoord<6> x(coords().value, ns, nv);
                            for(int d=0; d<3; ++d) buffer[ns*n_elem*3 + nv*3 + d] = x.v[d];
                        }
                    }
                });
        }
        */
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


struct VirtualHBondParams {
    CoordPair id;
    unsigned short residue_id;
};


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

    if(!((dotHOC > angular_cutoff) & (dotOHN > angular_cutoff))) return 0.f;

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
        float3 displace, float3 rHN, float3& d_displace, float3& drHN, float radius, float scale)
{
    const float cover_angular_cutoff = 0.174f;  // 80 degrees, rather arbitrary
    const float cover_angular_scale  = 2.865f;  // 20 degrees-ish, rather arbitrary

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


struct SCVirtualCoverageDeriv {
    int sc_index;
    int virtual_index;

    float3 d_sc;
    float3 d_rHN; // or d_rOC depending on donor or acceptor
    // float3 d_virtual;  // this deriv is the opposite of d_sc
};


struct HBondScoreDeriv {
    int donor_index, acceptor_index;
    float3 dH, drHN, dO, drOC;
};

struct HBondSidechainParams {
    CoordPair id;
    float radius, scale, cutoff2;
};
   


void count_hbond(
        float* potential,
        SysArray virtual_scores,
        const CoordArray virtual_pos,
        int n_donor,    int n_acceptor,
        const CoordPair* virtual_pair,
        const CoordArray sidechain_pos,
        int n_sidechain, const HBondSidechainParams * restrict sidechains,
        int n_system,
        float E_protein,
        float E_protein_solvent)
{
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        int n_virtual = n_donor + n_acceptor;
        VecArray vs = virtual_scores[ns];
        for(int nv=0; nv<n_virtual; ++nv) for(int d=0; d<2; ++d) vs(d,nv) = 0.f;
        if(potential) potential[ns] = 0.f;

        vector<float3> virtual_site(n_donor+n_acceptor);
        vector<float3> virtual_dir (n_donor+n_acceptor);

        for(int nv=0; nv<n_virtual; ++nv) {
            StaticCoord<6> x(virtual_pos.value, ns, nv);
            virtual_site[nv] = make_vec3(x.v[0],x.v[1],x.v[2]);
            virtual_dir [nv] = make_vec3(x.v[3],x.v[4],x.v[5]);
        }

        // Compute coverage and its derivative
        vector<SCVirtualCoverageDeriv> coverage_deriv;
        for(int n_sc=0; n_sc<n_sidechain; ++n_sc) {
            HBondSidechainParams p = sidechains[n_sc];
            float3 sc_pos = StaticCoord<3>(sidechain_pos.value, ns, p.id.index).f3();

            for(int nv=0; nv<n_virtual; ++nv) {
                float3 displace  = sc_pos-virtual_site[nv];
                float dist2 = mag2(displace);

                if(dist2<p.cutoff2) {
                    coverage_deriv.emplace_back();
                    auto& d = coverage_deriv.back();
                    d.sc_index = n_sc;
                    d.virtual_index = nv;

                    vs(1,nv) += coverage_score(displace, virtual_dir[nv], d.d_sc, d.d_rHN, p.radius, p.scale);
                }
            }
        }

        // Compute protein hbonding score and its derivative
        vector<HBondScoreDeriv> hbond_deriv;
        for(int nd=0; nd<n_donor; ++nd) {
            for(int na=n_donor; na<n_donor+n_acceptor; ++na) {
                if(mag2(virtual_site[nd] - virtual_site[na])<radial_cutoff2) {
                    hbond_deriv.emplace_back();
                    auto& d = hbond_deriv.back();
                    d.donor_index = nd;
                    d.acceptor_index = na;

                    float hb = hbond_score(virtual_site[nd], virtual_site[na], virtual_dir[nd], virtual_dir[na],
                                           d.dH,             d.dO,             d.drHN,          d.drOC);
                    float hb_log = hb>=1.f ? -1e10f : -logf(1.f-hb);  // work in multiplicative space
                    vs(0,nd) += hb_log;
                    vs(0,na) += hb_log;

                    float deriv_prefactor = min(1.f/(1.f-hb),1e5f); // FIXME this is a mess
                    d.dH   *= deriv_prefactor;
                    d.dO   *= deriv_prefactor;
                    d.drHN *= deriv_prefactor;
                    d.drOC *= deriv_prefactor;
                }
            }
        }

        // Compute probabilities of P and S states
        struct PSDeriv {float d_hbond,d_burial;};
        vector<PSDeriv> ps_deriv(n_virtual);
        for(int nv=0; nv<n_virtual; ++nv) {
            float zp = expf(-vs(0,nv));  // protein
            float zs = expf(-vs(1,nv));  // solvent
            float2 protein_hbond_prob = make_vec2(1.f-zp, zp);
            float2 solvation_fraction = make_vec2(zs,-zs);  // we summed log-coverage to get here

            float P = protein_hbond_prob.x();
            float S = (1.f-P) * solvation_fraction.x();
            vs(0,nv) = P;
            vs(1,nv) = S;

            if(potential) potential[ns] += P*E_protein + S*E_protein_solvent;  // FIXME add z-dependence

            ps_deriv[nv].d_hbond  = protein_hbond_prob.y()*(E_protein-solvation_fraction.x()*E_protein_solvent);
            ps_deriv[nv].d_burial = (1.f-P)*solvation_fraction.y()*E_protein_solvent;
        }

        // now I need to push those derivatives back and accumulate
        vector<MutableCoord<6>> virtual_deriv; virtual_deriv.reserve(n_virtual);
        vector<MutableCoord<3>> sc_deriv;      sc_deriv     .reserve(n_sidechain);
        for(int nv=0; nv<n_virtual;     ++nv) 
            virtual_deriv.emplace_back(virtual_pos.deriv, ns, virtual_pair[nv].slot, MutableCoord<6>::Zero);
        for(int nsc=0; nsc<n_sidechain; ++nsc) 
            sc_deriv.emplace_back(sidechain_pos.deriv, ns, sidechains[nsc].id.slot, MutableCoord<3>::Zero);

        // Push coverage derivatives
        for(auto& p: coverage_deriv) {
            float3 d = ps_deriv[p.virtual_index].d_burial * p.d_sc;
            float3 d_rHN = ps_deriv[p.virtual_index].d_burial * p.d_rHN;

            sc_deriv[p.sc_index] += d;
            virtual_deriv[p.virtual_index] += -d;
            virtual_deriv[p.virtual_index].v[3] += d_rHN.x();
            virtual_deriv[p.virtual_index].v[4] += d_rHN.y();
            virtual_deriv[p.virtual_index].v[5] += d_rHN.z();
        }

        // Push protein HBond derivatives
        for(auto& p: hbond_deriv) {
            float prefactor = ps_deriv[p.donor_index].d_hbond + ps_deriv[p.acceptor_index].d_hbond;
            float* d_donor    = virtual_deriv[p.donor_index   ].v;
            float* d_acceptor = virtual_deriv[p.acceptor_index].v;

            d_donor   [0] += prefactor * p.dH  .x();
            d_donor   [1] += prefactor * p.dH  .y();
            d_donor   [2] += prefactor * p.dH  .z();
            d_donor   [3] += prefactor * p.drHN.x();
            d_donor   [4] += prefactor * p.drHN.y();
            d_donor   [5] += prefactor * p.drHN.z();

            d_acceptor[0] += prefactor * p.dO  .x();
            d_acceptor[1] += prefactor * p.dO  .y();
            d_acceptor[2] += prefactor * p.dO  .z();
            d_acceptor[3] += prefactor * p.drOC.x();
            d_acceptor[4] += prefactor * p.drOC.y();
            d_acceptor[5] += prefactor * p.drOC.z();
        }

        // Write the derivatives to memory
        for(auto& d: virtual_deriv) d.flush();
        for(auto& d: sc_deriv     ) d.flush();
    }
}



struct HBondEnergy : public HBondCounter
{
    Infer_H_O& infer;
    CoordNode& sidechains;

    int n_donor;
    int n_acceptor;
    int n_sidechain;
    vector<float> virtual_score;
    vector<CoordPair> virtual_pair;
    vector<HBondSidechainParams> sidechain_params;

    float E_protein;
    float E_protein_solvent;

    HBondEnergy(hid_t grp, CoordNode& infer_, CoordNode& sidechains_):
        HBondCounter(infer_.n_system),
        infer(dynamic_cast<Infer_H_O&>(infer_)), 
        sidechains(sidechains_),
        n_donor    (infer.n_donor),
        n_acceptor (infer.n_acceptor),
        n_sidechain(sidechains.n_elem),
        virtual_score(2*(n_donor+n_acceptor)*n_system),
        virtual_pair (n_donor+n_acceptor),
        sidechain_params(sidechains.n_elem),

        E_protein(        read_attribute<float>(grp, ".", "protein_hbond_energy")),
        E_protein_solvent(read_attribute<float>(grp, ".", "solvent_hbond_energy"))
    {
        check_elem_width(sidechains, 3);

        check_size(grp, "sidechain_id",     n_sidechain);
        check_size(grp, "sidechain_radius", n_sidechain);
        check_size(grp, "sidechain_scale",  n_sidechain);

        traverse_dset<1,int>  (grp, "sidechain_id",     [&](int n_sc, int   x) {sidechain_params[n_sc].id.index = x;});
        traverse_dset<1,float>(grp, "sidechain_radius", [&](int n_sc, float x) {sidechain_params[n_sc].radius = x;});
        traverse_dset<1,float>(grp, "sidechain_scale",  [&](int n_sc, float x) {sidechain_params[n_sc].scale = x;});

        for(auto &p: sidechain_params) {
            p.cutoff2 = sqr(p.radius + 1.f/p.scale);
            sidechains.slot_machine.add_request(1,p.id);
        }

        for(int i=0; i<n_donor+n_acceptor; ++i) {
            virtual_pair[i].index = i;
            infer.slot_machine.add_request(1, virtual_pair[i]);
        }

        /*
        if(default_logger) {
            default_logger->add_logger<float>("hbond", {n_system,n_donor+n_acceptor,2}, [&](float* buffer) {
                    copy_n(virtual_score.data(), n_system*2*(n_donor+n_acceptor), buffer);});
        }
        */
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("hbond_energy"));

        count_hbond(
                potential.data(),
                SysArray(virtual_score.data(), (n_donor+n_acceptor)*2, n_donor+n_acceptor),
                infer.coords(),
                n_donor,    n_acceptor,
                virtual_pair.data(),
                sidechains.coords(),
                n_sidechain, sidechain_params.data(),
                n_system,
                E_protein,
                E_protein_solvent);

        n_hbond = -2.f; // FIXME handle this later / more generally report P, S, and N values
    }

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> virtual_coord_pairs(1);
        vector<vector<CoordPair>> sidechain_coord_pairs(1);
        virtual_coord_pairs.back() = virtual_pair;
        for(auto &p: sidechain_params) sidechain_coord_pairs.back().push_back(p.id);
        
        double virtual_dev   = compute_relative_deviation_for_node<6>(*this, infer, virtual_coord_pairs);
        double sidechain_dev = compute_relative_deviation_for_node<3>(*this, sidechains, sidechain_coord_pairs);
        return 0.5 * (virtual_dev+sidechain_dev);
    }
};


static RegisterNodeType<HBondEnergy,2> hbond_node("hbond_energy");
