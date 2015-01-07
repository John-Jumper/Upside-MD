#include "deriv_engine.h"
#include <string>
#include "timing.h"
#include <math.h>
#include "md_export.h"
#include <algorithm>

#include "random.h"
using namespace h5;
using namespace std;

struct VirtualParams {
    CoordPair atom[3];
    float bond_length;
};

struct VirtualHBondParams {
    CoordPair id;
    unsigned short residue_id;
    float  helix_energy_bonus;
};

struct NH_CO_Params {
    float H_bond_length, N_bond_length;
    unsigned short Cprev, N, CA, C, Nnext;
    unsigned short H_slots[9];
    unsigned short O_slots[9];
};


namespace {

inline void
hat_deriv(
        float3 v_hat, float v_invmag, 
        float3 &col0, float3 &col1, float3 &col2) // matrix is symmetric, so these are rows or cols
{
    float s = v_invmag;
    col0 = make_float3(s*(1.f-v_hat.x*v_hat.x), s*    -v_hat.y*v_hat.x , s*    -v_hat.z*v_hat.x );
    col1 = make_float3(s*    -v_hat.x*v_hat.y , s*(1.f-v_hat.y*v_hat.y), s*    -v_hat.z*v_hat.y );
    col2 = make_float3(s*    -v_hat.x*v_hat.z , s*    -v_hat.y*v_hat.z , s*(1.f-v_hat.z*v_hat.z));
}

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

    return make_float2( outer_sigmoid.x * inner_sigmoid.x,
            - inv_outer_width * outer_sigmoid.y * inner_sigmoid.x
            + inv_inner_width * inner_sigmoid.y * outer_sigmoid.x);
}


float2 hbond_angular_potential(float dotp) 
{
    const float wall_dp = 0.682f;  // half-height is at 47 degrees
    const float inv_dp_width = 1.f/0.05f;

    float2 v = sigmoid((dotp-wall_dp)*inv_dp_width);
    return make_float2(v.x, inv_dp_width*v.y);
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
    prev_c.set_deriv(3, -make_float3(dot(drow0,pcol0), dot(drow0,pcol1), dot(drow0,pcol2)));
    prev_c.set_deriv(4, -make_float3(dot(drow1,pcol0), dot(drow1,pcol1), dot(drow1,pcol2)));
    prev_c.set_deriv(5, -make_float3(dot(drow2,pcol0), dot(drow2,pcol1), dot(drow2,pcol2)));

    // position derivative is direction derivative times bond length
    for(int no=0; no<3; ++no) for(int nc=0; nc<3; ++nc) prev_c.d[no][nc] = bond_length*prev_c.d[no+3][nc];

    // next derivatives
    next_c.set_deriv(3, -make_float3(dot(drow0,ncol0), dot(drow0,ncol1), dot(drow0,ncol2)));
    next_c.set_deriv(4, -make_float3(dot(drow1,ncol0), dot(drow1,ncol1), dot(drow1,ncol2)));
    next_c.set_deriv(5, -make_float3(dot(drow2,ncol0), dot(drow2,ncol1), dot(drow2,ncol2)));

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
    hbond_pos.v[0] = curr_c.v[0] - bond_length*disp.x;
    hbond_pos.v[1] = curr_c.v[1] - bond_length*disp.y;
    hbond_pos.v[2] = curr_c.v[2] - bond_length*disp.z;
    hbond_pos.v[3] = -disp.x;
    hbond_pos.v[4] = -disp.y;
    hbond_pos.v[5] = -disp.z;
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



namespace {
float hbond_score(
        float  H[3], float  O[3], float  rHN[3], float  rOC[3],
        float dH[3], float dO[3], float drHN[3], float drOC[3],
        float score_multiplier)
{
    float HO[3] = {H[0]-O[0], H[1]-O[1], H[2]-O[2]};
    const float eps = 1e-6;  // a bit of paranoia to avoid division by zero later
    float magHO2 = HO[0]*HO[0] + HO[1]*HO[1] + HO[2]*HO[2] + eps;

    float invHOmag = rsqrtf(magHO2);
    float rHO[3] = {HO[0]*invHOmag, HO[1]*invHOmag, HO[2]*invHOmag};

    float dotHOC =  rHO[0]*rOC[0] + rHO[1]*rOC[1] + rHO[2]*rOC[2];
    float dotOHN = -rHO[0]*rHN[0] - rHO[1]*rHN[1] - rHO[2]*rHN[2];

    if(!((dotHOC > angular_cutoff) & (dotOHN > angular_cutoff))) return 0.f;

    float2 radial   = score_multiplier * hbond_radial_potential(sqrtf(magHO2));  // x has val, y has deriv
    float2 angular1 =                    hbond_angular_potential(dotHOC);
    float2 angular2 =                    hbond_angular_potential(dotOHN);

    float val =  radial.x * angular1.x * angular2.x;
    float c0  =  radial.y * angular1.x * angular2.x;
    float c1  =  radial.x * angular1.y * angular2.x;
    float c2  = -radial.x * angular1.x * angular2.y;

    drOC[0] += c1*rHO[0]; drHN[0] += c2*rHO[0];
    drOC[1] += c1*rHO[1]; drHN[1] += c2*rHO[1];
    drOC[2] += c1*rHO[2]; drHN[2] += c2*rHO[2];

    float dHO[3] = {
        c0*rHO[0] + (c1*invHOmag)*(rOC[0]-dotHOC*rHO[0]) + (c2*invHOmag)*(rHN[0]+dotOHN*rHO[0]),
        c0*rHO[1] + (c1*invHOmag)*(rOC[1]-dotHOC*rHO[1]) + (c2*invHOmag)*(rHN[1]+dotOHN*rHO[1]),
        c0*rHO[2] + (c1*invHOmag)*(rOC[2]-dotHOC*rHO[2]) + (c2*invHOmag)*(rHN[2]+dotOHN*rHO[2])};

    dH[0] += dHO[0]; dO[0] -= dHO[0];
    dH[1] += dHO[1]; dO[1] -= dHO[1];
    dH[2] += dHO[2]; dO[2] -= dHO[2];

    return val;
}


// void 
// test_infer_x_body()
// {
//     RandomGenerator gen(14, 10, 32, 101);
// 
//     float bond_length = 1.24f;
//     vector<float> output_storage(6);
//     vector<float> pos(9);
//     vector<float> deriv(10);
//     for(auto& x: pos) x = gen.normal().x;
// 
//     Coord<3,6> x1(pos.data(), deriv.data(), CoordPair(0,0));
//     Coord<3,6> x2(pos.data(), deriv.data(), CoordPair(1,0));
//     Coord<3,6> x3(pos.data(), deriv.data(), CoordPair(2,0));
// 
//     auto f = [&](Coord<3,6>& xprime) {
//         MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
//         infer_x_body(o, xprime, x2, x3, bond_length);
//         return o;
//     };
//     f(x1);
//     finite_difference(f, x1, &x1.d[0][0]);
// 
//     auto g = [&](Coord<3,6>& xprime) {
//         MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
//         infer_x_body(o, x1, xprime, x3, bond_length);
//         return o;
//     };
//     g(x2);
//     finite_difference(g, x2, &x2.d[0][0]);
// 
//     auto h = [&](Coord<3,6>& xprime) {
//         MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
//         infer_x_body(o, x1, x2, xprime, bond_length);
//         return o;
//     };
//     h(x3);
//     finite_difference(h, x3, &x3.d[0][0]);
// }
}


float count_hbond(
        const CoordArray virtual_pos,
        int n_donor,    const VirtualHBondParams * restrict donor_params,
        int n_acceptor, const VirtualHBondParams * restrict acceptor_params,
        const float hbond_energy, int n_system)
{
    float tot_hbond = 0.f;
    
#pragma omp parallel for reduction(+:tot_hbond)
    for(int ns=0; ns<n_system; ++ns) {
        vector<Coord<6>> donors;    donors   .reserve(n_donor);
        vector<Coord<6>> acceptors; acceptors.reserve(n_acceptor);

        for(int nd=0; nd<n_donor; ++nd) 
            donors   .emplace_back(virtual_pos, ns, donor_params   [nd].id);

        for(int na=0; na<n_acceptor; ++na)
            acceptors.emplace_back(virtual_pos, ns, acceptor_params[na].id);

        for(int nd=0; nd<n_donor; ++nd) {
            for(int na=0; na<n_acceptor; ++na) {
                if(mag2(donors[nd].f3()-acceptors[na].f3()) < radial_cutoff2) {
                    int res_disp = donor_params[nd].residue_id - acceptor_params[na].residue_id;
                    if(abs(res_disp)<2) continue;

                    const float energy = hbond_energy + (res_disp==4) * (donor_params[nd].helix_energy_bonus +
                            acceptor_params[na].helix_energy_bonus);

                    Coord<6> &H = donors[nd];
                    Coord<6> &O = acceptors[na];
                    tot_hbond += hbond_score(H.v, O.v, H.v+3, O.v+3, H.d[0], O.d[0], H.d[0]+3, O.d[0]+3, energy);
                }
            }
        }

        for(auto& d: donors)    for(int i=0; i<6; ++i) d.flush();
        for(auto& a: acceptors) for(int i=0; i<6; ++i) a.flush();
    }
    return tot_hbond;
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
    }

    CoordArray coords() {
        return CoordArray(SysArray(output.data(), n_virtual*6), slot_machine.accum_array());
    }

    virtual void compute_value() {
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


struct HBondEnergy : public HBondCounter
{
    int n_donor, n_acceptor;
    CoordNode& infer;
    vector<VirtualHBondParams> don_params;
    vector<VirtualHBondParams> acc_params;
    float hbond_energy;

    HBondEnergy(hid_t grp, CoordNode& infer_):
        HBondCounter(),
        n_donor   (get_dset_size(1, grp, "donors/residue_id")[0]), 
        n_acceptor(get_dset_size(1, grp, "acceptors/residue_id")[0]),
        infer(infer_), 
        don_params(n_donor), acc_params(n_acceptor), 
        hbond_energy(        read_attribute<float>(grp, ".", "hbond_energy"))
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

    virtual void compute_value() {
        Timer timer(string("hbond_energy"));
        n_hbond = (1.f/hbond_energy) * count_hbond(
                infer.coords(), 
                n_donor, don_params.data(), n_acceptor, acc_params.data(),
                hbond_energy, infer.n_system);
    }
};

static RegisterNodeType<Infer_H_O,1>  _16("infer_H_O");
static RegisterNodeType<HBondEnergy,1>_17("hbond_energy");

/*
void helical_probabilities(
        int n_residue, float * restrict helicity,  // size (n_residue,), corresponds to donor_params residue_id's
        const float * restrict virtual_pos,
        int n_donor,    const VirtualHBondParams * restrict donor_params,
        int n_acceptor, const VirtualHBondParams * restrict acceptor_params)
{
    vector<Coord<6>> donors;    donors   .reserve(n_donor);
    vector<Coord<6>> acceptors; acceptors.reserve(n_acceptor);

    for(int nd=0; nd<n_donor;    ++nd) donors   .emplace_back(virtual_pos, nullptr, donor_params   [nd].id);
    for(int na=0; na<n_acceptor; ++na) acceptors.emplace_back(virtual_pos, nullptr, acceptor_params[na].id);

    for(int i=0; i<n_residue; ++i) helicity[i] = 0.f;

    for(int nd=0; nd<n_donor; ++nd) {
        for(int na=0; na<n_acceptor; ++na) {
            int res_disp = donor_params[nd].residue_id - acceptor_params[na].residue_id;
            if(res_disp!=4) continue;

            if(mag2(donors[nd].f3()-acceptors[na].f3()) < radial_cutoff2) {
                if(abs(res_disp)<2) continue;

                Coord<6> &H = donors[nd];
                Coord<6> &O = acceptors[na];
                helicity[acceptor_params[na].residue_id] = hbond_score(
                        H.v, O.v, H.v+3, O.v+3, H.d[0], O.d[0], H.d[0]+3, O.d[0]+3, 1.f);
            }
        }
    }
}
*/

