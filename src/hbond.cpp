#include "hbond.h"
#include <math.h>
#include "md_export.h"
#include <algorithm>

#include "random.h"
using namespace std;



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
        float* HN_OC,
        const float* pos,
        float* pos_deriv,
        const VirtualParams* params,
        int n_term)
{
    for(int nt=0; nt<n_term; ++nt) {
        MutableCoord<6> hbond_pos(HN_OC, nt);
        Coord<3,6>      prev_c(pos, pos_deriv, params[nt].atom[0]);
        Coord<3,6>      curr_c(pos, pos_deriv, params[nt].atom[1]);
        Coord<3,6>      next_c(pos, pos_deriv, params[nt].atom[2]);

        infer_x_body(hbond_pos, prev_c,curr_c,next_c, params[nt].bond_length);

        hbond_pos.flush();
        prev_c.flush();
        curr_c.flush();
        next_c.flush();
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

    float invHOmag = rsqrt(magHO2);
    float rHO[3] = {HO[0]*invHOmag, HO[1]*invHOmag, HO[2]*invHOmag};

    float dotHOC =  rHO[0]*rOC[0] + rHO[1]*rOC[1] + rHO[2]*rOC[2];
    float dotOHN = -rHO[0]*rHN[0] - rHO[1]*rHN[1] - rHO[2]*rHN[2];

    if(!((dotHOC > angular_cutoff) & (dotOHN > angular_cutoff))) return 0.f;

    float2 radial   = score_multiplier * hbond_radial_potential(sqrt(magHO2));  // x has val, y has deriv
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


void 
test_infer_x_body()
{
    RandomGenerator gen(14, 10, 32, 101);

    float bond_length = 1.24f;
    vector<float> output_storage(6);
    vector<float> pos(9);
    vector<float> deriv(10);
    for(auto& x: pos) x = gen.normal().x;

    Coord<3,6> x1(pos.data(), deriv.data(), CoordPair(0,0));
    Coord<3,6> x2(pos.data(), deriv.data(), CoordPair(1,0));
    Coord<3,6> x3(pos.data(), deriv.data(), CoordPair(2,0));

    auto f = [&](Coord<3,6>& xprime) {
        MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
        infer_x_body(o, xprime, x2, x3, bond_length);
        return o;
    };
    f(x1);
    finite_difference(f, x1, &x1.d[0][0]);

    auto g = [&](Coord<3,6>& xprime) {
        MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
        infer_x_body(o, x1, xprime, x3, bond_length);
        return o;
    };
    g(x2);
    finite_difference(g, x2, &x2.d[0][0]);

    auto h = [&](Coord<3,6>& xprime) {
        MutableCoord<6> o(output_storage.data(),0,MutableCoord<6>::Zero);
        infer_x_body(o, x1, x2, xprime, bond_length);
        return o;
    };
    h(x3);
    finite_difference(h, x3, &x3.d[0][0]);
}
}


float count_hbond(
        const float * restrict virtual_pos,
        float       * restrict virtual_pos_deriv,
        int n_donor,    const VirtualHBondParams * restrict donor_params,
        int n_acceptor, const VirtualHBondParams * restrict acceptor_params,
        const float hbond_energy)
{
    vector<Coord<6>> donors;    donors   .reserve(n_donor);
    vector<Coord<6>> acceptors; acceptors.reserve(n_acceptor);

    for(int nd=0; nd<n_donor; ++nd) 
        donors   .emplace_back(virtual_pos, virtual_pos_deriv, donor_params   [nd].id);

    for(int na=0; na<n_acceptor; ++na)
        acceptors.emplace_back(virtual_pos, virtual_pos_deriv, acceptor_params[na].id);

    float tot_hbond = 0.f;
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
    return tot_hbond;
}


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
