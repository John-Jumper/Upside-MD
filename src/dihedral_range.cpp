#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"

using namespace h5;
using namespace std;

struct DihedralRangeParams {
    CoordPair atom[4];
    float  angle_range[2];
    float  scale;
    float  energy;
};

void dihedral_angle_range(
        const CoordArray   pos,
        const DihedralRangeParams* params,
        int n_terms, 
        int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        for(int nt=0; nt<n_terms; ++nt) {
            DihedralRangeParams p = params[nt];

            Coord<3> x1(pos, ns, p.atom[0]);
            Coord<3> x2(pos, ns, p.atom[1]);
            Coord<3> x3(pos, ns, p.atom[2]);
            Coord<3> x4(pos, ns, p.atom[3]);

            // potential function is 
            //   V(theta) = 

            float3 d1, d2, d3, d4;
            float theta = dihedral_germ(x1.f3(),x2.f3(),x3.f3(),x4.f3(), d1,d2,d3,d4);
            // d* contains the derivative of the dihedral with respect to the position
            //    of atom *

            // Compute correct image of angle to handle periodicity
            float center_angle = 0.5f * (p.angle_range[0] + p.angle_range[1]);
            float disp = theta - center_angle;   // disp is in range [0,2pi) 
            while(disp >  M_PI_F) disp -= 2.f*M_PI_F;
            while(disp < -M_PI_F) disp += 2.f*M_PI_F;
            theta = center_angle + disp;  // guarantees |theta-center_angle| <= pi

            float zl = expf(p.scale*(p.angle_range[0]-theta));
            float zu = expf(p.scale*(theta-p.angle_range[1]));
            float wl = 1.f/(1.f+zl);
            float wu = 1.f/(1.f+zu);
            
            float dV_dtheta = p.energy * p.scale * wl*wu * (wl*zl - zu*wu);

            x1.set_deriv(dV_dtheta * d1);
            x2.set_deriv(dV_dtheta * d2);
            x3.set_deriv(dV_dtheta * d3);
            x4.set_deriv(dV_dtheta * d4);

            x1.flush();
            x2.flush();
            x3.flush();
            x4.flush();
        }
    }
}


struct DihedralRange : public PotentialNode
{
    int n_elem;
    CoordNode& pos;
    vector<DihedralRangeParams> params;
    DihedralRange(hid_t grp, CoordNode& pos_):
        n_elem(get_dset_size(2, grp, "id")[0]),
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
    virtual void compute_value() {
            Timer timer(string("dihedral_range"));
            dihedral_angle_range(pos.coords(), params.data(),
                           n_elem, pos.n_system);
    }

};
static RegisterNodeType<DihedralRange,1>  dihedral_range_node("dihedral_range");
