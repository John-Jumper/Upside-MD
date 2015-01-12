#include "deriv_engine.h"
#include <string>
#include "coord.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "timing.h"

#include "coord.h"
#include "Float4.h"

using namespace std;
using namespace h5;


struct PointCloud {
    int   n_pts;   // must be divisible by 4
    float *x;  // rotated but not translated into position
    float *y;
    float *z;
    float *weight;
    int*  type;
    float3 translation;  
};

struct StericPoint {
    float3 pos;
    float  weight;
    int    type;
};


struct StericResidue {
    float3 center;
    float  radius;   // all points are within radius of the center
    int    start_point;
    int    n_pts;
};


struct StericParams {
    CoordPair loc;
    int       restype;
};


struct Interaction {
    float   largest_cutoff;
    int     n_types;
    int     n_bin;
    float   inv_dx;

    float*  cutoff2;
    float4* germ_arr;

    Interaction(int n_types_, int n_bin_, float dx_):
        largest_cutoff(0.f),
        n_types(n_types_),
        n_bin(n_bin_),
        inv_dx(1.f/dx_),
        cutoff2 (new float [n_types*n_types]),
        germ_arr(new float4[n_types*n_types*n_bin]) {
            for(int nb=0; nb<n_bin; ++nb) 
                germ_arr[nb] = make_float4(0.f,0.f,0.f,0.f);
        };


    float interaction(float deriv1[6], float deriv2[6], const PointCloud &r1, const PointCloud &r2) {
        Vec34 trans(r1.translation.x - r2.translation.x,
                    r1.translation.y - r2.translation.y,
                    r1.translation.z - r2.translation.z);

        Float4 pot;
        Vec34 c1;
        Vec34 t1;
        Vec34 negation_of_t2;
        Float4 cutoff_radius(largest_cutoff-1e-6f);

        alignas(16) int coord_bin[16];

        for(int i1=0; i1<r1.n_pts; i1+=4) {
            auto x1 = Vec34(r1.x+i1, r1.y+i1, r1.z+i1);
            auto w1 = Float4(r1.weight + i1);
            auto x1trans = x1 + trans;

            for(int i2=0; i2<r2.n_pts; i2+=4) {
                auto x2 = Vec34(r2.x+i2, r2.y+i2, r2.z+i2);
                auto w2 = Float4(r2.weight + i2);

                for(int it=0; it<4; ++it) {
                    auto disp = x1trans-x2;
                    auto r_mag = disp.mag(); r_mag = cutoff_radius.blendv(r_mag, (r_mag<cutoff_radius));
                    auto coord = Float4(inv_dx) * r_mag;

                    // variables are named for what they will contain *after* the transpose
                    // FIXME should use Int4 type here
                    coord.store_int(coord_bin+4*it);
                    Float4 pot1((float*)(germ_arr + (r1.type[i1+0]*n_types + r2.type[i2+(it+0)%4])*n_bin + coord_bin[4*it+0]));
                    Float4 der1((float*)(germ_arr + (r1.type[i1+1]*n_types + r2.type[i2+(it+1)%4])*n_bin + coord_bin[4*it+1]));
                    Float4 pot2((float*)(germ_arr + (r1.type[i1+2]*n_types + r2.type[i2+(it+2)%4])*n_bin + coord_bin[4*it+2]));
                    Float4 der2((float*)(germ_arr + (r1.type[i1+3]*n_types + r2.type[i2+(it+3)%4])*n_bin + coord_bin[4*it+3]));

                    auto r_excess = coord - coord.round<_MM_FROUND_TO_ZERO>();
                    auto l_excess = Float4(1.f) - r_excess;

                    // excess is multiplied by the weights as linear coefficients
                    auto w = w1*w2;
                    r_excess *= w;
                    l_excess *= w;

                    transpose4(pot1, der1, pot2, der2);

                    pot += l_excess*pot1 + r_excess*pot2;
                    auto deriv = (l_excess*der1 + r_excess*der2) * disp;

                    c1             +=       deriv;
                    t1             += cross(deriv, x1);
                    negation_of_t2 += cross(deriv, x2);

                    x2.left_rotate();
                }
            }
        }

        Float4 c1_sum = c1.sum();
        deriv1[0] += c1_sum.x();  deriv2[0] -= c1_sum.x();
        deriv1[1] += c1_sum.y();  deriv2[1] -= c1_sum.y();
        deriv1[2] += c1_sum.z();  deriv2[2] -= c1_sum.z();

        Float4 t1_sum = t1.sum();
        Float4 t2_sum = -negation_of_t2.sum();

        deriv1[3] += t1_sum.x();  deriv2[3] -= t2_sum.x();
        deriv1[4] += t1_sum.y();  deriv2[4] -= t2_sum.y();
        deriv1[5] += t1_sum.z();  deriv2[5] -= t2_sum.z();

        return pot.sum();
    }


    
    float2 germ(int loc, float r_mag) const {
        float coord = inv_dx*r_mag;
        int   coord_bin = int(coord);

        float4 vals = germ_arr[loc*n_bin + coord_bin]; 
        float r_excess = coord - coord_bin;
        float l_excess = 1.f-r_excess;

        return make_float2(l_excess * vals.x + r_excess * vals.y,
                           l_excess * vals.z + r_excess * vals.w);
    }

    ~Interaction() {
        delete [] cutoff2;
        delete [] germ_arr;
    }
};

/*
void steric_pairs(
        const CoordArray     rigid_body,
        const StericParams*  residues,  //  size (n_res,)
        const StericResidue* ref_res,
        const StericPoint*   ref_point,
        const Interaction    &interaction,
        const int*           point_starts,  // size (n_res+1,)
        int n_res, int n_system)
{
    int n_point = point_starts[n_res]; // total number of points
    for(int ns=0; ns<n_system; ++ns) {
        vector<AffineCoord<>> coords;  coords.reserve(n_res);
        vector<StericPoint> lab_points(n_point);
        vector<float> cdata(4*n_res);

        for(int nr=0; nr<n_res; ++nr) {
            int rt = residues[nr].restype;

            // Create coordinate
            coords.emplace_back(rigid_body, ns, residues[nr].loc);

            // Compute cutoff data
            float3 center = coords.back().apply(ref_res[rt].center);
            cdata[0*n_res + nr] = center.x;
            cdata[1*n_res + nr] = center.y;
            cdata[2*n_res + nr] = center.z;
            cdata[3*n_res + nr] = ref_res[rt].radius;

            // Apply rotation to reference points
            for(int np=0; np<ref_res[rt].n_pts; ++np) {
                const StericPoint& rp = ref_point[ref_res[rt].start_point + np];
                lab_points[point_starts[nr]+np].pos    = coords.back().apply(rp.pos);
                lab_points[point_starts[nr]+np].weight = rp.weight;
                lab_points[point_starts[nr]+np].type   = rp.type;
            }
        }

        for(int nr1=0; nr1<n_res; ++nr1) {
            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
                float cutoff = interaction.largest_cutoff + cdata[3*n_res+nr1] + cdata[3*n_res+nr2];
                float dist2 = (
                        (cdata[0*n_res+nr1]-cdata[0*n_res+nr2])*(cdata[0*n_res+nr1]-cdata[0*n_res+nr2]) +
                        (cdata[1*n_res+nr1]-cdata[1*n_res+nr2])*(cdata[1*n_res+nr1]-cdata[1*n_res+nr2]) +
                        (cdata[2*n_res+nr1]-cdata[2*n_res+nr2])*(cdata[2*n_res+nr1]-cdata[2*n_res+nr2]));

                if(dist2<cutoff*cutoff) {
                    for(int i1=point_starts[nr1]; i1<point_starts[nr1+1]; ++i1) {
                        for(int i2=point_starts[nr2]; i2<point_starts[nr2+1]; ++i2) {
                            int loc = lab_points[i1].type*interaction.n_types + lab_points[i2].type;
                            const float3 r = lab_points[i1].pos - lab_points[i2].pos;
                            const float rmag2 = mag2(r);
                            if(rmag2>=interaction.cutoff2[loc]) continue;
                            const float deriv_over_r = interaction.germ(loc, sqrtf(rmag2)).y;
                            const float3 g = (lab_points[i1].weight * lab_points[i2].weight * deriv_over_r)*r;

                            coords[nr1].add_deriv_at_location(lab_points[i1].pos,  g);
                            coords[nr2].add_deriv_at_location(lab_points[i2].pos, -g);
                        }
                    }
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr)
            coords[nr].flush();
    }
}
*/

void steric_pairs(
        const CoordArray     rigid_body,
        const StericParams*  residues,  //  size (n_res,)
        const StericResidue* ref_res,
        const StericPoint*   ref_point,
        const Interaction    &interaction,
        const int*           point_starts,  // size (n_res+1,)
        int n_res, int n_system)
{
    int n_point = point_starts[n_res]; // total number of points
    for(int ns=0; ns<n_system; ++ns) {
        vector<AffineCoord<>> coords;  coords.reserve(n_res);
        vector<StericPoint> lab_points(n_point);
        vector<float> cdata(4*n_res);

        for(int nr=0; nr<n_res; ++nr) {
            int rt = residues[nr].restype;

            // Create coordinate
            coords.emplace_back(rigid_body, ns, residues[nr].loc);

            // Compute cutoff data
            float3 center = coords.back().apply(ref_res[rt].center);
            cdata[0*n_res + nr] = center.x;
            cdata[1*n_res + nr] = center.y;
            cdata[2*n_res + nr] = center.z;
            cdata[3*n_res + nr] = ref_res[rt].radius;

            // Apply rotation to reference points
            for(int np=0; np<ref_res[rt].n_pts; ++np) {
                const StericPoint& rp = ref_point[ref_res[rt].start_point + np];
                lab_points[point_starts[nr]+np].pos    = coords.back().apply(rp.pos);
                lab_points[point_starts[nr]+np].weight = rp.weight;
                lab_points[point_starts[nr]+np].type   = rp.type;
            }
        }

        for(int nr1=0; nr1<n_res; ++nr1) {
            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
                float cutoff = interaction.largest_cutoff + cdata[3*n_res+nr1] + cdata[3*n_res+nr2];
                float dist2 = (
                        (cdata[0*n_res+nr1]-cdata[0*n_res+nr2])*(cdata[0*n_res+nr1]-cdata[0*n_res+nr2]) +
                        (cdata[1*n_res+nr1]-cdata[1*n_res+nr2])*(cdata[1*n_res+nr1]-cdata[1*n_res+nr2]) +
                        (cdata[2*n_res+nr1]-cdata[2*n_res+nr2])*(cdata[2*n_res+nr1]-cdata[2*n_res+nr2]));

                if(dist2<cutoff*cutoff) {
                    for(int i1=point_starts[nr1]; i1<point_starts[nr1+1]; ++i1) {
                        for(int i2=point_starts[nr2]; i2<point_starts[nr2+1]; ++i2) {
                            int loc = lab_points[i1].type*interaction.n_types + lab_points[i2].type;
                            const float3 r = lab_points[i1].pos - lab_points[i2].pos;
                            const float rmag2 = mag2(r);
                            if(rmag2>=interaction.cutoff2[loc]) continue;
                            const float deriv_over_r = interaction.germ(loc, sqrtf(rmag2)).y;
                            const float3 g = (lab_points[i1].weight * lab_points[i2].weight * deriv_over_r)*r;

                            coords[nr1].add_deriv_at_location(lab_points[i1].pos,  g);
                            coords[nr2].add_deriv_at_location(lab_points[i2].pos, -g);
                        }
                    }
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr)
            coords[nr].flush();
    }
}


struct StericInteraction : public PotentialNode
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
        r.n_pts = get_dset_size(1, grp, "weight")[0];

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
        n_res(h5::get_dset_size(1, grp, "restype")[0]), alignment(alignment_),
        params(n_res),
        pot(get_dset_size(2, grp, "atom_interaction/cutoff")[0], 
            get_dset_size(3, grp, "atom_interaction/potential")[2], 
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

    virtual void compute_value() {
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
static RegisterNodeType<StericInteraction,1> steric_node("steric");
