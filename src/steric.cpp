#include "steric.h"
#include "md_export.h"
#include "coord.h"
#include "affine.h"
#include <cmath>
#include <vector>
#include "timing.h"

using namespace std;


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
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        vector<AffineCoord> coords;  coords.reserve(n_res);
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
                            const float deriv_over_r = interaction.germ(loc, sqrt(rmag2)).y;
                            const float3 g = (lab_points[i1].weight * lab_points[i2].weight * deriv_over_r)*r;

                            coords[nr1].add_deriv_at_location(lab_points[i1].pos,  g);
                            coords[nr2].add_deriv_at_location(lab_points[i2].pos, -g);
                        }
                    }
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr) {
            coords[nr].flush();
        }
    }
}
