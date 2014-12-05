#include "attraction.h"
#include "affine.h"
#include <cmath>
#include <vector>

using namespace std;

struct AttractionResidue {
    float3      interaction_center;
    int         restype;
    AffineCoord coord;
};

void attraction_pairs(
        const CoordArray             rigid_body,
        const AttractionParams*      residue_param,
        const AttractionInteraction* interaction_params,
        int n_types, float cutoff,
        int n_res, int n_system)
{
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        vector<AttractionResidue> residues;  residues.reserve(n_res);

        for(int nr=0; nr<n_res; ++nr) {
            residues.emplace_back(); auto &r = residues.back();
            r.coord = AffineCoord(rigid_body, ns, residue_param[nr].loc);
            r.restype = residue_param[nr].restype;
            r.interaction_center = r.coord.apply(residue_param[nr].sc_ref_pos);
        }

        for(int nr1=0; nr1<n_res; ++nr1) {
            AttractionResidue &r1 = residues[nr1];

            for(int nr2=nr1+2; nr2<n_res; ++nr2) {  // do not interact with nearest neighbors
                AttractionResidue &r2 = residues[nr2];

                float3 disp = r1.interaction_center - r2.interaction_center;
                AttractionInteraction at = interaction_params[r1.restype*n_types + r2.restype];
                float dist2 = mag2(disp);
                float reduced_coord = at.scale * (dist2 - at.r0_squared);

                if(reduced_coord<cutoff) {
                    //printf("reduced_coord %.1f %.1f\n",reduced_coord,sqrtf(dist2));
                    float  z = expf(reduced_coord);
                    float  w = 1.f / (1.f + z);
                    float  deriv_over_r = -2.f*at.scale * at.energy * z * (w*w);
                    float3 deriv = deriv_over_r * disp;

                    r1.coord.add_deriv_at_location(r1.interaction_center,  deriv);
                    r2.coord.add_deriv_at_location(r2.interaction_center, -deriv);
                }
            }
        }

        for(int nr=0; nr<n_res; ++nr)
            residues[nr].coord.flush();
    }
}


void contact_energy(
        const CoordArray   rigid_body,
        const ContactPair* contact_param,
        int n_contacts, float cutoff, int n_system)
{
    #pragma omp parallel for
    for(int ns=0; ns<n_system; ++ns) {
        for(int nc=0; nc<n_contacts; ++nc) {
            ContactPair p = contact_param[nc];
            AffineCoord r1(rigid_body, ns, p.loc[0]);
            AffineCoord r2(rigid_body, ns, p.loc[1]);

            float3 x1 = r1.apply(p.sc_ref_pos[0]);
            float3 x2 = r2.apply(p.sc_ref_pos[1]);

            float3 disp = x1-x2;
            float  dist = mag(disp);
            float  reduced_coord = p.scale * (dist - p.r0);

            if(reduced_coord<cutoff) {
                float  z = expf(reduced_coord);
                float  w = 1.f / (1.f + z);
                float  deriv_over_r = -p.scale/dist * p.energy * z * (w*w);
                float3 deriv = deriv_over_r * disp;

                r1.add_deriv_at_location(x1,  deriv);
                r2.add_deriv_at_location(x2, -deriv);
            }

            r1.flush();
            r2.flush();
        }
    }
}
