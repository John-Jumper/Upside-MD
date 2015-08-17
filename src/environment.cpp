#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include "interaction_graph.h"

using namespace std;
using namespace h5;

namespace {
    struct EnvironmentCoverageInteraction {
        // radius scale angular_width angular_scale
        constexpr static const int n_param=6, n_dim1=6, n_dim2=4, n_deriv=7;

        static float cutoff(const Vec<n_param> &p) {
            return p[0] + compact_sigmoid_cutoff(p[1]);
        }

        static bool exclude_by_id(unsigned id1, unsigned id2) { 
            return id1==id2; // exclude by residue (id's should contain residues)
        }

        static float compute_edge(Vec<n_deriv> &d_base, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            // I will keep the hbond-like terminology, since this is mathematically almost identical
            float3 displace = extract<0,3>(sc_pos)-extract<0,3>(hb_pos);
            float3 rHN = extract<3,6>(hb_pos);
            float weight = sc_pos[3];
            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist = dist2*inv_dist;
            float3 displace_unitvec = inv_dist*displace;
            float  cos_coverage_angle = dot(rHN,displace_unitvec);

            float2 radial_cover    = compact_sigmoid(dist-p[0], p[1]);
            float2 angular_sigmoid = compact_sigmoid(p[2]-cos_coverage_angle, p[3]);
            float2 angular_cover   = make_vec2(p[4]+p[5]*angular_sigmoid.x(), p[5]*angular_sigmoid.y());

            float3 col0, col1, col2;
            hat_deriv(displace_unitvec, inv_dist, col0, col1, col2);
            float3 deriv_dir = make_vec3(dot(col0,rHN), dot(col1,rHN), dot(col2,rHN));

            float3 d_displace = (angular_cover.x()*radial_cover.y()) * displace_unitvec + 
                                (-radial_cover.x()*angular_cover.y()) * deriv_dir;
            float3 d_rHN = (-radial_cover.x()*angular_cover.y()) * displace_unitvec;

            float coverage = radial_cover.x() * angular_cover.x();

            store<0,3>(d_base, weight * d_displace);
            store<3,6>(d_base, weight * d_rHN);
            d_base[6] = coverage; // weight derivative

            return weight * coverage;
        }

        static void expand_deriv(Vec<n_dim1> &d_hb, Vec<n_dim2> &d_sc, const Vec<n_deriv> &d_base) {
            store<0,3>(d_hb, -extract<0,3>(d_base));  // opposite of d_sc by Newton's third
            store<3,6>(d_hb,  extract<3,6>(d_base));
            store<0,3>(d_sc,  extract<0,3>(d_base));
            d_sc[3] = d_base[6];
        }

        static void param_deriv(Vec<n_param> &d_param, const Vec<n_param> &p, 
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            float3 displace = extract<0,3>(sc_pos)-extract<0,3>(hb_pos);
            float3 rHN = extract<3,6>(hb_pos);
            float weight = sc_pos[3];

            float  dist2 = mag2(displace);
            float  inv_dist = rsqrt(dist2);
            float  dist = dist2*inv_dist;
            float3 displace_unitvec = inv_dist*displace;
            float  cos_coverage_angle = dot(rHN,displace_unitvec);

            float2 radial_cover    = compact_sigmoid(dist-p[0], p[1]);
            float2 angular_sigmoid = compact_sigmoid(p[2]-cos_coverage_angle, p[3]);
            float2 angular_cover   = make_vec2(p[4]+p[5]*angular_sigmoid.x(), p[5]*angular_sigmoid.y());

            float radial_cover_s =(dist-p[0])*compact_sigmoid((dist-p[0])*p[1],1.f).y();
            float angular_cover_s=p[5]*(p[2]-cos_coverage_angle)*compact_sigmoid((p[2]-cos_coverage_angle)*p[3],
                    1.f).y();

            d_param[0] = weight * -radial_cover.y() * angular_cover.x();
            d_param[1] = weight *  radial_cover_s   * angular_cover.x();
            d_param[2] = weight *  radial_cover.x() * angular_cover.y();
            d_param[3] = weight *  radial_cover.x() * angular_cover_s;
            d_param[4] = weight *  radial_cover.x();
            d_param[5] = weight *  radial_cover.x() * angular_sigmoid.x();
        }
    };
}


struct EnvironmentCoverage : public CoordNode {
    BetweenInteractionGraph<EnvironmentCoverageInteraction> igraph;
    int n_restype;

    EnvironmentCoverage(hid_t grp, CoordNode& rotamer_sidechains_, CoordNode& weighted_sidechains_):
        CoordNode(rotamer_sidechains_.n_system, 
                get_dset_size(1,grp,"index1")[0], 
                get_dset_size(3,grp,"interaction_param")[1]),
        igraph(grp, rotamer_sidechains_, weighted_sidechains_),
        n_restype(elem_width)
    {}

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("environment"));
        for(int ns=0; ns<n_system; ++ns) {
            VecArray env = coords().value[ns];
            fill(env, n_restype, n_elem, 0.f);

            // Compute coverage and its derivative
            igraph.compute_edges(ns, [&](int ne, float cov,
                        int index1, unsigned type1, unsigned id1,
                        int index2, unsigned type2, unsigned id2) {
                    env(type2,index1) += cov;});
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("environment_deriv"));
        for(int ns=0; ns<n_system; ++ns) {
            vector<float> sens(n_elem*n_restype, 0.f);
            VecArray accum = slot_machine.accum_array()[ns];

            for(auto tape_elem: slot_machine.deriv_tape)
                for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                    for(int rt: range(n_restype))
                        sens[tape_elem.atom*n_restype+rt] += accum(rt, tape_elem.loc+rec);

            // Push coverage derivatives
            for(int ned: range(igraph.n_edge[ns])) {
                int sc_num = igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 0];
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
static RegisterNodeType<EnvironmentCoverage,2> environment_coverage_node("environment");
