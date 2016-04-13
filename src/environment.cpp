#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include "spline.h"
#include "interaction_graph.h"
#include <algorithm>

using namespace std;
using namespace h5;

namespace {
    struct EnvironmentCoverageInteraction {
        // parameters are r0,r_sharpness, dot0,dot_sharpness

        constexpr static bool  symmetric = false;
        constexpr static int   n_param=4, n_dim1=6, n_dim2=4, simd_width=1;

        static float cutoff(const float* p) {
            return p[0] + compact_sigmoid_cutoff(p[1]);
        }

        static Int4 acceptable_id_pair(const Int4& id1, const Int4& id2) {
            auto sequence_exclude = Int4(2);  // exclude i,i, i,i+1, and i,i+2
            return (sequence_exclude < id1-id2) | (sequence_exclude < id2-id1);
        }

        static Float4 compute_edge(Vec<n_dim1,Float4> &d1, Vec<n_dim2,Float4> &d2, const float* p[4],
                const Vec<n_dim1,Float4> &cb_pos, const Vec<n_dim2,Float4> &sc_pos) {
            Float4 one(1.f);
            auto displace = extract<0,3>(sc_pos)-extract<0,3>(cb_pos);
            auto rvec1 = extract<3,6>(cb_pos);
            auto prob  = sc_pos[3];

            auto dist2 = mag2(displace);
            auto inv_dist = rsqrt(dist2);
            auto dist = dist2*inv_dist;
            auto displace_unitvec = inv_dist*displace;

            // read parameters then transpose
            Float4 r0(p[0]);
            Float4 r_sharpness(p[1]);
            Float4 dot0(p[2]);
            Float4 dot_sharpness(p[3]);
            transpose4(r0,r_sharpness,dot0,dot_sharpness);

            auto dp = dot(displace_unitvec,rvec1);
            auto radial_sig  = compact_sigmoid(dist-r0, r_sharpness);
            auto angular_sig = compact_sigmoid(dot0-dp, dot_sharpness);

            // now we compute derivatives (minus sign is from the derivative of angular_sig)
            auto d_displace = prob*(radial_sig.y()*angular_sig.x() * displace_unitvec -
                                    radial_sig.x()*angular_sig.y()* inv_dist*(rvec1 - dp*displace_unitvec));

            store<3,6>(d1, -prob*radial_sig.x()*angular_sig.y()*displace_unitvec);
            store<0,3>(d1, -d_displace);
            store<0,3>(d2,  d_displace);
            auto score = radial_sig.x() * angular_sig.x();
            d2[3] = score;
            return prob * score;
        }

        static void param_deriv(Vec<n_param> &d_param, const float* p,
                const Vec<n_dim1> &hb_pos, const Vec<n_dim2> &sc_pos) {
            d_param = make_zero<n_param>();   // not implemented currently
        }

        static bool is_compatible(const float* p1, const float* p2) {return true;};
    };


struct EnvironmentCoverage : public CoordNode {
    InteractionGraph<EnvironmentCoverageInteraction> igraph;

    EnvironmentCoverage(hid_t grp, CoordNode& cb_pos_, CoordNode& weighted_sidechains_):
        CoordNode(get_dset_size(1,grp,"index1")[0], 1),
        igraph(grp, &cb_pos_, &weighted_sidechains_)
    {
        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("environment_coverage", {n_elem}, [&](float* buffer) {
                    for(int ne: range(n_elem))
                            buffer[ne] = output(0,ne);});
        }
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("environment_coverage"));

        igraph.compute_edges();

        fill(output, 0.f);
        for(int ne=0; ne<igraph.n_edge; ++ne)  // accumulate for each cb
            output(0, igraph.edge_indices1[ne]) += igraph.edge_value[ne];
    }

    virtual void propagate_deriv() override {
        Timer timer(string("d_environment_coverage"));

        for(int ne: range(igraph.n_edge))
            igraph.edge_sensitivity[ne] = sens(0,igraph.edge_indices1[ne]);
        igraph.propagate_derivatives();
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const {return igraph.get_param();}
    virtual std::vector<float> get_param_deriv() const {return igraph.get_param_deriv();}
    virtual void set_param(const std::vector<float>& new_param) {igraph.set_param(new_param);}
#endif
};
static RegisterNodeType<EnvironmentCoverage,2> environment_coverage_node("environment_coverage");


struct WeightedPos : public CoordNode {
    CoordNode &pos;
    CoordNode &energy;

    struct Param {
        index_t index_pos;
        index_t index_weight;
    };
    vector<Param> params;

    WeightedPos(hid_t grp, CoordNode& pos_, CoordNode& energy_):
        CoordNode(get_dset_size(1,grp,"index_pos")[0], 4),
        pos(pos_),
        energy(energy_),
        params(n_elem)
    {
        traverse_dset<1,int>(grp,"index_pos"   ,[&](size_t ne, int x){params[ne].index_pos   =x;});
        traverse_dset<1,int>(grp,"index_weight",[&](size_t ne, int x){params[ne].index_weight=x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("weighted_pos");

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            output(0,ne) = pos.output(0,p.index_pos);
            output(1,ne) = pos.output(1,p.index_pos);
            output(2,ne) = pos.output(2,p.index_pos);
            output(3,ne) = expf(-energy.output(0,p.index_weight));
        }
    }

    virtual void propagate_deriv() override {
        Timer timer("d_weighted_pos");

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            pos.sens(0,p.index_pos) += sens(0,ne);
            pos.sens(1,p.index_pos) += sens(1,ne);
            pos.sens(2,p.index_pos) += sens(2,ne);
            energy.sens(0,p.index_weight) -= output(3,ne)*sens(3,ne); // exponential derivative
        }
    }
};
static RegisterNodeType<WeightedPos,2> weighted_pos_node("weighted_pos");

struct UniformTransform : public CoordNode {
    CoordNode& input;
    int n_coeff;
    float spline_offset;
    float spline_inv_dx;
    unique_ptr<float[]> bspline_coeff;
    unique_ptr<float[]> jac;
    

    UniformTransform(hid_t grp, CoordNode& input_):
        CoordNode(input_.n_elem, 1),
        input(input_),
        n_coeff(get_dset_size(1,grp,"bspline_coeff")[0]),
        spline_offset(read_attribute<float>(grp,"bspline_coeff","spline_offset")),
        spline_inv_dx(read_attribute<float>(grp,"bspline_coeff","spline_inv_dx")),
        bspline_coeff(new_aligned<float>(n_coeff,4)),
        jac          (new_aligned<float>(input.n_elem,4))
    {
        check_elem_width(input,1); // this restriction could be lifted
        fill_n(bspline_coeff.get(), round_up(n_coeff,4), 0.f);
        traverse_dset<1,float>(grp,"bspline_coeff",[&](size_t ne, float x){bspline_coeff[ne]=x;});
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("uniform_transform");
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            output(0,ne) = v[0];
            jac[ne]      = v[1]*spline_inv_dx;
        }
    }

    virtual void propagate_deriv() override {
        Timer timer("d_uniform_transform");
        for(int ne=0; ne<n_elem; ++ne)
            input.sens(0,ne) += jac[ne]*sens(0,ne);
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const override{
        vector<float> ret(2+n_coeff);
        ret[0] = spline_offset;
        ret[1] = spline_inv_dx;
        copy_n(bspline_coeff.get(), n_coeff, &ret[2]);
        return ret;
    }
        
    virtual std::vector<float> get_param_deriv() const override {
        vector<float> ret(2+n_coeff, 0.f);
        int starting_bin;
        float d[4];
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            clamped_deBoor_coeff_deriv(&starting_bin, d, bspline_coeff.get(), coord, n_coeff);

            ret[0] += v[1];                                    // derivative of offset
            ret[1] += v[1]*(input.output(0,ne)-spline_offset); // derivative of inv_dx
            for(int i: range(4)) ret[2+starting_bin+i] += d[i];
        }
        return ret;
    }

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() < size_t(2+4)) throw string("too small of size for spline");
        if(int(new_param.size())-2 != n_coeff) {
            n_coeff = int(new_param.size())-2;
            bspline_coeff = new_aligned<float>(n_coeff,4);
            fill_n(bspline_coeff.get(), round_up(n_coeff,4), 0.f);
        }
        spline_offset = new_param[0];
        spline_inv_dx = new_param[1];
        copy_n(begin(new_param)+2, n_coeff, bspline_coeff.get());
    }
#endif
};
static RegisterNodeType<UniformTransform,1> uniform_transform_node("uniform_transform");

struct LinearCoupling : public PotentialNode {
    CoordNode& input;
    vector<float> couplings;      // length n_restype
    vector<int>   coupling_types; // length input n_elem

    LinearCoupling(hid_t grp, CoordNode& input_):
        PotentialNode(),
        input(input_),
        couplings(get_dset_size(1,grp,"couplings")[0]),
        coupling_types(input.n_elem)
    {
        check_elem_width(input, 1);  // could be generalized

        check_size(grp, "coupling_types", input.n_elem);
        traverse_dset<1,float>(grp,"couplings",[&](size_t nt, float x){couplings[nt]=x;});
        traverse_dset<1,int>(grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        for(int i: coupling_types) if(i<0 || i>=int(couplings.size())) throw string("invalid coupling type");
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("linear_coupling");
        int n_elem = input.n_elem;
        float pot = 0.f;
        for(int ne=0; ne<n_elem; ++ne) {
            float c = couplings[coupling_types[ne]];
            pot += c*input.output(0,ne);
            input.sens(0,ne) += c;
        }
        potential = pot;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param() const override {
        return couplings;
    }
        
    virtual std::vector<float> get_param_deriv() const override {
        vector<float> deriv(couplings.size(), 0.f);

        int n_elem = input.n_elem;
        for(int ne=0; ne<n_elem; ++ne) {
            deriv[coupling_types[ne]] += input.output(0,ne);
        }
        return deriv;
    }

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() != couplings.size()) 
            throw string("attempting to change size of couplings vector on set_param");
        copy(begin(new_param),end(new_param), begin(couplings));
    }
#endif
};
static RegisterNodeType<LinearCoupling,1> linear_coupling_node("linear_coupling");

}
