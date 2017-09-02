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

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer(string("environment_coverage"));

        igraph.compute_edges();

        fill(output, 0.f);
        for(int ne=0; ne<igraph.n_edge; ++ne)  // accumulate for each cb
            output(0, igraph.edge_indices1[ne]) += igraph.edge_value[ne];
        return 0;
    }

    virtual int propagate_deriv(int round) override {
        Timer timer(string("d_environment_coverage"));
        VecArray sens_acc = sens.accum();

        for(int ne: range(igraph.n_edge))
            igraph.edge_sensitivity[ne] = sens_acc(0,igraph.edge_indices1[ne]);
        igraph.propagate_derivatives();
        return 0;
    }

    virtual std::vector<float> get_param() const override {return igraph.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {return igraph.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) override {igraph.set_param(new_param);}
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

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer("weighted_pos");

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            output(0,ne) = pos.output(0,p.index_pos);
            output(1,ne) = pos.output(1,p.index_pos);
            output(2,ne) = pos.output(2,p.index_pos);
            output(3,ne) = expf(-energy.output(0,p.index_weight));
        }
        return 0;
    }

    virtual int propagate_deriv(int round) override {
        Timer timer("d_weighted_pos");
        VecArray sens_acc = sens.accum();
        VecArray pos_sens = pos.sens.acquire();
        VecArray energy_sens = energy.sens.acquire();

        for(int ne=0; ne<n_elem; ++ne) {
            auto p = params[ne];
            pos_sens(0,p.index_pos) += sens_acc(0,ne);
            pos_sens(1,p.index_pos) += sens_acc(1,ne);
            pos_sens(2,p.index_pos) += sens_acc(2,ne);
            energy_sens(0,p.index_weight) -= output(3,ne)*sens_acc(3,ne); // exponential derivative
        }
        pos.sens.release(pos_sens);
        energy.sens.release(energy_sens);
        return 0;
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

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer("uniform_transform");
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            output(0,ne) = v[0];
            jac[ne]      = v[1]*spline_inv_dx;
        }
        return 0;
    }

    virtual int propagate_deriv(int round) override {
        Timer timer("d_uniform_transform");
        VecArray input_sens = input.sens.acquire();
        VecArray sens_acc = sens.accum();
        for(int ne=0; ne<n_elem; ++ne)
            input_sens(0,ne) += jac[ne]*sens_acc(0,ne);
        input.sens.release(input_sens);
        return 0;
    }

    virtual std::vector<float> get_param() const override{
        vector<float> ret(2+n_coeff);
        ret[0] = spline_offset;
        ret[1] = spline_inv_dx;
        copy_n(bspline_coeff.get(), n_coeff, &ret[2]);
        return ret;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {
        vector<float> ret(2+n_coeff, 0.f);
        int starting_bin;
        float d[4];
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(bspline_coeff.get(), coord, n_coeff);
            clamped_deBoor_coeff_deriv(&starting_bin, d, coord, n_coeff);

            ret[0] += v[1];                                    // derivative of offset
            ret[1] += v[1]*(input.output(0,ne)-spline_offset); // derivative of inv_dx
            for(int i: range(4)) ret[2+starting_bin+i] += d[i];
        }
        return ret;
    }
#endif

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
};
static RegisterNodeType<UniformTransform,1> uniform_transform_node("uniform_transform");

struct LinearCoupling : public PotentialNode {
    CoordNode& input;
    vector<float> couplings;      // length n_restype
    vector<int>   coupling_types; // length input n_elem
    CoordNode* inactivation;  // 0 to 1 variable to inactivate energy
    int inactivation_dim;

    LinearCoupling(hid_t grp, CoordNode& input_, CoordNode& inactivation_):
        LinearCoupling(grp, &input_, &inactivation_) {}

    LinearCoupling(hid_t grp, CoordNode& input_):
        LinearCoupling(grp, &input_, nullptr) {}

    LinearCoupling(hid_t grp, CoordNode* input_, CoordNode* inactivation_):
        PotentialNode(),
        input(*input_),
        couplings(get_dset_size(1,grp,"couplings")[0]),
        coupling_types(input.n_elem),
        inactivation(inactivation_),
        inactivation_dim(inactivation ? read_attribute<int>(grp, ".", "inactivation_dim") : 0)
    {
        check_elem_width(input, 1);  // could be generalized

        if(inactivation) {
            if(input.n_elem != inactivation->n_elem)
                throw string("Inactivation size must match input size");
            check_elem_width_lower_bound(*inactivation, inactivation_dim+1);
        }

        check_size(grp, "coupling_types", input.n_elem);
        traverse_dset<1,float>(grp,"couplings",[&](size_t nt, float x){couplings[nt]=x;});
        traverse_dset<1,int>(grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        for(int i: coupling_types) if(i<0 || i>=int(couplings.size())) throw string("invalid coupling type");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>(
                    (inactivation ? "linear_coupling_with_inactivation" : "linear_coupling_uniform"),
                    {input.n_elem}, [&](float* buffer) {
                    for(int ne: range(input.n_elem)) {
                        float c = couplings[coupling_types[ne]];
                        buffer[ne] = c*input.output(0,ne);
                    }});
        }
    }

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer("linear_coupling");
        VecArray input_sens = input.sens.acquire();
        VecArray inactivation_sens;
        if(inactivation) inactivation_sens = inactivation->sens.acquire();
        int n_elem = input.n_elem;
        float pot = 0.f;
        for(int ne=0; ne<n_elem; ++ne) {
            float c = couplings[coupling_types[ne]];
            float act = inactivation ? sqr(1.f-inactivation->output(inactivation_dim,ne)) : 1.f;
            float val = input.output(0,ne);
            pot += c * val * act;
            input_sens(0,ne) += c*act;
            if(inactivation) inactivation_sens(inactivation_dim,ne) -= c*val;
        }
        potential = pot;
        input.sens.release(input_sens);
        if(inactivation) inactivation->sens.release(inactivation_sens);
        return 0;
    }

    virtual std::vector<float> get_param() const override {
        return couplings;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {
        vector<float> deriv(couplings.size(), 0.f);

        int n_elem = input.n_elem;
        for(int ne=0; ne<n_elem; ++ne) {
            float act = inactivation ? 1.f - inactivation->output(inactivation_dim,ne) : 1.f;
            deriv[coupling_types[ne]] += input.output(0,ne) * act;
        }
        return deriv;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() != couplings.size())
            throw string("attempting to change size of couplings vector on set_param");
        copy(begin(new_param),end(new_param), begin(couplings));
    }
};
static RegisterNodeType<LinearCoupling,1> linear_coupling_node1("linear_coupling_uniform");
static RegisterNodeType<LinearCoupling,2> linear_coupling_node2("linear_coupling_with_inactivation");


struct NonlinearCoupling : public PotentialNode {
    CoordNode& input;
    int n_restype, n_coeff;
    float spline_offset, spline_inv_dx;
    vector<float> coeff;      // length n_restype*n_coeff
    vector<int>   coupling_types; // length input n_elem

    NonlinearCoupling(hid_t grp, CoordNode& input_):
        PotentialNode(),
        input(input_),
        n_restype(get_dset_size(2,grp,"coeff")[0]),
        n_coeff  (get_dset_size(2,grp,"coeff")[1]),
        spline_offset(read_attribute<float>(grp,"coeff","spline_offset")),
        spline_inv_dx(read_attribute<float>(grp,"coeff","spline_inv_dx")),
        coeff(n_restype*n_coeff),
        coupling_types(input.n_elem)
    {
        check_elem_width(input, 1);  // could be generalized

        check_size(grp, "coupling_types", input.n_elem);
        traverse_dset<2,float>(grp,"coeff",[&](size_t nt, size_t nc, float x){coeff[nt*n_coeff+nc]=x;});
        traverse_dset<1,int>(grp,"coupling_types",[&](size_t ne, int x){coupling_types[ne]=x;});
        for(int i: coupling_types) if(i<0 || i>=n_restype) throw string("invalid coupling type");

        if(logging(LOG_DETAILED)) {
            default_logger->add_logger<float>("nonlinear_coupling", {input.n_elem}, [&](float* buffer) {
                    for(int ne: range(input.n_elem)) {
                        auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
                        buffer[ne] = clamped_deBoor_value_and_deriv(
                                coeff.data() + coupling_types[ne]*n_coeff, coord, n_coeff)[0];
                    }});
        }
    }

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer("nonlinear_coupling");
        VecArray input_sens = input.sens.acquire();
        int n_elem = input.n_elem;
        float pot = 0.f;
        for(int ne=0; ne<n_elem; ++ne) {
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            auto v = clamped_deBoor_value_and_deriv(
                    coeff.data() + coupling_types[ne]*n_coeff, coord, n_coeff);
            pot              += v[0];
            input_sens(0,ne) += v[1]*spline_inv_dx;
        }
        potential = pot;
        input.sens.release(input_sens);
        return 0;
    }

    virtual std::vector<float> get_param() const override {
        return coeff;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() override {
        vector<float> deriv(coeff.size(), 0.f);

        int n_elem = input.n_elem;
        for(int ne=0; ne<n_elem; ++ne) {
            int starting_bin;
            float result[4];
            auto coord = (input.output(0,ne)-spline_offset)*spline_inv_dx;
            clamped_deBoor_coeff_deriv(&starting_bin, result, coord, n_coeff);
            for(int i: range(4)) deriv[coupling_types[ne]*n_coeff+starting_bin+i] += result[i];
        }
        return deriv;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) override {
        if(new_param.size() != coeff.size())
            throw string("attempting to change size of coeff vector on set_param");
        copy(begin(new_param),end(new_param), begin(coeff));
    }
};
static RegisterNodeType<NonlinearCoupling,1> nonlinear_coupling_node("nonlinear_coupling");

}
