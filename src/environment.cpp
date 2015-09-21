#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include "interaction_graph.h"
#include <Eigen/Dense>

using namespace std;
using namespace h5;
using namespace Eigen;

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
    {
        if(logging(LOG_EXTENSIVE)) {
            default_logger->add_logger<float>("environment_vector", {n_elem, n_restype}, [&](float* buffer) {
                    for(int ne: range(n_elem))
                        for(int rt: range(n_restype))
                            buffer[ne*n_restype+rt] = coords().value[0](rt,ne);});
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("environment_vector"));
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
        Timer timer(string("environment_vector_deriv"));
        for(int ns=0; ns<n_system; ++ns) {
            vector<float> sens(n_elem*n_restype, 0.f);
            VecArray accum = slot_machine.accum_array()[ns];

            for(auto tape_elem: slot_machine.deriv_tape)
                for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                    for(int rt: range(n_restype))
                        sens[tape_elem.atom*n_restype+rt] += accum(rt, tape_elem.loc+rec);

            // Push coverage derivatives
            for(int ned: range(igraph.n_edge[ns])) {
                int sc_num0 = igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 0];
                int sc_num1 = igraph.edge_indices[ns*2*igraph.max_n_edge + 2*ned + 1];
                int sc_idx0 = igraph.param1[sc_num0].index;
                int sc_type1= igraph.types2[sc_num1];
                igraph.use_derivative(ns, ned, sens[sc_idx0*n_restype+sc_type1]);
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
static RegisterNodeType<EnvironmentCoverage,2> environment_coverage_node("environment_vector");


struct SimpleEnvironmentEnergy : public PotentialNode {
    CoordNode &env_vector;

    int n_elem;
    int n_restype;

    vector<float> coeff;  // size (n_elem, n_restype)
    vector<slot_t> slots;

    SimpleEnvironmentEnergy(hid_t grp, CoordNode& env_vector_):
        PotentialNode(env_vector_.n_system),
        env_vector(env_vector_),
        n_elem(env_vector.n_elem),
        n_restype(env_vector.elem_width),
        coeff(n_elem*n_restype) 
    {
        if(n_system!=1) throw string("impossible");
        check_size(grp, "coefficients", n_elem , n_restype);
        traverse_dset<2,float>(grp, "coefficients", [&](size_t ne, size_t rt, float x) {coeff[ne*n_restype+rt]=x;});

        for(CoordPair p(0u,0u); p.index<unsigned(n_elem); ++p.index) {
            env_vector.slot_machine.add_request(1, p);
            slots.push_back(p.slot);
        }

        if(logging(LOG_DETAILED))
            default_logger->add_logger<float>("simple_environment_potential", {n_elem}, [&](float* buffer) {
                VecArray ev = env_vector.coords().value[0];
                for(int ne: range(n_elem)) {
                    buffer[ne] = 0.f;
                    for(int rt: range(n_restype))
                        buffer[ne] += coeff[ne*n_restype+rt] * ev(rt,ne);
                }});

    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer("simple_environment");

        if(mode==PotentialAndDerivMode) {
            VecArray ev = env_vector.coords().value[0];
            float pot = 0.f;
            for(int ne: range(n_elem))
                for(int rt: range(n_restype))
                    pot += coeff[ne*n_restype+rt] * ev(rt,ne);
            potential[0] = pot;
        }

        VecArray vec_accum = env_vector.coords().deriv[0];
        for(int ne: range(n_elem))
            for(int rt: range(n_restype))
                vec_accum(rt,slots[ne]) = coeff[ne*n_restype+rt];
    }
};
static RegisterNodeType<SimpleEnvironmentEnergy,1> simple_environment_node("simple_environment");


struct EnvironmentEnergy : public CoordNode {
    CoordNode &env_vector;

    int n_hidden;
    int n_restype;

    MatrixXf weight0;
    MatrixXf weight1;

    RowVectorXf shift0;
    RowVectorXf shift1;

    MatrixXf input;
    MatrixXf layer1;
    MatrixXf output;   // output probabilities
    MatrixXf all_prob;   // softmax probabilites on output

    vector<int> output_restype;

    VectorXf sens;
    vector<slot_t> slots;

    EnvironmentEnergy(hid_t grp, CoordNode &env_vector_):
        CoordNode (env_vector_.n_system, env_vector_.n_elem, 1),
        env_vector(env_vector_),

        n_hidden(get_dset_size(1, grp, "linear_shift0")[0]),
        n_restype(env_vector.elem_width),

        weight0(n_restype, n_hidden),
        weight1(n_hidden,  n_restype),

        shift0(n_hidden),
        shift1(n_restype),

        input (n_elem,    n_restype),
        layer1(n_elem,    n_hidden),
        output(n_elem,    n_restype),
        all_prob(n_elem,  n_restype),

        sens(n_elem)
    {
        setNbThreads(1); // disable Eigen threading
        check_size(grp, "linear_weight0",  n_restype, n_hidden);
        check_size(grp, "linear_weight1",  n_hidden,  n_restype);
        check_size(grp, "linear_shift0",  n_hidden);
        check_size(grp, "linear_shift1",  n_restype);  // this is actually not needed
        check_size(grp, "output_restype", n_elem);

        traverse_dset<2,float>(grp, "linear_weight0", [&](size_t i, size_t j, float x) {weight0(i,j) = x;});
        traverse_dset<2,float>(grp, "linear_weight1", [&](size_t i, size_t j, float x) {weight1(i,j) = x;});
        traverse_dset<1,float>(grp, "linear_shift0",  [&](size_t i, float x) {shift0[i] = x;});
        traverse_dset<1,float>(grp, "linear_shift1",  [&](size_t i, float x) {shift1[i] = x;});
        traverse_dset<1,int>  (grp, "output_restype", [&](size_t ne, int x) {output_restype.push_back(x);});

        for(CoordPair p(0u,0u); p.index<unsigned(n_elem); ++p.index) {
            env_vector.slot_machine.add_request(1, p);
            slots.push_back(p.slot);
        }
    }

    
    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("environment_energy"));

        VecArray v = env_vector.coords().value[0];  // only support 1 system
        VecArray e = coords().value[0];

        for(int ne: range(n_elem)) for(int d: range(n_restype)) input(ne,d) = v(d,ne);

        // apply first linear layer
        layer1.noalias() = (input*weight0).rowwise() + shift0;

        // apply relu
        for(int ne: range(n_elem)) for(int d: range(n_hidden)) layer1(ne,d) = max(0.f, layer1(ne,d));

        // apply second linear layer
        output.noalias() = (layer1*weight1).rowwise() + shift1;

        // apply negative log-softmax
        // -log(softmax) = -log(exp(x_j) / sum_k(exp(x_k))) 

        // FIXME most of my runtime is in the exponentials being calculated
        all_prob = (output.colwise()-output.rowwise().maxCoeff()).array().exp();
        all_prob.array().colwise() *= 1.f/all_prob.rowwise().sum().array();

        // to center the numbers near 0., let's subtract the naive guess of log(n_restype)
        const float offset = -logf(n_restype);

        for(int ne: range(n_elem))
            e(0,ne) = -logf(all_prob(ne, output_restype[ne])) + offset;
    }


    virtual void propagate_deriv() override {
        Timer timer(string("environment_energy_deriv"));

        sens = VectorXf::Zero(n_elem);
        VecArray accum = slot_machine.accum_array()[0];

        for(auto tape_elem: slot_machine.deriv_tape)
            for(int rec=0; rec<int(tape_elem.output_width); ++rec)
                sens[tape_elem.atom] += accum(0, tape_elem.loc+rec);

        // now let's run the backpropagation (the sensitivity will be used at the end)
        output.noalias() = all_prob;

        for(int ne: range(n_elem))
            output(ne, output_restype[ne]) -= 1.f;

        // also handle the relu
        layer1 = (layer1.array()>0.f).cast<float>().array() * (output * weight1.transpose()).array();

        input = layer1 * weight0.transpose();

        VecArray vec_accum = env_vector.coords().deriv[0];
        for(int ne: range(n_elem))
            for(int rt: range(n_restype))
                vec_accum(rt,slots[ne]) = sens[ne] * input(ne,rt);
    }

#ifdef PARAM_DERIV
    virtual vector<float> get_param() const override {
        // FIXME fake version just to extract the derivative
        vector<float> result(n_elem*n_restype);
        for(int ne: range(n_elem))
            for(int rt: range(n_restype))
                result[ne*n_restype + rt] = input(ne,rt);
        return result;
    }
#endif

};
static RegisterNodeType<EnvironmentEnergy,1> environment_energy_node("environment_energy");
