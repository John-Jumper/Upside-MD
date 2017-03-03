#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include <Eigen/Dense>

using namespace h5;
using namespace std;
using namespace Eigen;

template <typename T>
void printit(const T& x) {
    printf(" (%lu,%lu)", x.rows(), x.cols());
}

namespace {
    vector<float> copy_eigen_to_stl(const ArrayXXf& mat) {
        // Output is row-major order, which is the opposite of the Eigen default
        vector<float> ret(mat.rows()*mat.cols());

        int k=0;
        for(int i: range(mat.rows())) for(int j: range(mat.cols()))
            ret[k++] = mat(i,j);

        return ret;
    }

    void copy_stl_to_eigen(ArrayXXf& mat, const vector<float>& input) {
        // Input is row-major order, which is the opposite of the Eigen default
        if(int(input.size()) != mat.rows()*mat.cols())
            throw string("Invalid dimensions for conversion to matrix");

        int k=0;
        for(int i: range(mat.rows())) for(int j: range(mat.cols()))
            mat(i,j) = input[k++];
    }
}

struct FixedHMM : public PotentialNode
{
    struct Params {
        index_t index;
    };

    int n_residue;
    int n_state;
    CoordNode& node_1body;
    vector<Params> params;

    float    energy_offset;
    ArrayXXf transition_energy;
    MatrixXf transition_matrix;

    VecArrayStorage emission_prob;
    VecArrayStorage forward_belief;
    vector<float>   residue_potential;

#ifdef PARAM_DERIV
    // posterior expectation of edge transition counts
    // this is also the parameter derivative of transition energies
    ArrayXXf edge_transition_counts;  
#endif

    void update_transition_matrix_from_transition_energy() {
        // Offset by the expected value of the energy to increase numerical stability
        float e_min = transition_energy.minCoeff();
        energy_offset = (transition_energy.array() * (e_min - transition_energy.array()).exp()).sum() / 
                                                      ((e_min - transition_energy.array()).exp()).sum();
        transition_matrix = (energy_offset-transition_energy).exp();
    }

    FixedHMM(hid_t grp, CoordNode& node_1body_):
        PotentialNode(),
        n_residue(get_dset_size(1, grp, "index")[0]), 
        n_state(get_dset_size(2, grp, "transition_energy")[0]),
        node_1body(node_1body_), 
        params(n_residue),
        transition_energy(n_state, n_state),
        transition_matrix  (n_state, n_state),
        emission_prob    (n_state, n_residue),
        forward_belief   (n_state, n_residue),
        residue_potential(n_residue)
#ifdef PARAM_DERIV
        ,edge_transition_counts(n_state, n_state)
#endif
    {
        check_size(grp, "index", n_residue);
        check_size(grp, "transition_energy", n_state, n_state);

        traverse_dset<1,int>  (grp, "index",  [&](size_t i, int x) {params[i].index = x;});
        traverse_dset<2,float>(grp, "transition_energy", [&](size_t ix, size_t iy, float x) {
                transition_energy(ix,iy) = x;});
        update_transition_matrix_from_transition_energy();

        if(logging(LOG_DETAILED)) 
            default_logger->add_logger<float>("hmm_energy", {1}, [&](float* buffer) {
                    *buffer = potential;});

        if(logging(LOG_DETAILED)) 
            default_logger->add_logger<float>("hmm_energy_1body", {n_residue}, [&](float* buffer) {
                    // this sums to the free energy, except it is missing the mutual information correction
                for(int nr: range(n_residue)) 
                    buffer[nr] = residue_potential[nr];
                    });
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("hmm"));
        VecArray n1b = node_1body.output;

        float pot = energy_offset*(n_residue-1.f);  // correct for energy offset

        // Timer tprep("hmm_prep");
        // convert energies to emission probabilities
        for(int nr=0; nr<n_residue; ++nr) {
            float* x = &n1b(0,params[nr].index);

            // first find the minimum energy
            // note that this code must be correct even if n_state<4
            auto e_min = Float4(x[0]);
            // handle full groups after first group
            int ns=0;
            for(; ns<n_state-3; ns+=4)
                e_min = min(e_min, Float4(x+ns));
            // handle partial group at end
            for(; ns<n_state; ++ns)
                e_min = min(e_min, Float4(x[ns]));

            // now put minimum in all elements of e_min Float4
            e_min = min(shuffle<0,1,0,1>(e_min), shuffle<2,3,2,3>(e_min));
            e_min = min(e_min.broadcast<0>(), e_min.broadcast<1>());
            if(mode==PotentialAndDerivMode) pot += e_min.x();

            // write emmision probabilities
            float* p = &emission_prob(0,nr);
            for(int ns=0; ns<n_state; ns+=4)
                expf(e_min - Float4(x+ns)).store(p+ns);
        }
        // tprep.stop();

        // Timer tforw("hmm_forward");
        // compute forward probabilities
        RowVectorXf forward = RowVectorXf::Ones(n_state);
        for(int nr=0; nr<n_residue; ++nr) {
            if(nr) forward = forward*transition_matrix;
            forward.array() *= Map<RowVectorXf>(&emission_prob(0,nr), n_state).array();
            float norm = forward.sum();
            if(mode==PotentialAndDerivMode) pot -= logf(norm);
            forward *= rcp(norm);
            Map<RowVectorXf>(&forward_belief(0,nr), n_state) = forward;
        }
        if(mode==PotentialAndDerivMode) potential = pot;
        // tforw.stop();

        // Timer tback("hmm_backward");
        // compute backward probabilities and marginal probabilities
        VecArray sens = node_1body.sens;
        VectorXf backward = VectorXf::Ones(n_state);
        VectorXf marginal(n_state);
#ifdef PARAM_DERIV
        edge_transition_counts = MatrixXf::Zero(n_state,n_state);
#endif

        for(int nr=n_residue-1; nr>=0; --nr) {
            marginal.array() = Map<VectorXf>(&forward_belief(0,nr), n_state).array()*backward.array();
            marginal *= rcp(marginal.sum());
            Map<VectorXf>(&sens(0,nr), n_state) += marginal;

            if(mode==PotentialAndDerivMode) {
                Map<VectorXf> en(&n1b(0,params[nr].index), n_state);
                auto avg_energy_1body = marginal.dot(en);
                auto entropy_1body =  marginal.dot((marginal+1e-8f*VectorXf::Ones(n_state)).array().log().matrix());

                // // calculation edge mutual information
                // auto& left_belief = forward_belief(0,nr-1);
                // VectorXf right_belief = (Map<VectorXf>(&emission_prob(0,nr), n_state).array() * backward.array()).matrix();
                // MatrixXf prob = (left_belief*right_belief.transpose()).array() * transition_matrix;
                // float norm = prob.sum();
                // prob *= rcp(prob.sum());
                // don't forget the average energy of the edge matrices

                residue_potential[nr] = avg_energy_1body - entropy_1body + energy_offset*((n_residue-1.f)/n_residue);
            }


            if(nr>0) {
                backward.array() *= Map<VectorXf>(&emission_prob(0,nr), n_state).array();

                #ifdef PARAM_DERIV
                // parameter derivative of the transition energy is just the expected edge transition count
                // the expected edge transition count is just the renormalized element-wise product of the transition matrix
                //   and the outer product of the beliefs on either side.  Note that we want to catch the backward belief before
                //   we matrix multiply in another copy of the transition matrix.
                ArrayXXf unnormalized_counts = transition_matrix.array() * (
                        Map<VectorXf>(&forward_belief(0,nr-1), n_state) *
                        backward.transpose()).array();
                edge_transition_counts += unnormalized_counts * rcp(unnormalized_counts.sum());
                #endif

                backward = transition_matrix*backward;
                backward *= rcp(backward.sum());
            }

        }
        // tback.stop();
    }

    virtual vector<float> get_param() const override {return copy_eigen_to_stl(transition_energy);}
#ifdef PARAM_DERIV
    virtual vector<float> get_param_deriv() override {return copy_eigen_to_stl(edge_transition_counts);}
#endif
    virtual void set_param(const vector<float>& new_param) override {
        copy_stl_to_eigen(transition_energy,new_param);
        update_transition_matrix_from_transition_energy();
    }
};
static RegisterNodeType<FixedHMM,1> fixed_hmm_node("fixed_hmm");

struct TorusDBN_Emission : public CoordNode {
    struct Params {
        index_t residue;
    };

    CoordNode& rama;

    int n_restype;
    int n_residue;
    int n_state;
    vector<int> restypes;

    vector<Params> params;
    VecArrayStorage bp;             // size (6, n_state)
    VecArrayStorage prior_offset_energies;   // size (n_state, n_restype)
    MatrixXf prior_offset;                   // size (ru(n_state),n_residue)
    Matrix<float,Dynamic,6> cs;              // size (n_residue,6)
    Matrix<float,6,Dynamic> cs_to_emission;  // size (6,ru(n_state))
    Matrix<float,6,Dynamic> cs_sens;         // size (6,n_residue)

    unique_ptr<float> basin_param;

    void update_prior_offset() {
        for(int nr: range(n_residue)) for(int ns: range(n_state))
            prior_offset(ns,nr) = prior_offset_energies(ns,restypes[nr]) + bp(0,ns); // add log normalization of each basin
    }

    TorusDBN_Emission(hid_t grp, CoordNode& rama_):
        CoordNode(get_dset_size(1, grp, "id")[0], get_dset_size(2, grp, "basin_param")[0]),
        rama(rama_),

        n_restype(get_dset_size(2, grp, "prior_offset_energies")[0]),
        n_residue(get_dset_size(1, grp, "id")[0]),
        n_state  (get_dset_size(2, grp, "basin_param")[0]),
        restypes (n_residue),

        params(n_residue),
        bp(6,n_state),
        prior_offset_energies(n_state, n_restype),
        prior_offset  (MatrixXf               ::Zero(ru(n_state),n_residue)),
        cs            (Matrix<float,Dynamic,6>::Zero(n_residue,6)),
        cs_to_emission(Matrix<float,6,Dynamic>::Zero(6,ru(n_state))),
        cs_sens       (Matrix<float,6,Dynamic>::Zero(6,n_residue))
    {
        check_size(grp, "id", n_residue);
        check_size(grp, "restypes", n_residue);
        check_size(grp, "prior_offset_energies", n_restype, n_state);
        check_size(grp, "basin_param", n_state, 6);

        traverse_dset<1,int>(grp, "id", [&](int nr, int x) {params[nr].residue = x;});
        traverse_dset<1,int>(grp, "restypes", [&](int nr, int x) {restypes[nr] = x;});
        traverse_dset<2,float>(grp, "basin_param", [&](int ns, int np, float x) {bp(np,ns) = x;});

        traverse_dset<2,float>(grp, "prior_offset_energies", [&](int nrt, int ns, float x) {
                prior_offset_energies(ns,nrt) = x;});
        update_prior_offset();

        for(int ns: range(n_state)) {
            auto kappa_phi = bp(1,ns);
            auto angle_phi = bp(2,ns);

            auto kappa_psi = bp(3,ns);
            auto angle_psi = bp(4,ns);

            auto kappa_cor = bp(5,ns);
            auto angle_cor = angle_phi-angle_psi;

            auto f = [&](int i,float k,float a){
                cs_to_emission(i*2+0,ns) = k*cosf(a);
                cs_to_emission(i*2+1,ns) = k*sinf(a);
            };

            f(0, -kappa_phi,angle_phi);
            f(1, -kappa_psi,angle_psi);
            f(2,  kappa_cor,angle_cor); // this is the sign convention in the TorusDBN paper
        }
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("torus_dbn"));
        VecArray rpos = rama.output;
        for(int nr=0; nr<n_residue; ++nr) {
            float phi = rpos(0,params[nr].residue);
            float psi = rpos(1,params[nr].residue);

            cs(nr,0) = cosf(phi);
            cs(nr,1) = sinf(phi);
                   
            cs(nr,2) = cosf(psi);
            cs(nr,3) = sinf(psi);
                   
            cs(nr,4) = cosf(phi-psi);
            cs(nr,5) = sinf(phi-psi);
        }

        Map<Matrix<float,Dynamic,Dynamic,RowMajor>> output_matrix(output.x.get(), n_residue, ru(n_state));
        output_matrix = prior_offset.transpose() + cs*cs_to_emission;
    }

    virtual void propagate_deriv() {
        Timer timer(string("torus_dbn_deriv"));
        Map<Matrix<float,Dynamic,Dynamic,RowMajor>> state_sens(sens.x.get(), n_residue, ru(n_state));
        cs_sens = cs_to_emission*state_sens.transpose();

        VecArray rsens = rama.sens;
        for(int nr=0; nr<n_residue; ++nr) {
            int i = params[nr].residue;
            // as we all learned long ago, d(cos) = -sin and d(sin) = cos
            float phi_sens = -cs(nr,1)*cs_sens(0,nr) + cs(nr,0)*cs_sens(1,nr) 
                             -cs(nr,5)*cs_sens(4,nr) + cs(nr,4)*cs_sens(5,nr);
            float psi_sens = -cs(nr,3)*cs_sens(2,nr) + cs(nr,2)*cs_sens(3,nr) 
                             +cs(nr,5)*cs_sens(4,nr) - cs(nr,4)*cs_sens(5,nr);
            rsens(0,i) += phi_sens;
            rsens(1,i) += psi_sens;
        }
    }

    virtual vector<float> get_param() const {
        vector<float> ret(n_restype*n_state);
        for(int nrt: range(n_restype)) for(int ns: range(n_state))
            ret[nrt*n_state+ns] = prior_offset_energies(ns,nrt);
        return ret;
    }
#ifdef PARAM_DERIV
    virtual vector<float> get_param_deriv() {
        vector<float> ret(n_restype*n_state, 0.f);
        for(int nr: range(n_residue)) for(int ns: range(n_state))
            ret[restypes[nr]*n_state+ns] += sens(ns,nr);
        return ret;
    }
#endif
    virtual void set_param(const vector<float>& new_param) {
        if(new_param.size() != size_t(n_restype*n_state))
            throw string("Invalid dimensions for prior energy table");

        for(int nrt: range(n_restype)) for(int ns: range(n_state))
            prior_offset_energies(ns,nrt) = new_param[nrt*n_state+ns];
        update_prior_offset();
    }
};
static RegisterNodeType<TorusDBN_Emission,1> torus_dbn_node("torus_dbn");
