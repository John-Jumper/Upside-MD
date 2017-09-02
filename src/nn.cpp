#include "deriv_engine.h"
#include "timing.h"
#include "state_logger.h"
#include <Eigen/Dense>

using namespace std;
using namespace h5;
using namespace Eigen;

inline float relu(float x) {
    return x<0.f ? 0.f : x;
}


struct BackboneFeaturizer : public CoordNode
{
    CoordNode& rama;
    CoordNode& hbond;

    struct Param {
        int rama_idx;
        int donor_idx;
        int acceptor_idx;
    };
    vector<Param> params;

    BackboneFeaturizer(hid_t grp, CoordNode& rama_, CoordNode& hbond_):
        CoordNode(get_dset_size(1, grp, "rama_idx")[0], 6),
        rama(rama_),
        hbond(hbond_),
        params(n_elem)
    {
        check_size(grp, "rama_idx", n_elem);
        check_size(grp, "hbond_idx", n_elem, 2);

        traverse_dset<1,int>(grp, "rama_idx", [&](size_t ne, int x){params[ne].rama_idx=x;});

        // -1 is special code to indicate no hbond donor or acceptor
        traverse_dset<2,int>(grp, "hbond_idx", [&](size_t ne, size_t is_acceptor, int x){
                if(is_acceptor) params[ne].acceptor_idx=x;
                else            params[ne].donor_idx   =x;});
    }

    virtual int compute_value(int round, ComputeMode mode) override {
        VecArray ramac = rama.output;
        VecArray hbondc = hbond.output;

        for(int ne=0; ne<n_elem; ++ne) {
            auto& p = params[ne];

            float phi = ramac(0,p.rama_idx);
            float psi = ramac(1,p.rama_idx);
            float don_hb = p.donor_idx   ==-1 ? 0.f : hbondc(6,p.donor_idx);
            float acc_hb = p.acceptor_idx==-1 ? 0.f : hbondc(6,p.acceptor_idx);

            output(0,ne) = sin(phi);
            output(1,ne) = cos(phi);
            output(2,ne) = sin(psi);
            output(3,ne) = cos(psi);
            output(4,ne) = don_hb;
            output(5,ne) = acc_hb;
        }
        return 0;
    }

    virtual int propagate_deriv(int round) override {
        VecArray rama_s = rama.sens.acquire();
        VecArray hbond_s = hbond.sens.acquire();
        VecArray sens_acc = sens.accum();

        for(int ne=0; ne<n_elem; ++ne) {
            auto& p = params[ne];

            rama_s(0,p.rama_idx) += sens_acc(1,ne)*(-output(0,ne)) + sens_acc(0,ne)*output(1,ne);
            rama_s(1,p.rama_idx) += sens_acc(3,ne)*(-output(2,ne)) + sens_acc(2,ne)*output(3,ne);
            if(p.donor_idx   !=-1) hbond_s(6,p.donor_idx   ) += sens_acc(4,ne);
            if(p.acceptor_idx!=-1) hbond_s(6,p.acceptor_idx) += sens_acc(5,ne);
        }
        rama.sens.release(rama_s);
        hbond.sens.release(hbond_s);
        return 0;
    }
};
static RegisterNodeType<BackboneFeaturizer,2> backbone_featurizer_node("backbone_featurizer");


struct Conv1D : public CoordNode
{
    CoordNode& input;
    int n_elem_input;
    enum ActivationT {ReLU, Tanh, Identity};

    int conv_width, in_channels, out_channels;

    MatrixXf    weights;
    RowVectorXf bias;
    ActivationT activation;

    // temporaries
    MatrixXf input_conv_format;
    MatrixXf matmul_output;

    Conv1D(hid_t grp, CoordNode& input_):
        CoordNode   (input_.n_elem - get_dset_size(3, grp, "weights")[0] + 1,
                     get_dset_size(3, grp, "weights")[2]),
        input(input_),
        n_elem_input(input.n_elem),

        conv_width  (get_dset_size(3, grp, "weights")[0]),
        in_channels (get_dset_size(3, grp, "weights")[1]),
        out_channels(get_dset_size(3, grp, "weights")[2]),

        weights(conv_width*in_channels, out_channels),
        bias(out_channels),
        input_conv_format(n_elem, conv_width*in_channels),
        matmul_output    (n_elem, out_channels)
    {
        traverse_dset<3,float>(grp, "weights", [&](size_t nw, size_t in_c, size_t out_c, float x) {
                weights(nw*in_channels+in_c, out_c) = x;});

        traverse_dset<1,float>(grp, "bias", [&](size_t out_c, float x) {bias[out_c] = x;});

        // Read activation
        // This really should be a single string instead of a vector of them,
        // but I currently only have a reader for string vectors
        auto activation_str = read_attribute<vector<string>>(grp, ".", "activation");
        if(activation_str.size() != 1u) throw string("Invalid number of activations");

        if     (activation_str[0] == "ReLU") activation = ReLU;
        else if(activation_str[0] == "Tanh") activation = Tanh;
        else if(activation_str[0] == "Identity") activation = Identity;
        else throw string("Invalid activation name");
    }

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer(string("conv1d")); 
        VecArray inputc = input.output;
        
        int n_elem_output = n_elem;

        // Copy input into proper format so that convolution becomes matmul
        for(int nr=0; nr<n_elem_output; ++nr)
            for(int nw=0; nw<conv_width; ++nw)
                for(int nc=0; nc<in_channels; ++nc)
                    input_conv_format(nr,nw*in_channels+nc) = inputc(nc,nr+nw);
       
        // Multiply by weights to make matmul_output
        matmul_output.noalias() = input_conv_format * weights;
        matmul_output.rowwise() += bias;

        // Apply activation and store in output
        // We also keep matmul_output around for storing information for backprop
        switch(activation) {
            case Identity:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        output(nc,nr) = matmul_output(nr,nc);
                break;
            case ReLU:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        output(nc,nr) = relu(matmul_output(nr,nc));
                break;
            case Tanh:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        output(nc,nr) = tanh(matmul_output(nr,nc));
                break;
        }
        return 0;
    }

    virtual int propagate_deriv(int round) override {
        int n_elem_output = n_elem;
        VecArray sens_acc = sens.accum();

        // Backpropagate activations
        switch(activation) {
            case Identity:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        matmul_output(nr,nc) = sens_acc(nc,nr);
                break;
            case ReLU:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        matmul_output(nr,nc) = output(nc,nr)>0.f ? sens_acc(nc,nr) : 0.f;
                break;
            case Tanh:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        matmul_output(nr,nc) = sens_acc(nc,nr) * (1.f-sqr(output(nc,nr)));
                break;
        }

        // Backpropagate convolution
        input_conv_format.noalias() = matmul_output * weights.transpose();

        // Backpropagte into sens
        VecArray inp_sens = input.sens.acquire();
        for(int nr=0; nr<n_elem_output; ++nr)
            for(int nw=0; nw<conv_width; ++nw)
                for(int nc=0; nc<in_channels; ++nc)
                    inp_sens(nc,nr+nw) += input_conv_format(nr,nw*in_channels+nc);
        input.sens.release(inp_sens);
        return 0;
    }
};
static RegisterNodeType<Conv1D,1> conv1d_node("conv1d");


struct ScaledSum: public PotentialNode
{
    CoordNode& input;
    float scale;

    ScaledSum(hid_t grp, CoordNode& input_):
        PotentialNode(),
        input(input_),
        scale(read_attribute<float>(grp, ".", "scale"))
    {

        if(input.elem_width != 1u) throw string("Sum only works on elem width 1");
    }

    virtual int compute_value(int round, ComputeMode mode) override {
        Timer timer(string("scaled_sum")); 
        VecArray value = input.output;
        VecArray sens  = input.sens.acquire();
        int n_elem = input.n_elem;

        float pot = 0.f;
        for(int i=0; i<n_elem; ++i) pot += value(0,i);
        pot *= scale;
        potential = pot;

        for(int i=0; i<n_elem; ++i) sens(0,i) += scale;
        input.sens.release(sens);
        return 0;
    }
};
static RegisterNodeType<ScaledSum,1> scaled_sum_node("scaled_sum");
