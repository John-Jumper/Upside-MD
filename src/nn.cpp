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


struct Conv1D : public CoordNode
{
    CoordNode& input;
    int n_elem_input;
    enum ActivationT {ReLU, Identity};

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

        if(activation_str[0] == "ReLU") activation = ReLU;
        else if(activation_str[0] == "Identity") activation = Identity;
        else throw string("Invalid activation name");
    }

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("conv1d")); 
        VecArray inputc = input.output;
        
        int n_elem_output = n_elem;

        // Copy input into proper format so that convolution becomes matmul
        for(int nr=0; nr<n_elem_output; ++nr)
            for(int nw=0; nw<conv_width; ++nw)
                for(int nc=0; nc<in_channels; ++nc)
                    input_conv_format(nr,nw*in_channels+nc) = inputc(nc,nr);
       
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
        }
    }

    virtual void propagate_deriv() override {
        int n_elem_output = n_elem;

        // Backpropagate activations
        switch(activation) {
            case Identity:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        matmul_output(nr,nc) = sens(nc,nr);
                break;
            case ReLU:
                for(int nr=0; nr<n_elem_output; ++nr)
                    for(int nc=0; nc<out_channels; ++nc)
                        matmul_output(nr,nc) = output(nc,nr)>0.f ? sens(nc,nr) : 0.f;
                break;
        }

        // Backpropagate convolution
        input_conv_format.noalias() = matmul_output * weights.transpose();

        // Backpropagte into sens
        VecArray inp_sens = input.sens;
        for(int nr=0; nr<n_elem_output; ++nr)
            for(int nw=0; nw<conv_width; ++nw)
                for(int nc=0; nc<in_channels; ++nc)
                    inp_sens(nc,nr+nw) += input_conv_format(nr,nw*in_channels+nc);
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

    virtual void compute_value(ComputeMode mode) override {
        Timer timer(string("scaled_sum")); 
        VecArray value = input.output;
        VecArray sens  = input.sens;
        int n_elem = input.n_elem;

        float pot = 0.f;
        for(int i=0; i<n_elem; ++i) pot += value(0,i);
        pot *= scale;
        potential = pot;

        for(int i=0; i<n_elem; ++i) sens(0,i) += scale;
    }
};
static RegisterNodeType<ScaledSum,1> scaled_sum_node("scaled_sum");

