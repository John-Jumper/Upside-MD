#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"
#include <vector>
#include "spline.h"
#include "state_logger.h"
#include <initializer_list>

using namespace std;
using namespace h5;

enum class PlaceType {SCALAR, VECTOR, POINT};


template <int n_pos_dim> 
struct RigidPlacementNode: public CoordNode {
    struct Params {
        int layer_idx;
        index_t affine_residue;
        index_t rama_residue;
    };

    vector<PlaceType> signature;

    CoordNode& rama;
    CoordNode& alignment;

    vector<Params> params;
    LayeredPeriodicSpline2D<n_pos_dim> spline;
    VecArrayStorage rama_deriv;

    RigidPlacementNode(hid_t grp, CoordNode& rama_, CoordNode& alignment_):
        CoordNode(get_dset_size(1,grp,"layer_index")[0], n_pos_dim),
        rama(rama_), alignment(alignment_),
        params(n_elem), 
        spline(
                get_dset_size(4, grp, "placement_data")[0],
                get_dset_size(4, grp, "placement_data")[1],
                get_dset_size(4, grp, "placement_data")[2]),

        rama_deriv(2*n_pos_dim, n_elem) // first is all phi deriv then all psi deriv
    {
        // verify that the signature is as expected
        int n_pos_dim_input = 0;
        traverse_string_dset<1>(grp, "signature", [&](size_t i, string x){
                if(x == "scalar") {
                    signature.push_back(PlaceType::SCALAR);
                    n_pos_dim_input += 1;
                } else if(x == "vector") {
                    signature.push_back(PlaceType::VECTOR);
                    n_pos_dim_input += 3;
                } else if(x == "point") {
                    signature.push_back(PlaceType::POINT);
                    n_pos_dim_input += 3;
                } else {
                    throw string("unrecognized type in signature");
                }});
        if(n_pos_dim_input != n_pos_dim) 
            throw string("number of dimensions in input signature("+to_string(n_pos_dim_input)+") does not "
                    "match compiled n_pos_dim("+to_string(n_pos_dim)+").  Unable to continue.");

        check_size(grp, "layer_index",    n_elem);
        check_size(grp, "affine_residue", n_elem);
        check_size(grp, "rama_residue",   n_elem);
        check_size(grp, "placement_data", spline.n_layer, spline.nx, spline.ny, n_pos_dim);

        traverse_dset<1,int>(grp, "layer_index",    [&](size_t np, int x){params[np].layer_idx  = x;});
        traverse_dset<1,int>(grp, "affine_residue", [&](size_t np, int x){params[np].affine_residue = x;});
        traverse_dset<1,int>(grp, "rama_residue",   [&](size_t np, int x){params[np].rama_residue  = x;});

        {
            vector<double> all_data_to_fit;
            traverse_dset<4,double>(grp, "placement_data", [&](size_t nl, size_t ix, size_t iy,size_t d, double x) {
                    all_data_to_fit.push_back(x);});
            spline.fit_spline(all_data_to_fit.data());
        }

        if(logging(LOG_EXTENSIVE)) {
            // FIXME prepend the logging with the class name for disambiguation
            default_logger->add_logger<float>("placement_pos", {n_elem, n_pos_dim}, [&](float* buffer) {
                    VecArray pos = output;
                    for(int ne: range(n_elem))
                        for(int d: range(n_pos_dim))
                            buffer[ne*n_pos_dim + d] = pos(d,ne);});
        }
    }


    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("placement"));

        const float scale_x = spline.nx * (0.5f/M_PI_F - 1e-7f);
        const float scale_y = spline.ny * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        VecArray affine_pos = alignment.output;
        VecArray rama_pos   = rama.output;
        VecArray pos        = output;

        for(int ne: range(n_elem)) {
            auto aff = load_vec<7>(affine_pos, params[ne].affine_residue);
            auto r   = load_vec<2>(rama_pos,   params[ne].rama_residue);
            auto t   = extract<0,3>(aff);
            float U[9]; quat_to_rot(U, aff.v+3);

            float val[n_pos_dim*3];  // 3 here is deriv_x, deriv_y, value
            spline.evaluate_value_and_deriv(val, params[ne].layer_idx, 
                    (r[0]+shift)*scale_x, (r[1]+shift)*scale_y);

            int j = 0; // index of dimension that we are on

            #define READ3(i,j) make_vec3(val[((i)+0)*3+(j)], val[((i)+1)*3+(j)], val[((i)+2)*3+(j)])
            for(PlaceType type: signature) {
                switch(type) {
                    case PlaceType::SCALAR:
                        rama_deriv(j,ne)           = val[j*3+0] * scale_x;
                        rama_deriv(n_pos_dim+j,ne) = val[j*3+1] * scale_y;
                        pos  (j,ne) = val[j*3+2];
                        j += 1;
                        break;
                    case PlaceType::VECTOR:
                    case PlaceType::POINT:
                        // This if-statement is strictly not necessary as we already checked that n_pos_dim >= 3
                        // when we verified that the signature was consistent with n_pos_dim.  That being said,
                        // the if statement will be optimized out at compile time and it suppresses some warnings 
                        // from GCC at the time of writing.
                        if(n_pos_dim >= 3) { 
                            // point and vector differ only in shifting the final output
                            auto my_phi_deriv = scale_x * apply_rotation(U, READ3(j,0));
                            auto my_psi_deriv = scale_y * apply_rotation(U, READ3(j,1));
                            auto my_pos = (type==PlaceType::POINT
                                    ? apply_affine  (U,t, READ3(j,2))
                                    : apply_rotation(U,   READ3(j,2)));

                            for(int k: range(3)) rama_deriv(          j+k,ne) = my_phi_deriv[k];
                            for(int k: range(3)) rama_deriv(n_pos_dim+j+k,ne) = my_psi_deriv[k];
                            for(int k: range(3)) pos       (          j+k,ne) = my_pos[k];

                            j += 3;
                        }
                        break;
                }
            }
            #undef READ3
        }
    }

    virtual void propagate_deriv() {
        Timer timer(string("placement_deriv"));

        VecArray r_sens = rama.sens;
        VecArray a_sens = alignment.sens;
        VecArray affine_pos = alignment.output;

        for(int ne: range(n_elem)) {
            auto d = load_vec<n_pos_dim>(sens, ne);

            auto my_rama_deriv = load_vec<2*n_pos_dim>(rama_deriv, ne);
            auto rd = make_vec2(
                        dot(d, extract<        0,  n_pos_dim>(my_rama_deriv)),
                        dot(d, extract<n_pos_dim,2*n_pos_dim>(my_rama_deriv)));

            update_vec(r_sens, params[ne].rama_residue, rd);

            // only difference between points and vectors is whether to subtract off the translation
            auto z = make_zero<6>();
            int j=0;

            auto t  = load_vec<3>(affine_pos, params[ne].affine_residue);
            for(PlaceType type: signature) {
                switch(type) {
                    case PlaceType::SCALAR:
                        j+=1;  // no affine derivative
                        break;
                    case PlaceType::VECTOR:
                    case PlaceType::POINT:
                        // see note in compute_value for why this if-statement is unnecessary but harmless
                        if(n_pos_dim >= 3) { 
                            auto x  = make_vec3(output(j+0,ne), output(j+1,ne), output(j+2,ne));
                            auto dx = make_vec3(d[j+0], d[j+1], d[j+2]);

                            // torque relative to the residue center
                            auto tq = cross((type==PlaceType::POINT?x-t:x), dx);

                            // only points, not vectors, contribute to the CoM derivative
                            if(type==PlaceType::POINT) { z[0] += dx[0]; z[1] += dx[1]; z[2] += dx[2]; }
                            z[3] += tq[0]; z[4] += tq[1]; z[5] += tq[2];
                            j += 3;
                        }
                        break;
                }
            }
            update_vec(a_sens, params[ne].affine_residue, z);
        }
          
    }
};

static RegisterNodeType<RigidPlacementNode<1>,2> placement_scalar_node("placement_scalar");
static RegisterNodeType<RigidPlacementNode<3>,2> placement3_node("placement3");
static RegisterNodeType<RigidPlacementNode<4>,2> placement4_node("placement4");
static RegisterNodeType<RigidPlacementNode<6>,2> placement_rotamer_node("placement_rotamer");
static RegisterNodeType<RigidPlacementNode<6>,2> placement_cb_only_node("placement_cb_only");
