#include "deriv_engine.h"
#include "timing.h"
#include "affine.h"
#include <vector>
#include "spline.h"
#include "state_logger.h"
#include <initializer_list>

using namespace std;
using namespace h5;

namespace {

enum class PlaceT {SCALAR, VECTOR, POINT};

constexpr int placetype_size(PlaceT t) {
    return t==PlaceT::SCALAR ? 1 : 3;
}

template <int n_pos_dim>
struct RamaPlacement {
    struct Params {
        int layer_idx;
        index_t rama_residue;
    };
        
    CoordNode& rama;
    int n_elem;
    vector<Params> params;
    LayeredPeriodicSpline2D<n_pos_dim> spline;
    VecArrayStorage rama_deriv;

    RamaPlacement(hid_t grp, CoordNode& rama_):
        rama(rama_),
        n_elem(get_dset_size(1, grp, "layer_index")[0]),
        params(n_elem),
        spline(
                get_dset_size(4, grp, "placement_data")[0],
                get_dset_size(4, grp, "placement_data")[1],
                get_dset_size(4, grp, "placement_data")[2]),
        rama_deriv(2*n_pos_dim, n_elem) // first is all phi deriv then all psi deriv
    {
        check_size(grp, "layer_index",    n_elem);
        check_size(grp, "rama_residue",   n_elem);
        check_size(grp, "placement_data", spline.n_layer, spline.nx, spline.ny, n_pos_dim);

        traverse_dset<1,int>(grp, "layer_index",    [&](size_t np, int x){params[np].layer_idx  = x;});
        traverse_dset<1,int>(grp, "rama_residue",   [&](size_t np, int x){params[np].rama_residue  = x;});

        {
            vector<double> all_data_to_fit;
            traverse_dset<4,double>(grp, "placement_data", [&](size_t nl, size_t ix, size_t iy,size_t d, double x) {
                    all_data_to_fit.push_back(x);});
            spline.fit_spline(all_data_to_fit.data());
        }
    }

    void reset() {}

    Vec<n_pos_dim> evaluate(int ne) {
        const float scale_x = spline.nx * (0.5f/M_PI_F - 1e-7f);
        const float scale_y = spline.ny * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        VecArray rama_pos   = rama.output;

        auto r   = load_vec<2>(rama_pos,   params[ne].rama_residue);

        Vec<n_pos_dim> value;
        spline.evaluate_value_and_deriv(
                value.v, 
                &rama_deriv(        0,ne),
                &rama_deriv(n_pos_dim,ne),
                params[ne].layer_idx, 
                (r[0]+shift)*scale_x, (r[1]+shift)*scale_y);

        return value;
    }

    void propagate_deriv(const Vec<n_pos_dim> &sens, int ne) {
        const float scale_x = spline.nx * (0.5f/M_PI_F - 1e-7f);
        const float scale_y = spline.ny * (0.5f/M_PI_F - 1e-7f);

        VecArray r_sens = rama.sens;

        auto my_rama_deriv = load_vec<2*n_pos_dim>(rama_deriv, ne);
        auto rd = make_vec2(
                scale_x*dot(sens, extract<        0,  n_pos_dim>(my_rama_deriv)),
                scale_y*dot(sens, extract<n_pos_dim,2*n_pos_dim>(my_rama_deriv)));

        update_vec(r_sens, params[ne].rama_residue, rd);
    }

    virtual std::vector<float> get_param() const {return {};}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() {return {};}
#endif
    virtual void set_param(const std::vector<float>& new_param) {}
};


template <int n_pos_dim>
struct FixedPlacement {
    struct Params {
        int layer_idx;
    };
        
    int n_elem;
    int n_layer;
    vector<Params> params;
    VecArrayStorage data;

    #ifdef PARAM_DERIV
    VecArrayStorage param_deriv;
    #endif

    FixedPlacement(hid_t grp):
        n_elem (get_dset_size(1, grp, "layer_index")[0]),
        n_layer(get_dset_size(2, grp, "placement_data")[0]),
        params(n_elem),
        data(n_pos_dim, n_layer)
        #ifdef PARAM_DERIV
        ,param_deriv(n_pos_dim, n_layer)
        #endif
    {
        check_size(grp, "layer_index",    n_elem);
        check_size(grp, "placement_data", n_layer, n_pos_dim);

        traverse_dset<1,int>(grp, "layer_index",    [&](size_t np, int x){params[np].layer_idx  = x;});
        traverse_dset<2,float>(grp, "placement_data", [&](size_t nl, size_t d, double x) {data(d,nl) = x;});
    }

    void reset() {
        #ifdef PARAM_DERIV
        fill(param_deriv, 0.f);
        #endif
    }

    Vec<n_pos_dim> evaluate(int ne) {
        return load_vec<n_pos_dim>(data, params[ne].layer_idx);
    }

    void propagate_deriv(const Vec<n_pos_dim> &sens, int ne) {
        #ifdef PARAM_DERIV
        update_vec(param_deriv, params[ne].layer_idx, sens);
        #endif
    }

    virtual std::vector<float> get_param() const {
        auto ret = std::vector<float>(n_layer*n_pos_dim);
        for(int nl: range(n_layer)) for(int d: range(n_pos_dim)) ret[nl*n_pos_dim+d] = data(d,nl);
        return ret;
    }

#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() {
        auto ret = std::vector<float>(n_layer*n_pos_dim);
        for(int nl: range(n_layer)) for(int d: range(n_pos_dim)) ret[nl*n_pos_dim+d] = param_deriv(d,nl);
        return ret;
    }
#endif

    virtual void set_param(const std::vector<float>& new_param) {
        if(new_param.size() != size_t(n_layer*n_pos_dim)) throw string("wrong param size");
        for(int nl: range(n_layer)) for(int d: range(n_pos_dim)) data(d,nl) = new_param[nl*n_pos_dim+d];
    }
};


template<PlaceT first>
static constexpr int compute_pos_dim_from_signature() {
    return placetype_size(first);
}

template<PlaceT first, PlaceT second, PlaceT ...rest>
static constexpr int compute_pos_dim_from_signature() {
    return placetype_size(first) + compute_pos_dim_from_signature<second,rest...>();
}

template<int offset>
void do_transformations(const float* U, const float3& t, const float* val, float* pos) {}

template<int offset, PlaceT first, PlaceT ... rest>
void do_transformations(const float* U, const float3& t, const float* val, float* pos) {

    switch(first) {
        case PlaceT::SCALAR:
            pos[offset] = val[offset];
            break;

        case PlaceT::VECTOR:
            store_vec(pos+offset, apply_rotation(U,   make_vec3(val[offset],val[offset+1],val[offset+2])));
            break;

        case PlaceT::POINT:
            store_vec(pos+offset, apply_affine  (U,t, make_vec3(val[offset],val[offset+1],val[offset+2])));
            break;
    }

    do_transformations<offset+placetype_size(first), rest...>(U,t,val,pos);
}


template<int offset>
void do_sens_transformations(
        float* restrict ref_sens, float3& com_deriv, float3& torque, 
        const float* U, const float3& t, const float* x, const float* sens) {}

template<int offset, PlaceT first, PlaceT ... rest>
void do_sens_transformations(
        float* restrict ref_sens, float3& com_deriv, float3& torque, 
        const float* U, const float3& t, const float* x, const float* sens) {

    if(first==PlaceT::SCALAR) {
        ref_sens[offset] = sens[offset];
    } else if(first==PlaceT::POINT) {
        auto s  = make_vec3(sens[offset],sens[offset+1],sens[offset+2]);
        auto xv = make_vec3(x   [offset],x   [offset+1],x   [offset+2]);
        store_vec(ref_sens+offset, apply_inverse_rotation(U,s));
        com_deriv += s;
        torque    += cross(xv-t, s);
    } else if(first==PlaceT::VECTOR) {
        auto s = make_vec3(sens[offset],sens[offset+1],sens[offset+2]);
        auto xv = make_vec3(x   [offset],x   [offset+1],x   [offset+2]);
        store_vec(ref_sens+offset, apply_inverse_rotation(U,s));
        torque += cross(xv, s);
    }

    do_sens_transformations<offset+placetype_size(first), rest...>(ref_sens,com_deriv,torque, U,t,x,sens);
}


template <typename PlacementData, PlaceT ...signature>
struct PlacementNode: public CoordNode
{
    static constexpr int n_pos_dim = compute_pos_dim_from_signature<signature...>();

    PlacementData placement_data;
    CoordNode& alignment;

    vector<index_t> affine_residue;

    template<typename ... Args>
    PlacementNode(hid_t grp, CoordNode& alignment_, Args& ... placement_arguments):
        CoordNode(get_dset_size(1,grp,"layer_index")[0], n_pos_dim),
        placement_data(grp, placement_arguments...),
        alignment(alignment_),
        affine_residue(n_elem)
    {
        // static_assert(n_pos_dim == decltype(placement_data.evaluate(0)), "inconsistent n_pos_dim");
        check_size(grp, "affine_residue", n_elem);
        traverse_dset<1,int>(grp, "affine_residue", [&](size_t np, int x){affine_residue[np] = x;});

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

        VecArray affine_pos = alignment.output;
        VecArray pos        = output;

        placement_data.reset();

        for(int ne: range(n_elem)) {
            auto aff = load_vec<7>(affine_pos, affine_residue[ne]);
            auto t   = extract<0,3>(aff);
            float U[9]; quat_to_rot(U, aff.v+3);

            Vec<n_pos_dim> val = placement_data.evaluate(ne);

            do_transformations<0, signature...>(U,t, val.v, &pos(0,ne));
        }
    }

    virtual void propagate_deriv() {
      Timer timer(string("placement_deriv"));

      VecArray a_sens = alignment.sens;
      VecArray affine_pos = alignment.output;

      for(int ne: range(n_elem)) {
          auto d = load_vec<n_pos_dim>(sens, ne);
          float* x = &output(0,ne);

          auto aff = load_vec<7>(affine_pos, affine_residue[ne]);
          auto t   = extract<0,3>(aff);
          float U[9]; quat_to_rot(U, aff.v+3);
          
          Vec<n_pos_dim> ref_frame_sens;
          Vec<3> com_deriv = make_zero<3>();
          Vec<3> torque    = make_zero<3>();

          do_sens_transformations<0, signature...>(ref_frame_sens.v,com_deriv,torque, U,t,x, d.v);
          placement_data.propagate_deriv(ref_frame_sens, ne);

          update_vec(&a_sens(0,affine_residue[ne]), com_deriv);
          update_vec(&a_sens(3,affine_residue[ne]), torque);
      }
    }

    virtual std::vector<float> get_param() const {return placement_data.get_param();}
#ifdef PARAM_DERIV
    virtual std::vector<float> get_param_deriv() {return placement_data.get_param_deriv();}
#endif
    virtual void set_param(const std::vector<float>& new_param) {placement_data.set_param(new_param);}
};

// There is definitely too much template code generation here.  I should
// simplify this to be nicer to the instruction cache and improve compile
// times.
static RegisterNodeType<PlacementNode<RamaPlacement <1>, PlaceT::SCALAR               >,2> pl1("placement_scalar");
static RegisterNodeType<PlacementNode<FixedPlacement<1>, PlaceT::SCALAR               >,1> pl2("placement_fixed_scalar");
static RegisterNodeType<PlacementNode<RamaPlacement <3>, PlaceT::POINT                >,2> pl3("placement_point_only");
static RegisterNodeType<PlacementNode<FixedPlacement<3>, PlaceT::POINT                >,1> pl4("placement_fixed_point_only");
static RegisterNodeType<PlacementNode<RamaPlacement <6>, PlaceT::POINT, PlaceT::VECTOR>,2> pl5("placement_point_vector_only");
static RegisterNodeType<PlacementNode<FixedPlacement<6>, PlaceT::POINT, PlaceT::VECTOR>,1> pl6("placement_fixed_point_vector_only");
static RegisterNodeType<PlacementNode<FixedPlacement<7>, PlaceT::POINT, PlaceT::VECTOR, PlaceT::SCALAR>,1> pl7("placement_fixed_point_vector_scalar");
}
