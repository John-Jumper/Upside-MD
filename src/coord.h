#ifndef COORD_H
#define COORD_H

#include "vector_math.h"

//! Array containing data for each system in the simulation
struct SysArray {
    float *x;      //!< data pointer
    int   offset;  //!< offset per system, units of floats
    //! Initialize from existing data
    SysArray(float* x_, int offset_):
        x(x_), offset(offset_) {}
    //! Default initialization makes x the null pointer
    SysArray(): x(nullptr), offset(0) {}
};


//! Pair of value and derivative SysArray's 
struct CoordArray {
    SysArray value;  //!< Array containing value elements
    SysArray deriv;  //!< Array containing derivative slots

    //! Initialize from value and derivative SysArray's
    CoordArray(SysArray value_, SysArray deriv_):
        value(value_), deriv(deriv_) {}
};

typedef unsigned short index_t;  //!< type of coordinate indices
typedef unsigned short slot_t;   //!< type of derivative slot indices

//! Pair of coordinate index and slot index for its derivative

//! When the value of the coordinate given by index is used, 
//! a derivative should be recorded at the slot given by slot.
//! The class Coord should be used to read the value associated to 
//! index and to write the derivative associated to slot.
struct CoordPair {
    index_t index;  //!< Index into the value array to find the coordinate
    slot_t slot;    //!< Index into the derivative array to find the slot

    //! Initialize from existing data
    CoordPair(unsigned short index_, unsigned short slot_):
        index(index_), slot(slot_) {}

    //! Initialize with -1 for each member
    CoordPair(): index(-1), slot(-1) {}
};

//! Enum to choose whether to overwrite or sum to derivative slot 
enum CoordWritePolicy {
    OverwriteWritePolicy,  //!< overwrite derivative slot
    SumWritePolicy         //!< add to derivative slot
};


//! Class that represents a coordinate and its derivative
template <int N_DIM, int N_DIM_OUTPUT=1, CoordWritePolicy WRITE_POLICY=OverwriteWritePolicy>
struct Coord
{
    public:
        const static int n_dim = N_DIM;    //!< number of dimensions for the coordinate
        const static int n_dim_output = N_DIM_OUTPUT;    //!< number of dimensions for the output of the computation
        float v[N_DIM];   //!< value of the coordinate
        float d[N_DIM_OUTPUT][N_DIM];   //!< derivative of the coordinate with respect to each output dimension

    protected:
        float * const deriv_arr;  

    public:
        Coord(): deriv_arr(0) {};

        //! Initialize by specifying a coordinate array, a specific system, and the location within the coordinate array
        Coord(const CoordArray arr, int system, CoordPair c):
            deriv_arr(arr.deriv.x + system*arr.deriv.offset + c.slot*N_DIM)
        {
            for(int nd=0; nd<N_DIM; ++nd) 
                v[nd] = arr.value.x[system*arr.value.offset + c.index*N_DIM + nd];

            for(int no=0; no<N_DIM_OUTPUT; ++no) 
                for(int nd=0; nd<N_DIM; ++nd) 
                    d[no][nd] = 0.f;
        }

        //! Extract first 3 dimensions of the value as a float3
        float3 f3() const {
            return make_float3(v[0], v[1], v[2]);
        }

        //! Extract first 3 dimensions of the idx'th component of the derivative as a float3
        float3 df3(int idx) const {
            return make_float3(d[idx][0], d[idx][1], d[idx][2]);
        }

        //! Set the derivative for the 0'th component using a float3
        void set_deriv(float3 d_) {
            d[0][0] = d_.x;
            d[0][1] = d_.y;
            d[0][2] = d_.z;
        }

        //! Set the derivative for the i'th component using a float3
        void set_deriv(int i, float3 d_) {
            d[i][0] = d_.x;
            d[i][1] = d_.y;
            d[i][2] = d_.z;
        }

        //! Write the derivative to the derivative array in the CoordArray
        
        //! If the WRITE_POLICY is OverwriteWritePolicy (the default), the value of the d array
        //! overwrites the derivative at the slot specified by c.  If the WRITE_POLICY is 
        //! SumWritePolicy, the value of d is added to the slot specified by c.
        void flush() const {
            switch(WRITE_POLICY) {
                case OverwriteWritePolicy:
                    for(int no=0; no<N_DIM_OUTPUT; ++no) 
                        for(int nd=0; nd<N_DIM; ++nd) 
                            deriv_arr[no*N_DIM + nd]  = d[no][nd];
                    break;

                case SumWritePolicy:
                    for(int no=0; no<N_DIM_OUTPUT; ++no) 
                        for(int nd=0; nd<N_DIM; ++nd) 
                            deriv_arr[no*N_DIM + nd] += d[no][nd];
                    break;
            }
        }
};


//! Temporary coordinate that is discarded after use
template <int N_DIM>
struct TempCoord
{
    const static int n_dim = N_DIM;  //!< number of dimensions for the coordinate
    float v[N_DIM];  //!< value of the coordinate

    //! Default constructor initializes the value to 0.
    TempCoord() { for(int nd=0; nd<N_DIM; ++nd) v[nd] = 0.f; }

    //! Extract first 3 dimensions of the value as a float3
    float3 f3() const { return make_float3(v[0], v[1], v[2]); }

    //! Set first 3 dimensions of the value as a float3
    void set_value(float3 val) {
        v[0] = val.x;
        v[1] = val.y;
        v[2] = val.z;
    }

    //! Add float3 to first 3 dimensions of coordinate
    TempCoord<N_DIM>& operator+=(const float3 &o) {
        v[0] += o.x;
        v[1] += o.y;
        v[2] += o.z;
        return *this;
    }
};


//! Coordinate read from SysArray without derivative information
template <int N_DIM>
struct StaticCoord
{
    const static int n_dim = N_DIM; //!< number of dimensions for the coordinate
    float v[N_DIM];  //!< value of the coordinate

    StaticCoord() {};

    //! Initialize by specifying a SysArray, a specific system, and the location within the SysArray
    StaticCoord(SysArray value, int ns, int index)
    {
        for(int nd=0; nd<N_DIM; ++nd) 
            v[nd] = value.x[ns*value.offset + index*N_DIM + nd];
    }

    //! Extract first 3 dimensions of the value as a float3
    float3 f3() const {
        return make_float3(v[0], v[1], v[2]);
    }

    //! Add another StaticCoord's value to this object's value
    StaticCoord<N_DIM>& operator+=(const StaticCoord<N_DIM> &o) {for(int i=0; i<N_DIM; ++i) v[i]+=o.v[i]; return *this;}
};


//! Coordinate that is read and written to SysArray, but with no derivative information
template <int N_DIM>
struct MutableCoord
{
    public:
        const static int n_dim = N_DIM;   //!< number of dimensions for the coordinate
        // Note that allocating v with size 0 is rejected by some compilers (and maybe the standard)
        float v[(N_DIM==0) ? 1 : N_DIM];  //!< value of the coordinate
    protected:
        float * const value_arr;

    public:
        //! Enum representing whether to initialize by reading an array or just with zeros
        enum Init { ReadValue, Zero };

        MutableCoord(): value_arr(nullptr) {};

        //! Initialize by specifying a SysArray, a specific system, and the location within the SysArray
        MutableCoord(SysArray arr, int system, int index, Init init = ReadValue):
            value_arr(arr.x + system*arr.offset + index*N_DIM)
        {
            for(int nd=0; nd<N_DIM; ++nd) 
                v[nd] = init==ReadValue ? value_arr[nd] : 0.f;
        }

        //! Extract first 3 dimensions of the value as a float3
        float3 f3() const {
            return make_float3(v[0], v[1], v[2]);
        }

        //! Set first 3 dimensions of the value as a float3
        void set_value(float3 val) {
            v[0] = val.x;
            v[1] = val.y;
            v[2] = val.z;
        }

        //! Add float3 to first 3 dimensions of coordinate
        MutableCoord<N_DIM>& operator+=(const float3 &o) {
            v[0] += o.x;
            v[1] += o.y;
            v[2] += o.z;
            return *this;
        }

        //! Write the value to the SysArray
        void flush() {
            for(int nd=0; nd<N_DIM; ++nd) 
                value_arr[nd] = v[nd];
        }
};

/*
template <typename CoordT, typename FuncT>
void finite_difference(FuncT& f, CoordT& x, float* expected, float eps = 1e-2) 
{
    int ndim_output = decltype(f(x))::n_dim;
    auto y = x;
    int ndim_input  = decltype(y)::n_dim;

    vector<float> ret(ndim_output*ndim_input);
    for(int d=0; d<ndim_input; ++d) {
        CoordT x_prime1 = x; x_prime1.v[d] += eps;
        CoordT x_prime2 = x; x_prime2.v[d] -= eps;

        auto val1 = f(x_prime1);
        auto val2 = f(x_prime2);
        for(int no=0; no<ndim_output; ++no) ret[no*ndim_input+d] = (val1.v[no]-val2.v[no]) / (2*eps);
    }
    float z = 0.f;
    for(int no=0; no<ndim_output; ++no) {
        printf("exp:");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", expected[no*ndim_input+ni]);
        printf("\n");

        printf("fd: ");
        for(int ni=0; ni<ndim_input; ++ni) printf(" % f", ret     [no*ndim_input+ni]);
        printf("\n\n");
        for(int ni=0; ni<ndim_input; ++ni) {
            float t = expected[no*ndim_input+ni]-ret[no*ndim_input+ni];
            z += t*t;
        }
    }
    printf("rmsd % f\n\n\n", sqrtf(z/ndim_output/ndim_input));
}
*/

#endif
