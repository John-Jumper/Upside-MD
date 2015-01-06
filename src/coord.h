#ifndef COORD_H
#define COORD_H

#include "vector_math.h"

struct SysArray {
    float *x;
    int   offset;  // offset per system, units of floats
    SysArray(float* x_, int offset_):
        x(x_), offset(offset_) {}
    SysArray(): x(nullptr), offset(0) {}
};


struct CoordArray {
    SysArray value;
    SysArray deriv;
    CoordArray(SysArray value_, SysArray deriv_):
        value(value_), deriv(deriv_) {}
};

typedef unsigned short index_t;
typedef unsigned short slot_t;
struct CoordPair {
    index_t index;
    slot_t slot;
    CoordPair(unsigned short index_, unsigned short slot_):
        index(index_), slot(slot_) {}
    CoordPair(): index(-1), slot(-1) {}
};


enum CoordWritePolicy {
    OverwriteWritePolicy,
    SumWritePolicy
};


template <int N_DIM, int N_DIM_OUTPUT=1, CoordWritePolicy WRITE_POLICY=OverwriteWritePolicy>
struct Coord
{
    const static int n_dim = N_DIM;
    const static int n_dim_output = N_DIM_OUTPUT;
    float v[N_DIM];
    float d[N_DIM_OUTPUT][N_DIM];
    float * const deriv_arr;

    Coord(): deriv_arr(0) {};

    Coord(const CoordArray arr, int system, CoordPair c):
        deriv_arr(arr.deriv.x + system*arr.deriv.offset + c.slot*N_DIM)
    {
        for(int nd=0; nd<N_DIM; ++nd) 
            v[nd] = arr.value.x[system*arr.value.offset + c.index*N_DIM + nd];

        for(int no=0; no<N_DIM_OUTPUT; ++no) 
            for(int nd=0; nd<N_DIM; ++nd) 
                d[no][nd] = 0.f;
    }

    float3 f3() const {
        return make_float3(v[0], v[1], v[2]);
    }

    float3 df3(int idx) const {
        return make_float3(d[idx][0], d[idx][1], d[idx][2]);
    }

    void set_deriv(float3 d_) {
        d[0][0] = d_.x;
        d[0][1] = d_.y;
        d[0][2] = d_.z;
    }

    void set_deriv(int i, float3 d_) {
        d[i][0] = d_.x;
        d[i][1] = d_.y;
        d[i][2] = d_.z;
    }

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


template <int N_DIM>
struct TempCoord
{
    const static int n_dim = N_DIM;
    float v[N_DIM];

    TempCoord() { for(int nd=0; nd<N_DIM; ++nd) v[nd] = 0.f; }

    float3 f3() const { return make_float3(v[0], v[1], v[2]); }

    void set_value(float3 val) {
        v[0] = val.x;
        v[1] = val.y;
        v[2] = val.z;
    }

    TempCoord<N_DIM>& operator+=(const float3 &o) {
        v[0] += o.x;
        v[1] += o.y;
        v[2] += o.z;
        return *this;
    }
};


template <int N_DIM>
struct StaticCoord
{
    const static int n_dim = N_DIM;
    float v[N_DIM];

    StaticCoord() {};

    StaticCoord(SysArray value, int ns, int index)
    {
        for(int nd=0; nd<N_DIM; ++nd) 
            v[nd] = value.x[ns*value.offset + index*N_DIM + nd];
    }

    float3 f3() const {
        return make_float3(v[0], v[1], v[2]);
    }

    StaticCoord<N_DIM>& operator+=(const StaticCoord<N_DIM> &o) {for(int i=0; i<N_DIM; ++i) v[i]+=o.v[i]; return *this;}
};


template <int N_DIM>
struct MutableCoord
{
    const static int n_dim = N_DIM;
    float v[(N_DIM==0) ? 1 : N_DIM];  // specialization for a trivial case
    float * const value_arr;

    enum Init { ReadValue, Zero };

    MutableCoord(): value_arr(nullptr) {};

    MutableCoord(SysArray arr, int system, int index, Init init = ReadValue):
        value_arr(arr.x + system*arr.offset + index*N_DIM)
    {
        for(int nd=0; nd<N_DIM; ++nd) 
            v[nd] = init==ReadValue ? value_arr[nd] : 0.f;
    }

    float3 f3() const {
        return make_float3(v[0], v[1], v[2]);
    }

    void set_value(float3 val) {
        v[0] = val.x;
        v[1] = val.y;
        v[2] = val.z;
    }

    MutableCoord<N_DIM>& operator+=(const float3 &o) {
        v[0] += o.x;
        v[1] += o.y;
        v[2] += o.z;
        return *this;
    }

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
