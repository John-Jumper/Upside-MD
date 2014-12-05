#ifndef COORD_H
#define COORD_H

#include <cmath>
#include <Random123/threefry.h>
#include "md.h"
#include <vector>

struct float2 {
    float x,y;
    float2(): x(0.f),y(0.f) {};
    float2(float x_, float y_):
        x(x_), y(y_){}
};
struct float3 {
    float x,y,z;
    float3(): x(0.f),y(0.f),z(0.f) {};
    float3(float x_, float y_, float z_):
        x(x_), y(y_), z(z_) {}
};
struct float4 {
    float x,y,z,w;
    float4(): x(0.f),y(0.f),z(0.f),w(0.f) {};
    float4(float x_, float y_, float z_, float w_):
        x(x_), y(y_), z(z_), w(w_) {}
};

inline float& component(float3 &v, int dim) {
    switch(dim) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
    }
    // should never occur, but I don't want this function to throw
    fprintf(stderr, "the impossible occurred\n");
    return v.x;  
}


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


namespace{
static const float M_PI_F   = 3.141592653589793f;
static const float M_1_PI_F = 0.3183098861837907f;

inline float rsqrtf(float x) {return 1.f/sqrtf(x);}

inline float2 make_float2(float x, float y                  ) { return float2(x,y);     }
inline float3 make_float3(float x, float y, float z         ) { return float3(x,y,z);   }
inline float4 make_float4(float x, float y, float z, float w) { return float4(x,y,z,w); }
inline float4 make_float4(float3 v, float w) { return make_float4(v.x,v.y,v.z,w); }

inline float2 operator*(float a, const float2 &b) { return make_float2(a*b.x, a*b.y); }
inline float2 operator+(float a, const float2 &b) { return make_float2(a+b.x, a+b.y); }
inline float2 operator-(float a, const float2 &b) { return make_float2(a-b.x, a-b.y); }
inline float2 operator/(float a, const float2 &b) { return make_float2(a/b.x, a/b.y); }

inline float2 operator*(const float2 &a, const float2 &b) { return make_float2(a.x*b.x, a.y*b.y); }
inline float2 operator+(const float2 &a, const float2 &b) { return make_float2(a.x+b.x, a.y+b.y); }
inline float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x-b.x, a.y-b.y); }
inline float2 operator/(const float2 &a, const float2 &b) { return make_float2(a.x/b.x, a.y/b.y); }

inline float3 float3_from_float4(const float4& x) { return make_float3(x.x,x.y,x.z); }
inline float3 xyz(const float4& x) { return make_float3(x.x,x.y,x.z); }

inline float3 operator*(const float  &a, const float3 &b) { return make_float3(a*b.x, a*b.y, a*b.z); }
inline float3 operator+(const float  &a, const float3 &b) { return make_float3(a+b.x, a+b.y, a+b.z); }
inline float3 operator-(const float  &a, const float3 &b) { return make_float3(a-b.x, a-b.y, a-b.z); }
inline float3 operator/(const float  &a, const float3 &b) { return make_float3(a/b.x, a/b.y, a/b.z); }

inline float3 operator*(const float3 &a, const float  &b) { return make_float3(a.x*b, a.y*b, a.z*b); }
inline float3 operator+(const float3 &a, const float  &b) { return make_float3(a.x+b, a.y+b, a.z+b); }
inline float3 operator-(const float3 &a, const float  &b) { return make_float3(a.x-b, a.y-b, a.z-b); }
inline float3 operator/(const float3 &a, const float  &b) { return make_float3(a.x/b, a.y/b, a.z/b); }

inline float3 operator*(const float3 &a, const float3 &b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
inline float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline float3 operator-(const float3 &a, const float3 &b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline float3 operator/(const float3 &a, const float3 &b) { return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }

inline float3 operator*=(      float3 &a, const float  &b) { a.x*=b;   a.y*=b;   a.z*=b;   return a;}
inline float3 operator+=(      float3 &a, const float  &b) { a.x+=b;   a.y+=b;   a.z+=b;   return a;}
inline float3 operator-=(      float3 &a, const float  &b) { a.x-=b;   a.y-=b;   a.z-=b;   return a;}
inline float3 operator/=(      float3 &a, const float  &b) { a.x/=b;   a.y/=b;   a.z/=b;   return a;}

inline float3 operator*=(      float3 &a, const float3 &b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; return a;}
inline float3 operator+=(      float3 &a, const float3 &b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a;}
inline float3 operator-=(      float3 &a, const float3 &b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; return a;}
inline float3 operator/=(      float3 &a, const float3 &b) { a.x/=b.x; a.y/=b.y; a.z/=b.z; return a;}

inline float4 operator*(const float  &a, const float4 &b) { return make_float4(a*b.x, a*b.y, a*b.z, a*b.w); }
inline float4 operator+(const float  &a, const float4 &b) { return make_float4(a+b.x, a+b.y, a+b.z, a+b.w); }
inline float4 operator-(const float  &a, const float4 &b) { return make_float4(a-b.x, a-b.y, a-b.z, a-b.w); }
inline float4 operator/(const float  &a, const float4 &b) { return make_float4(a/b.x, a/b.y, a/b.z, a/b.w); }

inline float4 operator*(const float4 &a, const float  &b) { return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }
inline float4 operator+(const float4 &a, const float  &b) { return make_float4(a.x+b, a.y+b, a.z+b, a.w+b); }
inline float4 operator-(const float4 &a, const float  &b) { return make_float4(a.x-b, a.y-b, a.z-b, a.w-b); }
inline float4 operator/(const float4 &a, const float  &b) { return make_float4(a.x/b, a.y/b, a.z/b, a.w/b); }

inline float4 operator*(const float4 &a, const float4 &b) { return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
inline float4 operator+(const float4 &a, const float4 &b) { return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
inline float4 operator-(const float4 &a, const float4 &b) { return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
inline float4 operator/(const float4 &a, const float4 &b) { return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }

inline float4 operator*=(      float4 &a, const float  &b) { a.x*=b;   a.y*=b;   a.z*=b;   a.w*=b;   return a;}
inline float4 operator+=(      float4 &a, const float  &b) { a.x+=b;   a.y+=b;   a.z+=b;   a.w+=b;   return a;}
inline float4 operator-=(      float4 &a, const float  &b) { a.x-=b;   a.y-=b;   a.z-=b;   a.w-=b;   return a;}
inline float4 operator/=(      float4 &a, const float  &b) { a.x/=b;   a.y/=b;   a.z/=b;   a.w/=b;   return a;}
                                                                                                     
inline float4 operator*=(      float4 &a, const float4 &b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w; return a;}
inline float4 operator+=(      float4 &a, const float4 &b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; return a;}
inline float4 operator-=(      float4 &a, const float4 &b) { a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w; return a;}
inline float4 operator/=(      float4 &a, const float4 &b) { a.x/=b.x; a.y/=b.y; a.z/=b.z; a.w/=b.w; return a;}

inline float3 operator-(const float3 &a) {return make_float3(-a.x, -a.y, -a.z);}
inline float4 operator-(const float4 &a) {return make_float4(-a.x, -a.y, -a.z, -a.w);}

inline float inv_mag(float3 a){return rsqrtf(a.x*a.x + a.y*a.y + a.z*a.z);}

inline float3 cross(float3 a, float3 b){
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
}

inline float mag(float3 a){return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);}

inline float mag2(float3 a){return a.x*a.x + a.y*a.y + a.z*a.z;}

inline float inv_mag2(float3 a){float m=inv_mag(a); return m*m;}

inline float3 normalize3(float3 a){return a*inv_mag(a);}

inline float dot(float3 a, float3 b){return a.x*b.x + a.y*b.y + a.z*b.z;}

inline float2 sigmoid(float x) {
#ifdef APPROX_SIGMOID
    float z = rsqrtf(4.f+x*x);
    return make_float2(0.5f*(1.f + x*z), (2.f*z)*(z*z));
#else
    //return make_float2(0.5f*(tanh(0.5f*x) + 1.f), 0.5f / (1.f + cosh(x)));
    float z = exp(-x);
    float w = 1.f/(1.f+z);
    return make_float2(w, z*w*w);
#endif
}

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
                        deriv_arr[no*N_DIM + nd] = d[no][nd];
                break;

            case SumWritePolicy:
                for(int no=0; no<N_DIM_OUTPUT; ++no) 
                    for(int nd=0; nd<N_DIM; ++nd) 
                        deriv_arr[no*N_DIM + nd] = d[no][nd];
                break;
        }
    }
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


inline float dihedral_germ(
        float3  r1, float3  r2, float3  r3, float3  r4,
        float3 &d1, float3 &d2, float3 &d3, float3 &d4)
    // Formulas and notation taken from Blondel and Karplus, 1995
{
    float3 F = r1-r2;
    float3 G = r2-r3;
    float3 H = r4-r3;

    float3 A = cross(F,G);
    float3 B = cross(H,G);
    float3 C = cross(B,A);

    float inv_Amag2 = inv_mag2(A);
    float inv_Bmag2 = inv_mag2(B);

    float Gmag2    = mag2(G);
    float inv_Gmag = rsqrtf(Gmag2);
    float Gmag     = Gmag2 * inv_Gmag;

    d1 = -Gmag * inv_Amag2 * A;
    d4 =  Gmag * inv_Bmag2 * B;

    float3 f_mid =  dot(F,G)*inv_Amag2*inv_Gmag * A - dot(H,G)*inv_Bmag2*inv_Gmag * B;

    d2 = -d1 + f_mid;
    d3 = -d4 - f_mid;

    return atan2(dot(C,G), dot(A,B) * Gmag);
}
}

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


template <int my_width, int width1, int width2>
void reverse_autodiff(
        const SysArray accum,
        SysArray deriv1,
        SysArray deriv2,
        const DerivRecord* tape,
        const AutoDiffParams* p,
        int n_tape,
        int n_atom, 
        int n_system)
{
    for(int ns=0; ns<n_system; ++ns) {
        std::vector<TempCoord<my_width>> sens(n_atom);
        for(int nt=0; nt<n_tape; ++nt) {
            auto tape_elem = tape[nt];
            for(int rec=0; rec<tape_elem.output_width; ++rec) {
                auto val = StaticCoord<my_width>(accum, ns, tape_elem.loc + rec);
                for(int d=0; d<my_width; ++d)
                    sens[tape_elem.atom].v[d] += val.v[d];
            }
        }

        for(int na=0; na<n_atom; ++na) {
            if(width1) {
                for(int nsl=0; nsl<p[na].n_slots1; ++nsl) {
                    for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                        MutableCoord<width1> c(deriv1, ns, p[na].slots1[nsl]+sens_dim);
                        for(int d=0; d<width1; ++d) c.v[d] *= sens[na].v[sens_dim];
                        c.flush();
                    }
                }
            }

            if(width2) {
                for(int nsl=0; nsl<p[na].n_slots2; ++nsl) {
                    for(int sens_dim=0; sens_dim<my_width; ++sens_dim) {
                        MutableCoord<width2> c(deriv2, ns, p[na].slots2[nsl]+sens_dim);
                        for(int d=0; d<width2; ++d) c.v[d] *= sens[na].v[sens_dim];
                        c.flush();
                    }
                }
            }
        }
    }
}

#endif
