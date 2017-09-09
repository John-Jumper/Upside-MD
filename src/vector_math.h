#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include <cmath>
#include <type_traits>
#include <memory>
#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>
#include "Float4.h"
#include <mutex>


static constexpr int default_alignment=4; // suitable for SSE

// the function below is used to allow a version of max to be called in a constexpr context
constexpr inline int maxint(int i, int j) {
    return (i>j) ? i : j;
}

constexpr inline int round_up(int i, int alignment) {
    return ((i+alignment-1)/alignment)*alignment; // probably alignment is a power of 2
}

constexpr inline int ru(int i) {
    return i==1 ? i : round_up(i,4);
}


template <typename T>
static std::unique_ptr<T[]> new_aligned(int n_elem, int alignment_elems=default_alignment) {
    // round up allocation to ensure that you can also read to the end without
    //   overstepping the array, if needed
    T* ptr = new T[round_up(n_elem, alignment_elems)];
    // printf("aligning %i elements at %p\n", round_up(n_elem, alignment_elems), (void*)ptr);
    // if((unsigned long)(ptr)%(unsigned long)(alignment_elems*sizeof(T))) throw "bad alignment on string";
    return std::unique_ptr<T[]>(ptr);
}

struct VecArray {
    float* x;
    int row_width;

    VecArray(): x(nullptr), row_width(0) {}

    VecArray(float* x_, int row_width_):
        x(x_), row_width(row_width_) {}

    float& operator()(int i_comp, int i_elem) {
        return x[i_comp+ i_elem*row_width];
    }

    const float& operator()(int i_comp, int i_elem) const {
        return x[i_comp + i_elem*row_width];
    }
};


struct VecArrayStorage {
    int n_elem;
    int row_width;
    std::unique_ptr<float[]> x;

    VecArrayStorage(int elem_width_, int n_elem_):
        n_elem(n_elem_), row_width(ru(elem_width_)),
        x(new_aligned<float>(n_elem*row_width)) {
            std::fill_n(x.get(), n_elem*row_width, 0.f);
        }

    VecArrayStorage(VecArrayStorage&& o):
        n_elem(o.n_elem), row_width(o.row_width),
        x(std::move(o.x))
    {}

    VecArrayStorage(const VecArrayStorage& o):
        n_elem(o.n_elem), row_width(o.row_width),
        x(new_aligned<float>(n_elem*row_width))
    {
        std::copy_n(o.x.get(), n_elem*row_width, x.get());
    }

    VecArrayStorage(): VecArrayStorage(1,1) {}

    float& operator()(int i_comp, int i_elem) {
        return x[i_elem*row_width + i_comp];
    }

    const float& operator()(int i_comp, int i_elem) const {
        return x[i_elem*row_width + i_comp];
    }

    operator VecArray() {return VecArray(x.get(), row_width);}

    void reset(int elem_width_, int n_elem_) {
        row_width = ru(elem_width_);
        n_elem = n_elem_;
        x.reset(new float[n_elem*row_width]);
    }
};


static void copy(VecArrayStorage& v_src, VecArrayStorage& v_dst) {
    assert(v_src.n_elem    == v_dst.n_elem);
    assert(v_src.row_width == v_dst.row_width);
    std::copy_n(v_src.x.get(), v_src.n_elem*v_src.row_width, v_dst.x.get()); 
}

inline void swap(VecArrayStorage& a, VecArrayStorage& b) {
    assert(a.n_elem==b.n_elem);
    assert(a.row_width==b.row_width);
    a.x.swap(b.x);
}

static void fill(VecArrayStorage& v, float fill_value) {
    std::fill_n(v.x.get(), v.n_elem*v.row_width, fill_value);
}

static void fill(VecArray v, int n_dim, int n_elem, float fill_value) {
    for(int d=0; d<n_dim; ++d) 
        for(int ne=0; ne<n_elem; ++ne) 
            v(d,ne) = fill_value;
}


struct VecArrayAccum {
    // Class representing accumulation of vectors
    // For thread-safety, it provides multiple copies that are further accumulated
    // the entire public API is thread-safe

    protected:
        std::mutex mut;
        // FIXME for reasons that make no sense to me, code is consistently
        // 5% *faster* on ubiquitin when reclaim is false, even on a single core
        // where reclaim should be strictly less work.  I do not understand.
        constexpr static bool reclaim = true;

        enum class ActiveState {
            Never,
            Previous,
            Current};

        int n_store;
        std::vector<ActiveState> state;
        // We must store unique_ptr's because we hand out the indices of the stores
        // and a vector resize would perform a dangerous move.  
        std::vector<std::unique_ptr<VecArrayStorage>> store;

        void push_back_store() {
            ++n_store;
            state.push_back(ActiveState::Never);
            store.emplace_back(
                    std::unique_ptr<VecArrayStorage>(
                        new VecArrayStorage(elem_width, n_elem)));
        }

        void zero_store(int ns) {
            // reset for later accumulation
            fill(*store[ns], 0.f);
            state[ns] = ActiveState::Never;
        }


    public:
        int n_elem;
        int elem_width;

        VecArrayAccum(int elem_width_, int n_elem_):
            n_store(0),
            n_elem(n_elem_),
            elem_width(elem_width_)
        {
            push_back_store();  // always have at least one store
        }

        VecArray acquire() {
            std::lock_guard<std::mutex> g(mut);

            for(int i=0; i<n_store; ++i) {
                if(state[i]!=ActiveState::Current) {
                    // We cannot accumulate multiple threads into previously released
                    // arrays when we want deterministic results because random 
                    // associativity may occur.
                    if(!reclaim && state[i] == ActiveState::Previous) continue;

                    state[i] = ActiveState::Current;
                    return *store[i];
                }
            }

            // if we reach here then we have no inactive store
            push_back_store();
            state.back() = ActiveState::Current;
            return *store.back();
        }

        void release(VecArray va) {
            std::lock_guard<std::mutex> g(mut);
            // I will figure out which to release from the pointer
            // It would be better but more annoying to make the user keep the index

            for(int i=0; i<n_store; ++i) {
                if(va.x == store[i]->x.get()) {
                    state[i] = ActiveState::Previous;
                    return;
                }
            }

            // We cannot reach here on a proper release;
            throw std::string("VecArrayAccum release does not match any acquire");
        }

        VecArray accum() {
            // accumulate all arrays into the zeroth array
            std::lock_guard<std::mutex> g(mut);

            float* base = store[0]->x.get();
            int n_entry = store[0]->n_elem*store[0]->row_width;

            for(int ns=1; ns<n_store; ++ns) {
                switch(state[ns]) {
                    case ActiveState::Never:
                        continue;

                    case ActiveState::Current:
                        throw std::string("VecArrayAccum missing release before accum");

                    case ActiveState::Previous:
                        float* acc = store[ns]->x.get();
                        for(int i=0; i<n_entry; ++i)
                            base[i] += acc[i];
                        zero_store(ns);
                }
            }
            
            return *store[0];
        }

        void zero_accum() {
            std::lock_guard<std::mutex> g(mut);
            for(int ns=0; ns<n_store; ++ns)
                if(ns==0 || state[ns] != ActiveState::Never)
                    zero_store(ns);
        }
};


struct range {
    int start;
    int stop;
    int stride;

    range(int stop_): start(0), stop(stop_), stride(1) {}
    range(int start_, int stop_): start(start_), stop(stop_), stride(1) {}
    range(int start_, int stop_, int stride_): start(start_), stop(stop_), stride(stride_) {}

    struct iterator {
        int index;
        int stride;

        iterator(int index_, int stride_): index(index_), stride(stride_) {}
        // this operator!= is designed to reprod
        bool operator!=(iterator other) {return index<other.index;}
        int  operator*() {return index;}
        iterator& operator++() {
            index+=stride;
            return *this;
        }
    };

   iterator begin() {return iterator(start, stride);}
   iterator end()   {return iterator(stop,  stride);}
};


template <int ndim, typename ScalarT = float>
struct
// alignas(std::alignment_of<ScalarT>::value)  // GCC 4.8.1 does not like this line
 Vec {
     typedef ScalarT scalar_t;
     ScalarT v[ndim>=1 ? ndim : 1];  // avoid 0-size arrays since this is not allowed

     ScalarT&       x()       {return           v[0];}
     const ScalarT& x() const {return           v[0];}
     ScalarT&       y()       {return ndim>=1 ? v[1] : v[0];}
     const ScalarT& y() const {return ndim>=1 ? v[1] : v[0];}
     ScalarT&       z()       {return ndim>=2 ? v[2] : v[0];}
     const ScalarT& z() const {return ndim>=2 ? v[2] : v[0];}
     ScalarT&       w()       {return ndim>=3 ? v[3] : v[0];}
     const ScalarT& w() const {return ndim>=3 ? v[3] : v[0];}

     ScalarT& operator[](int i) {return v[i];}
     const ScalarT& operator[](int i) const {return v[i];}
 };

typedef Vec<2,float> float2;
typedef Vec<3,float> float3;
typedef Vec<4,float> float4;

// template <int D>
// inline Vec<D,float> load_vec(const VecArray& a, int idx) {
//     Vec<D,float> r;
//     #pragma unroll
//     for(int d=0; d<D; ++d) r[d] = a(d,idx);
//     return r;
// }

template <int D>
inline Vec<D,float> load_vec(const VecArray& a, int idx) {
    Vec<D,float> r;
    #pragma unroll
    for(int d=0; d<D; ++d) r[d] = a(d,idx);
    return r;
}

template<int D>
inline Vec<D,float> load_vec(const float* a) {
    Vec<D,float> r;
    #pragma unroll
    for(int d=0; d<D; ++d) r[d] = a[d];
    return r;
}
template<int D>
inline Vec<D,Float4> load_vec(const float* a, Alignment align) {
    Vec<D,Float4> r;
    #pragma unroll
    for(int d=0; d<D; ++d) r[d] = Float4(a+4*d);
    return r;
}


template <int D>
inline void store_vec(VecArrayStorage& a, int idx, const Vec<D,float>& r) {
    #pragma unroll
    for(int d=0; d<D; ++d) a(d,idx) = r[d];
}

template <int D>
inline void store_vec(VecArray& a, int idx, const Vec<D,float>& r) {
    #pragma unroll
    for(int d=0; d<D; ++d) a(d,idx) = r[d];
}

template <int D>
inline void store_vec(float* a, const Vec<D,float>& r) {
    #pragma unroll
    for(int d=0; d<D; ++d) a[d] = r[d];
}

template <int D>
inline void store_vec(float* a, const Vec<D,Float4>& r, Alignment align=Alignment::aligned) {
    #pragma unroll
    for(int d=0; d<D; ++d) r[d].store(a+4*d, align);
}

template <int D, typename multype>
inline void update_vec_scale(VecArray a, int idx, const multype &r) {
    store_vec(a,idx, load_vec<D>(a,idx) * r);
}

template <int D, typename VArray>
inline void update_vec(VArray& a, int idx, const Vec<D> &r) {
    store_vec(a,idx, load_vec<D>(a,idx) + r);
}

template <int D>
inline void update_vec(float* a, const Vec<D> &r) {
    store_vec(a, load_vec<D>(a) + r);
}


//! Get component of vector by index

static const float M_PI_F   = 3.141592653589793f;   //!< value of pi as float
static const float M_1_PI_F = 0.3183098861837907f;  //!< value of 1/pi as float

inline float approx_rsqrt(float x) {return 1.f/sqrtf(x);}  //!< reciprocal square root at lower accuracy
inline float rsqrt(float x) {return 1.f/sqrtf(x);}  //!< reciprocal square root (1/sqrt(x))
template <typename D> constexpr inline D sqr(D x) {return x*x;}  //!< square a number (x^2)
inline float rcp       (float x) {return 1.f/x;}  //!< reciprocal of number
inline float approx_rcp(float x) {return 1.f/x;}  //!< reciprocal of number
inline float ternary(bool b, float x, float y) {return b ? x : y;}

template <int D, typename S>
inline Vec<D,S> vec_rcp(const Vec<D,S>& x) {
    Vec<D,S> y;
    for(int i=0; i<D; ++i) y[i] = rcp(x[i]);
    return y;
}

template <int D, typename S>
inline Vec<D,S> approx_vec_rcp(const Vec<D,S>& x) {
    Vec<D,S> y;
    for(int i=0; i<D; ++i) y[i] = approx_rcp(x[i]);
    return y;
}



// FIXME more general blendv needed
template <int D>
inline Vec<D,float> blendv(bool which, const Vec<D,float>& a, const Vec<D,float>& b) {
    return which ? a : b;
}




template<typename S> inline Vec<1,S> make_vec1(const S& x                                    ) {Vec<1,S> a; a[0]=x;                         return a;}
template<typename S> inline Vec<2,S> make_vec2(const S& x, const S& y                        ) {Vec<2,S> a; a[0]=x; a[1]=y;                 return a;}
template<typename S> inline Vec<3,S> make_vec3(const S& x, const S& y, const S& z            ) {Vec<3,S> a; a[0]=x; a[1]=y; a[2]=z;         return a;}
template<typename S> inline Vec<4,S> make_vec4(const S& x, const S& y, const S& z, const S& w) {Vec<4,S> a; a[0]=x; a[1]=y; a[2]=z; a[3]=w; return a;}
//! make float4 from float3 (as x,y,z) and scalar (as w)
inline float4 make_vec4(float3 v, float w) { return make_vec4(v.x(),v.y(),v.z(),w); } 

inline float3 xyz(const float4& x) { return make_vec3(x.x(),x.y(),x.z()); } //!< return x,y,z as float3 from a float4

//! \cond
template <int D, typename S> 
inline Vec<D,S> operator+(const Vec<D,S>& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]+b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator-(const Vec<D,S>& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]-b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator*(const Vec<D,S>& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]*b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator/(const Vec<D,S>& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]/b[i];
    return c;
}


template <int D, typename S> 
inline Vec<D,S> operator+(const S& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a   +b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator-(const S& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a   -b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator*(const S& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a   *b[i];
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator/(const S& a, const Vec<D,S>& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a   /b[i];
    return c;
}


template <int D, typename S> 
inline Vec<D,S> operator+(const Vec<D,S>& a, const S& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]+b;
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator-(const Vec<D,S>& a, const S& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]-b;
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator*(const Vec<D,S>& a, const S& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]*b;
    return c;
}
template <int D, typename S> 
inline Vec<D,S> operator/(const Vec<D,S>& a, const S& b) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = a[i]/b;
    return c;
}


template <int D, typename S> 
inline Vec<D,S>& operator+=(Vec<D,S>& a, const Vec<D,S>& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]+=b[i];
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator-=(Vec<D,S>& a, const Vec<D,S>& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]-=b[i];
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator*=(Vec<D,S>& a, const Vec<D,S>& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]*=b[i];
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator/=(Vec<D,S>& a, const Vec<D,S>& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]/=b[i];
    return a;
}


template <int D, typename S> 
inline Vec<D,S>& operator+=(Vec<D,S>& a, const S& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]+=b;
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator-=(Vec<D,S>& a, const S& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]-=b;
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator*=(Vec<D,S>& a, const S& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]*=b;
    return a;
}
template <int D, typename S> 
inline Vec<D,S>& operator/=(Vec<D,S>& a, const S& b) {
    #pragma unroll
    for(int i=0; i<D; ++i) a[i]/=b;
    return a;
}


template <int D, typename S> 
inline Vec<D,S> operator-(const Vec<D,S>& a) {
    Vec<D,S> c;
    #pragma unroll
    for(int i=0; i<D; ++i) c[i] = -a[i];
    return c;
}
//! \endcond

// FIXME how to handle sqrtf, rsqrtf for simd types?
// I will assume there are functions rsqrt(s) and rcp(s) and sqrt(s) and I will write sqr(s)

template <typename S> inline S zero() {return S(0.f);} 
template <typename S> inline S one () {return S(1.f);} 
template <typename S> inline S two () {return S(2.f);} 
template <typename S> inline S half() {return S(0.5f);} 

template <int D, typename S = float>
inline Vec<D,S> make_zero() {
    Vec<D,S> ret;
    for(int i=0; i<D; ++i) ret[i] = zero<S>();
    return ret;
}

template <int D, typename S = float>
inline Vec<D,S> make_one () {
    Vec<D,S> ret;
    for(int i=0; i<D; ++i) ret[i] = one <S>();
    return ret;
}



template<int start_loc, int stop_loc, int D, typename S>
inline Vec<stop_loc-start_loc,S> extract(const Vec<D,S>& x) {
    static_assert(0<=start_loc, "extract start must be non-negative");
    static_assert(start_loc<=stop_loc, "extract start must be <= extract stop");
    static_assert(stop_loc<=D, "extract stop must be <= dimension");

    Vec<stop_loc-start_loc,S> ret;
    #pragma unroll
    for(int i=0; i<stop_loc-start_loc; ++i) ret[i] = x[i+start_loc];
    return ret;
}

template<int start_loc, int stop_loc, int D, typename S>
inline void store(Vec<D,S>& x, const Vec<stop_loc-start_loc,S> &y) {
    static_assert(0<=start_loc, "extract start must be non-negative");
    static_assert(start_loc<=stop_loc, "extract start must be <= extract stop");
    static_assert(stop_loc<=D, "extract stop must be <= dimension");

    #pragma unroll
    for(int i=0; i<stop_loc-start_loc; ++i) x[i+start_loc] = y[i];
}

template <typename S>
inline S a_sqrt(const S& a) {
    return a * rsqrt(a);
}

// template <int D, typename S>
// inline S mag2(const Vec<D,S>& a) {
//     S m = zero<S>();
//     #pragma unroll
//     for(int i=0; i<D; ++i) m += sqr(a[i]);
//     return m;
// }

inline float fmadd(float a, float b, float c) {
    return a*b+c;
}

template <int D, typename S>
inline S mag2(const Vec<D,S>& a) {
    // not valid for length-0 vectors
    S m = a[0]*a[0];
    #pragma unroll
    for(int i=1; i<D; ++i) m = fmadd(a[i],a[i], m);
    return m;
}

template <int ndim, typename S>
inline S inv_mag(const Vec<ndim,S>& a) {
    return rsqrt(mag2(a));
}

template <int ndim, typename S>
inline S inv_mag2(const Vec<ndim,S>& a) {
    return rcp(mag2(a));
}

template <int ndim, typename S>
inline S mag(const Vec<ndim,S>& a) {
    return a_sqrt(mag2(a));
}

template <int D, typename S>
inline S sum(const Vec<D,S>& a) {
    S m = a[0];
    #pragma unroll
    for(int i=1; i<D; ++i) m += a[i];
    return m;
}

// derivative of (v/v_mag), which is otherwise know as a hat vector
template<typename S>
inline void
hat_deriv(
        const Vec<3,S>& v_hat, const S& v_invmag, 
        Vec<3,S> &col0, Vec<3,S> &col1, Vec<3,S> &col2) // matrix is symmetric, so these are rows or cols
{
    auto s = v_invmag;
    auto one = S(1.f);

    col0 = make_vec3(s*(one-v_hat.x()*v_hat.x()), s*    -v_hat.y()*v_hat.x() , s*    -v_hat.z()*v_hat.x() );
    col1 = make_vec3(s*    -v_hat.x()*v_hat.y() , s*(one-v_hat.y()*v_hat.y()), s*    -v_hat.z()*v_hat.y() );
    col2 = make_vec3(s*    -v_hat.x()*v_hat.z() , s*    -v_hat.y()*v_hat.z() , s*(one-v_hat.z()*v_hat.z()));
}


//! cross-product of the vectors a and b
template <typename S>
inline Vec<3,S> cross(const Vec<3,S>& a, const Vec<3,S>& b){
    Vec<3,S> c;
    c[0] = a.y()*b.z() - a.z()*b.y();
    c[1] = a.z()*b.x() - a.x()*b.z();
    c[2] = a.x()*b.y() - a.y()*b.x();
    return c;
}

template <int D, typename S>
inline Vec<D,S> normalized(const Vec<D,S>& a) { return a*inv_mag(a); }

template <int D, typename S>
inline Vec<D,S> approx_normalized(const Vec<D,S>& a) {return a*approx_rsqrt(mag2(a)); }

template <int D, typename S>
inline Vec<D,S> prob_normalized(const Vec<D,S>& a) { return a*rcp(sum(a)); }

template <int D, typename S>
inline S dot(const Vec<D,S>& a, const Vec<D,S>& b){
    S c = a[0]*b[0];
    #pragma unroll
    for(int i=1; i<D; ++i) c = fmadd(a[i],b[i], c);
    return c;
}


template <int D, typename S>
Vec<D,S> left_multiply_matrix(Vec<D*D,S> m, Vec<D,S> v) {
    Vec<D,S> mv;
    for(int i=0; i<D; ++i) {
        auto x = m[i*D+0]*v[0];
        for(int j=1; j<D; ++j)
            x = fmadd(m[i*D+j],v[j], x);
        mv[i] = x;
    }
    return mv;
}

template <int D, typename S>
Vec<D,S> right_multiply_matrix(Vec<D,S> v, Vec<D*D,S> m) {
    Vec<D,S> vm;
    for(int j=0; j<D; ++j) {
        auto x = v[0]*m[0*D+j];
        for(int i=1; i<D; ++i)
            x = fmadd(v[i],m[i*D+j], x);
        vm[j] = x;
    }
    return vm;
}

template <int D, typename S>
S min(Vec<D,S> y) {
    using namespace std;
    S x = y[0];
    // for(int i=1; i<D; ++i) x = ternary((y[i]<x), y[i], x);
    for(int i=1; i<D; ++i) x = min(y[i],x);
    return x;
}

template <int D, typename S>
S max(Vec<D,S> y) {
    using namespace std;
    S x = y[0];
    // for(int i=1; i<D; ++i) x = ternary((x<=y[i]), y[i], x);
    for(int i=1; i<D; ++i) x = max(y[i],x);
    return x;
}


//! sigmoid function and its derivative 

//! Value of function is 1/(1+exp(x)) and the derivative is 
//! exp(x)/(1+exp(x))^2
template <typename S>
inline Vec<2,S> sigmoid(S x) {
    S z = expf(-x);
    S w = rcp(S(1.f)+z);
    return make_vec2(w, z*w*w);
}

inline bool any(bool x) {return x;} // scalar any function is trivial
inline bool none(bool x) {return !x;} // scalar none function is trivial

// Sigmoid-like function that has zero derivative outside (-1/sharpness,1/sharpness)
// This function is 1 for large negative values and 0 for large positive values
template <typename S>
inline Vec<2,S> compact_sigmoid(const S& x, const S& sharpness) {
    // FIXME this sigmoid is both narrower and reversed direction from a normal sigmoid
    S y = x*sharpness;
    Vec<2,S> z = make_vec2(
            S(0.25f)*(y+S(2.f))*(y-one<S>())*(y-one<S>()),
            (sharpness*S(0.75f))*(sqr(y)-one<S>()));

    auto too_big   = S( 1.f)<y;
    auto too_small = y<S(-1.f);

    // apply cutoffs
    z.x() = ternary(too_small, one<S>(), ternary(too_big, zero<S>(), z.x()));
    z.y() = ternary(too_small | too_big, zero<S>(), z.y());
    return z;
}

inline float compact_sigmoid_cutoff(float sharpness) {
#ifdef NONCOMPACT_SIGMOID
    return 2.f/sharpness;  // 0.002
#else
    return 1.f/sharpness;  // exactly 0.
#endif
}


//! Sigmoid-like function that has zero derivative outside the two intervals (-half_width-1/sharpness,-half_width+1/sharpness)
//! and (half_width-1/sharpness,half_width+1/sharpness).  The function is the product of opposing half-sigmoids.
template <typename S>
inline Vec<2,S> compact_double_sigmoid(const S& x, const S& half_width, const S& sharpness) {
    Vec<2,S> v1 = compact_sigmoid( x-half_width, sharpness);
    Vec<2,S> v2 = compact_sigmoid(-x-half_width, sharpness);
    return make_vec2(v1.x()*v2.x(), v1.y()*v2.x()-v1.x()*v2.y());
}


//! compact_double_sigmoid that also handles periodicity
//! note that both theta and center must be in the range (-PI,PI)
template <typename S>
inline Vec<2,S> angular_compact_double_sigmoid(const S& theta, const S& center, const S& half_width, const S& sharpness) {
    S dev = theta - center;
    dev = blendv((dev <-M_PI_F), dev + S(2.f*M_PI_F), dev);
    dev = blendv((dev > M_PI_F), dev - S(2.f*M_PI_F), dev);
    return compact_double_sigmoid(dev, half_width, sharpness);
}

//! order is value, then dvalue/dphi, dvalue/dpsi
template <typename S>
inline Vec<3,S> rama_box(const Vec<2,S>& rama, const Vec<2,S>& center, const Vec<2,S>& half_width, const S& sharpness) {
    Vec<2,S> phi = angular_compact_double_sigmoid(rama.x(), center.x(), half_width.x(), sharpness);
    Vec<2,S> psi = angular_compact_double_sigmoid(rama.y(), center.y(), half_width.y(), sharpness);
    return make_vec3(phi.x()*psi.x(), phi.y()*psi.x(), phi.x()*psi.y());
}


//! Compute a dihedral angle and its derivative from four positions

//! The angle is always in the range [-pi,pi].  If the arguments are NaN or Inf, the result is undefined.
//! The arguments d1,d2,d3,d4 are output values, and d1 corresponds to the derivative of the dihedral angle
//! with respect to r1.
template <typename V>  // parameterize on vector type (probably Vec<3> or Float4)
static typename V::scalar_t dihedral_germ(
        V  r1, V  r2, V  r3, V  r4,
        V &d1, V &d2, V &d3, V &d4)
    // Formulas and notation taken from Blondel and Karplus, 1995
{
    typedef typename V::scalar_t S;  // scalar type associated to vector type

    V F = r1-r2;
    V G = r2-r3;
    V H = r4-r3;

    V A = cross(F,G);
    V B = cross(H,G);
    V C = cross(B,A);

    S inv_Amag2 = inv_mag2(A);
    S inv_Bmag2 = inv_mag2(B);

    S Gmag2    = mag2(G);
    S inv_Gmag = rsqrt(Gmag2);
    S Gmag     = Gmag2 * inv_Gmag;

    d1 = -Gmag * inv_Amag2 * A;
    d4 =  Gmag * inv_Bmag2 * B;

    V f_mid =  dot(F,G)*inv_Amag2*inv_Gmag * A - dot(H,G)*inv_Bmag2*inv_Gmag * B;

    d2 = -d1 + f_mid;
    d3 = -d4 - f_mid;

    return atan2f(extract_float(dot(C,G)), extract_float(dot(A,B) * Gmag));
}


static void print(const VecArray &a, int n_dim, int n_elem, const char* txt) {
    for(int ne: range(n_elem)) {
        printf("%s% 4i  ", txt, ne);
        for(int nd: range(n_dim)) printf(" % .2f", a(nd,ne));
        printf("\n");
    }
}


template<int D>
inline Vec<D,Float4> aligned_gather_vec(const float* data, const Int4& offsets) {
    Vec<D,Float4> ret;

    const float* p0 = data+offsets.x();
    const float* p1 = data+offsets.y();
    const float* p2 = data+offsets.z();
    const float* p3 = data+offsets.w();

    Float4 extra[3]; // scratch space to do the transpose

    #pragma unroll
    for(int d=0; d<D; d+=4) {
        ret[d  ]                      = Float4(p0+d);
        (d+1<D ? ret[d+1] : extra[0]) = Float4(p1+d);
        (d+2<D ? ret[d+2] : extra[1]) = Float4(p2+d);
        (d+3<D ? ret[d+3] : extra[2]) = Float4(p3+d);

        transpose4(
                ret[d  ],
                (d+1<D ? ret[d+1] : extra[0]),
                (d+2<D ? ret[d+2] : extra[1]),
                (d+3<D ? ret[d+3] : extra[2]));
    }

    return ret;
}


template<int D>
inline void aligned_scatter_store_vec_destructive(float* data, const Int4& offsets, Vec<D,Float4>& v) {
    // note that this function changes the vector v

    float* p0 = data+offsets.x();
    float* p1 = data+offsets.y();
    float* p2 = data+offsets.z();
    float* p3 = data+offsets.w();

    Float4 extra[3]; // scratch space to do the transpose

    #pragma unroll
    for(int d=0; d<D; d+=4) {
        transpose4(
                v[d  ],
                (d+1<D ? v[d+1] : extra[0]),
                (d+2<D ? v[d+2] : extra[1]),
                (d+3<D ? v[d+3] : extra[2]));

        // this writes must be done sequentially in case some of the 
        // offsets are equal (and hence point to the same memory location)
        v[d  ]                     .store(p0+d);
        (d+1<D ? v[d+1] : extra[0]).store(p1+d);
        (d+2<D ? v[d+2] : extra[1]).store(p2+d);
        (d+3<D ? v[d+3] : extra[2]).store(p3+d);
    }
}


template<int D>
inline void aligned_scatter_update_vec_destructive(float* data, const Int4& offsets, Vec<D,Float4>& v) {
    // note that this function changes the vector v

    float* p0 = data+offsets.x();
    float* p1 = data+offsets.y();
    float* p2 = data+offsets.z();
    float* p3 = data+offsets.w();

    Float4 extra[3]; // scratch space to do the transpose

    #pragma unroll
    for(int d=0; d<D; d+=4) {
        transpose4(
                v[d  ],
                (d+1<D ? v[d+1] : extra[0]),
                (d+2<D ? v[d+2] : extra[1]),
                (d+3<D ? v[d+3] : extra[2]));

        // this writes must be done sequentially in case some of the 
        // offsets are equal (and hence point to the same memory location)
        (Float4(p0+d) + v[d  ]                     ).store(p0+d);
        (Float4(p1+d) + (d+1<D ? v[d+1] : extra[0])).store(p1+d);
        (Float4(p2+d) + (d+2<D ? v[d+2] : extra[1])).store(p2+d);
        (Float4(p3+d) + (d+3<D ? v[d+3] : extra[2])).store(p3+d);
    }
}


#endif
