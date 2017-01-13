#ifndef FLOAT4_H
#define FLOAT4_H

// Author: John Jumper

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
    
enum class Alignment {unaligned, aligned};

struct Int4;
struct Float4;

static void print_vector(const char* nm, const Float4& val);

alignas(16) static const uint8_t left_pack_control_vector[16*16] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
  4, 5, 6, 7, 0, 1, 2, 3, 8, 9,10,11,12,13,14,15,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
  8, 9,10,11, 0, 1, 2, 3, 4, 5, 6, 7,12,13,14,15,
  0, 1, 2, 3, 8, 9,10,11, 4, 5, 6, 7,12,13,14,15,
  4, 5, 6, 7, 8, 9,10,11, 0, 1, 2, 3,12,13,14,15,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
 12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
  0, 1, 2, 3,12,13,14,15, 4, 5, 6, 7, 8, 9,10,11,
  4, 5, 6, 7,12,13,14,15, 0, 1, 2, 3, 8, 9,10,11,
  0, 1, 2, 3, 4, 5, 6, 7,12,13,14,15, 8, 9,10,11,
  8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7,
  0, 1, 2, 3, 8, 9,10,11,12,13,14,15, 4, 5, 6, 7,
  4, 5, 6, 7, 8, 9,10,11,12,13,14,15, 0, 1, 2, 3,
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
};

alignas(16) static const uint32_t  abs_mask[4] = {0x7fffffffu, 0x7fffffffu, 0x7fffffffu, 0x7fffffffu};
alignas(16) static const uint32_t sign_mask[4] = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};


inline int popcnt_nibble(int value) {
    return _mm_popcnt_u32((unsigned) value);
}


struct alignas(16) Int4
{
    protected:
        __m128i vec;
        Int4(__m128i vec_):
            vec(vec_)
        {};

    public:
        Int4(): vec(_mm_setzero_si128()) {}

        // constructor from aligned storage
        explicit Int4(const int32_t* vec_, Alignment align = Alignment::aligned):
            vec(align==Alignment::aligned ? _mm_load_si128((__m128i*)vec_) : _mm_loadu_si128((__m128i*)vec_)) {}

        // broadcast constructor
        // explicit Int4(const int32_t& val): 
        //     vec(_mm_castps_si128(_mm_broadcast_ss((float*)(&val)))) {}
        explicit Int4(const int val):
            vec(_mm_set1_epi32(val)) {}

        // gather constructor from offsets
        // Not particularly efficient
        Int4(const int32_t* base, const Int4& offsets) {
            alignas(16) int32_t data[4] = {
                base[offsets.x()], base[offsets.y()], base[offsets.z()], base[offsets.w()]};
            vec = _mm_load_si128((const __m128i*)data);
        }

        Int4 left_pack(int mask) const {
            return Int4(_mm_shuffle_epi8(vec, ((__m128i*)left_pack_control_vector)[mask]));
        }

        Int4 left_pack_inplace(int mask) {
            vec = _mm_shuffle_epi8(vec, ((__m128i*)left_pack_control_vector)[mask]);
            return *this;
        }

        Int4 operator+(const Int4 &o) const {return Int4(_mm_add_epi32  (vec, o.vec));}
        Int4 operator-(const Int4 &o) const {return Int4(_mm_sub_epi32  (vec, o.vec));}
        Int4 operator-()                const {return _mm_sub_epi32(_mm_setzero_si128(), vec);}
        Int4 operator*(const Int4 &o) const {return Int4(_mm_mullo_epi32  (vec, o.vec));}
        Int4 operator<(const Int4 &o) const {return Int4(_mm_cmplt_epi32(vec,o.vec));}
        Int4 operator==(const Int4 &o) const {return Int4(_mm_cmpeq_epi32(vec,o.vec));}
        Int4 operator&(const Int4 &o) const {return Int4(_mm_and_si128(vec,o.vec));}
        Int4 operator|(const Int4 &o) const {return Int4(_mm_or_si128(vec,o.vec));}
        Int4 operator!=(const Int4 &o) const {
            __m128i all_zero = _mm_setzero_si128();
            __m128i all_one  = _mm_cmpeq_epi32(all_zero, all_zero);
            return Int4(_mm_xor_si128(all_one, _mm_cmpeq_epi32(vec,o.vec)));
        }

        Int4 operator|=(const Int4 &o) {return vec = _mm_or_si128(vec,o.vec);}

        bool any () const {return !_mm_testz_si128(vec,vec);}
        bool none() const {return  _mm_testz_si128(vec,vec);}

        Int4 sum_in_all_entries() const {
            // the sum of the vector is now in all entries
            __m128i vec2 = _mm_hadd_epi32(vec,vec);
            return _mm_hadd_epi32(vec2,vec2);
        }

        int x() const {return _mm_extract_epi32(vec,0);}
        int y() const {return _mm_extract_epi32(vec,1);}
        int z() const {return _mm_extract_epi32(vec,2);}
        int w() const {return _mm_extract_epi32(vec,3);}

        int movemask() {return _mm_movemask_ps(_mm_castsi128_ps(vec));}

        friend Float4;
        // Float4 Int4::cast_float() const {
        //     // bit-equivalent cast to float
        //     return Float4(_mm_castps_si128(vec));
        // } 

        void store(int32_t* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm_store_si128 ((__m128i*)vec_, vec); 
            else 
                _mm_storeu_si128((__m128i*)vec_,vec);
        }

        Int4 srl(int shift_count) const {return Int4(_mm_srli_epi32(vec,shift_count));} // right logical shift
        Int4 sll(int shift_count) const {return Int4(_mm_slli_epi32(vec,shift_count));} // left  logical shift
};




struct alignas(16) Float4 
{
    protected:
        __m128 vec;
        Float4(__m128 vec_):
            vec(vec_)
        {};

    public:
        typedef Float4 scalar_t; // functions that return scalars (like dot) give constant Float4's
        Float4(): vec(_mm_setzero_ps()) {}

        // constructor from aligned storage
        explicit Float4(const float* vec_, Alignment align = Alignment::aligned):
            vec(align==Alignment::aligned ? _mm_load_ps(vec_) : _mm_loadu_ps(vec_)) {}

        // broadcast constructor
        // Float4(const float& val):   
        //     vec(_mm_broadcast_ss(&val)) {}
        Float4(const float val):   
            vec(_mm_set1_ps(val)) {}

        // gather constructor from offsets
        // Not particularly efficient
        Float4(const float* base, const Int4& offsets) {
            alignas(16) float data[4] = {
                base[offsets.x()], base[offsets.y()], base[offsets.z()], base[offsets.w()]};
            vec = _mm_load_ps(data);
        }

        template <int i>
        Float4 broadcast() const {
            return Float4(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE(i,i,i,i)));
        }

        template <int out_mask0, int out_mask1, int out_mask2, int out_mask3,
                  int sum_mask0, int sum_mask1, int sum_mask2, int sum_mask3>
        Float4 dp(const Float4& o) const {
            return Float4(_mm_dp_ps(vec,o.vec, 
                    (out_mask0<<0) | 
                    (out_mask1<<1) | 
                    (out_mask2<<2) | 
                    (out_mask3<<3) | 
                    (sum_mask0<<4) | 
                    (sum_mask1<<5) | 
                    (sum_mask2<<6) | 
                    (sum_mask3<<7)));
        }

        Float4 operator+ (const Float4 &o) const {return Float4(_mm_add_ps  (vec, o.vec));}
        Float4 operator- (const Float4 &o) const {return Float4(_mm_sub_ps  (vec, o.vec));}
        Float4 operator- ()                const {return _mm_sub_ps(_mm_setzero_ps(), vec);}
        Float4 operator* (const Float4 &o) const {return Float4(_mm_mul_ps  (vec, o.vec));}
        Float4 operator< (const Float4 &o) const {return Float4(_mm_cmplt_ps(vec,o.vec));}
        Float4 operator<=(const Float4 &o) const {return Float4(_mm_cmple_ps(vec,o.vec));}
        Float4 operator!=(const Float4 &o) const {return Float4(_mm_cmpneq_ps(vec,o.vec));}
        Float4 operator==(const Float4 &o) const {return Float4(_mm_cmpeq_ps(vec,o.vec));}
        Float4 operator& (const Float4 &o) const {return Float4(_mm_and_ps(vec,o.vec));}
        Float4 operator| (const Float4 &o) const {return Float4(_mm_or_ps(vec,o.vec));}

        Float4 approx_rsqrt() const { 
            // 12-bit accuracy (about to about 0.02%)
            return _mm_rsqrt_ps(vec);
        }
        Float4 approx_rcp() const { 
            // 12-bit accuracy (about to about 0.02%)
            return _mm_rcp_ps(vec);
        }
        Float4 rsqrt() const { 
            // one round of newton-raphson, see online documentation for _mm_rsqrt_ps for details
            Float4 a = _mm_rsqrt_ps(vec);   // 12-bit approximation
            return Float4(1.5f)*a - (Float4(0.5f)*(*this)) * a * (a*a);
        }
        Float4 rcp() const {
            // one round of newton-raphson, see online documentation for _mm_rcp_ps for details
            auto x = approx_rcp();
            return x*(Float4(2.f) - (*this)*x);
        }
        // Float4 sqrt() const {return (*this) * this->rsqrt();}  // faster than _mm_sqrt_ps but NaN for vec==0.
        Float4 sqrt() const {return _mm_sqrt_ps(vec);}

        Float4 abs()  const {
            // this function assumes well-behaved, finite floats
            // I am not sure if it works sanely for denormals, inf, and NaN
            return _mm_and_ps(vec, _mm_load_ps((const float*)abs_mask));
        }

        Float4 copysign(const Float4 &o)  const {
            // this function assumes well-behaved, finite floats
            // I am not sure if it works sanely for denormals, inf, and NaN
            return _mm_or_ps(
                    _mm_and_ps(  vec, _mm_load_ps((const float*) abs_mask)),
                    _mm_and_ps(o.vec, _mm_load_ps((const float*)sign_mask)));
        }

        Float4 operator+=(const Float4 &o) {return vec = _mm_add_ps(vec, o.vec);}
        Float4 operator-=(const Float4 &o) {return vec = _mm_sub_ps(vec, o.vec);}
        Float4 operator*=(const Float4 &o) {return vec = _mm_mul_ps(vec, o.vec);}
        Float4 operator|=(const Float4 &o) {return vec = _mm_or_ps(vec,o.vec);}

        int movemask() const {return _mm_movemask_ps(vec);}
        bool none() const {__m128i v = _mm_castps_si128(vec); return  _mm_testz_si128(v,v);}
        bool any() const  {return !none();}

        const Float4 right_rotate() const { return Float4(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE(2,1,0,3))); }
        const Float4 left_rotate()  const { return Float4(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE(0,3,2,1))); }

        void store(float* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm_store_ps(vec_, vec); 
            else 
                _mm_storeu_ps(vec_,vec);
        }

        void store_scalar(float* x) {
            _mm_store_ss(x,vec);
        }

        Float4 update(float* vec_, Alignment align=Alignment::aligned) const { 
            Float4 new_val = *this + Float4(vec_,align);
            new_val.store(vec_,align);
            return new_val;
        }
        Float4 scale_update(Float4& scale_val, float* vec_, Alignment align=Alignment::aligned) const { 
            Float4 new_val = fmadd(scale_val,*this, Float4(vec_,align));
            new_val.store(vec_,align);
            return new_val;
        }

        Float4 sum_in_all_entries() const {
            // the sum of the vector is now in all entries
            __m128 vec2 = _mm_hadd_ps(vec,vec);
            return _mm_hadd_ps(vec2,vec2);
        }

        // choose from values whenever the equivalent element of mask is true
        // *this is the mask 
        //   c.ternary(a,b)  ==  c ? a : b;
        Float4 ternary(const Float4& a, const Float4& b) const {
            return Float4(_mm_blendv_ps(b.vec, a.vec, vec));
        }

        template <int round_mode = _MM_FROUND_TO_NEAREST_INT>
        Float4 round() const {return Float4(_mm_round_ps(vec, round_mode));}

        Int4 truncate_to_int() const {
            return Int4(_mm_cvttps_epi32(vec));
        }

        template <int m0, int m1, int m2, int m3>
        Float4 blend(const Float4& o) const {
            constexpr int mask = (m0<<0) + (m1<<1) + (m2<<2) + (m3<<3);
            return _mm_blend_ps(vec, o.vec, mask);
        }

        template <int m0, int m1, int m2, int m3>
        Float4 zero_entries() const {
            return this->blend<m0,m1,m2,m3>(Float4());
        }

        Float4 horizontal_max() const {
            // FIXME improve this implementation
            return max(max(broadcast<0>(), broadcast<1>()), max(broadcast<2>(),broadcast<3>()));
        }
        Float4 horizontal_min() const {
            // FIXME improve this implementation
            return min(min(broadcast<0>(), broadcast<1>()), min(broadcast<2>(),broadcast<3>()));
        }

        float x() const { float val; _MM_EXTRACT_FLOAT(val, vec, 0); return val;}
        float y() const { float val; _MM_EXTRACT_FLOAT(val, vec, 1); return val;}
        float z() const { float val; _MM_EXTRACT_FLOAT(val, vec, 2); return val;}
        float w() const { float val; _MM_EXTRACT_FLOAT(val, vec, 3); return val;}

        friend Int4;
        friend void transpose4(Float4&, Float4&, Float4&, Float4&);
        template <int i3, int i2, int i1, int i0> friend Float4 shuffle(Float4 m1, Float4 m2);

        Int4 cast_int() const {
            // bit-equivalent cast to int
            return Int4(_mm_castps_si128(vec));
        } 

        friend inline Float4 fmadd(const Float4& a1, const Float4& a2, const Float4& b);
        friend inline Float4 fmsub(const Float4& a1, const Float4& a2, const Float4& b);
        friend inline Float4 min(const Float4& a, const Float4& b);
        friend inline Float4 max(const Float4& a, const Float4& b);
        friend inline Float4 horizontal_add(const Float4& x1, const Float4& x2);
};

/*
struct alignas(32) Float8 
{
    protected:
        __m256 vec;
        Float8(__m256 vec_):
            vec(vec_)
        {};

    public:
        Float8(): vec(_mm256_setzero_ps()) {}

        // constructor from aligned storage
        explicit Float8(const float* vec, Alignment align = Alignment::aligned):
            vec(align==Alignment::aligned ? _mm256_load_ps(vec) : _mm256_loadu_ps(vec)) {}

        // broadcast constructor
        Float8(float val):   
            vec(_mm256_broadcast_ps(val)) {}

        Float8 operator+(const Float8 &o) const {return Float8(_mm256_add_ps  (vec, o.vec));}
        Float8 operator-(const Float8 &o) const {return Float8(_mm256_sub_ps  (vec, o.vec));}
        Float8 operator*(const Float8 &o) const {return Float8(_mm256_mul_ps  (vec, o.vec));}
        Float8 operator<(const Float8 &o) const {return Float8(_mm256_cmp_ps(vec,o.vec,_CMP_LT_OQ));}
        Float8 operator!=(const Float8 &o) const {return Float8(_mm256_cmp_ps(vec,o.vec,_CMP_NEQ_OQ));}
        Float8 operator==(const Float8 &o) const {return Float8(_mm256_cmp_ps(vec,o.vec,_CMP_EQ_OQ));}
        Float8 operator&(const Float8 &o) const {return Float8(_mm256_and_ps(vec,o.vec));}
        Float8 operator|(const Float8 &o) const {return Float8(_mm256_or_ps(vec,o.vec));}
        Float8 sqrt() const {return Float8(_mm256_sqrt_ps(vec));}

        Float8 operator+=(const Float8 &o) {return vec = _mm256_add_ps(vec, o.vec);}
        Float8 operator-=(const Float8 &o) {return vec = _mm256_sub_ps(vec, o.vec);}
        Float8 operator*=(const Float8 &o) {return vec = _mm256_mul_ps(vec, o.vec);}

        // int movemask() {return _mm256_movemask_ps(vec);}
        // bool any() {return movemask();}

        void store(float* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm256_store_ps(vec_, vec); 
            else 
                _mm256_storeu_ps(vec_,vec);
        }


        Float8 swap_halves() const {return Float8(_mm256_permute2f128_ps(vec,vec,1));}

        float sum() const {
            __m256 half_sum = _mm256_add_ps(vec, swap_halves().vec);
            __m256 x = _mm256_hadd_ps(half_sum, half_sum);
            x = _mm256_hadd_ps(x, x);

            float ret;
            _mm_store_ss(&ret, _mm256_castps256_ps128(x));
            return ret;
        }
};
*/

static void print_vector(const char* nm, const Float4& val) {
    printf("%s % .2f % .2f % .2f % .2f\n", nm, val.x(), val.y(), val.z(), val.w());
}

static void print_vector(const char* nm, const Int4& val) {
    printf("%s %3i %3i %3i %3i\n", nm, val.x(), val.y(), val.z(), val.w());
}

// convenience functions to match float interface
inline Float4 rsqrt(const Float4& x) { return x.rsqrt(); }
inline Float4 sqrtf(const Float4& x) { return x.sqrt(); }
inline Float4 rcp(const Float4& x) { return x.rcp();}

// this function uses a left-to-right convention, unlike the _MM_SHUFFLE macro
template <int i0, int i1, int i2, int i3>
Float4 shuffle(Float4 m1, Float4 m2)
{
    return Float4(_mm_shuffle_ps(m1.vec,m2.vec, _MM_SHUFFLE(i3,i2,i1,i0)));
}

// this function uses a left-to-right convention, unlike the _MM_SHUFFLE macro
// convenience function where shuffle(x) == shuffle(x,x)
template <int i0, int i1, int i2, int i3>
Float4 shuffle(Float4 m)
{
    return shuffle<i0,i1,i2,i3>(m,m);
}

inline void transpose4(Float4 &x, Float4 &y, Float4 &z, Float4 &w)
{
    _MM_TRANSPOSE4_PS(x.vec,y.vec,z.vec,w.vec);
}

inline Float4 fmadd(const Float4& a1, const Float4& a2, const Float4& b) {
#ifdef __AVX2__
    return Float4(_mm_fmadd_ps(a1.vec,a2.vec, b.vec));
#else
    return Float4(_mm_add_ps(_mm_mul_ps(a1.vec,a2.vec), b.vec));
#endif
}

inline Float4 fmsub(const Float4& a1, const Float4& a2, const Float4& b) {
#ifdef __AVX2__
    return Float4(_mm_fmsub_ps(a1.vec,a2.vec, b.vec));
#else
    return Float4(_mm_sub_ps(_mm_mul_ps(a1.vec,a2.vec), b.vec));
#endif
}

// FIXME I should put in an efficient, approximate SSE expf
//   but for now I will just make it work
inline Float4 expf(const Float4& x) {
    alignas(16) float result[4] = {expf(x.x()), expf(x.y()), expf(x.z()), expf(x.w())};
    return Float4(result, Alignment::aligned);
}

// FIXME I should put in an efficient, approximate SSE logf
//   but for now I will just make it work
inline Float4 logf(const Float4& x) {
    alignas(16) float result[4] = {logf(x.x()), logf(x.y()), logf(x.z()), logf(x.w())};
    return Float4(result, Alignment::aligned);
}

inline bool any(const Float4& x) {return x.any();}
inline bool none(const Float4& x) {return x.none();}

inline Float4 ternary(const Float4& which, const Float4& a, const Float4& b) {
    // compatibility with float version
    return which.ternary(a,b);
}

inline Float4 min(const Float4& a, const Float4& b) {
    return _mm_min_ps(a.vec, b.vec);
}

inline Float4 max(const Float4& a, const Float4& b) {
    return _mm_max_ps(a.vec, b.vec);
}

inline Float4 approx_rsqrt(const Float4& x) {return x.approx_rsqrt();}
inline Float4 approx_rcp  (const Float4& x) {return x.approx_rcp  ();}


inline Float4 horizontal_add(const Float4& x1, const Float4& x2) {
    return Float4(_mm_hadd_ps(x1.vec, x2.vec));
}

template <int out_mask0, int out_mask1, int out_mask2, int out_mask3,
          int sum_mask0, int sum_mask1, int sum_mask2, int sum_mask3>
Float4 dp(const Float4& x, const Float4& y) {
    return x.dp<out_mask0, out_mask1, out_mask2, out_mask3,
                sum_mask0, sum_mask1, sum_mask2, sum_mask3>(y);
}

inline Float4 dot(const Float4& x, const Float4& y) {
    return x.dp<1,1,1,1, 1,1,1,1>(y);
}

inline Float4 dot3(const Float4& x, const Float4& y) {
    return x.dp<1,1,1,0, 1,1,1,0>(y);
}

inline Float4 mag2(const Float4& x) {
    return dot(x,x);
}

inline Float4 inv_mag (const Float4& x) {
    return rsqrt(dot(x,x));
}

inline Float4 inv_mag2(const Float4& x) {
    return rcp(dot(x,x));
}

inline Float4 cross(const Float4& x, const Float4& y) {
    return fmsub(shuffle<1,2,0,3>(x),  shuffle<2,0,1,3>(y),
                 shuffle<2,0,1,3>(x) * shuffle<1,2,0,3>(y));
}

inline float extract_float(const Float4& x) { return x.x(); }
inline float extract_float(const Float4& x, int d) { 
    // this function is slow but can be convenient
    alignas(16) float data[4];
    x.store(data);
    return data[d];
}

inline float extract_float(float x) { return x; }


inline Float4 fabsf(const Float4 &x) {
    return x.abs();
}

inline Float4 copysignf(const Float4 &mag_carrier, const Float4 &sign_carrier) {
    return mag_carrier.copysign(sign_carrier);
}

inline Float4 approx_max_normalize3(const Float4& o) {
    auto m = max(o.broadcast<0>(), max(o.broadcast<1>(), o.broadcast<2>()));
    return o*approx_rcp(m);
}

inline Float4 isnan(const Float4& o) {return o!=o;}

#endif
