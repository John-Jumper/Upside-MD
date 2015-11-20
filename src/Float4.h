#ifndef FLOAT4_H
#define FLOAT4_H

// Author: John Jumper

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
    
enum class Alignment {unaligned, aligned};

struct Int4;
struct Float4;

alignas(16) static uint8_t left_pack_control_vector[16*16] = {
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
        explicit Int4(const int32_t* vec, Alignment align = Alignment::aligned):
            vec(align==Alignment::aligned ? _mm_load_si128((__m128i*)vec) : _mm_loadu_si128((__m128i*)vec)) {}

        // broadcast constructor
        explicit Int4(const int32_t& val): 
            vec(_mm_castps_si128(_mm_broadcast_ss((float*)(&val)))) {}

        Int4 left_pack(int mask) {
            return Int4(_mm_shuffle_epi8(vec, ((__m128i*)left_pack_control_vector)[mask]));
        }

        Int4 operator+(const Int4 &o) const {return Int4(_mm_add_epi32  (vec, o.vec));}
        Int4 operator-(const Int4 &o) const {return Int4(_mm_sub_epi32  (vec, o.vec));}
        Int4 operator-()                const {return _mm_sub_epi32(_mm_setzero_si128(), vec);}
        Int4 operator*(const Int4 &o) const {return Int4(_mm_mul_epi32  (vec, o.vec));}
        Int4 operator<(const Int4 &o) const {return Int4(_mm_cmplt_epi32(vec,o.vec));}
        Int4 operator==(const Int4 &o) const {return Int4(_mm_cmpeq_epi32(vec,o.vec));}
        Int4 operator&(const Int4 &o) const {return Int4(_mm_and_si128(vec,o.vec));}
        Int4 operator|(const Int4 &o) const {return Int4(_mm_or_si128(vec,o.vec));}
        Int4 operator!=(const Int4 &o) const {
            __m128i all_zero = _mm_setzero_si128();
            __m128i all_one  = _mm_cmpeq_epi32(all_zero, all_zero);
            return Int4(_mm_xor_si128(all_one, _mm_cmpeq_epi32(vec,o.vec)));
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
        Float4(): vec(_mm_setzero_ps()) {}

        // constructor from aligned storage
        explicit Float4(const float* vec, Alignment align = Alignment::aligned):
            vec(align==Alignment::aligned ? _mm_load_ps(vec) : _mm_loadu_ps(vec)) {}

        // broadcast constructor
        Float4(const float& val):   
            vec(_mm_broadcast_ss(&val)) {}

        Float4 operator+(const Float4 &o) const {return Float4(_mm_add_ps  (vec, o.vec));}
        Float4 operator-(const Float4 &o) const {return Float4(_mm_sub_ps  (vec, o.vec));}
        Float4 operator-()                const {return _mm_sub_ps(_mm_setzero_ps(), vec);}
        Float4 operator*(const Float4 &o) const {return Float4(_mm_mul_ps  (vec, o.vec));}
        Float4 operator<(const Float4 &o) const {return Float4(_mm_cmplt_ps(vec,o.vec));}
        Float4 operator<=(const Float4 &o) const {return Float4(_mm_cmple_ps(vec,o.vec));}
        Float4 operator!=(const Float4 &o) const {return Float4(_mm_cmpneq_ps(vec,o.vec));}
        Float4 operator==(const Float4 &o) const {return Float4(_mm_cmpeq_ps(vec,o.vec));}
        Float4 operator&(const Float4 &o) const {return Float4(_mm_and_ps(vec,o.vec));}
        Float4 operator|(const Float4 &o) const {return Float4(_mm_or_ps(vec,o.vec));}
        Float4 rsqrt() const { 
            // one round of newton-raphson, see online documentation for _mm_rsqrt_ps for details
            Float4 a = _mm_rsqrt_ps(vec);   // 12-bit approximation
            return Float4(1.5f)*a - (Float4(0.5f)*(*this)) * a * (a*a);
        }
        Float4 sqrt() const {return (*this) * this->rsqrt();}  // faster than _mm_sqrt_ps

        Float4 operator+=(const Float4 &o) {return vec = _mm_add_ps(vec, o.vec);}
        Float4 operator-=(const Float4 &o) {return vec = _mm_sub_ps(vec, o.vec);}
        Float4 operator*=(const Float4 &o) {return vec = _mm_mul_ps(vec, o.vec);}

        int movemask() {return _mm_movemask_ps(vec);}
        bool none() {__m128i v = _mm_castps_si128(vec); return  _mm_testz_si128(v,v);}
        bool any()  {return !none();}

        const Float4 right_rotate() const { return Float4(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE(2,1,0,3))); }
        const Float4 left_rotate()  const { return Float4(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE(0,3,2,1))); }

        void store(float* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm_store_ps(vec_, vec); 
            else 
                _mm_storeu_ps(vec_,vec);
        }

        float sum() const {
            __m128 vec2 = _mm_hadd_ps(vec,vec);
            vec2 = _mm_hadd_ps(vec2,vec2);
            float ret;
            _mm_store_ss(&ret, vec2);
            return ret;
        }

        // choose from values whenever the equivalent element of mask is true
        Float4 blendv(Float4 values, Float4 mask) const {
            return Float4(_mm_blendv_ps(vec, values.vec, mask.vec));
        }

        template <int round_mode = _MM_FROUND_TO_NEAREST_INT>
        Float4 round() const {return Float4(_mm_round_ps(vec, round_mode));}

        Int4 truncate_to_int() const {
            return Int4(_mm_cvttps_epi32(vec));
        }

        template <int m3, int m2, int m1, int m0>
        Float4 zero_entries() const
        {
            // requires SSE 4.1
            constexpr int mask = (m3<<0) + (m2<<1) + (m1<<2) + (m0<<3);
            return _mm_insert_ps(vec,vec, mask);
        }

        float x() const { float val; _MM_EXTRACT_FLOAT(val, vec, 0); return val;}
        float y() const { float val; _MM_EXTRACT_FLOAT(val, vec, 1); return val;}
        float z() const { float val; _MM_EXTRACT_FLOAT(val, vec, 2); return val;}
        float w() const { float val; _MM_EXTRACT_FLOAT(val, vec, 3); return val;}

        friend Int4;
        friend void transpose4(Float4&, Float4&, Float4&, Float4&);
        template <int i3, int i2, int i1, int i0> friend Float4 shuffle_ps(Float4 m1, Float4 m2);

        Int4 cast_int() const {
            // bit-equivalent cast to int
            return Int4(_mm_castps_si128(vec));
        } 

        friend inline Float4 fmadd(const Float4& a1, const Float4& a2, const Float4& b);
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


template <int i3, int i2, int i1, int i0>
Float4 shuffle_ps(Float4 m1, Float4 m2)
{
    return Float4(_mm_shuffle_ps(m1.vec,m2.vec, _MM_SHUFFLE(i3,i2,i1,i0)));
}


// shuffle using a left-to-right convention for value ordering instead of 
// the right-to-left ordering of the SSE API
template <int i0, int i1, int i2, int i3>
inline Float4 sane_shuffle_ps(Float4 m1, Float4 m2)
{
    return shuffle_ps<3-i1, 3-i0, 3-i3, 3-i2>(m2,m1);
}


inline void transpose4(Float4 &x, Float4 &y, Float4 &z, Float4 &w)
{
    _MM_TRANSPOSE4_PS(x.vec,y.vec,z.vec,w.vec);
}

inline Float4 fmadd(const Float4& a1, const Float4& a2, const Float4& b) {
    return Float4(_mm_fmadd_ps(a1.vec,a2.vec, b.vec));
}

// inline int left_pack_simd(Float4 x, Float4 mask);

#endif
