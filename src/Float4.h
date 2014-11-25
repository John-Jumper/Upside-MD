#ifndef __FLOAT4_H__
#define __FLOAT4_H__

// Author: John Jumper

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
    

enum class Alignment {unaligned, aligned};

struct Vec34;

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
        Float4(float val):   
            vec(_mm_set1_ps(val)) {}

        Float4 operator+(const Float4 &o) const {return Float4(_mm_add_ps  (vec, o.vec));}
        Float4 operator-(const Float4 &o) const {return Float4(_mm_sub_ps  (vec, o.vec));}
        Float4 operator-()                const {return _mm_sub_ps(_mm_set1_ps(0.f), vec);}
        Float4 operator*(const Float4 &o) const {return Float4(_mm_mul_ps  (vec, o.vec));}
        Float4 operator<(const Float4 &o) const {return Float4(_mm_cmplt_ps(vec,o.vec));}
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
        bool any() {return movemask();}

        void right_rotate() { vec = _mm_shuffle_ps(vec,vec, _MM_SHUFFLE(2,1,0,3)); }
        void left_rotate()  { vec = _mm_shuffle_ps(vec,vec, _MM_SHUFFLE(0,3,2,1)); }

        void store(float* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm_store_ps(vec_, vec); 
            else 
                _mm_storeu_ps(vec_,vec);
        }

        void store_int(int* vec_, Alignment align=Alignment::aligned) const { 
            if(align==Alignment::aligned) 
                _mm_store_si128((__m128i*)vec_, _mm_cvtps_epi32(vec)); 
            else 
                _mm_storeu_si128((__m128i*)vec_,_mm_cvtps_epi32(vec));
        }

        float sum() const {
            __m128 vec2 = _mm_hadd_ps(vec,vec);
            vec2 = _mm_hadd_ps(vec2,vec2);
            float ret;
            _mm_store_ss(&ret, vec2);
            return ret;
        }

        // choose from values whenever the equivalent element of mask is true
        Float4 blendv(Float4 values, Float4 mask) {
            return Float4(_mm_blendv_ps(vec, values.vec, mask.vec));
        }

        template <int round_mode = _MM_FROUND_TO_NEAREST_INT>
        Float4 round() const {return Float4(_mm_round_ps(vec, round_mode));}

        template <int m3, int m2, int m1, int m0>
        Float4 zero_entries() const
        {
            // requires SSE 4.1
            constexpr int mask = (m3<<0) + (m2<<1) + (m1<<2) + (m0<<3);
            return _mm_insert_ps(vec,vec, mask);
        }

        // convenience function, not at all efficient
        float w() const { float val; _MM_EXTRACT_FLOAT(val, vec, 0); return val;}
        float x() const { float val; _MM_EXTRACT_FLOAT(val, vec, 1); return val;}
        float y() const { float val; _MM_EXTRACT_FLOAT(val, vec, 2); return val;}
        float z() const { float val; _MM_EXTRACT_FLOAT(val, vec, 3); return val;}

        friend void transpose4(Float4&, Float4&, Float4&, Float4&);
        template <int i3, int i2, int i1, int i0> friend Float4 shuffle_ps(Float4 m1, Float4 m2);
        friend Vec34;
};

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
            vec(_mm256_set1_ps(val)) {}

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

        // void right_rotate() { vec = _mm256_shuffle_ps(vec,vec, _MM_SHUFFLE(2,1,0,3)); }
        // void left_rotate()  { vec = _mm256_shuffle_ps(vec,vec, _MM_SHUFFLE(0,3,2,1)); }

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


inline void transpose4(Float4 &w, Float4 &x, Float4 &y, Float4 &z)
{
    _MM_TRANSPOSE4_PS(w.vec,x.vec,y.vec,z.vec);
}

struct alignas(16) Vec34
{
    public:
        Float4 x;
        Float4 y;
        Float4 z;

        Vec34() {};

        // construct from 3 memory locations
        Vec34(const float* x, const float* y, const float* z, Alignment align = Alignment::aligned):
            x(x,align),
            y(y,align),
            z(z,align) {}

        // construct from 3 memory locations
        Vec34(const Float4 &x, const Float4 &y, const Float4 &z):
            x(x),
            y(y),
            z(z) {}

        // construct from 3 values
        Vec34(const float x, const float y, const float z):
            x(x),
            y(y),
            z(z) {}

        // construct from 1 memory location and a stride
        Vec34(const float* pos, int stride, Alignment align = Alignment::aligned):
            x(pos + 0*stride, align),
            y(pos + 1*stride, align),
            z(pos + 2*stride, align) {}

        Float4 mag2() const { return (x*x + y*y) + z*z; }
        Float4 mag () const { return mag2().sqrt(); }

        Vec34 operator+(const Vec34  &o) const {return Vec34(x+o.x, y+o.y, z+o.z);}
        Vec34 operator-(const Vec34  &o) const {return Vec34(x-o.x, y-o.y, z-o.z);}
        Vec34 operator-()                const {return Vec34(-x, -y, -z);}
        Vec34 operator*(const Float4 &o) const {return Vec34(x*o,   y*o,   z*o);}
        Float4 operator==(const Vec34 &o) const {return (x==o.x) & (y==o.y) & (z==o.z);}
        Float4 operator!=(const Vec34 &o) const {return (x!=o.x) | (y!=o.y) | (z!=o.z);}
        Vec34 operator+=(const Vec34  &o) { x+=o.x; y+=o.y; z+=o.z; return *this; }

        void left_rotate() {
            x.left_rotate();
            y.left_rotate();
            z.left_rotate();
        }

        void right_rotate() {
            x.right_rotate();
            y.right_rotate();
            z.right_rotate();
        }

        void store(float* x, float* y, float* z) {
            this->x.store(x);
            this->y.store(y);
            this->z.store(z);
        }

        // construct from 1 memory location and a stride
        void store(float* pos, int stride) {
            x.store(pos + 0*stride);
            y.store(pos + 1*stride);
            z.store(pos + 2*stride);
        }

        Float4 sum() const {
            __m128 xy = _mm_hadd_ps(x.vec, y.vec);            // (x[0]+x[1],     x[2]+x[3],     y[0]+y[1],     y[2]+y[3])
            __m128 zs = _mm_hadd_ps(z.vec, _mm_setzero_ps()); // (z[0]+z[1],     z[2]+z[3],     0.,            0.)
            return _mm_hadd_ps(xy, zs);                       // (x[0]+...+x[3], y[0]+...+y[3], z[0]+...+z[3], 0.) 
        }

};

inline Vec34 operator*(const Float4 &a, const Vec34 &b) { return b*a; }


inline Vec34 cross(const Vec34&a, const Vec34 &b) {
    return Vec34(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
}


struct alignas(32) Vec38
{
    public:
        Float8 x;
        Float8 y;
        Float8 z;

        // construct from 3 memory locations
        Vec38(const float* x, const float* y, const float* z, Alignment align = Alignment::aligned):
            x(x,align),
            y(y,align),
            z(z,align) {}

        // construct from 3 memory locations
        Vec38(const Float8 &x, const Float8 &y, const Float8 &z):
            x(x),
            y(y),
            z(z) {}

        // construct from 3 values
        Vec38(const float x, const float y, const float z):
            x(x),
            y(y),
            z(z) {}

        // construct from 1 memory location and a stride
        Vec38(const float* pos, int stride, Alignment align = Alignment::aligned):
            x(pos + 0*stride, align),
            y(pos + 1*stride, align),
            z(pos + 2*stride, align) {}

        Float8 mag2() const
        {
            return (x*x + y*y) + z*z;
        }

        Vec38 operator-(const Vec38  &o) const {return Vec38(x-o.x, y-o.y, z-o.z);}
        Vec38 operator*(const Float8 &o) const {return Vec38(x*o,   y*o,   z*o);}
        Float8 operator==(const Vec38 &o) const {return (x==o.x) & (y==o.y) & (z==o.z);}
        Float8 operator!=(const Vec38 &o) const {return (x!=o.x) | (y!=o.y) | (z!=o.z);}

        // void left_rotate() {
        //     x.left_rotate();
        //     y.left_rotate();
        //     z.left_rotate();
        // }

        // void right_rotate() {
        //     x.right_rotate();
        //     y.right_rotate();
        //     z.right_rotate();
        // }

        void store(float* x, float* y, float* z) {
            this->x.store(x);
            this->y.store(y);
            this->z.store(z);
        }

        // construct from 1 memory location and a stride
        void store(float* pos, int stride) {
            x.store(pos + 0*stride);
            y.store(pos + 1*stride);
            z.store(pos + 2*stride);
        }

};

#endif
