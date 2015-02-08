#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include <cmath>

//! Two dimensional vector (x,y)
struct float2 {
    float x,y;
    float2(): x(0.f),y(0.f) {};
    float2(float x_, float y_):
        x(x_), y(y_){}
};
//! Three dimensional vector (x,y,z)
struct float3 {
    float x,y,z;
    float3(): x(0.f),y(0.f),z(0.f) {};
    float3(float x_, float y_, float z_):
        x(x_), y(y_), z(z_) {}
};
//! Four dimensional vector (x,y,z,w)
struct float4 {
    float x,y,z,w;
    float4(): x(0.f),y(0.f),z(0.f),w(0.f) {};
    float4(float x_, float y_, float z_, float w_):
        x(x_), y(y_), z(z_), w(w_) {}
};

//! Get component of float3 by index

//! component(v,0) == v.x, component(v,1) == v.y, component(v,2) == v.z,
//! any other value for dim is an error
inline float& component(float3 &v, int dim) {
    switch(dim) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
    }
    throw -1;  // abort and die
    return v.x;  
}

//! Get component of float4 by index

//! component(v,0) == v.x, component(v,1) == v.y, component(v,2) == v.z, component(v,3) == v.w,
//! any other value for dim is an error
inline float& component(float4 &v, int dim) {
    switch(dim) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
    }
    throw -1;  // abort and die
    return v.x;  
}


static const float M_PI_F   = 3.141592653589793f;   //!< value of pi as float
static const float M_1_PI_F = 0.3183098861837907f;  //!< value of 1/pi as float

inline float rsqrtf(float x) {return 1.f/sqrtf(x);}  //!< reciprocal square root (1/sqrt(x))
inline float sqr(float x) {return x*x;}  //!< square a number (x^2)


inline float2 make_float2(float x, float y                  ) {return float2(x,y);    } //!< make float2 from scalars
inline float3 make_float3(float x, float y, float z         ) {return float3(x,y,z);  } //!< make float3 from scalars
inline float4 make_float4(float x, float y, float z, float w) {return float4(x,y,z,w);} //!< make float4 from scalars
inline float4 make_float4(float3 v, float w) { return make_float4(v.x,v.y,v.z,w); } //!< make float4 from float3 (as x,y,z) and scalar (as w)

inline float3 xyz(const float4& x) { return make_float3(x.x,x.y,x.z); } //!< return x,y,z as float3 from a float4

//! \cond
inline float2 operator*(float a, const float2 &b) { return make_float2(a*b.x, a*b.y); }
inline float2 operator+(float a, const float2 &b) { return make_float2(a+b.x, a+b.y); }
inline float2 operator-(float a, const float2 &b) { return make_float2(a-b.x, a-b.y); }
inline float2 operator/(float a, const float2 &b) { return make_float2(a/b.x, a/b.y); }

inline float2 operator*(const float2 &a, const float2 &b) { return make_float2(a.x*b.x, a.y*b.y); }
inline float2 operator+(const float2 &a, const float2 &b) { return make_float2(a.x+b.x, a.y+b.y); }
inline float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x-b.x, a.y-b.y); }
inline float2 operator/(const float2 &a, const float2 &b) { return make_float2(a.x/b.x, a.y/b.y); }

inline float2 operator*(const float2 &a, const float  &b) { return make_float2(a.x*b, a.y*b); }
inline float2 operator+(const float2 &a, const float  &b) { return make_float2(a.x+b, a.y+b); }
inline float2 operator-(const float2 &a, const float  &b) { return make_float2(a.x-b, a.y-b); }
inline float2 operator/(const float2 &a, const float  &b) { return make_float2(a.x/b, a.y/b); }

inline float2 operator*=(      float2 &a, const float2 &b) { a.x*=b.x; a.y*=b.y; return a;}
inline float2 operator+=(      float2 &a, const float2 &b) { a.x+=b.x; a.y+=b.y; return a;}
inline float2 operator-=(      float2 &a, const float2 &b) { a.x-=b.x; a.y-=b.y; return a;}
inline float2 operator/=(      float2 &a, const float2 &b) { a.x/=b.x; a.y/=b.y; return a;}

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
//! \endcond

inline float inv_mag(float3 a){return rsqrtf(a.x*a.x + a.y*a.y + a.z*a.z);} //!< 1/sqrt(|v|), where |v| is the magnitude of v

//! cross-product of the vectors a and b
inline float3 cross(float3 a, float3 b){
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x);
}

inline float mag(float3 a){return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);}  //!< vector magnitude

inline float mag2(float3 a){return a.x*a.x + a.y*a.y + a.z*a.z;} //!< vector magnitude squared (faster than mag)
inline float mag2(float4 a){return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w;} //!< vector magnitude squared (faster than mag)

inline float inv_mag2(float3 a){float m=inv_mag(a); return m*m;} //!< reciprocal of vector magnitude squared (1/|v|^2)

inline float3 normalize3(float3 a){return a*inv_mag(a);} //!< unit vector (v/|v|)

inline float dot(float3 a, float3 b){return a.x*b.x + a.y*b.y + a.z*b.z;} //!< dot product of vectors

//! sigmoid function and its derivative 

//! Value of function is 1/(1+exp(x)) and the derivative is 
//! exp(x)/(1+exp(x))^2
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


// Sigmoid-like function that has zero derivative outside (-1/sharpness,1/sharpness)
// This function is 1 for large negative values and 0 for large positive values
inline float2 compact_sigmoid(float x, float sharpness) {
    float y = x*sharpness;
    if     (y> 1.f) return make_float2(0.f, 0.f);
    else if(y<-1.f) return make_float2(1.f, 0.f);
    else            return make_float2(0.25f*(y+2.f)*(y-1.f)*(y-1.f), (sharpness*0.75f)*(y*y-1.f));
}


//! Sigmoid-like function that has zero derivative outside the two intervals (-half_width-1/sharpness,-half_width+1/sharpness)
//! and (half_width-1/sharpness,half_width+1/sharpness).  The function is the product of opposing half-sigmoids.
inline float2 compact_double_sigmoid(float x, float half_width, float sharpness) {
    float2 v1 = compact_sigmoid( x-half_width, sharpness);
    float2 v2 = compact_sigmoid(-x-half_width, sharpness);
    return make_float2(v1.x*v2.x, v1.y*v2.x-v1.x*v2.y);
}


//! compact_double_sigmoid that also handles periodicity
//! note that both theta and center must be in the range (-PI,PI)
inline float2 angular_compact_double_sigmoid(float theta, float center, float half_width, float sharpness) {
    float dev = theta - center;
    if(dev <-M_PI_F) dev += 2.f*M_PI_F;
    if(dev > M_PI_F) dev -= 2.f*M_PI_F;
    return compact_double_sigmoid(dev, half_width, sharpness);
}

//! order is value, then dvalue/dphi, dvalue/dpsi
inline float3 rama_box(float2 rama, float2 center, float2 half_width, float sharpness) {
    float2 phi = angular_compact_double_sigmoid(rama.x, center.x, half_width.x, sharpness);
    float2 psi = angular_compact_double_sigmoid(rama.y, center.y, half_width.y, sharpness);
    return make_float3(phi.x*psi.x, phi.y*psi.x, phi.x*psi.y);
}
    


//! Compute a dihedral angle and its derivative from four positions

//! The angle is always in the range [-pi,pi].  If the arguments are NaN or Inf, the result is undefined.
//! The arguments d1,d2,d3,d4 are output values, and d1 corresponds to the derivative of the dihedral angle
//! with respect to r1.
static float dihedral_germ(
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

    return atan2f(dot(C,G), dot(A,B) * Gmag);
}
#endif
