#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include <cmath>

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
    throw -1;  // abort and die
    return v.x;  
}


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
#endif
