#ifndef MD_H
#define MD_H

#include <cstdint>
#include <initializer_list>

#ifdef __OPENCL_VERSION__
// To make this header useful for both host and device code, I must translate
// the typenames on the device
typedef uchar  cl_uchar;
typedef ushort cl_ushort;
typedef short  cl_short;
typedef float  cl_float;
typedef int    cl_int;
#else
#include <cstdint>
typedef uint8_t  cl_uchar;
typedef uint16_t cl_ushort;
typedef int16_t  cl_short;
typedef float    cl_float;
typedef int32_t  cl_int;
#endif

struct DerivRecord {
    unsigned short atom, loc, output_width, unused;
    DerivRecord(unsigned short atom_, unsigned short loc_, unsigned short output_width_):
        atom(atom_), loc(loc_), output_width(output_width_) {}
};

struct CoordPair {
    unsigned short index, slot;
    CoordPair(unsigned short index_, unsigned short slot_):
        index(index_), slot(slot_) {}
    CoordPair(): index(-1), slot(-1) {}
};

#define HMM_ALIGNMENT 32

struct AutoDiffParams {
    unsigned char  n_slots1, n_slots2;
    unsigned short slots1[6];      
    unsigned short slots2[5];        

    AutoDiffParams(
            const std::initializer_list<unsigned short> &slots1_,
            const std::initializer_list<unsigned short> &slots2_):
        n_slots1(slots1_.size()), n_slots2(slots2_.size()) 
    {
        unsigned loc1=0;
        for(auto i: slots1_) slots1[loc1++] = i;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        for(auto i: slots2_) slots1[loc2++] = i;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

    explicit AutoDiffParams(const std::initializer_list<unsigned short> &slots1_):
        n_slots1(slots1_.size()), n_slots2(0u)
    { 
        unsigned loc1=0;
        for(auto i: slots1_) slots1[loc1++] = i;
        while(loc1<sizeof(slots1)/sizeof(slots1[0])) slots1[loc1++] = -1;

        unsigned loc2=0;
        // for(auto i: slots2_) slots1[loc2++] = i;
        while(loc2<sizeof(slots2)/sizeof(slots2[0])) slots2[loc2++] = -1;
    }

} ;  // struct for implementing reverse autodifferentiation

typedef struct {
    CoordPair atom[3];
    float     ref_geom[9];
} AffineAlignmentParams;

typedef struct {
    CoordPair residue;
} AffineParams;

// struct is 4 words
typedef struct {
    CoordPair atom[2];
    cl_float equil_dist;
    cl_float spring_constant;
    cl_int padding[4];
} DistSpringParams;

struct ZFlatBottomParams {
    CoordPair atom;
    float     z0;
    float     radius;
    float     spring_constant;
};

// struct is 5 words, padded to 8 words
typedef struct {
    CoordPair atom[1];
    cl_float x,y,z;
    cl_float spring_constant;
    cl_int padding[3];
} PosSpringParams;

struct VirtualParams {
    CoordPair atom[3];
    float bond_length;
};


struct VirtualHBondParams {
    CoordPair id;
    cl_ushort residue_id;
    cl_float  helix_energy_bonus;
};


// struct is 5 words, padded to 8 words
typedef struct {
    CoordPair atom[3];
    cl_float equil_dp;
    cl_float spring_constant;
    cl_int padding[3];
} AngleSpringParams;


// struct is 6 words, padded to 8 words
typedef struct {
    CoordPair atom[4];
    cl_float equil_dihedral;
    cl_float spring_constant;
    cl_int padding[2];
} DihedralSpringParams;


// struct is 5 words, padded to 8 words
typedef struct {
    CoordPair atom[5];
    cl_int padding[3];
} HMMParams;


// struct is 12 words
typedef struct {
    cl_ushort n_atom_in_group;
    cl_short  group_type;  // not really necesary, but helps with alignment
    cl_float  inv_n_atom_in_group; 
    cl_ushort group_inds[ 5];
    cl_ushort slot      [15];  // 3 slots for each id since vector output
} GroupCOMParams;

// struct is 14 words, padded to 16 words
typedef struct {
    float H_bond_length, N_bond_length;
    cl_ushort Cprev, N, CA, C, Nnext;
    cl_ushort H_slots[9];
    cl_ushort O_slots[9];
    cl_ushort padding[5];
} NH_CO_Params;

#endif
