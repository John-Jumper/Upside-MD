#ifndef COORD_H
#define COORD_H

#include "vector_math.h"

inline void copy_vec_array_to_buffer(VecArray arr, int n_elem, int n_dim, float* buffer) {
        for(int i=0; i<n_elem; ++i)
            for(int d=0; d<n_dim; ++d) 
                buffer[i*n_dim+d] = arr(d,i);
}


//! Pair of value and derivative VecArray's 
struct CoordArray {
    VecArray value;  //!< Array containing value elements
    VecArray deriv;  //!< Array containing derivative slots

    //! Initialize from value and derivative VecArray's
    CoordArray(VecArray value_, VecArray deriv_):
        value(value_), deriv(deriv_) {}
};

typedef unsigned int index_t;  //!< type of coordinate indices
typedef unsigned int slot_t;   //!< type of derivative slot indices

//! Pair of coordinate index and slot index for its derivative

//! When the value of the coordinate given by index is used, 
//! a derivative should be recorded at the slot given by slot.
//! The class Coord should be used to read the value associated to 
//! index and to write the derivative associated to slot.
struct CoordPair {
    index_t index;  //!< Index into the value array to find the coordinate
    slot_t slot;    //!< Index into the derivative array to find the slot

    //! Initialize from existing data
    CoordPair(index_t index_, slot_t slot_):
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
        int i_slot;
        mutable VecArray deriv_arr;

    public:
        Coord(): i_slot(0), deriv_arr(nullptr,0) {};

        //! Initialize by specifying a coordinate array and the location within the coordinate array
        Coord(CoordArray arr, CoordPair c):
            i_slot(c.slot), deriv_arr(arr.deriv)
        {
            for(int nd=0; nd<N_DIM; ++nd) 
                v[nd] = arr.value(nd,c.index);

            for(int no=0; no<N_DIM_OUTPUT; ++no) 
                for(int nd=0; nd<N_DIM; ++nd) 
                    d[no][nd] = 0.f;
        }

        //! Extract first 3 dimensions of the value as a float3
        float3 f3() const {
            return make_vec3(v[0], v[1], v[2]);
        }

        //! Extract first 3 dimensions of the idx'th component of the derivative as a float3
        float3 df3(int idx) const {
            return make_vec3(d[idx][0], d[idx][1], d[idx][2]);
        }

        //! Set the derivative for the 0'th component using a float3
        template <int D>
        void set_deriv(Vec<D,float> d_) {
            #pragma unroll
            for(int j=0; j<D; ++j) d[0][j] = d_[j];
        }

        //! Set the derivative for the i'th component using a float3
        template <int D>
        void set_deriv(int i, Vec<D,float> d_) {
            #pragma unroll
            for(int j=0; j<D; ++j) d[i][j] = d_[j];
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
                            deriv_arr(nd,i_slot+no)  = d[no][nd];
                    break;

                case SumWritePolicy:
                    for(int no=0; no<N_DIM_OUTPUT; ++no) 
                        for(int nd=0; nd<N_DIM; ++nd) 
                            deriv_arr(nd,i_slot+no) += d[no][nd];
                    break;
            }
        }
};

#endif
