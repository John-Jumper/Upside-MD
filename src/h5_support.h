#ifndef H5_SUPPORT_H
#define H5_SUPPORT_H

#include <string>
#include <vector>
#include <array>
#include <functional>
#include <memory>

#include <hdf5.h>

//! NB this library is not threadsafe.  It is up to the user to deal with this fact
namespace h5 {

//! \cond
template <typename T> // DO NOT ADD BOOL TO THIS TYPE -- unwitting users might hit vector<bool>
inline hid_t select_predtype () { return T::NO_SPECIALIZATION_AVAILABLE; }
template<> inline hid_t select_predtype<float> (){ return H5T_NATIVE_FLOAT;  }
template<> inline hid_t select_predtype<double>(){ return H5T_NATIVE_DOUBLE; }
template<> inline hid_t select_predtype<int>   (){ return H5T_NATIVE_INT;    }
template<> inline hid_t select_predtype<long>  (){ return H5T_NATIVE_LONG;    }
template<> inline hid_t select_predtype<unsigned>(){ return H5T_NATIVE_UINT;    }
//! \endcond

hid_t h5_noerr(hid_t i); //!< if i<0 (signalling failed H5 function), throw an error
hid_t h5_noerr(const char* nm, hid_t i); //!< if i<0 (signalling failed H5 function), throw an error

inline bool h5_bool_return(htri_t value) {
    if(value<0) throw std::string("hdf5 error");
    return value>0;  // true condition for htri_t
}

//! Wrapper to make hid_t compatible with smart pointers
struct Hid_t {   // special type for saner hid_t
    hid_t ref;
    Hid_t(std::nullptr_t = nullptr): ref(-1) {}
    Hid_t(hid_t ref_): ref(ref_) {}
    operator hid_t() const {return ref;}
    explicit operator bool() const {return ref>=0;}
    bool operator==(const Hid_t &o) const {return o.ref == ref;}
    bool operator!=(const Hid_t &o) const {return o.ref != ref;}
};


//! Type of a function to delete a hid_t reference
typedef herr_t H5DeleterFunc(hid_t);
//! Custom deleter for hid_t references
struct H5Deleter
{
    typedef Hid_t  pointer;
    H5DeleterFunc *deleter;

    H5Deleter(): deleter(nullptr) {}
    H5Deleter(H5DeleterFunc *deleter_): deleter(deleter_) {}
    void operator()(pointer p) {if(deleter) (*deleter)(p);} // no error check since destructors can't throw
};

//! Wrapper for hid_t reference with custom deleter
typedef std::unique_ptr<Hid_t,H5Deleter> H5Obj;

//! Wrap a raw hid_t reference so that the object is released when it is not needed

//! The argument deleter is most commonly a function 
//! such as H5Gclose or h5Dclose
H5Obj h5_obj(H5DeleterFunc &deleter, hid_t obj);

//! Duplicate an H5Obj by increasing the reference count of the underlying object
inline H5Obj duplicate_obj(H5Obj& obj) {
    H5Iinc_ref(obj.get());
    return H5Obj(obj.get(), obj.get_deleter());
}

static H5Obj open_file(char* path, decltype(H5F_ACC_RDONLY) flags) {
    if(flags == H5F_ACC_RDONLY || flags == H5F_ACC_RDWR)
        return h5_obj(H5Fclose, H5Fopen(path, flags, H5P_DEFAULT));
    else 
        return h5_obj(H5Fclose, H5Fcreate(path, flags, H5P_DEFAULT, H5P_DEFAULT));

}

//! Check that an HDF object exists at a given path

//! If check_valid is true, then the function will also 
//! ensure that the path is not a dangling link.
bool h5_exists(hid_t base, const char* nm);

// Read the dimension sizes of a dataset
std::vector<hsize_t> get_dset_size(int ndims, hid_t group, const char* name);

// Attempt to read a scalar attribute.  Returns true if the attribute is present, false otherwise.
// See below for more typesafe overloads.  attr_value_output is overwritten with the value of the 
// attribute if it is present, otherwise it is unmodified.  If the underlying path does not exist,
// an exception is thrown.
bool read_attribute(void* attr_value_output, hid_t h5, const char* path, const char* attr_name, hid_t predtype);


//! Read a scalar attribute with exception if not present
template<class T>
T read_attribute(hid_t h5, const char* path, const char* attr_name) {
    T value;
    if(!read_attribute(&value, h5, path, attr_name, select_predtype<T>()))
        throw "attribute "+ std::string(attr_name) + " not present";
    return value;
}

//! Read a scalar attribute with default value
template<class T>
T read_attribute(hid_t h5, const char* path, const char* attr_name, const T& default_value) {
    T value;
    if(!read_attribute(&value, h5, path, attr_name, select_predtype<T>()))
        value = default_value;
    return value;
}

//! Read an attribute containing a list of strings
template<>
std::vector<std::string> read_attribute<std::vector<std::string>>
(hid_t h5, const char* path, const char* attr_name);

void write_string_attribute(
        hid_t h5, const char* path, const char* attr_name,
        const std::string& value);

void check_size(hid_t group, const char* name, std::vector<size_t> sz); //!< Check the dimension sizes of an arbitrary dataset
void check_size(hid_t group, const char* name, size_t sz); //!< Check the dimension sizes of an 1D dataset
void check_size(hid_t group, const char* name, size_t sz1, size_t sz2); //!< Check the dimension sizes of an 2D dataset
void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3); //!< Check the dimension sizes of an 3D dataset
void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4); //!< Check the dimension sizes of an 4D dataset
void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5); //!< Check the dimension sizes of an 5D dataset

H5Obj ensure_group(hid_t loc, const char* nm);    //!< Ensure that a group of a specific name exists
H5Obj open_group(hid_t loc, const char* nm);      //!< Open an existing group
void ensure_not_exist(hid_t loc, const char* nm); //!< Delete a group if it exists


//! Create a new dataset
H5Obj create_earray(hid_t group, const char* name, hid_t dtype,
        const std::initializer_list<int> &dims, // any direction that is extendable must have dims == 0
        const std::initializer_list<int> &chunk_dims,
        bool compression_level=1);  // 1 is often recommended

H5Obj create_earray(hid_t group, const char* name, hid_t dtype,
        const std::vector<hsize_t>& dims_v, // any direction that is extendable must have dims == 0
        const std::vector<hsize_t>& chunk_dims_v,
        bool compression_level=1);  // 1 is often recommended

//! Append a raw data buffer to a dataset
void append_to_dset(hid_t dset, hid_t hdf_predtype, size_t n_new_data_elems, const void* new_data, int append_dim);


//! Append vector of data to a dataset
template <typename T>
void append_to_dset(hid_t dset, const std::vector<T> &new_data, int append_dim) {
    append_to_dset(dset, select_predtype<T>(), new_data.size(), (const void*)(new_data.data()), append_dim);
}


//! Read list of names within a group
std::vector<std::string> 
node_names_in_group(const hid_t loc, const std::string grp_name);


//! \cond
template <int ndim, typename T, typename F>
struct traverse_dataset_iteraction_helper { 
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {T::NO_SPECIALIZATION_AVAILABLE();}
};

template <typename T, typename F>
struct traverse_dataset_iteraction_helper<1,T,F> {
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {
        size_t loc = 0;
        for(size_t i0=0; i0<dims[0]; ++i0)
            f(i0, data[stride*loc++]);
    }
};

template <typename T, typename F>
struct traverse_dataset_iteraction_helper<2,T,F> {
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {
        size_t loc = 0;
        for(size_t i0=0; i0<dims[0]; ++i0)
            for(size_t i1=0; i1<dims[1]; ++i1)
                f(i0,i1, data[stride*loc++]);
    }
};

template <typename T, typename F>
struct traverse_dataset_iteraction_helper<3,T,F> {
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {
        size_t loc = 0;
        for(size_t i0=0; i0<dims[0]; ++i0)
            for(size_t i1=0; i1<dims[1]; ++i1)
                for(size_t i2=0; i2<dims[2]; ++i2)
                    f(i0,i1,i2, data[stride*loc++]);
    }
};

template <typename T, typename F>
struct traverse_dataset_iteraction_helper<4,T,F> {
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {
        size_t loc = 0;
        for(size_t i0=0; i0<dims[0]; ++i0)
            for(size_t i1=0; i1<dims[1]; ++i1)
                for(size_t i2=0; i2<dims[2]; ++i2)
                    for(size_t i3=0; i3<dims[3]; ++i3)
                        f(i0,i1,i2,i3, data[stride*loc++]);
    }
};


template <typename T, typename F>
struct traverse_dataset_iteraction_helper<5,T,F> {
    void operator()(T* data, hsize_t* dims, const F& f, size_t stride=1) {
        size_t loc = 0;
        for(size_t i0=0; i0<dims[0]; ++i0)
            for(size_t i1=0; i1<dims[1]; ++i1)
                for(size_t i2=0; i2<dims[2]; ++i2)
                    for(size_t i3=0; i3<dims[3]; ++i3)
                        for(size_t i4=0; i4<dims[4]; ++i4)
                            f(i0,i1,i2,i3,i4, data[stride*loc++]);
    }
};
//! \endcond


//! Execute a function f for each element of a dataset
template<int ndims, typename T, typename F>
void traverse_dset(hid_t group, const char* name, const F& f)
try {
    auto dset  = h5_obj(H5Dclose, H5Dopen2(group, name, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Dget_space(dset.get()));

    if(H5Sget_simple_extent_ndims(space.get()) != ndims) 
        throw std::string("wrong shape for array, expected " +
            std::to_string(ndims) + " dimension(s) but got " +
            std::to_string(H5Sget_simple_extent_ndims(space.get())) +
            " dimension(s).");
    hsize_t dims[ndims]; h5_noerr(H5Sget_simple_extent_dims(space.get(), dims, NULL));

    size_t dim_product = 1;
    for(int i=0; i<ndims; ++i) dim_product *= dims[i];

    auto tmp = std::unique_ptr<T[]>(new T[dim_product]);

    h5_noerr(H5Dread(dset.get(), select_predtype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.get()));
    traverse_dataset_iteraction_helper<ndims,T,F>()(tmp.get(), dims, f);
} catch(const std::string &e) {
    throw "while traversing '" + std::string(name) + "', " + e;
}


//! Execute a function f for each string in a dataset
template<int ndims, typename F>
void traverse_string_dset(
        hid_t group, const char* name, const F& f)
try {
    auto dset  = h5_obj(H5Dclose, H5Dopen2(group, name, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Dget_space(dset.get()));
    auto dtype = h5_obj(H5Tclose, H5Dget_type(dset.get()));
    if(H5Tis_variable_str(dtype.get())) throw std::string("variable-length strings not supported");

    size_t maxchars = H5Tget_size(dtype.get());
    if(maxchars==0) throw std::string("H5Tget_size error"); // defined error value

    if(H5Sget_simple_extent_ndims(space.get()) != ndims) throw std::string("wrong size for id array");
    hsize_t dims[ndims]; h5_noerr(H5Sget_simple_extent_dims(space.get(), dims, NULL));

    size_t dim_product = 1;
    for(int i=0; i<ndims; ++i) dim_product *= dims[i];

    // the extra 1+ accounts for the space to hold the NULL-terminator
    auto tmp = std::unique_ptr<char[]>(new char[dim_product*(1u+maxchars)]);
    h5_noerr(H5Dread(dset.get(), dtype.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.get()));

    // I must wrap the traversal function to provide the requested std::string
    // FIXME this only works for 1 dimension
    auto g = [&](int i, char& s) {
        std::string tmp(maxchars,'\0');
        std::copy(&s, &s+maxchars, begin(tmp));
        while(tmp.size() && tmp.back() == '\0') tmp.pop_back();
        f(i,tmp);  // I need to forward an unknown number of arguments here, in general
    };
    traverse_dataset_iteraction_helper<ndims,char,decltype(g)>()(tmp.get(), dims, g, maxchars);
} catch(const std::string &e) {
    throw "while traversing '" + std::string(name) + "', " + e;
}

}
#endif
