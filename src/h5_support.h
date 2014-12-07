#ifndef H5_SUPPORT_H
#define H5_SUPPORT_H

#include <string>
#include <vector>
#include <array>
#include <functional>
#include <memory>

#include <hdf5.h>
#include <hdf5_hl.h>

namespace h5 {

template <typename T> // DO NOT ADD BOOL TO THIS TYPE -- unwitting users might hit vector<bool>
inline const hid_t select_predtype () { return T::NO_SPECIALIZATION_AVAILABLE; }
template<> inline const hid_t select_predtype<float> (){ return H5T_NATIVE_FLOAT;  }
template<> inline const hid_t select_predtype<double>(){ return H5T_NATIVE_DOUBLE; }
template<> inline const hid_t select_predtype<int>   (){ return H5T_NATIVE_INT;    }
template<> inline const hid_t select_predtype<long>  (){ return H5T_NATIVE_LONG;    }

inline int h5_noerr(int i) {
    if(i<0) {
        // H5Eprint2(H5E_DEFAULT, stderr);
        throw std::string("error " + std::to_string(i));
    }
    return i;
}

inline int h5_noerr(const char* nm, int i) {
    if(i<0) {
        // H5Eprint2(H5E_DEFAULT, stderr);
        throw std::string("error " + std::to_string(i) + " in " + nm);
    }
    return i;
}


struct Hid_t {   // special type for saner hid_t
    hid_t ref;
    Hid_t(std::nullptr_t = nullptr): ref(-1) {}
    Hid_t(hid_t ref_): ref(ref_) {}
    operator hid_t() const {return ref;}
    explicit operator bool() const {return ref>=0;}
    bool operator==(const Hid_t &o) const {return o.ref == ref;}
    bool operator!=(const Hid_t &o) const {return o.ref != ref;}
};


typedef herr_t H5DeleterFunc(hid_t);
struct H5Deleter
{
    typedef Hid_t  pointer;
    H5DeleterFunc *deleter;

    H5Deleter(): deleter(nullptr) {}
    H5Deleter(H5DeleterFunc *deleter_): deleter(deleter_) {}
    void operator()(pointer p) {if(deleter) (*deleter)(p);} // no error check since destructors can't throw
};

typedef std::unique_ptr<Hid_t,H5Deleter> H5Obj;


inline H5Obj h5_obj(H5DeleterFunc &deleter, hid_t obj)
{
    return H5Obj(h5_noerr(obj), H5Deleter(&deleter));
}


inline bool h5_exists(hid_t base, const char* nm, bool check_valid=true) {
    return h5_noerr(H5LTpath_valid(base, nm, check_valid));
}

template <int ndims>
std::array<hsize_t,ndims> get_dset_size(hid_t group, const char* name) 
try {
    std::array<hsize_t,ndims> ret;
    auto dset  = h5_obj(H5Dclose, H5Dopen2(group, name, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Dget_space(dset.get()));

    if(h5_noerr(H5Sget_simple_extent_ndims(space.get())) != ndims) throw std::string("wrong number of dimensions");
    h5_noerr("H5Sget_simple_extent_dims", H5Sget_simple_extent_dims(space.get(), ret.data(), NULL));
    return ret;
} catch(const std::string &e) {
    throw "while getting size of '" + std::string(name) + "', " + e;
}



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


template<int ndims, typename T, typename F>
void traverse_dset(
        hid_t group, const char* name, const F& f)
try {
    auto dset  = h5_obj(H5Dclose, H5Dopen2(group, name, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Dget_space(dset.get()));

    if(H5Sget_simple_extent_ndims(space.get()) != ndims) throw std::string("wrong size for id array");
    hsize_t dims[ndims]; h5_noerr(H5Sget_simple_extent_dims(space.get(), dims, NULL));

    size_t dim_product = 1;
    for(int i=0; i<ndims; ++i) dim_product *= dims[i];

    auto tmp = std::unique_ptr<T[]>(new T[dim_product]);

    h5_noerr(H5Dread(dset.get(), select_predtype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.get()));
    traverse_dataset_iteraction_helper<ndims,T,F>()(tmp.get(), dims, f);
} catch(const std::string &e) {
    throw "while traversing '" + std::string(name) + "', " + e;
}


// Partial specialization would be better, but that is forbidden for functions.  In the 
// future, this should be wrapped in a class
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


template<class T>
T read_attribute(hid_t h5, const char* path, const char* attr_name) 
try {
    T retval;
    h5_noerr(H5LTget_attribute(h5, path, attr_name, select_predtype<T>(), &retval));
    return retval;
} catch(const std::string &e) {
    throw "while reading attribute '" + std::string(attr_name) + "' of '" + std::string(path) + "', " + e;
}

template<>
std::vector<std::string> read_attribute<std::vector<std::string>>
(hid_t h5, const char* path, const char* attr_name);

template <size_t ndim>
inline void check_size(hid_t group, const char* name, std::array<size_t,ndim> sz)
{
    auto dims = get_dset_size<ndim>(group, name);
    for(size_t d=0; d<ndim; ++d) {
        if(dims[d] != sz[d]) {
            std::string msg = std::string("dimensions of '") + name + std::string("', expected (");
            for(size_t i=0; i<ndim; ++i) msg += std::to_string(sz  [i]) + std::string((i<ndim-1) ? ", " : "");
            msg += ") but got (";
            for(size_t i=0; i<ndim; ++i) msg += std::to_string(dims[i]) + std::string((i<ndim-1) ? ", " : "");
            msg += ")";
            throw msg;
        }
    }
}

inline void check_size(hid_t group, const char* name, size_t sz) 
{ check_size(group, name, std::array<size_t,1>{{sz}}); }

inline void check_size(hid_t group, const char* name, size_t sz1, size_t sz2) 
{ check_size(group, name, std::array<size_t,2>{{sz1,sz2}}); }

inline void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3) 
{ check_size(group, name, std::array<size_t,3>{{sz1,sz2,sz3}}); }

inline void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4) 
{ check_size(group, name, std::array<size_t,4>{{sz1,sz2,sz3,sz4}}); }

inline void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5) 
{ check_size(group, name, std::array<size_t,5>{{sz1,sz2,sz3,sz4,sz5}}); }


inline H5Obj ensure_group(hid_t loc, const char* nm) {
    return h5_obj(H5Gclose, h5_exists(loc, nm) 
            ? H5Gopen2(loc, nm, H5P_DEFAULT)
            : H5Gcreate2(loc, nm, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
}

inline H5Obj open_group(hid_t loc, const char* nm) {
    return h5_obj(H5Gclose, H5Gopen2(loc, nm, H5P_DEFAULT));
}

inline void ensure_not_exist(hid_t loc, const char* nm) {
    if(h5_exists(loc, nm)) H5Ldelete(loc, nm, H5P_DEFAULT);
}


inline H5Obj create_earray(hid_t group, const char* name, hid_t dtype,
        const std::initializer_list<int> &dims, // any direction that is extendable must have dims == 0
        const std::initializer_list<int> &chunk_dims,
        bool compression_level=0)  // 1 is often recommended
{
    if(dims.size() != chunk_dims.size()) throw std::string("invalid chunk dims");

    hsize_t ndims = dims.size();
    std::vector<hsize_t> dims_v(ndims);
    std::vector<hsize_t> max_dims_v(ndims);
    std::vector<hsize_t> chunk_dims_v(ndims);

    for(size_t d=0; d<ndims; ++d) {
        dims_v[d]       = *(begin(dims      )+d);
        max_dims_v[d]   = dims_v[d] ? dims_v[d] : H5S_UNLIMITED;
        chunk_dims_v[d] = *(begin(chunk_dims)+d);
    }

    auto space_id = h5_obj(H5Sclose, H5Screate_simple(ndims, dims_v.data(), max_dims_v.data()));

    // setup chunked, possibly compressed storage
    auto dcpl_id = h5_obj(H5Pclose, H5Pcreate(H5P_DATASET_CREATE));
    h5_noerr(H5Pset_chunk(dcpl_id.get(), ndims, chunk_dims_v.data()));
    h5_noerr(H5Pset_shuffle(dcpl_id.get()));     // improves data compression
    h5_noerr(H5Pset_fletcher32(dcpl_id.get()));  // for verifying data integrity
    if(compression_level) h5_noerr(H5Pset_deflate(dcpl_id.get(), compression_level));

    return h5_obj(H5Dclose, H5Dcreate2(group, name, dtype, space_id.get(), 
                H5P_DEFAULT, dcpl_id.get(), H5P_DEFAULT));
}


// I will check that the vector size is cleanly divided by the product of all other dimensions
template <typename T>
void append_to_dset(hid_t dset, const std::vector<T> &new_data, int append_dim)
try {
    // Load current data size
    auto space = h5_obj(H5Sclose, H5Dget_space(dset));
    int ndims = h5_noerr(H5Sget_simple_extent_ndims(space.get()));
    std::vector<hsize_t> dims(ndims);
    h5_noerr("H5Sget_simple_extent_dims", H5Sget_simple_extent_dims(space.get(), dims.data(), NULL));

    // Compute number of records in new_data
    long int record_size = 1;  // product of all dim sizes except the append_dim
    for(int nd=0; nd<(int)dims.size(); ++nd) if(nd!=append_dim) record_size *= dims[nd];

    long int n_records = new_data.size() / record_size;
    if(n_records*record_size != (long)new_data.size()) 
        throw std::string("new data is not an integer number of records long");

    // Enlarge dataset for new data
    auto enlarged_dims = dims;
    enlarged_dims[append_dim] += n_records;
    h5_noerr("H5Dset_extent", H5Dset_extent(dset, enlarged_dims.data()));
    auto enlarged_space = h5_obj(H5Sclose, H5Dget_space(dset));

    // Dataspaces are required for the source and destination

    // Source is just a simple dataspace
    auto mem_dims = dims;
    mem_dims[append_dim] = n_records;
    auto mem_space = h5_obj(H5Sclose, H5Screate_simple(mem_dims.size(), mem_dims.data(), NULL));

    // Destination is a hyperslab selection starting at the old length in the append direction
    std::vector<hsize_t> starts(ndims); 
    for(int d=0; d<ndims; ++d) starts[d] = (d==append_dim) ? dims[d] : 0u;

    h5_noerr("H5Sselect_hyperslab", 
            H5Sselect_hyperslab(enlarged_space.get(), H5S_SELECT_SET, starts.data(), nullptr, mem_dims.data(), nullptr));

    // Perform write
    h5_noerr(H5Dwrite(dset, select_predtype<T>(), mem_space.get(), enlarged_space.get(), H5P_DEFAULT, new_data.data()));
}
catch(const std::string &e) {
    throw "while appending to '" + std::string("a dataset") + "', " + e;
}


static std::vector<std::string> 
node_names_in_group(const hid_t loc, const std::string grp_name) {
    std::vector<std::string> names;

    H5G_info_t info;
    h5_noerr(H5Gget_info_by_name(loc, grp_name.c_str(), &info, H5P_DEFAULT));

    for(int i=0; i<int(info.nlinks); ++i) {
        // 1+ is for NULL terminator
        size_t name_size = 1 + h5_noerr(
                 H5Lget_name_by_idx(loc, grp_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, i,
                    nullptr, 0, H5P_DEFAULT));

        auto tmp = std::unique_ptr<char[]>(new char[name_size]);
        h5_noerr(H5Lget_name_by_idx(loc, grp_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, i,
                    tmp.get(), name_size, H5P_DEFAULT));

        names.emplace_back(tmp.get());
    }
    return names;
}









}
#endif
