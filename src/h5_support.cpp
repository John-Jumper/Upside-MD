#include "h5_support.h"


namespace h5 {

hid_t h5_noerr(hid_t i) {
    if(i<0) {
        // H5Eprint2(H5E_DEFAULT, stderr);
        throw std::string("error " + std::to_string(i));
    }
    return i;
}

hid_t h5_noerr(const char* nm, hid_t i) {
    if(i<0) {
        // H5Eprint2(H5E_DEFAULT, stderr);
        throw std::string("error " + std::to_string(i) + " in " + nm);
    }
    return i;
}

H5Obj h5_obj(H5DeleterFunc &deleter, hid_t obj)
{
    return H5Obj(h5_noerr(obj), H5Deleter(&deleter));
}

std::vector<hsize_t> get_dset_size(int ndims, hid_t group, const char* name) 
try {
    std::vector<hsize_t> ret(ndims, 0);
    auto dset  = h5_obj(H5Dclose, H5Dopen2(group, name, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Dget_space(dset.get()));

    int ndims_actual = h5_noerr(H5Sget_simple_extent_ndims(space.get()));
    if(ndims_actual != ndims) 
        throw std::string("wrong number of dimensions (expected ") + 
            std::to_string(ndims) + ", but got " + std::to_string(ndims_actual) + ")";
    h5_noerr("H5Sget_simple_extent_dims", H5Sget_simple_extent_dims(space.get(), ret.data(), NULL));
    return ret;
} catch(const std::string &e) {
    throw "while getting size of '" + std::string(name) + "', " + e;
}


bool h5_exists(hid_t base, const char* nm) {
    // Note that this function does not do the full dance specified in 
    // the documentation for H5Oexists_by_name.  I don't think this will cause
    // false results but if there is a problem, consult the h5 docs.

    // There is some special case problem with '.', so we 
    // will always return true since base is assumed to exist
    if(nm==std::string(".")) return true; 
    
    return h5_noerr(H5Lexists(base, nm, H5P_DEFAULT)) &&
        h5_noerr(H5Oexists_by_name(base, nm, H5P_DEFAULT));
}

bool read_attribute(void* attr_value_output, hid_t h5, const char* path, const char* attr_name, hid_t predtype)
try {
    if(!h5_exists(h5, path))
        throw std::string("path does not exist in h5 file");

    if(!h5_bool_return(H5Aexists_by_name(h5, path, attr_name, H5P_DEFAULT)))
        return false;

    // From here, everything should exist
    auto attr  = h5_obj(H5Aclose, H5Aopen_by_name(h5, path, attr_name, H5P_DEFAULT, H5P_DEFAULT));
    h5_noerr(H5Aread(attr.get(), predtype, attr_value_output));
    return true;
}
catch(const std::string &e) {
    throw "while reading attribute '" + std::string(attr_name) + "' of '" + std::string(path) + "', " + e;
}

template <>
std::vector<std::string> read_attribute<std::vector<std::string>>(hid_t h5, const char* path, const char* attr_name) 
try {
    if(!h5_exists(h5, path))
        throw std::string("path does not exist in h5 file");

    auto attr  = h5_obj(H5Aclose, H5Aopen_by_name(h5, path, attr_name, H5P_DEFAULT, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Aget_space(attr.get()));
    auto dtype = h5_obj(H5Tclose, H5Aget_type (attr.get()));
    if(H5Tis_variable_str(dtype.get())) throw std::string("variable-length strings not supported");

    size_t maxchars = H5Tget_size(dtype.get());
    if(maxchars==0) throw std::string("H5Tget_size error"); // defined error value

    if(H5Sget_simple_extent_ndims(space.get()) != 1) throw std::string("wrong size for attribute");
    hsize_t dims[1]; h5_noerr(H5Sget_simple_extent_dims(space.get(), dims, NULL));

    auto tmp = std::unique_ptr<char[]>(new char[dims[0]*maxchars+1]);
    std::fill(tmp.get(), tmp.get()+dims[0]*maxchars+1, '\0');
    h5_noerr(H5Aread(attr.get(), dtype.get(), tmp.get()));

    std::vector<std::string> ret;

    auto g = [&](int i, char& s) {
        std::string tmp(maxchars,'\0');
        std::copy(&s, &s+maxchars, begin(tmp));
        while(tmp.size() && tmp.back() == '\0') tmp.pop_back();
        ret.push_back(tmp);
    };
    traverse_dataset_iteraction_helper<1,char,decltype(g)>()(tmp.get(), dims, g, maxchars);
    return ret;
} catch(const std::string &e) {
    throw "while reading attribute '" + std::string(attr_name) + "' of '" + std::string(path) + "', " + e;
}


void write_string_attribute(
        hid_t h5, const char* path, const char* attr_name,
        const std::string& value) 
try {
    // Create a string datatype of the appropriate size and null-terminated
    auto attr_type = h5_obj(H5Tclose, H5Tcopy(H5T_C_S1));
    h5_noerr(H5Tset_size(attr_type.get(), 1+value.size()));  // include 0 byte in size
    h5_noerr(H5Tset_strpad(attr_type.get(), H5T_STR_NULLTERM));
    auto attr_space = h5_obj(H5Sclose, H5Screate(H5S_SCALAR));

    // Create and write attribute
    auto attr = h5_obj(H5Aclose, H5Acreate_by_name(
                h5, path, attr_name,
                attr_type.get(), attr_space.get(), 
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    h5_noerr(H5Awrite(attr.get(), attr_type.get(), value.c_str()));
} catch(const std::string &e) {
    throw "while writing attribute '" + std::string(attr_name) + "' of '" +
        std::string(path) + "', " + e;
}

void check_size(hid_t group, const char* name, std::vector<size_t> sz)
{
    size_t ndim = sz.size();
    auto dims = get_dset_size(ndim, group, name);
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

void check_size(hid_t group, const char* name, size_t sz) 
{ check_size(group, name, std::vector<size_t>(1,sz)); }

void check_size(hid_t group, const char* name, size_t sz1, size_t sz2) 
{ check_size(group, name, std::vector<size_t>{{sz1,sz2}}); }

void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3) 
{ check_size(group, name, std::vector<size_t>{{sz1,sz2,sz3}}); }

void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4) 
{ check_size(group, name, std::vector<size_t>{{sz1,sz2,sz3,sz4}}); }

void check_size(hid_t group, const char* name, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5) 
{ check_size(group, name, std::vector<size_t>{{sz1,sz2,sz3,sz4,sz5}}); }


H5Obj ensure_group(hid_t loc, const char* nm) {
    return h5_obj(H5Gclose, h5_exists(loc, nm) 
            ? H5Gopen2(loc, nm, H5P_DEFAULT)
            : H5Gcreate2(loc, nm, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
}

H5Obj open_group(hid_t loc, const char* nm) 
try {
    return h5_obj(H5Gclose, H5Gopen2(loc, nm, H5P_DEFAULT));
} catch(std::string &s) {
    throw std::string("unable to open group ") + nm + " (does it exist?), " + s;
}

void ensure_not_exist(hid_t loc, const char* nm) {
    if(h5_exists(loc, nm)) H5Ldelete(loc, nm, H5P_DEFAULT);
}


H5Obj create_earray(hid_t group, const char* name, hid_t dtype,
        const std::initializer_list<int> &dims, // any direction that is extendable must have dims == -1
        const std::initializer_list<int> &chunk_dims,
        bool compression_level){ // 1 is often recommended
    hsize_t ndims = dims.size();
    std::vector<hsize_t> dims_v(ndims);
    std::vector<hsize_t> chunk_dims_v(ndims);

    for(size_t d=0; d<ndims; ++d) {
        int x = *(begin(dims)+d);
        dims_v[d]       = (x==-1) ? H5S_UNLIMITED : x;
        chunk_dims_v[d] = *(begin(chunk_dims)+d);
    }
    return create_earray(group, name, dtype, dims_v, chunk_dims_v, compression_level);
}


H5Obj create_earray(hid_t group, const char* name, hid_t dtype,
        const std::vector<hsize_t>& dims_v, // any direction that is extendable must have dims == H5S_UNLIMITED
        const std::vector<hsize_t>& chunk_dims_v,
        bool compression_level)  // 1 is often recommended
{
    if(dims_v.size() != chunk_dims_v.size()) throw std::string("invalid chunk dims");
    std::vector<hsize_t> dims = dims_v;
    std::vector<hsize_t> chunk_dims = chunk_dims_v;

    hsize_t ndims = dims_v.size();
    std::vector<hsize_t> max_dims_v(ndims);

    for(size_t d=0; d<ndims; ++d) {
        if(dims[d]==H5S_UNLIMITED) {
            dims[d] = 0;
            max_dims_v[d] = H5S_UNLIMITED;
        } else {
            max_dims_v[d] = dims[d];
        }
        if(chunk_dims[d]==0u) chunk_dims[d]=1;
    }

    auto space_id = h5_obj(H5Sclose, H5Screate_simple(ndims, dims.data(), max_dims_v.data()));

    // setup chunked, possibly compressed storage
    auto dcpl_id = h5_obj(H5Pclose, H5Pcreate(H5P_DATASET_CREATE));
    h5_noerr(H5Pset_chunk(dcpl_id.get(), ndims, chunk_dims.data()));
    h5_noerr(H5Pset_shuffle(dcpl_id.get()));     // improves data compression
    h5_noerr(H5Pset_fletcher32(dcpl_id.get()));  // for verifying data integrity
    if(compression_level) h5_noerr(H5Pset_deflate(dcpl_id.get(), compression_level));

    return h5_obj(H5Dclose, H5Dcreate2(group, name, dtype, space_id.get(), 
                H5P_DEFAULT, dcpl_id.get(), H5P_DEFAULT));
}

void append_to_dset(hid_t dset, hid_t hdf_predtype, size_t n_new_data_elems, const void* new_data, int append_dim)
try {
    // Load current data size
    auto space = h5_obj(H5Sclose, H5Dget_space(dset));
    int ndims = h5_noerr(H5Sget_simple_extent_ndims(space.get()));
    std::vector<hsize_t> dims(ndims);
    h5_noerr("H5Sget_simple_extent_dims", H5Sget_simple_extent_dims(space.get(), dims.data(), NULL));

    // Compute number of records in new_data
    long int record_size = 1;  // product of all dim sizes except the append_dim
    for(int nd=0; nd<(int)dims.size(); ++nd) if(nd!=append_dim) record_size *= dims[nd];

    long int n_records = n_new_data_elems / record_size;
    if(n_records*record_size != (long)n_new_data_elems) 
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
    h5_noerr(H5Dwrite(dset, hdf_predtype, mem_space.get(), enlarged_space.get(), H5P_DEFAULT, new_data));
}
catch(const std::string &e) {
    throw "while appending to '" + std::string("a dataset") + "', " + e;
}


std::vector<std::string> 
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
