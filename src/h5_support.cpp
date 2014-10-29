#include "h5_support.h"

namespace h5 {

template <>
std::vector<std::string> read_attribute<std::vector<std::string>>(hid_t h5, const char* path, const char* attr_name) 
try {
    auto attr  = h5_obj(H5Aclose, H5Aopen_by_name(h5, path, attr_name, H5P_DEFAULT, H5P_DEFAULT));
    auto space = h5_obj(H5Sclose, H5Aget_space(attr.get()));
    auto dtype = h5_obj(H5Tclose, H5Aget_type (attr.get()));
    if(H5Tis_variable_str(dtype.get())) throw std::string("variable-length strings not supported");

    size_t maxchars = H5Tget_size(dtype.get());
    if(maxchars==0) throw std::string("H5Tget_size error"); // defined error value

    if(H5Sget_simple_extent_ndims(space.get()) != 1) throw std::string("wrong size for attribute");
    hsize_t dims[1]; h5_noerr(H5Sget_simple_extent_dims(space.get(), dims, NULL));

    auto tmp = std::unique_ptr<char>(new char[dims[0]*maxchars+1]);
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
} 
catch(const std::string &e) {
    throw "while reading attribute '" + std::string(attr_name) + "' of '" + std::string(path) + "', " + e;
}



}
