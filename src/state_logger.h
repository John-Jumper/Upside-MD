#ifndef STATE_LOGGER_H
#define STATE_LOGGER_H

#include "deriv_engine.h"
#include "h5_support.h"
#include <initializer_list>
#include <memory>
#include "timing.h"

struct SingleLogger {
    virtual void collect_samples() = 0;
    virtual void dump_samples   () = 0;
    virtual ~SingleLogger() {};
};


template <typename T, typename F>
struct SpecializedSingleLogger: public SingleLogger {
    h5::H5Obj data_set;
    std::vector<hsize_t> dims;
    std::vector<T> data_buffer;
    F sample_function;
    hsize_t row_size;

    SpecializedSingleLogger(hid_t logging_group, const char* loc, 
            F sample_function_, const std::initializer_list<int>& dims_):
        sample_function(sample_function_), row_size(1u)
    {
        dims.push_back(0);
        std::vector<hsize_t> chunk_shape;
        chunk_shape.push_back(100);
        for(auto i: dims_) {
            dims.push_back(i);
            chunk_shape.push_back(i);
            row_size *= i;
        }
        data_set = h5::create_earray(logging_group, loc, h5::select_predtype<T>(), dims, chunk_shape);
    }

    virtual void collect_samples() {
        data_buffer.resize(data_buffer.size()+row_size);
        T* current_data = data_buffer.data() + data_buffer.size() - row_size;
        sample_function(current_data);
    }

    virtual void dump_samples() {
        if(data_buffer.size()) h5::append_to_dset(data_set.get(), data_buffer, 0);
        data_buffer.resize(0);
    }

    virtual ~SpecializedSingleLogger() {
        dump_samples();
    }
};

        
struct H5Logger {
    h5::H5Obj config;
    h5::H5Obj logging_group;
    std::vector<std::unique_ptr<SingleLogger>> state_loggers;
    size_t n_samples_collected;

    H5Logger(): config(0u), logging_group(0u), n_samples_collected(0u) {}

    H5Logger(h5::H5Obj& config_, const char* loc): 
        config(h5::duplicate_obj(config_)),
        logging_group(h5::ensure_group(config.get(), loc)),
        n_samples_collected(0u)
    {}

    void collect_samples() {
        Timer timer(std::string("logger"));
        for(auto &sl: state_loggers) 
            sl->collect_samples();

        n_samples_collected++;
        if(!(n_samples_collected % 100)) flush();
    }

    void flush() {
        for(auto &sl: state_loggers) 
            sl->dump_samples();
        H5Fflush(config.get(), H5F_SCOPE_LOCAL);
    }

    template <typename T, typename F>
    void add_logger(
            const char* relative_path, 
            const std::initializer_list<int>& data_shape, 
            const F&& sample_function) {
        auto logger = std::unique_ptr<SingleLogger>(
                new SpecializedSingleLogger<T,F>(logging_group.get(), relative_path, sample_function, data_shape));
        state_loggers.emplace_back(std::move(logger));
    }

    template <typename T, typename F>
    void log_once(
            const char* relative_path, 
            const std::initializer_list<int>& data_shape, 
            const F&& sample_function) {
        std::vector<hsize_t> dims;
        hsize_t data_size = 1u;
        for(auto i: data_shape) {
            dims.push_back(i);
            data_size *= i;
        }
        std::vector<hsize_t> fake_dims = dims;
        fake_dims[0] = 0u;  // extensible dimension

        std::vector<T> data_buffer(data_size);
        sample_function(data_buffer.data());

        auto data_set = h5::create_earray(logging_group.get(), relative_path, h5::select_predtype<T>(), 
                fake_dims, dims);
        h5::append_to_dset(data_set.get(), data_buffer, 0);
    }

    virtual ~H5Logger() {
        flush();
    }
};

extern H5Logger* default_logger;

/*
template <typename T>
void log_once(
        const char* relative_path,
        T* data, const std::initializer_list<int>& data_shape);

template <typename T>
void log_append(
        const char* relative_path,
        T* data, 
        const std::initializer_list<int>& data_shape,
        int data_offset = -1);

template <typename T>
void log_automatically(
        const char* relative_path,
        std::function<void(T*)> data_fill_callback, 
        const std::initializer_list<int>& data_shape);
        */

#endif
