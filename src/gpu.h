#ifndef GPU_H
#define GPU_H

template <class T>
struct shared_vector 
{
    protected:
        mutable vector<T> host_data;
        mutable bool owned_by_device;
        mutable T* device_ptr;
        mutable size_t device_size;

        void ensure_data_on_host() const {
#ifdef USE_CUDA
            if(!owned_by_device) return;
            
            // invariant: vector size and device array size are always equal when data
            // is owned by the device
            
            // because of this invariant, we can do a simple cuda copy
            cudaNoError(cudaMemcpy(host_data.data(), device_ptr, device_size*sizeof(T), cudaMemcpyDeviceToHost));
            owned_by_device = false;
#endif
        }

        void ensure_data_on_device() const {
#ifdef USE_CUDA
            if(owned_by_device) return;

            if(device_size != host_data.size()) {
                cudaNoError(device_ptr);
                cudaNoError(cudaMalloc(&device_ptr, host_data.size()*sizeof(T)));
                device_size = host_data.size();
            }

            cudaNoError(cudaMemcpy(device_ptr, host_data.data(), device_size*sizeof(T), cudaMemcpyHostToDevice));
            owned_by_device = true;
#endif
        }

    public:
        shared_vector(size_t n_elem):
            host_data(n_elem),
            owned_by_device(false),
            device_ptr(nullptr),
            device_size(0) {}

        shared_vector(vector<T>& init):
            host_data(init),
            owned_by_device(false),
            device_ptr(nullptr),
            device_size(0) {}

        shared_vector(): 
            owned_by_device(false), 
            device_ptr(nullptr), 
            device_size(0) {}

        vector<T>&       host() {
            ensure_data_on_host();
            return host_data;
        }

        const vector<T>& host() const {
            ensure_data_on_host();
            return host_data;
        }

        T*       device() {
            ensure_data_on_device();
            return device_ptr;
        }

        const T* device() const {
            ensure_data_on_device();
            return device_ptr;
        }

        size_t size() const {
            return host_data.size();  // host size is always correct
        }
};


#endif

