#include <torch/script.h>
#include "pyt_all_reduce_kernel.hh"
#define BLOCKX_DIM 256

template<typename scalar_t>
void cpu_all_reduce(float * sum, scalar_t* data, int n){
    scalar_t temp_sum = 0;
    for (int i=0; i<n; ++i){
        temp_sum += data[i];
    }
    *sum = temp_sum;
}

template<typename scalar_t>
__global__
void gpu_all_reduce(float *sum, scalar_t *data, int n){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    scalar_t temp = 0;
    for (int i=idx; i < n; i += stride){
        temp += data[i];
    }

    atomicAdd(sum, temp);
}


torch::Tensor all_reduce_launcher(torch::Tensor input){
    torch::Device device(torch::kCUDA, 0);
    torch::Tensor output = torch::zeros(1, torch::kFloat);
    if (input.device() == device){
        output = output.to(device);
        dim3 blockSize(BLOCKX_DIM);
        dim3 gridSize((input.size(0)+BLOCKX_DIM-1)/BLOCKX_DIM);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "gpu_all_reduce", ([&] {
            gpu_all_reduce<scalar_t><<<gridSize, blockSize, 0, stream>>>(output.data_ptr<float>(), 
                input.data_ptr<scalar_t>(), 
                input.size(0));
        } ));
    }
    else{
        cpu_all_reduce<int>(output.data_ptr<float>(), input.data_ptr<int>(), input.size(0));
    }
    return output;
}

