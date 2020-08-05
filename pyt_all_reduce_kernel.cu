#include <torch/script.h>
#include "pyt_all_reduce_kernel.hh"
#define BLOCKX_DIM 256

void cpu_all_reduce(int* sum, int* data, int n){
    int temp_sum = 0;
    for (int i=0; i<n; ++i){
        temp_sum += data[i];
    }
    *sum = temp_sum;
}

__global__
void gpu_all_reduce(int *sum, int* data, int n){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    int temp = 0;
    for (int i=idx; i < n; i += stride){
        temp += data[i];
    }

    atomicAdd(sum, temp);
}


torch::Tensor all_reduce_launcher(torch::Tensor input){
    torch::Device device(torch::kCUDA, 0);
    torch::Tensor output = torch::zeros(1, torch::kInt);
    if (input.device() == device){
        output = output.to(device);
        dim3 blockSize(BLOCKX_DIM);
        dim3 gridSize((input.size(0)+BLOCKX_DIM-1)/BLOCKX_DIM);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gpu_all_reduce<<<gridSize, blockSize, 0, stream>>>(output.data_ptr<int>(), 
                input.data_ptr<int>(), 
                input.size(0));
    }
    else{
        cpu_all_reduce(output.data_ptr<int>(), input.data_ptr<int>(), input.size(0));
    }
    return output;
           
}

