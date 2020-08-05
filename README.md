This is short tutorial on how to hook custom operators in PyTorch.
Here are the steps:
1. First, we need a custom operator(duh!) which we want to add to PyTorch.
For the sake of this tutorial let's take the example of `all_reduce` kernel. We will add a GPU and CPU version of `all_reduce` op in this tutorial.

CPU:
```
void cpu_all_reduce(int* sum, int* data, int n){
    int temp_sum = 0;
    for (int i=0; i<n; ++i){
        temp_sum += data[i];
    }
    *sum = temp_sum;
}
```

GPU:
```
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
```

2. Now that we have a custom operator, next step is the create a laucher function which will call the appropriate CPU/GPU op.

```
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
```

3. Alright, that was fun! Now comes the part where we define a PyTorch op which will call the `all_reduce_launcher` function.
```
static torch::Tensor custom_allreduce(torch::Tensor input) {
    return all_reduce_launcher(input);·
}
```

4. We are almost done! We just have to register this OP with PyTorch so that PyTorch can recognize this as a valid operator.
```
TORCH_LIBRARY (my_ops, m){
    m.def("custom_allreduce", &custom_allreduce);
}
```


