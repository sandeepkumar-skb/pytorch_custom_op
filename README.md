## This is short End to End tutorial on how to hook custom operators in PyTorch.

This Tutorial is devided into 3 Parts.
Part 1: Creating an op and registering it to PyTorch.
Part 2: Building the op into a shared library.
Part 3: Testing out the custom op.

## Part 1: Creating an op and registering it to PyTorch.
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
## Part 2: Building the custom op
Now we have to build the custom op into a library which can be imported and used as a PyTorch operator. Here I have used the CMake recipe to build the op. If you want to use the python way of building then you refer to 
https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#building-the-custom-operator for more details.

1. Create a `CMakeList.txt` file(See the comments for explanation):
```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_allreduce_op LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_allreduce_op SHARED pyt_all_reduce_op.cpp pyt_all_reduce_kernel.cu)
# Enable C++14
target_compile_features(custom_allreduce_op PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(custom_allreduce_op "${TORCH_LIBRARIES}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
```
2. Make a `build` directory and run the following command inside the `build` directory:
   ```
   mkdir build; cd build;
   cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" ..
   ```
3. Now `make -j$(nproc)`
    ```
    Scanning dependencies of target custom_allreduce_op
    [ 33%] Building CXX object CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_op.cpp.o
    [ 66%] Building CUDA object CMakeFiles/custom_allreduce_op.dir/pyt_all_reduce_kernel.cu.o
    [100%] Linking CXX shared library libcustom_allreduce_op.so
    [100%] Built target custom_allreduce_op
    ```
Done! Now you library is created. Let's test the Op.
## Part 3: Testing the Custom Op:
This step is easy, simply import the library which is created in the previous part and use the operator other PyTorch operators. Since the custom op is registers into torch.ops we will have to call `torch.ops.my_ops.custom_allreduce(input)`
```
import torch
torch.ops.load_library("build/libcustom_allreduce_op.so")
A = torch.ones(1024, dtype=torch.int, device='cuda')
b = torch.ops.my_ops.custom_allreduce(A)
print(b.to('cpu'))
```






