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
