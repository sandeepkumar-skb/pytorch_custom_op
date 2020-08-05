#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>


torch::Tensor all_reduce_launcher(torch::Tensor input);

