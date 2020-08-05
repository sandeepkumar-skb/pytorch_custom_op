#include <torch/torch.h>
#include <torch/script.h>
#include "pyt_all_reduce_kernel.hh"

static torch::Tensor custom_allreduce(torch::Tensor input) {
    return all_reduce_launcher(input); 
}
//static auto registry = torch::RegisterOperators("myop::skbmm", &skbmm);
TORCH_LIBRARY (my_ops, m){
    m.def("custom_allreduce", &custom_allreduce);
}

