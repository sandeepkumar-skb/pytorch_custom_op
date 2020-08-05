import torch
torch.ops.load_library("build/libcustom_allreduce_op.so")
A = torch.ones(1024, dtype=torch.int)
b = torch.ops.my_ops.custom_allreduce(A)
print(b)

