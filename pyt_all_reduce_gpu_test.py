import torch
torch.ops.load_library("build/libcustom_allreduce_op.so")
A = torch.ones(1024, dtype=torch.float32, device='cuda')
b = torch.ops.my_ops.custom_allreduce(A.half())
print(b[0]) #.to('cpu'))

