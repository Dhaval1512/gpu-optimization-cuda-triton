import torch
from extensions.cuda_ops import gelu as cuda_gelu
from extensions.cuda_ops import swish as cuda_swish

def cuda_gelu_activation(x: torch.Tensor):
    return cuda_gelu(x)

def cuda_swish_activation(x: torch.Tensor):
    return cuda_swish(x)
