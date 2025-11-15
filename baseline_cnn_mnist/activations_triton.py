import torch
from triton_kernels.triton_ops import (
    triton_gelu,
    triton_swish,
    triton_fused_ln_gelu
)

def triton_gelu_activation(x):
    return triton_gelu(x)

def triton_swish_activation(x):
    return triton_swish(x)

def triton_fused_ln_gelu_activation(x, gamma=None, beta=None):
    # for LN+GELU fusion
    if gamma is None or beta is None:
        raise ValueError("gamma and beta required for fused LN+GELU")
    return triton_fused_ln_gelu(x, gamma, beta)
