import torch
from triton_kernels.layernorm import layernorm_fw as triton_layernorm
from triton_kernels.gelu import gelu_fw as triton_gelu
from triton_kernels.swish import swish_fw as triton_swish
from triton_kernels.fused_layernorm_gelu import ln_gelu_fw as triton_ln_gelu

def layernorm(x, gamma, beta, eps=1e-5):
    return triton_layernorm(x, gamma, beta, eps)

def gelu(x):  return triton_gelu(x)
def swish(x): return triton_swish(x)

def fused_ln_gelu(x, gamma, beta, eps=1e-5):
    return triton_ln_gelu(x, gamma, beta, eps)
