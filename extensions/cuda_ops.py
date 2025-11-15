import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda_ops as _C
import torch


def gelu(x: torch.Tensor):
    # x: [N] or any shape tensor on CUDA
    return _C.gelu_forward(x)


def swish(x: torch.Tensor):
    # x: [N] or any shape tensor on CUDA
    return _C.swish_forward(x)


def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    # x: [R, C], gamma/beta: [C]
    return _C.layernorm_forward(x, gamma, beta, eps)


def fused_ln_gelu(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    # x: [R, C], gamma/beta: [C]
    return _C.fused_ln_gelu_forward(x, gamma, beta, eps)
