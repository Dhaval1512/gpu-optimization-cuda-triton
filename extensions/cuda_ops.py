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

def focal_loss(log_probs: torch.Tensor, 
               targets: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0):
    """
    Focal Loss using CUDA
    
    Args:
        log_probs: [N, C] tensor of log probabilities (from log_softmax)
        targets: [N] tensor of target class indices
        alpha: weighting factor (default 0.25)
        gamma: focusing parameter (default 2.0)
    
    Returns:
        Scalar loss value (mean over batch)
    """
    return _C.focal_loss_forward(log_probs, targets, alpha, gamma)