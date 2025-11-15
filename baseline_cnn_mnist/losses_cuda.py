import torch
from extensions.cuda_ops import focal_loss as cuda_focal_loss

def focal_loss_cuda(log_probs: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0):
    """CUDA Focal Loss wrapper"""
    return cuda_focal_loss(log_probs, targets, alpha, gamma)