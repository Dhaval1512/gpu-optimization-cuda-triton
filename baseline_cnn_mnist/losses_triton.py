from triton_kernels.triton_ops import triton_focal_loss

def focal_loss_triton(log_probs, targets, alpha=0.25, gamma=2.0):
    return triton_focal_loss(log_probs, targets, alpha, gamma)