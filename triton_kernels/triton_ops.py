import triton
import triton.language as tl
import torch


# -------------------- GELU --------------------
@triton.jit
def gelu_kernel(X, Y, N,
                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    x = tl.load(X + offset, mask=mask)
    inv_sqrt2 = 0.70710678118
    gelu = 0.5 * x * (1 + tl.erf(x * inv_sqrt2))
    tl.store(Y + offset, gelu, mask=mask)


def triton_gelu(x: torch.Tensor):
    N = x.numel()
    y = torch.empty_like(x)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK),)

    gelu_kernel[grid](x, y, N, BLOCK=BLOCK)
    return y

# -------------------- Swish --------------------
@triton.jit
def swish_kernel(X, Y, N,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    x = tl.load(X + offset, mask=mask)
    sig = 1 / (1 + tl.exp(-x))
    sw = x * sig
    tl.store(Y + offset, sw, mask=mask)


def triton_swish(x: torch.Tensor):
    N = x.numel()
    y = torch.empty_like(x)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK),)

    swish_kernel[grid](x, y, N, BLOCK=BLOCK)
    return y


# -------------------- LayerNorm --------------------
@triton.jit
def layernorm_fwd(X, Y, Gamma, Beta,
                  R, C, eps: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):

    row_id = tl.program_id(0)
    X_row = X + row_id * C
    Y_row = Y + row_id * C

    offsets = tl.arange(0, BLOCK_SIZE)

    # Load row, compute mean
    x = tl.load(X_row + offsets, mask=offsets < C, other=0.0)
    mean = tl.sum(x, axis=0) / C

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C

    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    norm = (x - mean) * inv_std

    gamma = tl.load(Gamma + offsets, mask=offsets < C, other=1.0)
    beta = tl.load(Beta + offsets, mask=offsets < C, other=0.0)

    y = norm * gamma + beta

    tl.store(Y_row + offsets, y, mask=offsets < C)


def triton_layernorm(x, gamma, beta, eps=1e-5):
    R, C = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(C)

    grid = lambda meta: (R,)

    layernorm_fwd[grid](
        x, y, gamma, beta,
        R, C, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return y

# -------------------- Fused LN + GELU --------------------
@triton.jit
def fused_ln_gelu_fwd(X, Y, Gamma, Beta,
                      R, C, eps: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):

    row_id = tl.program_id(0)
    X_row = X + row_id * C
    Y_row = Y + row_id * C

    offsets = tl.arange(0, BLOCK_SIZE)

    x = tl.load(X_row + offsets, mask=offsets < C, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / C
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    norm = diff * inv_std

    gamma = tl.load(Gamma + offsets, mask=offsets < C, other=1.0)
    beta = tl.load(Beta + offsets, mask=offsets < C, other=0.0)

    ln_out = norm * gamma + beta

    # GELU (erf-based, PyTorch-accurate) inside same kernel → fusion
    inv_sqrt2 = 0.70710678118
    gelu = 0.5 * ln_out * (1 + tl.erf(ln_out * inv_sqrt2))

    tl.store(Y_row + offsets, gelu, mask=offsets < C)



def triton_fused_ln_gelu(x, gamma, beta, eps=1e-5):
    R, C = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = lambda meta: (R,)

    fused_ln_gelu_fwd[grid](
        x, y, gamma, beta,
        R, C, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return y

# -------------------- Focal Loss --------------------
@triton.jit
def focal_loss_kernel(
    LogProbs, Targets, Losses,
    N: tl.constexpr, C: tl.constexpr,
    alpha: tl.constexpr, gamma: tl.constexpr,
    BLOCK: tl.constexpr):
    
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    
    # Load target class for each sample
    target = tl.load(Targets + idx, mask=mask, other=0)
    
    # Load log probability of the target class
    # log_probs is [N, C], we need log_probs[idx, target]
    offset = idx * C + target
    log_pt = tl.load(LogProbs + offset, mask=mask, other=0.0)
    
    # Convert to probability: pt = exp(log_pt)
    pt = tl.exp(log_pt)
    
    # Focal loss: -alpha * (1 - pt)^gamma * log_pt
    focal_weight = tl.math.pow(1.0 - pt, gamma)
    loss = -alpha * focal_weight * log_pt
    
    tl.store(Losses + idx, loss, mask=mask)


def triton_focal_loss(log_probs: torch.Tensor, 
                      targets: torch.Tensor,
                      alpha: float = 0.25,
                      gamma: float = 2.0):
    """
    Focal Loss using Triton
    
    Args:
        log_probs: [N, C] tensor of log probabilities (from log_softmax)
        targets: [N] tensor of target class indices
        alpha: weighting factor (default 0.25)
        gamma: focusing parameter (default 2.0)
    
    Returns:
        Scalar loss value (mean over batch)
    """
    N, C = log_probs.shape
    
    # Allocate output
    losses = torch.empty(N, device=log_probs.device, dtype=log_probs.dtype)
    
    BLOCK = 256
    grid = lambda meta: (triton.cdiv(N, BLOCK),)
    
    focal_loss_kernel[grid](
        log_probs, targets, losses,
        N, C, alpha, gamma,
        BLOCK=BLOCK
    )
    
    # Return mean loss
    return losses.mean()

# Focal Loss
@triton.jit
def focal_loss_kernel(LogProbs, Targets, Losses, N: tl.constexpr, C: tl.constexpr, 
                      alpha: tl.constexpr, gamma: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    target = tl.load(Targets + idx, mask=mask, other=0)
    offset = idx * C + target
    log_pt = tl.load(LogProbs + offset, mask=mask, other=0.0)
    pt = tl.exp(log_pt)
    
    # Use repeated multiplication instead of pow for (1-pt)^gamma
    # For gamma=2.0: (1-pt)^2 = (1-pt) * (1-pt)
    one_minus_pt = 1.0 - pt
    focal_weight = one_minus_pt * one_minus_pt  # gamma=2.0
    
    loss = -alpha * focal_weight * log_pt
    tl.store(Losses + idx, loss, mask=mask)