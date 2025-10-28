import cupy as cp
import torch
from pathlib import Path

def _load_kernel(fname: str, kname: str):
    code = Path(fname).read_text()
    return cp.RawKernel(code, kname)

# cache kernels
_K_LN  = _load_kernel("cuda_kernels/layernorm.cu", "layernorm_forward")
_K_GELU = _load_kernel("cuda_kernels/gelu.cu", "gelu_forward")
_K_SWISH = _load_kernel("cuda_kernels/swish.cu", "swish_forward")
_K_LN_GELU = _load_kernel("cuda_kernels/fused_layernorm_gelu.cu","fused_ln_gelu_forward")

def to_cu(t: torch.Tensor): return cp.asarray(t.detach().contiguous().cpu().numpy())
def from_cu(a: cp.ndarray, like: torch.Tensor): return torch.from_numpy(cp.asnumpy(a)).to(like.device)

def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5):
    rows, cols = x.shape
    X = to_cu(x); G = to_cu(gamma); B = to_cu(beta)
    Y = cp.empty_like(X)
    block_x = min(256, 1 << (int(cols - 1).bit_length()))  # next pow2 <= 256
    block = (block_x, 1, 1)
    grid  = (rows, 1, 1)
    shmem = 2 * block_x * cp.dtype(cp.float32).itemsize
    _K_LN(grid, block, (X, G, B, Y, rows, cols, eps), shared_mem=shmem)
    return from_cu(Y, x)

def gelu(x: torch.Tensor):
    N = x.numel()
    X = to_cu(x); Y = cp.empty_like(X)
    block = (256,1,1); grid = ((N + 255)//256,1,1)
    _K_GELU(grid, block, (X, Y, N))
    return from_cu(Y, x)

def swish(x: torch.Tensor):
    N = x.numel()
    X = to_cu(x); Y = cp.empty_like(X)
    block = (256,1,1); grid = ((N + 255)//256,1,1)
    _K_SWISH(grid, block, (X, Y, N))
    return from_cu(Y, x)

def fused_ln_gelu(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-5):
    rows, cols = x.shape
    X = to_cu(x); G = to_cu(gamma); B = to_cu(beta)
    Y = cp.empty_like(X)
    block_x = min(256, 1 << (int(cols - 1).bit_length()))
    block = (block_x, 1, 1)
    grid  = (rows, 1, 1)
    shmem = 2 * block_x * cp.dtype(cp.float32).itemsize
    _K_LN_GELU(grid, block, (X, G, B, Y, rows, cols, eps), shared_mem=shmem)
    return from_cu(Y, x)