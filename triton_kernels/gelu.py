import triton, triton.language as tl

@triton.jit
def _gelu_fw(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1. + tl.erf(x * inv_sqrt2))
    tl.store(Y + offs, y, mask=mask)

def gelu_fw(x):
    N = x.numel()
    y = x.new_empty(N)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _gelu_fw[grid](x.view(-1), y, N, BLOCK=BLOCK)
    return y.view_as(x)
