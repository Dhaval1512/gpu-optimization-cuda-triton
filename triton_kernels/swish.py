import triton, triton.language as tl

@triton.jit
def _swish_fw(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y + offs, x * s, mask=mask)

def swish_fw(x):
    N = x.numel()
    y = x.new_empty(N)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _swish_fw[grid](x.view(-1), y, N, BLOCK=BLOCK)
    return y.view_as(x)
