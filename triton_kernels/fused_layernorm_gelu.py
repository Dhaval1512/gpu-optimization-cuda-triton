import triton, triton.language as tl

@triton.jit
def _ln_gelu_fw(X, G, B, Y, cols, eps: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = r * cols + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < cols

    x = tl.load(X + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + tl.arange(0, BLOCK), mask=mask, other=1.0)
    b = tl.load(B + tl.arange(0, BLOCK), mask=mask, other=0.0)
    h = (xc * inv_std) * g + b

    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * h * (1. + tl.erf(h * inv_sqrt2))

    tl.store(Y + offs, y, mask=mask)

def ln_gelu_fw(x, gamma, beta, eps=1e-5):
    rows, cols = x.shape
    y = x.new_empty(x.shape)
    BLOCK = triton.next_power_of_2(cols)
    grid = (rows,)
    _ln_gelu_fw[grid](x, gamma, beta, y, cols, eps, BLOCK=BLOCK)
    return y
