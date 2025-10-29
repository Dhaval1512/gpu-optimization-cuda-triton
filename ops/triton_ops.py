import torch

try:
    from triton_kernels.layernorm import layernorm_fw as triton_layernorm
    from triton_kernels.gelu import gelu_fw as triton_gelu
    from triton_kernels.swish import swish_fw as triton_swish
    from triton_kernels.fused_layernorm_gelu import ln_gelu_fw as triton_ln_gelu
    _TRITON_AVAILABLE = True
except ModuleNotFoundError:
    _TRITON_AVAILABLE = False

    def _missing(*_args, **_kwargs):
        raise RuntimeError("Triton kernels are not available on this platform.")

def layernorm(x, gamma, beta, eps=1e-5):
    if not _TRITON_AVAILABLE:
        _missing()
    return triton_layernorm(x, gamma, beta, eps)

def gelu(x):
    if not _TRITON_AVAILABLE:
        _missing()
    return triton_gelu(x)

def swish(x):
    if not _TRITON_AVAILABLE:
        _missing()
    return triton_swish(x)

def fused_ln_gelu(x, gamma, beta, eps=1e-5):
    if not _TRITON_AVAILABLE:
        _missing()
    return triton_ln_gelu(x, gamma, beta, eps)
