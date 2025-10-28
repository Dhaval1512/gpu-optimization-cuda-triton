import torch, torch.nn.functional as F
from ops.cuda_ops import layernorm as cuda_ln, gelu as cuda_gelu, swish as cuda_swish, fused_ln_gelu as cuda_ln_gelu
from ops.triton_ops import layernorm as tri_ln, gelu as tri_gelu, swish as tri_swish, fused_ln_gelu as tri_ln_gelu

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

def check_close(a, b, name, atol=1e-4, rtol=1e-4):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"{name}: {'OK' if ok else 'MISMATCH'}  max|Δ|={float((a-b).abs().max())}")
    return ok

# shape: rows x cols (LN across last dim)
rows, cols = 128, 512
x = torch.randn(rows, cols, device=device, dtype=torch.float32)
gamma = torch.randn(cols, device=device, dtype=torch.float32)
beta  = torch.randn(cols, device=device, dtype=torch.float32)

# PyTorch references
y_ln_ref = F.layer_norm(x, normalized_shape=(cols,), weight=gamma, bias=beta, eps=1e-5)
y_gelu_ref = F.gelu(x)
y_swish_ref = x * torch.sigmoid(x)
y_fused_ref = F.gelu(F.layer_norm(x, (cols,), gamma, beta, 1e-5))

# CUDA
y_ln_cu   = cuda_ln(x, gamma, beta, 1e-5)
y_gelu_cu = cuda_gelu(x)
y_swish_cu= cuda_swish(x)
y_fused_cu= cuda_ln_gelu(x, gamma, beta, 1e-5)

# Triton
y_ln_tr   = tri_ln(x, gamma, beta, 1e-5)
y_gelu_tr = tri_gelu(x)
y_swish_tr= tri_swish(x)
y_fused_tr= tri_ln_gelu(x, gamma, beta, 1e-5)

# Checks
check_close(y_ln_cu,   y_ln_ref,   "CUDA LayerNorm")
check_close(y_ln_tr,   y_ln_ref,   "Triton LayerNorm")
check_close(y_gelu_cu, y_gelu_ref, "CUDA GELU")
check_close(y_gelu_tr, y_gelu_ref, "Triton GELU")
check_close(y_swish_cu,y_swish_ref,"CUDA Swish")
check_close(y_swish_tr,y_swish_ref,"Triton Swish")
check_close(y_fused_cu,y_fused_ref,"CUDA Fused LN+GELU")
check_close(y_fused_tr,y_fused_ref,"Triton Fused LN+GELU")
