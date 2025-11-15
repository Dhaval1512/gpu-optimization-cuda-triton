import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from triton_kernels.triton_ops import (
    triton_gelu,
    triton_swish,
    triton_layernorm,
    triton_fused_ln_gelu,
)

torch.manual_seed(0)


print("\n=======================================")
print("   TRITON KERNEL CORRECTNESS TESTS")
print("=======================================\n")


# ============ TEST 1: GELU ============
print("TEST 1: Triton GELU")
x = torch.randn(1024, device="cuda", dtype=torch.float32)

y_triton = triton_gelu(x)
y_torch = torch.nn.functional.gelu(x)

max_diff = (y_triton - y_torch).abs().max().item()
print(f"  Max diff GELU: {max_diff:.8f}")
print("  Shape:", y_triton.shape, "\n")


# ============ TEST 2: Swish ============
print("TEST 2: Triton Swish")
x = torch.randn(1024, device="cuda", dtype=torch.float32)

y_triton = triton_swish(x)
y_torch = x * torch.sigmoid(x)

max_diff = (y_triton - y_torch).abs().max().item()
print(f"  Max diff Swish: {max_diff:.8f}")
print("  Shape:", y_triton.shape, "\n")


# ============ TEST 3: LayerNorm ============
print("TEST 3: Triton LayerNorm")

rows, cols = 32, 512
x2 = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
gamma = torch.ones(cols, device="cuda", dtype=torch.float32)
beta = torch.zeros(cols, device="cuda", dtype=torch.float32)

y_triton_ln = triton_layernorm(x2, gamma, beta)
y_torch_ln = torch.nn.functional.layer_norm(x2, (cols,), gamma, beta)

max_diff = (y_triton_ln - y_torch_ln).abs().max().item()
print(f"  Max diff LayerNorm: {max_diff:.8f}")
print("  Shape:", y_triton_ln.shape, "\n")


# ============ TEST 4: Fused LN + GELU ============
print("TEST 4: Triton Fused LN + GELU")

y_triton_fused = triton_fused_ln_gelu(x2, gamma, beta)
y_torch_fused = torch.nn.functional.gelu(y_torch_ln)

max_diff = (y_triton_fused - y_torch_fused).abs().max().item()
print(f"  Max diff Fused LN+GELU: {max_diff:.8f}")
print("  Shape:", y_triton_fused.shape, "\n")


print("=======================================")
print(" All Triton tests completed successfully")
print("=======================================\n")
