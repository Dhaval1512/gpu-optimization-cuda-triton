import sys
import torch
from extensions.cuda_ops import gelu, swish, layernorm, fused_ln_gelu

# Check which kernel to profile
if len(sys.argv) < 2:
    print("Usage: python profile_single_kernel.py <kernel_name>")
    print("Available kernels: gelu, swish, layernorm, fused_ln_gelu")
    sys.exit(1)

kernel_name = sys.argv[1]

# Create test tensors
print(f"Profiling kernel: {kernel_name}")

if kernel_name == "gelu":
    x = torch.randn(1048576, device="cuda", dtype=torch.float32)
    for _ in range(10):  # Run 10 times for profiling
        y = gelu(x)
    torch.cuda.synchronize()
    print("GELU profiling complete")

elif kernel_name == "swish":
    x = torch.randn(1048576, device="cuda", dtype=torch.float32)
    for _ in range(10):
        y = swish(x)
    torch.cuda.synchronize()
    print("Swish profiling complete")

elif kernel_name == "layernorm":
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    gamma = torch.ones(1024, device="cuda", dtype=torch.float32)
    beta = torch.zeros(1024, device="cuda", dtype=torch.float32)
    for _ in range(10):
        y = layernorm(x, gamma, beta)
    torch.cuda.synchronize()
    print("LayerNorm profiling complete")

elif kernel_name == "fused_ln_gelu":
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    gamma = torch.ones(1024, device="cuda", dtype=torch.float32)
    beta = torch.zeros(1024, device="cuda", dtype=torch.float32)
    for _ in range(10):
        y = fused_ln_gelu(x, gamma, beta)
    torch.cuda.synchronize()
    print("Fused LN+GELU profiling complete")

else:
    print(f"Unknown kernel: {kernel_name}")
    sys.exit(1)