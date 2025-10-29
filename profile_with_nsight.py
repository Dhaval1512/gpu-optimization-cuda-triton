"""
Simple kernel runner for Nsight profiling
Run this with Nsight Systems or Nsight Compute wrapper
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import argparse
from ops.cuda_ops import (
    layernorm as cuda_ln,
    gelu as cuda_gelu,
    swish as cuda_swish,
    fused_ln_gelu as cuda_ln_gelu
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["layernorm", "gelu", "swish", "ln_gelu"], required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--cols", type=int, default=784)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"Profiling {args.op} on {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create test data
    x = torch.randn(args.batch, args.cols, device=device, dtype=torch.float32)
    gamma = torch.ones(args.cols, device=device, dtype=torch.float32)
    beta = torch.zeros(args.cols, device=device, dtype=torch.float32)

    # Select operation
    ops = {
        "layernorm": lambda: cuda_ln(x, gamma, beta),
        "gelu": lambda: cuda_gelu(x),
        "swish": lambda: cuda_swish(x),
        "ln_gelu": lambda: cuda_ln_gelu(x, gamma, beta)
    }
    
    fn = ops[args.op]
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = fn()
    torch.cuda.synchronize()
    
    # Main profiling loop
    print(f"Running {args.iters} iterations...")
    for i in range(args.iters):
        _ = fn()
        if i % 20 == 0:
            print(f"  Progress: {i}/{args.iters}")
    torch.cuda.synchronize()
    
    print(f"✅ Completed {args.iters} iterations")

if __name__ == "__main__":
    main()