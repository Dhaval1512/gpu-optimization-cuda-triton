#!/usr/bin/env python3
"""Correctness test for Fusion 3: LayerNorm + GELU + Swish"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from triton_kernels.triton_ops import triton_fused_ln_gelu_swish
import cuda_ops

def test_fusion3_correctness():
    print("\n" + "="*80)
    print("  FUSION 3 CORRECTNESS TEST: LayerNorm + GELU + Swish")
    print("="*80 + "\n")
    
    device = 'cuda'
    eps = 1e-5
    
    # Test configurations
    configs = [
        (4, 8, "Small"),
        (32, 64, "Medium"),
        (128, 256, "Large"),
        (256, 512, "Extra Large"),
    ]
    
    all_passed = True
    
    for M, N, name in configs:
        print(f"Testing {name}: [{M}, {N}]")
        print("-" * 80)
        
        # Create test data
        torch.manual_seed(42)
        x = torch.randn(M, N, device=device)
        gamma = torch.ones(N, device=device)
        beta = torch.zeros(N, device=device)
        
        # === PyTorch reference (unfused) ===
        with torch.no_grad():
            # LayerNorm
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + eps)
            x_ln = x_norm * gamma + beta
            
            # GELU
            x_gelu = F.gelu(x_ln)
            
            # Swish
            x_swish = x_ln * torch.sigmoid(x_ln)
            
            # Combine
            y_ref = x_gelu + x_swish
        
        # === Triton fused ===
        y_triton = triton_fused_ln_gelu_swish(x, gamma, beta, eps=eps)
        
        # === CUDA fused ===
        y_cuda = cuda_ops.cuda_fused_ln_gelu_swish(x, gamma, beta, eps)
        
        # === Compare Triton vs PyTorch ===
        triton_max_diff = torch.max(torch.abs(y_triton - y_ref)).item()
        triton_mean_diff = torch.mean(torch.abs(y_triton - y_ref)).item()
        
        print(f"  Triton vs PyTorch:")
        print(f"    Max difference:  {triton_max_diff:.2e}")
        print(f"    Mean difference: {triton_mean_diff:.2e}")
        
        # === Compare CUDA vs PyTorch ===
        cuda_max_diff = torch.max(torch.abs(y_cuda - y_ref)).item()
        cuda_mean_diff = torch.mean(torch.abs(y_cuda - y_ref)).item()
        
        print(f"  CUDA vs PyTorch:")
        print(f"    Max difference:  {cuda_max_diff:.2e}")
        print(f"    Mean difference: {cuda_mean_diff:.2e}")
        
        # === Tolerance check ===
        tolerance = 1e-4
        triton_pass = triton_max_diff < tolerance
        cuda_pass = cuda_max_diff < tolerance
        
        if triton_pass and cuda_pass:
            print(f"  ✅ PASSED\n")
        else:
            print(f"  ❌ FAILED")
            if not triton_pass:
                print(f"    Triton difference {triton_max_diff} exceeds tolerance {tolerance}")
            if not cuda_pass:
                print(f"    CUDA difference {cuda_max_diff} exceeds tolerance {tolerance}")
            print()
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("  ✅ ALL CORRECTNESS TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED - SEE ABOVE")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = test_fusion3_correctness()
    sys.exit(0 if success else 1)