#!/usr/bin/env python3
"""Correctness test for Fusion 2: LayerNorm + Swish + Dropout"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from triton_kernels.triton_ops import triton_fused_ln_swish_dropout
import cuda_ops

def test_fusion2_correctness():
    print("\n" + "="*80)
    print("  FUSION 2 CORRECTNESS TEST: LayerNorm + Swish + Dropout")
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
        
        # === PyTorch reference (unfused, no dropout for numerical comparison) ===
        with torch.no_grad():
            # LayerNorm
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + eps)
            x_scaled = x_norm * gamma + beta
            
            # Swish
            y_ref = x_scaled * torch.sigmoid(x_scaled)
        
        # === Triton fused (no dropout for comparison) ===
        y_triton, mask_triton = triton_fused_ln_swish_dropout(
            x, gamma, beta, 
            dropout_p=0.0,
            training=False,
            seed=42,
            eps=eps
        )
        
        # === CUDA fused (no dropout for comparison) ===
        y_cuda, mask_cuda = cuda_ops.cuda_fused_ln_swish_dropout(
            x, gamma, beta,
            0.0,  # dropout_p
            42,   # seed
            eps
        )
        
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
    
    # === Test dropout functionality ===
    print("="*80)
    print("Testing Dropout Functionality")
    print("="*80)
    
    x = torch.randn(1000, 512, device=device)
    gamma = torch.ones(512, device=device)
    beta = torch.zeros(512, device=device)
    
    dropout_rates = [0.1, 0.3, 0.5]
    
    for target_rate in dropout_rates:
        print(f"\nTarget dropout rate: {target_rate*100:.0f}%")
        
        # Triton
        _, mask_triton = triton_fused_ln_swish_dropout(
            x, gamma, beta,
            dropout_p=target_rate,
            training=True,
            seed=42
        )
        actual_triton = (mask_triton == 0).float().mean().item()
        
        # CUDA
        _, mask_cuda = cuda_ops.cuda_fused_ln_swish_dropout(
            x, gamma, beta,
            target_rate,
            42,
            eps
        )
        actual_cuda = (mask_cuda == 0).float().mean().item()
        
        print(f"  Triton actual: {actual_triton*100:.1f}%")
        print(f"  CUDA actual:   {actual_cuda*100:.1f}%")
        
        # Allow 5% tolerance
        triton_ok = abs(actual_triton - target_rate) < 0.05
        cuda_ok = abs(actual_cuda - target_rate) < 0.05
        
        if triton_ok and cuda_ok:
            print(f"  ✅ Both within tolerance")
        else:
            print(f"  ⚠️  Warning: Dropout rate differs from target")
            if not triton_ok:
                print(f"     Triton: {abs(actual_triton - target_rate)*100:.1f}% off")
            if not cuda_ok:
                print(f"     CUDA: {abs(actual_cuda - target_rate)*100:.1f}% off")
    
    print("\n" + "="*80)
    if all_passed:
        print("  ✅ ALL CORRECTNESS TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED - SEE ABOVE")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = test_fusion2_correctness()
    sys.exit(0 if success else 1)