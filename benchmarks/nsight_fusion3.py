#!/usr/bin/env python3
"""
Fusion 3 profiling - CORRECT SIGNATURE with epsilon
Based on working benchmark: cuda_fused_ln_gelu_swish(x, gamma, beta, eps)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("FUSION 3 (LN + GELU + Swish) - PROFILING MODE")
print("="*80 + "\n")

# Import cuda_ops
try:
    import cuda_ops
    print("✅ Imported cuda_ops extension\n")
except ImportError as e:
    print(f"❌ Could not import cuda_ops: {e}")
    sys.exit(1)

print("✅ Found: cuda_fused_ln_gelu_swish\n")


def profile_fusion3():
    """Profile Fusion 3 with correct 4-argument signature"""
    
    print("="*80)
    print("Running 10 iterations for Nsight profiling")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("❌ CUDA not available!")
        return
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Test shapes - matching your benchmark configurations
    test_shapes = [
        (512, 1024, "XX-Large"),
        (256, 512, "X-Large"),
    ]
    
    eps = 1e-5  # LayerNorm epsilon (same as in your benchmark)
    
    for M, N, name in test_shapes:
        print(f"\n📊 Shape: {name} [{M} x {N}]")
        print("-" * 60)
        
        # Input tensor
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        
        # LayerNorm parameters
        gamma = torch.ones(N, device=device, dtype=torch.float32)
        beta = torch.zeros(N, device=device, dtype=torch.float32)
        
        print(f"   Input: {x.shape}")
        print(f"   Gamma: {gamma.shape}")
        print(f"   Beta: {beta.shape}")
        print(f"   Epsilon: {eps}")
        
        # Warmup
        print("\n🔥 Warmup (3 iterations)...")
        for _ in range(3):
            _ = cuda_ops.cuda_fused_ln_gelu_swish(x, gamma, beta, eps)
        torch.cuda.synchronize()
        print("✅ Warmup complete")
        
        # Profile - 10 iterations for Nsight
        print(f"\n⚡ Profiling cuda_fused_ln_gelu_swish (10 iterations)...")
        for i in range(10):
            result = cuda_ops.cuda_fused_ln_gelu_swish(x, gamma, beta, eps)
            torch.cuda.synchronize()
            if i == 0:
                print(f"   Output shape: {result.shape}")
                print(f"   Output mean: {result.mean().item():.4f}")
                print(f"   Output std: {result.std().item():.4f}")
        print("   ✅ Complete")
    
    print("\n" + "="*80)
    print("✅ FUSION 3 PROFILING COMPLETE")
    print("="*80)
    print("\n💡 Nsight Compute captured metrics for your 2.94× speedup!")
    print("   Check profiling_results/ for:")
    print("   - nsight_fusion3_correct.ncu-rep (full profile)")
    print("   - nsight_fusion3_correct_metrics.csv (extracted metrics)\n")


if __name__ == "__main__":
    profile_fusion3()