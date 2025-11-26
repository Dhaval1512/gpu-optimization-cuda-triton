#!/usr/bin/env python3
"""Kernel-level benchmarks for Fusion 2"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from triton_kernels.triton_ops import triton_fused_ln_swish_dropout
import cuda_ops
import csv
import statistics as stats

def benchmark(fn, *args, warmup=10, iters=100):
    """Benchmark a function using CUDA events"""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return stats.mean(times), stats.pstdev(times)

def main():
    print("\n" + "="*80)
    print("  FUSION 2 KERNEL BENCHMARK: LayerNorm + Swish + Dropout")
    print("="*80 + "\n")
    
    device = 'cuda'
    results = []
    
    # Test configurations
    configs = [
        (32, 64, "Small"),
        (64, 128, "Medium"),
        (128, 256, "Large"),
        (256, 512, "X-Large"),
        (512, 1024, "XX-Large"),
    ]
    
    for M, N, name in configs:
        print(f"Testing {name}: [{M}, {N}]")
        print("-" * 80)
        
        x = torch.randn(M, N, device=device)
        gamma = torch.ones(N, device=device)
        beta = torch.zeros(N, device=device)
        
        # === PyTorch unfused ===
        def pytorch_unfused(x, gamma, beta):
            x_norm = F.layer_norm(x, (N,), gamma, beta)
            x_swish = x_norm * torch.sigmoid(x_norm)
            return F.dropout(x_swish, p=0.1, training=True)
        
        mean_pt, std_pt = benchmark(pytorch_unfused, x, gamma, beta)
        print(f"  PyTorch (unfused):  {mean_pt:>8.4f} ± {std_pt:>6.4f} ms")
        results.append(("ln_swish_dropout", "pytorch_unfused", f"{M}x{N}", M*N, mean_pt, std_pt))
        
        # === Triton fused ===
        mean_tr, std_tr = benchmark(
            lambda: triton_fused_ln_swish_dropout(x, gamma, beta, dropout_p=0.1, training=True)[0]
        )
        speedup_tr = mean_pt / mean_tr
        print(f"  Triton (fused):     {mean_tr:>8.4f} ± {std_tr:>6.4f} ms  [{speedup_tr:.2f}× speedup]")
        results.append(("ln_swish_dropout", "triton_fused", f"{M}x{N}", M*N, mean_tr, std_tr))
        
        # === CUDA fused ===
        mean_cu, std_cu = benchmark(
            lambda: cuda_ops.cuda_fused_ln_swish_dropout(x, gamma, beta, 0.1, 42, 1e-5)[0]
        )
        speedup_cu = mean_pt / mean_cu
        print(f"  CUDA (fused):       {mean_cu:>8.4f} ± {std_cu:>6.4f} ms  [{speedup_cu:.2f}× speedup]")
        results.append(("ln_swish_dropout", "cuda_fused", f"{M}x{N}", M*N, mean_cu, std_cu))
        
        print(f"\n  Best speedup: {max(speedup_tr, speedup_cu):.2f}×\n")
    
    # Save results
    os.makedirs("report", exist_ok=True)
    with open("report/fusion2_kernel_benchmarks.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "implementation", "shape", "elements", "mean_ms", "std_ms"])
        writer.writerows(results)
    
    print("="*80)
    print("✅ Results saved to: report/fusion2_kernel_benchmarks.csv")
    print("="*80 + "\n")
    
    # Summary table
    print("SUMMARY:")
    print("="*80)
    print(f"{'Shape':<15} {'PyTorch':<12} {'Triton':<12} {'CUDA':<12} {'Best Speedup':<15}")
    print("-"*80)
    
    for i in range(0, len(results), 3):
        shape = results[i][2]
        pt = results[i][4]
        tr = results[i+1][4]
        cu = results[i+2][4]
        best = max(pt/tr, pt/cu)
        print(f"{shape:<15} {pt:>8.4f} ms  {tr:>8.4f} ms  {cu:>8.4f} ms  {best:>10.2f}×")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()