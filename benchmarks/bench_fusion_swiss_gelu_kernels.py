import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from triton_kernels.triton_ops import triton_fused_gelu_swish
import cuda_ops
import csv
import statistics as stats

def benchmark_kernel(fn, *args, warmup=10, iters=100):
    """Benchmark a kernel function"""
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return stats.mean(times), stats.pstdev(times)

def main():
    print("\n" + "="*80)
    print("  KERNEL BENCHMARK: GELU+SWISH FUSION ON MNIST DATA")
    print("="*80 + "\n")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Reference function
    def pytorch_gelu_swish(x):
        gelu = F.gelu(x)
        swish = x * torch.sigmoid(x)
        return gelu + swish
    
    results = []
    
    # Test across different batch sizes (real MNIST data)
    batch_sizes = [16, 32, 64, 128, 256, 512]
    
    for batch_size in batch_sizes:
        print(f"{'='*80}")
        print(f"Testing Batch Size: {batch_size}")
        print(f"{'='*80}")
        
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        data, _ = next(iter(loader))
        data = data.cuda()
        
        print(f"  Tensor shape: {data.shape}")
        print(f"  Elements: {data.numel():,}\n")
        
        # PyTorch unfused
        mean_pt, std_pt = benchmark_kernel(pytorch_gelu_swish, data)
        print(f"  PyTorch (unfused):  {mean_pt:>8.4f} ± {std_pt:>6.4f} ms")
        results.append(("gelu_swish", "pytorch", batch_size, data.numel(), mean_pt, std_pt))
        
        # Triton fused
        mean_tr, std_tr = benchmark_kernel(triton_fused_gelu_swish, data)
        speedup_tr = mean_pt / mean_tr
        print(f"  Triton (fused):     {mean_tr:>8.4f} ± {std_tr:>6.4f} ms  [{speedup_tr:.2f}× speedup]")
        results.append(("gelu_swish", "triton_fused", batch_size, data.numel(), mean_tr, std_tr))
        
        # CUDA fused
        mean_cu, std_cu = benchmark_kernel(cuda_ops.cuda_fused_gelu_swish, data)
        speedup_cu = mean_pt / mean_cu
        print(f"  CUDA (fused):       {mean_cu:>8.4f} ± {std_cu:>6.4f} ms  [{speedup_cu:.2f}× speedup]")
        results.append(("gelu_swish", "cuda_fused", batch_size, data.numel(), mean_cu, std_cu))
        
        print(f"\n  Best speedup: {max(speedup_tr, speedup_cu):.2f}×\n")
    
    # Save results
    os.makedirs("report", exist_ok=True)
    with open("report/kernel_fusion1_mnist.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "implementation", "batch_size", "elements", "mean_ms", "std_ms"])
        writer.writerows(results)
    
    print("="*80)
    print("✅ Results saved to: report/kernel_fusion1_mnist.csv")
    print("="*80 + "\n")
    
    # Summary
    print("SUMMARY:")
    print("="*80)
    print(f"{'Batch':<10} {'Elements':<12} {'PyTorch':<12} {'Triton':<12} {'CUDA':<12} {'Best':<10}")
    print("-"*80)
    
    for i in range(0, len(results), 3):
        bs = results[i][2]
        elem = results[i][3]
        pt = results[i][4]
        tr = results[i+1][4]
        cu = results[i+2][4]
        best = max(pt/tr, pt/cu)
        print(f"{bs:<10} {elem:<12,} {pt:>8.4f} ms {tr:>8.4f} ms {cu:>8.4f} ms {best:>6.2f}×")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()