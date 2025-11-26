#!/usr/bin/env python3
"""CNN inference benchmark for Fusion 3: LayerNorm + GELU + Swish"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fused_models.model_fused_ln_swiss_gelu import Fusion3CNN
import csv
import statistics as stats

def benchmark_model(model, data_loader, warmup=50, iters=200):
    """Benchmark model inference on MNIST"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= warmup:
                break
            data = data.cuda()
            _ = model(data)
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= iters:
                break
            
            data = data.cuda()
            
            start_event.record()
            _ = model(data)
            end_event.record()
            
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    
    # Remove outliers
    times_sorted = sorted(times)
    trim = len(times) // 20
    times_trimmed = times_sorted[trim:-trim] if trim > 0 else times_sorted
    
    return stats.mean(times_trimmed), stats.pstdev(times_trimmed)

def main():
    print("\n" + "="*80)
    print("  FUSION 3 CNN INFERENCE BENCHMARK ON MNIST")
    print("  LayerNorm + GELU + Swish")
    print("="*80 + "\n")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    results = []
    
    for batch_size in batch_sizes:
        print(f"{'='*80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*80}")
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # PyTorch unfused
        print("\n1. PyTorch (Unfused)")
        model_unfused = Fusion3CNN(use_fusion=False).cuda().eval()
        mean_unfused, std_unfused = benchmark_model(model_unfused, test_loader)
        print(f"   Time: {mean_unfused:>8.4f} ± {std_unfused:>6.4f} ms/batch")
        results.append(("fusion3_cnn", "pytorch_unfused", batch_size, mean_unfused, std_unfused))
        
        # Triton fused
        print("\n2. Triton (Fused)")
        model_triton = Fusion3CNN(use_fusion=True, backend='triton').cuda().eval()
        mean_triton, std_triton = benchmark_model(model_triton, test_loader)
        speedup_triton = mean_unfused / mean_triton
        print(f"   Time: {mean_triton:>8.4f} ± {std_triton:>6.4f} ms/batch")
        print(f"   Speedup: {speedup_triton:.2f}×")
        results.append(("fusion3_cnn", "triton_fused", batch_size, mean_triton, std_triton))
        
        # CUDA fused
        print("\n3. CUDA (Fused)")
        model_cuda = Fusion3CNN(use_fusion=True, backend='cuda').cuda().eval()
        mean_cuda, std_cuda = benchmark_model(model_cuda, test_loader)
        speedup_cuda = mean_unfused / mean_cuda
        print(f"   Time: {mean_cuda:>8.4f} ± {std_cuda:>6.4f} ms/batch")
        print(f"   Speedup: {speedup_cuda:.2f}×")
        results.append(("fusion3_cnn", "cuda_fused", batch_size, mean_cuda, std_cuda))
        
        print(f"\nBest speedup: {max(speedup_triton, speedup_cuda):.2f}×\n")
    
    # Save results
    os.makedirs("report", exist_ok=True)
    with open("report/fusion3_cnn_inference.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "implementation", "batch_size", "mean_ms", "std_ms"])
        writer.writerows(results)
    
    print("="*80)
    print("✅ Results saved to: report/fusion_ln_swiss_gelu_cnn_inference.csv")
    print("="*80 + "\n")
    
    # Summary
    print("SUMMARY:")
    print("="*80)
    print(f"{'Batch':<10} {'PyTorch':<15} {'Triton':<15} {'CUDA':<15} {'Best Speedup':<15}")
    print("-"*80)
    
    for i in range(0, len(results), 3):
        bs = results[i][2]
        pt = results[i][3]
        tr = results[i+1][3]
        cu = results[i+2][3]
        best = max(pt/tr, pt/cu)
        print(f"{bs:<10} {pt:>10.4f} ms  {tr:>10.4f} ms  {cu:>10.4f} ms  {best:>10.2f}×")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()