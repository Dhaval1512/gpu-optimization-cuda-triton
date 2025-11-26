import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from triton_kernels.triton_ops import triton_fused_gelu_swish
import cuda_ops

def test_correctness():
    print("\n" + "="*80)
    print("  CORRECTNESS TEST: FUSED GELU+SWISH ON MNIST DATA")
    print("="*80 + "\n")
    
    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Get a batch of real MNIST data
    data, labels = next(iter(test_loader))
    data = data.cuda()
    
    print(f"Testing on real MNIST batch:")
    print(f"  Shape: {data.shape}")
    print(f"  Size: {data.numel():,} elements")
    print(f"  Min: {data.min().item():.4f}, Max: {data.max().item():.4f}\n")
    
    # Reference implementation (PyTorch unfused)
    def pytorch_gelu_swish(x):
        gelu = F.gelu(x)
        swish = x * torch.sigmoid(x)
        return gelu + swish
    
    output_ref = pytorch_gelu_swish(data)
    
    # Triton fused
    output_triton = triton_fused_gelu_swish(data)
    
    # CUDA fused
    output_cuda = cuda_ops.cuda_fused_gelu_swish(data)
    
    # Compare results
    print("Correctness Check:")
    print("-" * 80)
    
    # Triton vs PyTorch
    triton_max_err = torch.max(torch.abs(output_triton - output_ref)).item()
    triton_mean_err = torch.mean(torch.abs(output_triton - output_ref)).item()
    triton_rel_err = torch.mean(torch.abs((output_triton - output_ref) / (output_ref + 1e-8))).item()
    
    print(f"Triton vs PyTorch:")
    print(f"  Max absolute error:  {triton_max_err:.2e}")
    print(f"  Mean absolute error: {triton_mean_err:.2e}")
    print(f"  Mean relative error: {triton_rel_err:.2e}")
    
    # CUDA vs PyTorch
    cuda_max_err = torch.max(torch.abs(output_cuda - output_ref)).item()
    cuda_mean_err = torch.mean(torch.abs(output_cuda - output_ref)).item()
    cuda_rel_err = torch.mean(torch.abs((output_cuda - output_ref) / (output_ref + 1e-8))).item()
    
    print(f"\nCUDA vs PyTorch:")
    print(f"  Max absolute error:  {cuda_max_err:.2e}")
    print(f"  Mean absolute error: {cuda_mean_err:.2e}")
    print(f"  Mean relative error: {cuda_rel_err:.2e}")
    
    # Assertions
    tolerance = 1e-4
    assert triton_max_err < tolerance, f"Triton error too large: {triton_max_err}"
    assert cuda_max_err < tolerance, f"CUDA error too large: {cuda_max_err}"
    
    print("\n" + "="*80)
    print("  ✅ ALL CORRECTNESS TESTS PASSED!")
    print("="*80 + "\n")
    
    # Test on multiple batches
    print("Testing on 10 random MNIST batches...")
    print("-" * 80)
    
    passed = 0
    for i, (data, _) in enumerate(test_loader):
        if i >= 10:
            break
        
        data = data.cuda()
        
        ref = pytorch_gelu_swish(data)
        triton_out = triton_fused_gelu_swish(data)
        cuda_out = cuda_ops.cuda_fused_gelu_swish(data)
        
        triton_err = torch.max(torch.abs(triton_out - ref)).item()
        cuda_err = torch.max(torch.abs(cuda_out - ref)).item()
        
        if triton_err < tolerance and cuda_err < tolerance:
            passed += 1
            print(f"  Batch {i+1}: ✅ PASSED (Triton: {triton_err:.2e}, CUDA: {cuda_err:.2e})")
        else:
            print(f"  Batch {i+1}: ❌ FAILED (Triton: {triton_err:.2e}, CUDA: {cuda_err:.2e})")
    
    print(f"\nPassed {passed}/10 batches")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_correctness()