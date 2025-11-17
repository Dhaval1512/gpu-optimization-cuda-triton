import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import csv
import statistics as stats
import torch.nn.functional as F
from extensions.cuda_ops import gelu as cuda_gelu, swish as cuda_swish, layernorm as cuda_layernorm, fused_ln_gelu as cuda_fused_ln_gelu
from triton_kernels.triton_ops import triton_gelu, triton_swish, triton_layernorm, triton_fused_ln_gelu

DEVICE = "cuda"
WARMUP = 10
ITERS = 50

def benchmark(fn, *args):
    torch.cuda.synchronize()
    for _ in range(WARMUP): fn(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    
    for _ in range(ITERS):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return stats.mean(times), stats.pstdev(times)

results = []

print("\n" + "="*80)
print("  COMPREHENSIVE WORKLOAD VARIATION BENCHMARKS")
print("  Testing: Batch Size × Sequence Length × Hidden Dimension")
print("="*80 + "\n")

# Configuration matrix
batch_sizes = [16, 32, 64, 128]
sequence_lengths = [256, 512, 1024]
hidden_dims = [128, 256, 512, 768]

total_tests = len(batch_sizes) * len(sequence_lengths) * len(hidden_dims) * 4 * 3  # 4 ops × 3 impls
current_test = 0

for bs in batch_sizes:
    for seq_len in sequence_lengths:
        for hidden_dim in hidden_dims:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Batch={bs}, Seq={seq_len}, Dim={hidden_dim}")
            
            # Create tensors
            x_3d = torch.randn(bs, seq_len, hidden_dim, device=DEVICE)
            x_flat = x_3d.reshape(-1)  # For elementwise ops
            x_2d = x_3d.reshape(-1, hidden_dim)  # For LayerNorm
            
            gamma = torch.ones(hidden_dim, device=DEVICE)
            beta = torch.zeros(hidden_dim, device=DEVICE)
            
            # ========== GELU ==========
            print("  Testing GELU...")
            mean_ms, std_ms = benchmark(lambda: F.gelu(x_flat))
            results.append(("gelu", "pytorch", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: cuda_gelu(x_flat))
            results.append(("gelu", "cuda", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: triton_gelu(x_flat))
            results.append(("gelu", "triton", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            # ========== Swish ==========
            print("  Testing Swish...")
            mean_ms, std_ms = benchmark(lambda: x_flat * torch.sigmoid(x_flat))
            results.append(("swish", "pytorch", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: cuda_swish(x_flat))
            results.append(("swish", "cuda", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: triton_swish(x_flat))
            results.append(("swish", "triton", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            # ========== LayerNorm ==========
            print("  Testing LayerNorm...")
            mean_ms, std_ms = benchmark(lambda: F.layer_norm(x_2d, (hidden_dim,), gamma, beta))
            results.append(("layernorm", "pytorch", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: cuda_layernorm(x_2d, gamma, beta))
            results.append(("layernorm", "cuda", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: triton_layernorm(x_2d, gamma, beta))
            results.append(("layernorm", "triton", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            # ========== Fused LN+GELU ==========
            print("  Testing Fused LN+GELU...")
            def torch_fused(x, g, b):
                return F.gelu(F.layer_norm(x, (hidden_dim,), g, b))
            
            mean_ms, std_ms = benchmark(torch_fused, x_2d, gamma, beta)
            results.append(("fused_ln_gelu", "pytorch", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: cuda_fused_ln_gelu(x_2d, gamma, beta))
            results.append(("fused_ln_gelu", "cuda", bs, seq_len, hidden_dim, mean_ms, std_ms))
            
            mean_ms, std_ms = benchmark(lambda: triton_fused_ln_gelu(x_2d, gamma, beta))
            results.append(("fused_ln_gelu", "triton", bs, seq_len, hidden_dim, mean_ms, std_ms))

print("\n" + "="*80)
print("  SAVING RESULTS...")
print("="*80 + "\n")

os.makedirs("report", exist_ok=True)
with open("report/workload_variations.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["operation", "implementation", "batch_size", "seq_length", "hidden_dim", "mean_ms", "std_ms"])
    writer.writerows(results)

print(f"✅ Results saved to: report/workload_variations.csv")
print(f"   Total rows: {len(results)}")
print("\nDone!\n")