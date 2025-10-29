"""
Phase 2 – CUDA-Only Kernel Benchmark (Windows Compatible)
---------------------------------------------------------
Runs LayerNorm, GELU, Swish, and Fused LN+GELU on MNIST tensors 
using CUDA kernels only (skips Triton for Windows compatibility).

Outputs:
  profiling/phase2_cuda_results.json
  profiling/phase2_cuda_results.csv
  profiling/phase2_cuda_results.xlsx
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torchvision, torchvision.transforms as transforms
import time, json
import pandas as pd
from ops.cuda_ops import (
    layernorm as cuda_ln, 
    gelu as cuda_gelu, 
    swish as cuda_swish, 
    fused_ln_gelu as cuda_ln_gelu
)

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
ITERS = 200
WARMUP = 20
SAVE_DIR = "profiling"

# ----------------------------------------------------------------------
# RUN BENCHMARK
# ----------------------------------------------------------------------
def run_kernel(op, impl, x_mnist, gamma, beta, ops_dict):
    fn = ops_dict[op][impl]

    # warmup
    print(f"  ⏳ Warming up...")
    for _ in range(WARMUP):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timing
    print(f"  ⚡ Running {ITERS} iterations...")
    t0 = time.time()
    for _ in range(ITERS):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = 1000 * (t1 - t0) / ITERS
    result = {
        "op": op,
        "impl": impl,
        "batch": BATCH,
        "cols": x_mnist.shape[1],
        "iters": ITERS,
        "avg_ms_per_iter": round(avg_ms, 4),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }
    
    return result

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------------
    # DATASET
    # ----------------------------------------------------------------------
    print("\n📊 Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # FIXED: num_workers=0 to avoid multiprocessing issues on Windows
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=0)
    imgs, _ = next(iter(trainloader))
    x_mnist = imgs.to(device).view(imgs.size(0), -1)  # [B, 784]
    rows, cols = x_mnist.shape
    gamma = torch.ones(cols, device=device)
    beta = torch.zeros(cols, device=device)

    print(f"✅ Data shape: {x_mnist.shape} (batch={BATCH}, features={cols})")

    # ----------------------------------------------------------------------
    # OPS DICTIONARY (CUDA ONLY)
    # ----------------------------------------------------------------------
    ops_dict = {
        "layernorm": {"cuda": lambda: cuda_ln(x_mnist, gamma, beta)},
        "gelu": {"cuda": lambda: cuda_gelu(x_mnist)},
        "swish": {"cuda": lambda: cuda_swish(x_mnist)},
        "ln_gelu": {"cuda": lambda: cuda_ln_gelu(x_mnist, gamma, beta)}
    }

    results = []
    
    print("\n" + "="*70)
    print("🚀 Starting CUDA Kernel Benchmarks")
    print("="*70)
    
    for op in ops_dict.keys():
        print(f"\n{'─'*70}")
        print(f"📌 Operation: {op.upper()}")
        print(f"{'─'*70}")
        
        for impl in ["cuda"]:
            res = run_kernel(op, impl, x_mnist, gamma, beta, ops_dict)
            print(f"✅ {op}/{impl}: {res['avg_ms_per_iter']:.4f} ms per iteration")
            results.append(res)

    # Save JSON
    json_path = f"{SAVE_DIR}/phase2_cuda_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV + Excel
    df = pd.DataFrame(results)
    csv_path = f"{SAVE_DIR}/phase2_cuda_results.csv"
    xlsx_path = f"{SAVE_DIR}/phase2_cuda_results.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"  • JSON:  {json_path}")
    print(f"  • CSV:   {csv_path}")
    print(f"  • Excel: {xlsx_path}")
    print("\n📊 Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()