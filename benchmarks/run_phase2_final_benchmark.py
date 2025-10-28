"""
Phase 2 – Individual Kernel Benchmark + Nsight Profiling
---------------------------------------------------------
Runs LayerNorm, GELU, Swish, and Fused LN+GELU
on MNIST tensors using both CUDA & Triton kernels.

Outputs:
  profiling/phase2_results.json
  profiling/phase2_results.csv
  profiling/phase2_results.xlsx
  profiling/nsight/*.qdrep / *.ncu-rep
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch, torchvision, torchvision.transforms as transforms
import time, json, os, subprocess
import pandas as pd
from ops.cuda_ops import layernorm as cuda_ln, gelu as cuda_gelu, swish as cuda_swish, fused_ln_gelu as cuda_ln_gelu
from ops.triton_ops import layernorm as tri_ln, gelu as tri_gelu, swish as tri_swish, fused_ln_gelu as tri_ln_gelu

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
ITERS = 200
WARMUP = 20
SAVE_DIR = "profiling"
NSIGHT_DIR = os.path.join(SAVE_DIR, "nsight")
os.makedirs(NSIGHT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)
imgs, _ = next(iter(trainloader))
x_mnist = imgs.to(device).view(imgs.size(0), -1)  # [B, 784]
rows, cols = x_mnist.shape
gamma = torch.ones(cols, device=device)
beta = torch.zeros(cols, device=device)

# ----------------------------------------------------------------------
# OPS DICTIONARY
# ----------------------------------------------------------------------
OPS = {
    "layernorm": {"cuda": lambda: cuda_ln(x_mnist, gamma, beta),
                  "triton": lambda: tri_ln(x_mnist, gamma, beta)},
    "gelu": {"cuda": lambda: cuda_gelu(x_mnist),
             "triton": lambda: tri_gelu(x_mnist)},
    "swish": {"cuda": lambda: cuda_swish(x_mnist),
              "triton": lambda: tri_swish(x_mnist)},
    "ln_gelu": {"cuda": lambda: cuda_ln_gelu(x_mnist, gamma, beta),
                "triton": lambda: tri_ln_gelu(x_mnist, gamma, beta)}
}

# ----------------------------------------------------------------------
# UTILITY: PROFILE WITH NSIGHT
# ----------------------------------------------------------------------
def run_nsight(op, impl):
    base = f"{NSIGHT_DIR}/{op}_{impl}"
    print(f"⚙️  Nsight profiling {op} ({impl}) ...")
    # Nsight Systems
    subprocess.run([
        "nsys", "profile", "--force-overwrite=true", "-t", "cuda", "-o", base,
        "python3", "benchmarks/run_phase2_individuals_mnist.py",
        "--only", op, "--impl", impl
    ], check=False)

    # Nsight Compute
    subprocess.run([
        "ncu", "--set", "full", "--target-processes", "all",
        "--force-overwrite=true", "--export", base,
        "python3", "benchmarks/run_phase2_individuals_mnist.py",
        "--only", op, "--impl", impl
    ], check=False) 

    return {
        "nsys_report": f"{base}.qdrep",
        "ncu_report":  f"{base}.ncu-rep"
    }

# ----------------------------------------------------------------------
# RUN BENCHMARK
# ----------------------------------------------------------------------
def run_kernel(op, impl):
    fn = OPS[op][impl]

    # warmup
    for _ in range(WARMUP):
        _ = fn()
    torch.cuda.synchronize()

    # timing
    t0 = time.time()
    for _ in range(ITERS):
        _ = fn()
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = 1000 * (t1 - t0) / ITERS
    result = {
        "op": op,
        "impl": impl,
        "batch": BATCH,
        "cols": cols,
        "iters": ITERS,
        "avg_ms_per_iter": round(avg_ms, 4),
        "device": torch.cuda.get_device_name(0)
    }

    # Attach Nsight report paths (comment out if Nsight not available)
    result.update(run_nsight(op, impl))
    return result

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    results = []
    for op in OPS.keys():
        for impl in ["cuda", "triton"]:
            print(f"\n🚀 Running {op} ({impl}) on GPU ...")
            res = run_kernel(op, impl)
            print(res)
            results.append(res)

    # Save JSON
    os.makedirs(SAVE_DIR, exist_ok=True)
    json_path = f"{SAVE_DIR}/phase2_results.json"
    json.dump(results, open(json_path, "w"), indent=2)

    # Save CSV + Excel
    df = pd.DataFrame(results)
    csv_path = f"{SAVE_DIR}/phase2_results.csv"
    xlsx_path = f"{SAVE_DIR}/phase2_results.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"\n✅ All results saved:\n  {json_path}\n  {csv_path}\n  {xlsx_path}")
    print(f"📁 Nsight reports stored in: {NSIGHT_DIR}/")
