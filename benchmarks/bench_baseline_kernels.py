import os
import math
import csv
import statistics as stats

import torch
import torch.nn.functional as F

# ---- Make sure project root is on sys.path ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.cuda_ops import (
    gelu as cuda_gelu,
    swish as cuda_swish,
    layernorm as cuda_layernorm,
    fused_ln_gelu as cuda_fused_ln_gelu,
)

from triton_kernels.triton_ops import (
    triton_gelu,
    triton_swish,
    triton_layernorm,
    triton_fused_ln_gelu,
)


DEVICE = "cuda"
DTYPE = torch.float32

WARMUP_ITERS = 10
BENCH_ITERS = 100

OUTPUT_CSV = "report/kernel_benchmarks.csv"  # will be created if not exists


def benchmark_kernel(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a single kernel using CUDA events. Returns list of times in ms."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []

    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))  # in milliseconds

    return times_ms


def summarize_times(times_ms):
    mean_ms = stats.mean(times_ms)
    std_ms = stats.pstdev(times_ms)
    return mean_ms, std_ms


def ensure_report_dir(path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def main():
    print("\n=======================================")
    print("    KERNEL BENCHMARKS (CUDA vs Triton)")
    print("=======================================\n")

    assert torch.cuda.is_available(), "CUDA is not available!"

    results = []

    # ----------------- Config: Shapes -----------------
    # You can tweak these if you want to simulate larger transformer shapes
    N_ELEMS = 1 << 20  # 1,048,576 elements for GELU / Swish
    LN_ROWS, LN_COLS = 4096, 1024  # LayerNorm & fused LN+GELU

    # ----------- Allocate common tensors -----------
    # Elementwise tensors
    x_elem = torch.randn(N_ELEMS, device=DEVICE, dtype=DTYPE)

    # LayerNorm tensors
    x_ln = torch.randn(LN_ROWS, LN_COLS, device=DEVICE, dtype=DTYPE)
    gamma = torch.ones(LN_COLS, device=DEVICE, dtype=DTYPE)
    beta = torch.zeros(LN_COLS, device=DEVICE, dtype=DTYPE)

    # ===================== GELU =====================
    print("Benchmarking GELU...")

    # PyTorch
    def torch_gelu(x):
        return F.gelu(x)

    times = benchmark_kernel(torch_gelu, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  PyTorch GELU:  mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("gelu", "pytorch", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    # CUDA
    times = benchmark_kernel(cuda_gelu, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  CUDA GELU:     mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("gelu", "cuda", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    # Triton
    times = benchmark_kernel(triton_gelu, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  Triton GELU:   mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("gelu", "triton", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    print()

    # ===================== SWISH =====================
    print("Benchmarking Swish...")

    def torch_swish(x):
        return x * torch.sigmoid(x)

    times = benchmark_kernel(torch_swish, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  PyTorch Swish:  mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("swish", "pytorch", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    times = benchmark_kernel(cuda_swish, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  CUDA Swish:     mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("swish", "cuda", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    times = benchmark_kernel(triton_swish, x_elem)
    mean_ms, std_ms = summarize_times(times)
    print(f"  Triton Swish:   mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(("swish", "triton", f"N={N_ELEMS}", mean_ms, std_ms, BENCH_ITERS))

    print()

    # ===================== LayerNorm =====================
    print("Benchmarking LayerNorm...")

    def torch_layernorm(x, gamma, beta):
        return F.layer_norm(x, (LN_COLS,), gamma, beta)

    times = benchmark_kernel(torch_layernorm, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  PyTorch LN:     mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("layernorm", "pytorch", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    times = benchmark_kernel(cuda_layernorm, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  CUDA LN:        mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("layernorm", "cuda", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    times = benchmark_kernel(triton_layernorm, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  Triton LN:      mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("layernorm", "triton", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    print()

    # ===================== Fused LN + GELU =====================
    print("Benchmarking Fused LN + GELU...")

    def torch_fused_ln_gelu(x, gamma, beta):
        y = F.layer_norm(x, (LN_COLS,), gamma, beta)
        return F.gelu(y)

    times = benchmark_kernel(torch_fused_ln_gelu, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  PyTorch LN+GELU:   mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("fused_ln_gelu", "pytorch", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    times = benchmark_kernel(cuda_fused_ln_gelu, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  CUDA LN+GELU:      mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("fused_ln_gelu", "cuda", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    times = benchmark_kernel(triton_fused_ln_gelu, x_ln, gamma, beta)
    mean_ms, std_ms = summarize_times(times)
    print(f"  Triton LN+GELU:    mean={mean_ms:.4f} ms  std={std_ms:.4f} ms")
    results.append(
        ("fused_ln_gelu", "triton", f"{LN_ROWS}x{LN_COLS}", mean_ms, std_ms, BENCH_ITERS)
    )

    print("\n=======================================")
    print("   Saving results to CSV")
    print("=======================================\n")

    ensure_report_dir(OUTPUT_CSV)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "implementation", "shape", "mean_ms", "std_ms", "iters"])
        for row in results:
            writer.writerow(row)

    print(f"Results written to: {OUTPUT_CSV}\n")
    print("Done.")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
