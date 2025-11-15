import matplotlib.pyplot as plt
import numpy as np
import os

# ========= YOUR RESULTS (FROM BENCHMARK) ==========
batch_sizes = np.array([16, 32, 64, 128])

pytorch_times = np.array([2.1791, 0.4270, 0.4219, 0.7777])
cuda_times    = np.array([0.3530, 0.3646, 0.4238, 0.7059])
triton_times  = np.array([1.8914, 0.4211, 0.3977, 0.6969])

os.makedirs("plots", exist_ok=True)

# ================================
# 1. BAR CHART — LATENCY COMPARISON
# ================================
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = np.arange(len(batch_sizes))

plt.bar(index, pytorch_times, bar_width, label="PyTorch CNN")
plt.bar(index + bar_width, cuda_times, bar_width, label="CUDA CNN")
plt.bar(index + 2*bar_width, triton_times, bar_width, label="Triton CNN")

plt.xlabel("Batch Size", fontsize=12)
plt.ylabel("Latency (ms/batch)", fontsize=12)
plt.title("Latency Comparison (Bar Chart)", fontsize=14)
plt.xticks(index + bar_width, batch_sizes)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.4)

plt.savefig("plots/bar_latency_comparison.png", dpi=300)
plt.close()


# =================================
# 2. BAR CHART — SPEEDUP COMPARISON
# =================================
cuda_speedup   = pytorch_times / cuda_times
triton_speedup = pytorch_times / triton_times

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(batch_sizes))

plt.bar(index, cuda_speedup, bar_width, label="CUDA Speedup")
plt.bar(index + bar_width, triton_speedup, bar_width, label="Triton Speedup")

plt.xlabel("Batch Size", fontsize=12)
plt.ylabel("Speedup ×", fontsize=12)
plt.title("Speedup Comparison (CUDA & Triton vs PyTorch)", fontsize=14)
plt.xticks(index + bar_width / 2, batch_sizes)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.4)

plt.savefig("plots/bar_speedup_comparison.png", dpi=300)
plt.close()


# ===================================================
# 3. RELATIVE PERFORMANCE (%) — LINE PLOT
# ===================================================
pytorch_norm = pytorch_times / pytorch_times.max() * 100
cuda_norm = cuda_times / cuda_times.max() * 100
triton_norm = triton_times / triton_times.max() * 100

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, pytorch_norm, marker='o', label="PyTorch (%)")
plt.plot(batch_sizes, cuda_norm, marker='o', label="CUDA (%)")
plt.plot(batch_sizes, triton_norm, marker='o', label="Triton (%)")

plt.xlabel("Batch Size")
plt.ylabel("Relative Runtime (%)")
plt.title("Relative Performance (%) vs Batch Size")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.savefig("plots/relative_performance.png", dpi=300)
plt.close()


# ===================================================
# 4. EFFICIENCY PLOT — LOWER BETTER
# ===================================================
eff_pytorch = 1 / pytorch_times
eff_cuda = 1 / cuda_times
eff_triton = 1 / triton_times

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, eff_pytorch, marker='o', label="PyTorch Efficiency")
plt.plot(batch_sizes, eff_cuda, marker='o', label="CUDA Efficiency")
plt.plot(batch_sizes, eff_triton, marker='o', label="Triton Efficiency")

plt.title("Model Efficiency (1/Latency)", fontsize=14)
plt.xlabel("Batch Size")
plt.ylabel("Efficiency Score")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.savefig("plots/efficiency_plot.png", dpi=300)
plt.close()


# ===================================================
# 5. SCATTER PLOT — LATENCY TREND
# ===================================================
plt.figure(figsize=(10, 6))
plt.scatter(batch_sizes, pytorch_times, label="PyTorch", s=100)
plt.scatter(batch_sizes, cuda_times, label="CUDA", s=100)
plt.scatter(batch_sizes, triton_times, label="Triton", s=100)

plt.plot(batch_sizes, pytorch_times, linestyle='--', alpha=0.5)
plt.plot(batch_sizes, cuda_times, linestyle='--', alpha=0.5)
plt.plot(batch_sizes, triton_times, linestyle='--', alpha=0.5)

plt.title("Latency Trend (Scatter Plot)", fontsize=14)
plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.savefig("plots/scatter_latency_trend.png", dpi=300)
plt.close()

print("All graphs generated successfully inside /plots folder!")
