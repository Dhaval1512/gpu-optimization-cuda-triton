import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
#  MANUALLY FILL YOUR RESULTS HERE
# ===============================

batch_sizes = np.array([16, 32, 64, 128])

pytorch_times = np.array([2.1791, 0.4270, 0.4219, 0.7777])
cuda_times    = np.array([0.3530, 0.3646, 0.4238, 0.7059])
triton_times  = np.array([1.8914, 0.4211, 0.3977, 0.6969])

# ===============================
#  CREATE OUTPUT FOLDER
# ===============================
os.makedirs("plots", exist_ok=True)


# ===============================
#  PLOT 1 — Latency vs Batch Size
# ===============================

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, pytorch_times, marker='o', label="PyTorch CNN", linewidth=2)
plt.plot(batch_sizes, cuda_times, marker='o', label="CUDA CNN", linewidth=2)
plt.plot(batch_sizes, triton_times, marker='o', label="Triton CNN", linewidth=2)

plt.title("Inference Latency vs Batch Size (PyTorch vs CUDA vs Triton)", fontsize=14)
plt.xlabel("Batch Size", fontsize=12)
plt.ylabel("Latency (ms per batch)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.savefig("plots/inference_latency_comparison.png", dpi=300)
plt.close()


# ===============================
#  PLOT 2 — Speedup vs Batch Size
# ===============================

cuda_speedup   = pytorch_times / cuda_times
triton_speedup = pytorch_times / triton_times

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, cuda_speedup, marker='o', label="CUDA Speedup (vs PyTorch)", linewidth=2)
plt.plot(batch_sizes, triton_speedup, marker='o', label="Triton Speedup (vs PyTorch)", linewidth=2)

plt.title("Speedup over PyTorch Baseline", fontsize=14)
plt.xlabel("Batch Size", fontsize=12)
plt.ylabel("Speedup ×", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.savefig("plots/inference_speedup_comparison.png", dpi=300)
plt.close()

print("Plots generated successfully in /plots folder!")
