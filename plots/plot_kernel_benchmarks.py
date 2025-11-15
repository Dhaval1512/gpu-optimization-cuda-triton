import csv
import os
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "report/kernel_benchmarks.csv"


# ----------------------
# Load CSV Data
# ----------------------
kernels = ["gelu", "swish", "layernorm", "fused_ln_gelu"]
impls = ["pytorch", "cuda", "triton"]

data = {k: {} for k in kernels}

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        k = row["kernel"]
        impl = row["implementation"]
        mean_ms = float(row["mean_ms"])
        if k in data and impl in impls:
            data[k][impl] = mean_ms


# ----------------------
# Helper function to plot
# ----------------------
def plot_kernel(kernel_name, save_path):
    values = [
        data[kernel_name]["pytorch"],
        data[kernel_name]["cuda"],
        data[kernel_name]["triton"],
    ]
    labels = ["PyTorch", "CUDA", "Triton"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)

    plt.title(f"{kernel_name.upper()} – Runtime Comparison")
    plt.ylabel("Time (ms)")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val,
                 f"{val:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ----------------------
# Generate individual plots
# ----------------------
os.makedirs("plots", exist_ok=True)

plot_kernel("gelu", "plots/gelu_runtime.png")
plot_kernel("swish", "plots/swish_runtime.png")
plot_kernel("layernorm", "plots/layernorm_runtime.png")
plot_kernel("fused_ln_gelu", "plots/fused_ln_gelu_runtime.png")


# ----------------------
# Speedup plot (vs PyTorch)
# ----------------------
plt.figure(figsize=(10, 6))

kernel_labels = []
cuda_speedup = []
triton_speedup = []

for k in kernels:
    pt = data[k]["pytorch"]
    cu = data[k]["cuda"]
    tr = data[k]["triton"]

    kernel_labels.append(k.upper())

    cuda_speedup.append(pt / cu)
    triton_speedup.append(pt / tr)

x = np.arange(len(kernels))
width = 0.35

plt.bar(x - width/2, cuda_speedup, width, label="CUDA Speedup vs PyTorch")
plt.bar(x + width/2, triton_speedup, width, label="Triton Speedup vs PyTorch")

plt.xticks(x, kernel_labels)
plt.ylabel("Speedup Ratio")
plt.title("Speedup Comparison (Higher = Faster)")

for i in range(len(kernels)):
    plt.text(x[i] - width/2, cuda_speedup[i], f"{cuda_speedup[i]:.2f}",
             ha='center', va='bottom')
    plt.text(x[i] + width/2, triton_speedup[i], f"{triton_speedup[i]:.2f}",
             ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig("plots/speedup_plot.png")
plt.close()

print("All plots saved in /plots folder.")
