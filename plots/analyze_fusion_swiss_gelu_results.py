import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df_kernel = pd.read_csv("report/kernel_fusion_swiss_gelu_mnist.csv")
df_cnn = pd.read_csv("report/mnist_cnn_fusion_swiss_gelu.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Kernel Performance
ax = axes[0, 0]
batch_sizes = [16, 32, 64, 128, 256, 512]
pytorch_k = []
triton_k = []
cuda_k = []

for bs in batch_sizes:
    pytorch_k.append(df_kernel[(df_kernel['batch_size'] == bs) & (df_kernel['implementation'] == 'pytorch')]['mean_ms'].values[0])
    triton_k.append(df_kernel[(df_kernel['batch_size'] == bs) & (df_kernel['implementation'] == 'triton_fused')]['mean_ms'].values[0])
    cuda_k.append(df_kernel[(df_kernel['batch_size'] == bs) & (df_kernel['implementation'] == 'cuda_fused')]['mean_ms'].values[0])

x = np.arange(len(batch_sizes))
width = 0.25
ax.bar(x - width, pytorch_k, width, label='PyTorch', color='coral', alpha=0.8)
ax.bar(x, triton_k, width, label='Triton', color='steelblue', alpha=0.8)
ax.bar(x + width, cuda_k, width, label='CUDA', color='green', alpha=0.8)
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Kernel Time (ms)', fontweight='bold')
ax.set_title('Kernel-Level Performance (GELU+Swish)', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Kernel Speedup
ax = axes[0, 1]
speedup_triton = [p/t for p, t in zip(pytorch_k, triton_k)]
speedup_cuda = [p/c for p, c in zip(pytorch_k, cuda_k)]

ax.plot(batch_sizes, speedup_triton, marker='o', linewidth=2, markersize=8, label='Triton', color='steelblue')
ax.plot(batch_sizes, speedup_cuda, marker='s', linewidth=2, markersize=8, label='CUDA', color='green')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Speedup vs PyTorch', fontweight='bold')
ax.set_title('Kernel Speedup Factor', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Add values
for i, (tr, cu) in enumerate(zip(speedup_triton, speedup_cuda)):
    if cu > 1.0:  # Only show CUDA (better performer)
        ax.text(batch_sizes[i], cu + 0.05, f'{cu:.2f}×', ha='center', fontsize=9, color='green', fontweight='bold')

# Plot 3: CNN Inference Time
ax = axes[1, 0]
batch_sizes_cnn = [16, 32, 64, 128, 256]
pytorch_cnn = []
triton_cnn = []
cuda_cnn = []

for bs in batch_sizes_cnn:
    pytorch_cnn.append(df_cnn[(df_cnn['batch_size'] == bs) & (df_cnn['implementation'] == 'pytorch_unfused')]['mean_ms'].values[0])
    triton_cnn.append(df_cnn[(df_cnn['batch_size'] == bs) & (df_cnn['implementation'] == 'triton_fused')]['mean_ms'].values[0])
    cuda_cnn.append(df_cnn[(df_cnn['batch_size'] == bs) & (df_cnn['implementation'] == 'cuda_fused')]['mean_ms'].values[0])

x = np.arange(len(batch_sizes_cnn))
ax.bar(x - width, pytorch_cnn, width, label='PyTorch', color='coral', alpha=0.8)
ax.bar(x, triton_cnn, width, label='Triton', color='steelblue', alpha=0.8)
ax.bar(x + width, cuda_cnn, width, label='CUDA', color='green', alpha=0.8)
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Inference Time (ms)', fontweight='bold')
ax.set_title('End-to-End CNN Inference on MNIST', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes_cnn)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: CNN Speedup
ax = axes[1, 1]
speedup_triton_cnn = [p/t if t < p else p/t for p, t in zip(pytorch_cnn, triton_cnn)]
speedup_cuda_cnn = [p/c if c < p else p/c for p, c in zip(pytorch_cnn, cuda_cnn)]

ax.plot(batch_sizes_cnn, speedup_triton_cnn, marker='o', linewidth=2, markersize=8, label='Triton', color='steelblue')
ax.plot(batch_sizes_cnn, speedup_cuda_cnn, marker='s', linewidth=2, markersize=8, label='CUDA', color='green')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
ax.set_xlabel('Batch Size', fontweight='bold')
ax.set_ylabel('Speedup vs PyTorch', fontweight='bold')
ax.set_title('CNN Inference Speedup', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/fusion1_complete_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: plots/fusion1_complete_analysis.png")