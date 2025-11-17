import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
df = pd.read_csv("report/workload_variations.csv")

os.makedirs("plots", exist_ok=True)

print("\n" + "="*80)
print("  GENERATING WORKLOAD VARIATION PLOTS")
print("="*80 + "\n")

# ========== 1. Heatmap: GELU performance across seq_length × hidden_dim ==========
print("Generating heatmap: GELU performance...")

gelu_cuda = df[(df['operation'] == 'gelu') & (df['implementation'] == 'cuda')]
pivot = gelu_cuda.pivot_table(values='mean_ms', index='seq_length', columns='hidden_dim', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Latency (ms)'})
plt.title('CUDA GELU: Latency vs Sequence Length × Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Sequence Length')
plt.tight_layout()
plt.savefig("plots/workload_gelu_heatmap.png", dpi=300)
plt.close()

# ========== 2. Line plot: Scaling with sequence length ==========
print("Generating line plot: Scaling with sequence length...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ops = ['gelu', 'swish', 'layernorm', 'fused_ln_gelu']

for idx, op in enumerate(ops):
    ax = axes[idx // 2, idx % 2]
    
    op_data = df[(df['operation'] == op) & (df['hidden_dim'] == 512) & (df['batch_size'] == 32)]
    
    for impl in ['pytorch', 'cuda', 'triton']:
        impl_data = op_data[op_data['implementation'] == impl]
        ax.plot(impl_data['seq_length'], impl_data['mean_ms'], marker='o', label=impl.upper())
    
    ax.set_title(f'{op.upper()} (Batch=32, Dim=512)')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/workload_seq_scaling.png", dpi=300)
plt.close()

# ========== 3. Bar chart: Speedup across dimensions ==========
print("Generating bar chart: Speedup across dimensions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, op in enumerate(ops):
    ax = axes[idx // 2, idx % 2]
    
    op_data = df[(df['operation'] == op) & (df['batch_size'] == 32) & (df['seq_length'] == 512)]
    
    dims = sorted(op_data['hidden_dim'].unique())
    cuda_speedup = []
    triton_speedup = []
    
    for dim in dims:
        pt = op_data[(op_data['implementation'] == 'pytorch') & (op_data['hidden_dim'] == dim)]['mean_ms'].values[0]
        cu = op_data[(op_data['implementation'] == 'cuda') & (op_data['hidden_dim'] == dim)]['mean_ms'].values[0]
        tr = op_data[(op_data['implementation'] == 'triton') & (op_data['hidden_dim'] == dim)]['mean_ms'].values[0]
        
        cuda_speedup.append(pt / cu)
        triton_speedup.append(pt / tr)
    
    x = np.arange(len(dims))
    width = 0.35
    
    ax.bar(x - width/2, cuda_speedup, width, label='CUDA Speedup')
    ax.bar(x + width/2, triton_speedup, width, label='Triton Speedup')
    
    ax.set_title(f'{op.upper()} Speedup (Batch=32, Seq=512)')
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Speedup vs PyTorch')
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plots/workload_dim_speedup.png", dpi=300)
plt.close()

# ========== 4. 3D surface plot: Fused LN+GELU performance ==========
print("Generating 3D surface plot: Fused LN+GELU performance...")

from mpl_toolkits.mplot3d import Axes3D

fused_cuda = df[(df['operation'] == 'fused_ln_gelu') & (df['implementation'] == 'cuda') & (df['batch_size'] == 32)]

seq_lens = sorted(fused_cuda['seq_length'].unique())
dims = sorted(fused_cuda['hidden_dim'].unique())

X, Y = np.meshgrid(seq_lens, dims)
Z = np.zeros_like(X, dtype=float)

for i, dim in enumerate(dims):
    for j, seq in enumerate(seq_lens):
        val = fused_cuda[(fused_cuda['seq_length'] == seq) & (fused_cuda['hidden_dim'] == dim)]['mean_ms'].values
        Z[i, j] = val[0] if len(val) > 0 else 0

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Hidden Dimension')
ax.set_zlabel('Latency (ms)')
ax.set_title('CUDA Fused LN+GELU: Performance Landscape (Batch=32)')
fig.colorbar(surf)
plt.savefig("plots/workload_3d_surface.png", dpi=300)
plt.close()

# ========== 5. Summary table ==========
print("Generating summary table...")

summary = []
for op in ops:
    for impl in ['pytorch', 'cuda', 'triton']:
        op_data = df[(df['operation'] == op) & (df['implementation'] == impl)]
        summary.append({
            'Operation': op,
            'Implementation': impl,
            'Min (ms)': op_data['mean_ms'].min(),
            'Max (ms)': op_data['mean_ms'].max(),
            'Mean (ms)': op_data['mean_ms'].mean(),
            'Std (ms)': op_data['mean_ms'].std()
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("report/workload_summary.csv", index=False)

print("\n✅ All plots generated successfully!")
print("   - plots/workload_gelu_heatmap.png")
print("   - plots/workload_seq_scaling.png")
print("   - plots/workload_dim_speedup.png")
print("   - plots/workload_3d_surface.png")
print("   - report/workload_summary.csv")
print("\nDone!\n")