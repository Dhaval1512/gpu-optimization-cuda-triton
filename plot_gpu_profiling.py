import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n" + "="*80)
print("  GENERATING GPU PROFILING VISUALIZATIONS")
print("="*80 + "\n")

# Load data
kernel_df = pd.read_csv("report/gpu_kernel_profiling.csv")

os.makedirs("plots", exist_ok=True)

# ========== 1. Kernel Performance Comparison ==========
print("Generating kernel performance comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Duration comparison
operations = kernel_df['Operation'].str[:25]  # Truncate long names
durations = kernel_df['Duration_ms']

ax1.barh(operations, durations, color='steelblue')
ax1.set_xlabel('Duration (ms)', fontsize=12)
ax1.set_title('Kernel Execution Time', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(durations):
    ax1.text(v, i, f'  {v:.3f}ms', va='center', fontsize=10)

# Throughput comparison
x = np.arange(len(operations))
width = 0.35

compute = kernel_df['Compute_Throughput_%']
memory = kernel_df['Memory_Throughput_%']

ax2.bar(x - width/2, compute, width, label='Compute Throughput', color='coral')
ax2.bar(x + width/2, memory, width, label='Memory Throughput', color='skyblue')

ax2.set_ylabel('Throughput (%)', fontsize=12)
ax2.set_title('Compute vs Memory Throughput', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(operations, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plots/gpu_kernel_performance.png", dpi=300)
plt.close()

# ========== 2. Occupancy and Efficiency ==========
print("Generating occupancy and efficiency plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Occupancy
occupancy = kernel_df['Occupancy_%']
colors = plt.cm.RdYlGn(occupancy / 100)

bars = ax1.barh(operations, occupancy, color=colors)
ax1.set_xlabel('Occupancy (%)', fontsize=12)
ax1.set_title('GPU Occupancy by Kernel', fontsize=14, fontweight='bold')
ax1.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Target: 60%')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

for i, v in enumerate(occupancy):
    ax1.text(v, i, f'  {v:.1f}%', va='center', fontsize=10)

# SM Efficiency
sm_eff = kernel_df['SM_Efficiency_%']
warp_eff = kernel_df['Warp_Execution_Efficiency_%']

x = np.arange(len(operations))
width = 0.35

ax2.bar(x - width/2, sm_eff, width, label='SM Efficiency', color='mediumpurple')
ax2.bar(x + width/2, warp_eff, width, label='Warp Efficiency', color='lightcoral')

ax2.set_ylabel('Efficiency (%)', fontsize=12)
ax2.set_title('SM and Warp Execution Efficiency', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(operations, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([90, 100])

plt.tight_layout()
plt.savefig("plots/gpu_occupancy_efficiency.png", dpi=300)
plt.close()

# ========== 3. Resource Usage ==========
print("Generating resource usage plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Register usage
registers = kernel_df['Register_Usage']
block_size = kernel_df['Block_Size']

colors = plt.cm.viridis(registers / registers.max())
bars = ax1.barh(operations, registers, color=colors)
ax1.set_xlabel('Registers per Thread', fontsize=12)
ax1.set_title('Register Usage by Kernel', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(registers):
    ax1.text(v, i, f'  {v}', va='center', fontsize=10)

# Block size
colors = plt.cm.plasma(block_size / block_size.max())
bars = ax2.barh(operations, block_size, color=colors)
ax2.set_xlabel('Threads per Block', fontsize=12)
ax2.set_title('Block Size Configuration', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(block_size):
    ax2.text(v, i, f'  {v}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("plots/gpu_resource_usage.png", dpi=300)
plt.close()

# ========== 4. Memory vs Compute Bottleneck Analysis ==========
print("Generating bottleneck analysis...")

plt.figure(figsize=(10, 8))

compute = kernel_df['Compute_Throughput_%']
memory = kernel_df['Memory_Throughput_%']

plt.scatter(compute, memory, s=300, alpha=0.6, c=range(len(operations)), cmap='tab10')

for i, op in enumerate(operations):
    plt.annotate(op, (compute.iloc[i], memory.iloc[i]), 
                 fontsize=9, ha='right', va='bottom')

# Add diagonal line (balanced point)
plt.plot([0, 100], [0, 100], 'r--', alpha=0.3, linewidth=2, label='Balanced')

# Add regions
plt.axhspan(80, 100, alpha=0.1, color='blue', label='Memory-Bound Region')
plt.axvspan(80, 100, alpha=0.1, color='red', label='Compute-Bound Region')

plt.xlabel('Compute Throughput (%)', fontsize=12)
plt.ylabel('Memory Throughput (%)', fontsize=12)
plt.title('Memory vs Compute Bottleneck Analysis', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.xlim([0, 100])
plt.ylim([0, 100])

plt.tight_layout()
plt.savefig("plots/gpu_bottleneck_analysis.png", dpi=300)
plt.close()

# ========== 5. Speedup Analysis (Fusion) ==========
print("Generating fusion speedup analysis...")

# Manual data for fusion analysis
operations_fusion = ['LN (separate)', 'GELU (separate)', 'LN+GELU (fused)', 'Theoretical Minimum']
times = [0.045, 0.020, 0.052, 0.065]  # LN + GELU separate = 0.065ms
speedups = [times[3]/t for t in times]  # vs theoretical unfused

plt.figure(figsize=(10, 6))

colors = ['coral', 'coral', 'green', 'gray']
bars = plt.bar(operations_fusion, times, color=colors, alpha=0.7)

plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('Kernel Fusion Impact: LayerNorm + GELU', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add speedup annotations
for i, (bar, time, speedup) in enumerate(zip(bars, times, speedups)):
    if i == 2:  # Fused kernel
        plt.text(bar.get_x() + bar.get_width()/2, time, 
                f'{time:.3f}ms\n{speedups[3]/speedup:.2f}× speedup', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, time, 
                f'{time:.3f}ms', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig("plots/gpu_fusion_speedup.png", dpi=300)
plt.close()

print("\n✅ All GPU profiling visualizations generated successfully!")
print("   - plots/gpu_kernel_performance.png")
print("   - plots/gpu_occupancy_efficiency.png")
print("   - plots/gpu_resource_usage.png")
print("   - plots/gpu_bottleneck_analysis.png")
print("   - plots/gpu_fusion_speedup.png")
print("\nDone!\n")