import subprocess
import os
import pandas as pd

print("\n" + "="*80)
print("  EXTRACTING GPU PROFILING METRICS FROM NSIGHT SYSTEMS")
print("="*80 + "\n")

# You already have: profiling/baseline_cnn.nsys-rep
# Extract CUDA API statistics

nsys_file = "profiling/baseline_cnn.nsys-rep"

if not os.path.exists(nsys_file):
    print("❌ ERROR: baseline_cnn.nsys-rep not found!")
    print("   Please ensure you have the Nsight Systems profile data.")
    exit(1)

# Extract CUDA API stats
print("Extracting CUDA API statistics...")
cmd = f"nsys stats --report cuda_api_sum {nsys_file} --format csv --output profiling/cuda_api_stats"
os.system(cmd)

# Extract kernel stats
print("Extracting kernel execution statistics...")
cmd = f"nsys stats --report cuda_gpu_kern_sum {nsys_file} --format csv --output profiling/kernel_stats"
os.system(cmd)

# Extract memory operations
print("Extracting memory operation statistics...")
cmd = f"nsys stats --report cuda_gpu_mem_time_sum {nsys_file} --format csv --output profiling/memory_stats"
os.system(cmd)

print("\n" + "="*80)
print("  ANALYZING EXTRACTED DATA")
print("="*80 + "\n")

# Read and summarize kernel stats
kernel_df = pd.read_csv("profiling/kernel_stats.csv")

print("TOP 10 KERNELS BY TOTAL TIME:")
print("-" * 80)
top_kernels = kernel_df.nlargest(10, 'Total Time (ns)')
for idx, row in top_kernels.iterrows():
    print(f"{row['Name'][:60]:60s} | {row['Total Time (ns)']/1e6:8.2f} ms | {row['Instances']:6d} calls")

print("\n" + "="*80)
print("  COMPUTING GPU EFFICIENCY METRICS")
print("="*80 + "\n")

# Calculate estimated metrics based on timing data
total_kernel_time = kernel_df['Total Time (ns)'].sum()
total_instances = kernel_df['Instances'].sum()

print(f"Total Kernel Execution Time: {total_kernel_time/1e9:.4f} seconds")
print(f"Total Kernel Launches: {total_instances}")
print(f"Average Kernel Duration: {(total_kernel_time/total_instances)/1e6:.4f} ms")

# Create summary report
summary = {
    'Metric': [
        'Total Kernel Time (s)',
        'Total Kernel Launches',
        'Average Kernel Duration (ms)',
        'Unique Kernels',
        'Most Called Kernel',
        'Longest Running Kernel'
    ],
    'Value': [
        f"{total_kernel_time/1e9:.4f}",
        f"{total_instances}",
        f"{(total_kernel_time/total_instances)/1e6:.4f}",
        f"{len(kernel_df)}",
        kernel_df.loc[kernel_df['Instances'].idxmax(), 'Name'][:50],
        kernel_df.loc[kernel_df['Total Time (ns)'].idxmax(), 'Name'][:50]
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv("report/gpu_profiling_summary.csv", index=False)

print("\n✅ GPU profiling metrics extracted successfully!")
print("   - profiling/cuda_api_stats.csv")
print("   - profiling/kernel_stats.csv")
print("   - profiling/memory_stats.csv")
print("   - report/gpu_profiling_summary.csv")
print("\nDone!\n")