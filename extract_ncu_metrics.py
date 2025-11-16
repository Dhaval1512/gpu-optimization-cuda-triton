import subprocess
import os
import pandas as pd

print("\n" + "="*80)
print("  EXTRACTING METRICS FROM NSIGHT COMPUTE REPORTS")
print("="*80 + "\n")

# Nsight Compute executable path
NCU_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"

# Report files
kernels = {
    "gelu": "profiling/nsight/gelu_profile.ncu-rep",
    "swish": "profiling/nsight/swish_profile.ncu-rep",
    "layernorm": "profiling/nsight/layernorm_profile.ncu-rep",
    "fused_ln_gelu": "profiling/nsight/fused_ln_gelu_profile.ncu-rep"
}

# Metrics to extract
metrics = [
    "gpu__time_duration.sum",           # Duration
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # Compute throughput
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", # Memory throughput
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # Occupancy
]

results = []

for kernel_name, report_path in kernels.items():
    if not os.path.exists(report_path):
        print(f"⚠️  Warning: {report_path} not found, skipping...")
        continue
    
    print(f"Processing: {kernel_name}")
    
    # Extract metrics using ncu CLI
    cmd = [
        NCU_PATH,
        "--csv",
        "--page", "raw",
        f"--metrics", ",".join(metrics),
        "-i", report_path
    ]
    
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        
        # Parse CSV output
        lines = output.strip().split('\n')
        if len(lines) < 2:
            print(f"  ❌ No data extracted")
            continue
        
        # Get metric values (last row of CSV)
        values = lines[-1].split(',')
        
        kernel_metrics = {
            'Kernel': kernel_name,
            'Duration_ms': float(values[0]) / 1e6 if len(values) > 0 else 0,  # ns to ms
            'Compute_Throughput_%': float(values[1]) if len(values) > 1 else 0,
            'Memory_Throughput_%': float(values[2]) if len(values) > 2 else 0,
            'Occupancy_%': float(values[3]) if len(values) > 3 else 0
        }
        
        results.append(kernel_metrics)
        print(f"  ✓ Extracted {len(kernel_metrics)} metrics")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

if not results:
    print("\n❌ No metrics extracted. Check that .ncu-rep files exist.")
    exit(1)

# Save to CSV
os.makedirs("report", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("report/ncu_profiling_metrics.csv", index=False)

print("\n" + "="*80)
print("  METRICS EXTRACTED SUCCESSFULLY")
print("="*80 + "\n")

print("PROFILING RESULTS:")
print(df.to_string(index=False))

print(f"\n✅ Results saved to: report/ncu_profiling_metrics.csv\n")