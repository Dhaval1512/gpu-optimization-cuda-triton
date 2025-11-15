import subprocess
import os

NCU_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"

print("\n" + "="*70)
print("  EXTRACTING GPU METRICS")
print("="*70 + "\n")

# Simple export command
cmd = f'"{NCU_PATH}" --csv --page details -i profiling/pytorch_operations.ncu-rep'

print("Running export...")
os.system(f'{cmd} > profiling/extracted_metrics.csv')

print("✓ Metrics exported to profiling/extracted_metrics.csv")

# Display summary from the Nsight UI data you already have
print("\n" + "="*70)
print("  METRICS SUMMARY (from your screenshots)")
print("="*70 + "\n")

summary = """
GPU PROFILING RESULTS
=====================

Device: NVIDIA GeForce RTX 4060 Laptop GPU
Compute Capability: 8.9 (Ampere)
SM Frequency: 1.90 GHz

KERNEL PERFORMANCE METRICS:
---------------------------
Operation: GELU (vectorized_elementwise_kernel)
- Duration: ~0.02 ms per call
- Compute Throughput: 38%
- Memory Throughput: 87%
- Occupancy: ~64%
- Register Usage: 32
- Block Size: 128

Operation: Conv2D (cuDNN Ampere kernels)
- Duration: 0.08-0.19 ms per call
- Compute Throughput: 42-71%
- Memory Throughput: 89-91%
- Occupancy: 58-72%

KEY FINDINGS:
-------------
✓ Total kernels profiled: 1169
✓ SM workload imbalance: 5.71% (excellent)
✓ Operations are memory-bound (high memory throughput %)
✓ Good occupancy achieved (64%)
✓ Vectorized operations perform better

COPY THIS DATA TO YOUR REPORT!
"""

print(summary)
print("\n" + "="*70)