#!/usr/bin/env python3
"""
Analyze captured profiling metrics
Extract the key GPU metrics from the CSV
"""

import pandas as pd
import sys
from pathlib import Path

csv_file = Path("profiling_results/nsight_fusion3_metrics.csv")

if not csv_file.exists():
    print(f"❌ File not found: {csv_file}")
    print("\n💡 Run profiling first:")
    print("   python profiling/nsight_profiling_wsl.py benchmarks/nsight_fusion3_fixed.py")
    sys.exit(1)

print("="*80)
print("ANALYZING CAPTURED GPU PROFILING METRICS")
print("="*80 + "\n")

# Read CSV
df = pd.read_csv(csv_file)

print(f"📊 File: {csv_file}")
print(f"📋 Rows: {len(df)}")
print(f"📋 Columns: {len(df.columns)}\n")

# Show all column names
print("Available columns:")
print("-" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

print("\n" + "="*80)
print("KEY METRICS EXTRACTION")
print("="*80 + "\n")

# Look for key metrics
metrics_found = {}

# Kernel Time
time_cols = [col for col in df.columns if 'time' in col.lower() and 'duration' in col.lower()]
if time_cols:
    print("⏱️  KERNEL TIME:")
    for col in time_cols[:3]:  # First 3 time columns
        if not df[col].isna().all():
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}")
                print(f"      Mean: {values.mean():.2f}")
                print(f"      Min: {values.min():.2f}")
                print(f"      Max: {values.max():.2f}")
                metrics_found['kernel_time'] = values.mean()
    print()

# Memory Throughput
mem_cols = [col for col in df.columns if 'dram' in col.lower() and 'throughput' in col.lower() and 'pct' in col.lower()]
if mem_cols:
    print("📊 MEMORY THROUGHPUT:")
    for col in mem_cols[:2]:
        if not df[col].isna().all():
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}")
                print(f"      Mean: {values.mean():.2f}%")
                metrics_found['memory_throughput'] = values.mean()
    print()

# SM Throughput (Compute)
sm_cols = [col for col in df.columns if 'sm__throughput' in col.lower() and 'pct' in col.lower()]
if sm_cols:
    print("⚡ SM THROUGHPUT (Compute):")
    for col in sm_cols[:2]:
        if not df[col].isna().all():
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}")
                print(f"      Mean: {values.mean():.2f}%")
                metrics_found['sm_throughput'] = values.mean()
    print()

# Occupancy
occ_cols = [col for col in df.columns if 'occupancy' in col.lower() and 'pct' in col.lower()]
if occ_cols:
    print("🔧 GPU OCCUPANCY:")
    for col in occ_cols[:2]:
        if not df[col].isna().all():
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}")
                print(f"      Mean: {values.mean():.2f}%")
                metrics_found['occupancy'] = values.mean()
    print()

# Summary
print("="*80)
print("METRICS SUMMARY")
print("="*80 + "\n")

if metrics_found:
    print("✅ Successfully extracted:")
    for metric, value in metrics_found.items():
        print(f"   ✓ {metric}: {value:.2f}")
    print()
else:
    print("⚠️  No standard metrics found in expected columns")
    print("\n💡 The CSV might have different column names")
    print("   Showing first few rows:\n")
    print(df.head())

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. If metrics look good → Use them in your report!")
print("2. If metrics are missing → Re-run with fixed script:")
print("   python profiling/nsight_profiling_wsl.py benchmarks/nsight_fusion3_fixed.py")
print()