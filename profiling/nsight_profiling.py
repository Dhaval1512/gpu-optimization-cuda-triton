#!/usr/bin/env python3
"""
Complete Automated Nsight Compute Profiling
Extracts ALL required metrics for JLR Hackathon grading
"""

import subprocess
import pandas as pd
import os
from pathlib import Path

class AutomatedNsightProfiler:
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Key metrics matching grading rubric requirements
        self.metrics = [
            "gpu__time_duration.avg",                                       # Kernel time (ns)
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",          # Memory throughput (%)
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",            # SM compute throughput (%)
            "launch__occupancy_per_block_size.avg.pct_of_peak_sustained_active",  # Occupancy (%)
            "smsp__warps_active.avg.pct_of_peak_sustained_active",         # Warp efficiency (%)
            "dram__bytes_read.sum",                                         # Memory read (bytes)
            "dram__bytes_write.sum",                                        # Memory write (bytes)
        ]
    
    def profile_benchmark(self, benchmark_script, output_name):
        """
        Profile a benchmark script and extract all metrics
        
        Args:
            benchmark_script: Path to Python benchmark script
            output_name: Base name for output files
        """
        print(f"\n{'='*80}")
        print(f"PROFILING: {benchmark_script}")
        print(f"{'='*80}\n")
        
        # Step 1: Run profiler and save .ncu-rep file
        print("Step 1: Running Nsight Compute profiler...")
        
        rep_file = self.output_dir / f"{output_name}.ncu-rep"
        
        cmd_profile = [
            "ncu",
            "--set", "full",
            "--export", str(rep_file.with_suffix('')),
            "--target-processes", "all",
            "python", benchmark_script
        ]
        
        try:
            print(f"   Command: {' '.join(cmd_profile)}")
            result = subprocess.run(cmd_profile, capture_output=True, text=True, timeout=600)
            
            # Check if profile was created
            if rep_file.exists():
                size_mb = rep_file.stat().st_size / (1024*1024)
                print(f"✅ Profile saved: {rep_file.name} ({size_mb:.2f} MB)\n")
            else:
                print(f"❌ Profile file not created\n")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️  Profiling timed out (10 minutes)")
            return None
        except FileNotFoundError:
            print("❌ Error: 'ncu' command not found!")
            print("   Install: module load nsight-compute (on Delta)")
            print("   Or add Nsight Compute to PATH")
            return None
        except Exception as e:
            print(f"❌ Error during profiling: {e}")
            return None
        
        # Step 2: Extract metrics to CSV
        print("Step 2: Extracting metrics from profile...")
        
        csv_file = self.output_dir / f"{output_name}_metrics.csv"
        
        cmd_extract = [
            "ncu",
            "--import", str(rep_file),
            "--csv",
            "--page", "details",
            "--metrics", ",".join(self.metrics)
        ]
        
        try:
            result = subprocess.run(cmd_extract, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                with open(csv_file, 'w') as f:
                    f.write(result.stdout)
                print(f"✅ Metrics extracted: {csv_file.name}\n")
                
                # Parse and display summary
                self.display_summary(csv_file)
                return csv_file
            else:
                print(f"⚠️  Could not extract metrics")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting metrics: {e}")
            return None
    
    def display_summary(self, csv_file):
        """
        Display a formatted summary of profiling metrics
        """
        try:
            df = pd.read_csv(csv_file)
            
            print("📊 PROFILING SUMMARY:")
            print("-" * 80)
            
            # Kernel Time
            time_col = [c for c in df.columns if 'duration' in c.lower()]
            if time_col and time_col[0] in df.columns:
                time_ns = df[time_col[0]].mean()
                time_ms = time_ns / 1_000_000
                print(f"  ⏱️  Kernel Time:        {time_ms:.4f} ms")
            
            # Memory Throughput
            mem_col = [c for c in df.columns if 'dram' in c.lower() and 'throughput' in c.lower()]
            if mem_col and mem_col[0] in df.columns:
                mem_throughput = df[mem_col[0]].mean()
                status = "✅" if mem_throughput > 80 else "⚠️" if mem_throughput > 50 else "❌"
                print(f"  {status} Memory Throughput:  {mem_throughput:.2f}% (>80% = memory-bound)")
            
            # SM Throughput (Compute)
            sm_col = [c for c in df.columns if 'sm__throughput' in c.lower()]
            if sm_col and sm_col[0] in df.columns:
                sm_throughput = df[sm_col[0]].mean()
                status = "✅" if sm_throughput > 70 else "⚠️" if sm_throughput > 40 else "❌"
                print(f"  {status} SM Throughput:      {sm_throughput:.2f}% (>70% = compute-bound)")
            
            # Occupancy
            occ_col = [c for c in df.columns if 'occupancy' in c.lower()]
            if occ_col and occ_col[0] in df.columns:
                occupancy = df[occ_col[0]].mean()
                status = "✅" if occupancy > 60 else "⚠️" if occupancy > 40 else "❌"
                print(f"  {status} Occupancy:          {occupancy:.2f}% (>60% = good)")
            
            # Warp Efficiency
            warp_col = [c for c in df.columns if 'warps_active' in c.lower()]
            if warp_col and warp_col[0] in df.columns:
                warp_eff = df[warp_col[0]].mean()
                status = "✅" if warp_eff > 80 else "⚠️" if warp_eff > 60 else "❌"
                print(f"  {status} Warp Efficiency:    {warp_eff:.2f}% (>80% = efficient)")
            
            # Memory Traffic
            read_col = [c for c in df.columns if 'bytes_read' in c.lower()]
            write_col = [c for c in df.columns if 'bytes_write' in c.lower()]
            
            if read_col and read_col[0] in df.columns:
                bytes_read = df[read_col[0]].sum()
                mb_read = bytes_read / (1024*1024)
                print(f"  📥 Memory Read:        {mb_read:.2f} MB")
            
            if write_col and write_col[0] in df.columns:
                bytes_write = df[write_col[0]].sum()
                mb_write = bytes_write / (1024*1024)
                print(f"  📤 Memory Write:       {mb_write:.2f} MB")
            
            print("-" * 80 + "\n")
            
        except Exception as e:
            print(f"⚠️  Could not parse CSV: {e}\n")
    
    def profile_all_benchmarks(self):
        """
        Profile all key benchmarks for the hackathon
        """
        benchmarks = [
            # Baseline operations
            ("benchmarks/bench_baseline_kernels.py", "baseline_kernels"),
            
            # Fusions
            ("benchmarks/bench_fusion_swiss_gelu_kernels.py", "fusion1_kernels"),
            ("benchmarks/bench_fusion_ln_swiss_dropout_kernels.py", "fusion2_kernels"),
            ("benchmarks/bench_fusion_ln_swiss_gelu_kernels.py", "fusion3_kernels"),
            
            # CNN inference (if available)
            ("benchmarks/bench_mnist_inference_swiss_gelu.py", "fusion1_cnn"),
            ("benchmarks/bench_fusion_ln_swiss_dropout_cnn.py", "fusion2_cnn"),
        ]
        
        results = []
        
        for script, name in benchmarks:
            if not os.path.exists(script):
                print(f"⚠️  Benchmark not found: {script}, skipping...")
                continue
            
            csv_file = self.profile_benchmark(script, name)
            if csv_file:
                results.append((name, csv_file))
        
        print(f"\n{'='*80}")
        print(f"PROFILING COMPLETE")
        print(f"{'='*80}\n")
        print(f"✅ Successfully profiled {len(results)} benchmarks")
        print(f"📁 Results saved to: {self.output_dir}/\n")
        
        for name, csv_file in results:
            print(f"   - {name}: {csv_file.name}")
        print()
        
        return results


def create_master_summary_table(profiling_dir="profiling_results", output_dir="report"):
    """
    Create a master summary table from all profiling results
    """
    print("\n" + "="*80)
    print("CREATING MASTER GPU PROFILING SUMMARY TABLE")
    print("="*80 + "\n")
    
    prof_dir = Path(profiling_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    csv_files = list(prof_dir.glob("*_metrics.csv"))
    
    if not csv_files:
        print("⚠️  No profiling result CSV files found")
        return
    
    summary_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract operation name
            operation = csv_file.stem.replace("_metrics", "")
            
            # Get average metrics
            time_col = [c for c in df.columns if 'duration' in c.lower()]
            mem_col = [c for c in df.columns if 'dram' in c.lower() and 'throughput' in c.lower()]
            sm_col = [c for c in df.columns if 'sm__throughput' in c.lower()]
            occ_col = [c for c in df.columns if 'occupancy' in c.lower()]
            warp_col = [c for c in df.columns if 'warps_active' in c.lower()]
            
            row = {'Operation': operation}
            
            if time_col and time_col[0] in df.columns:
                row['Kernel_Time_ms'] = round(df[time_col[0]].mean() / 1_000_000, 4)
            
            if mem_col and mem_col[0] in df.columns:
                row['Memory_Throughput_%'] = round(df[mem_col[0]].mean(), 2)
            
            if sm_col and sm_col[0] in df.columns:
                row['SM_Throughput_%'] = round(df[sm_col[0]].mean(), 2)
            
            if occ_col and occ_col[0] in df.columns:
                row['Occupancy_%'] = round(df[occ_col[0]].mean(), 2)
            
            if warp_col and warp_col[0] in df.columns:
                row['Warp_Efficiency_%'] = round(df[warp_col[0]].mean(), 2)
            
            summary_data.append(row)
            
        except Exception as e:
            print(f"⚠️  Error processing {csv_file.name}: {e}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = out_dir / "gpu_profiling_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Master summary table saved: {summary_path}\n")
        print(summary_df.to_string(index=False))
        print()
        
        # Also save in a more readable format
        readable_path = out_dir / "gpu_profiling_summary_readable.txt"
        with open(readable_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GPU PROFILING SUMMARY - ALL OPERATIONS\n")
            f.write("="*80 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
            f.write("✅ All Required Metrics Captured:\n")
            f.write("   1. Kernel Time (ms)\n")
            f.write("   2. Memory Throughput (%)\n")
            f.write("   3. SM Usage / GPU Compute (%)\n")
            f.write("   4. Occupancy (%)\n")
            f.write("   5. Warp Efficiency (%)\n")
            f.write("="*80 + "\n")
        
        print(f"✅ Readable summary saved: {readable_path.name}\n")
    else:
        print("⚠️  No data to create summary table")


def quick_profile_single(benchmark_path, operation_name):
    """
    Quick profile of a single benchmark (for testing)
    """
    profiler = AutomatedNsightProfiler()
    profiler.profile_benchmark(benchmark_path, operation_name)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         COMPLETE AUTOMATED NSIGHT COMPUTE PROFILING SYSTEM                  ║
║              Extracts All Required GPU Metrics for Grading                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🎯 Required Metrics (Page 3 Grading Rubric - 20%):")
    print("   1. ⏱️  Kernel Time")
    print("   2. 📊 Memory Throughput")
    print("   3. 🔧 GPU Occupancy")
    print("   4. ⚡ SM Usage (GPU Compute)")
    print("   5. 📈 Inference Time per Batch\n")
    
    import sys
    
    if len(sys.argv) > 1:
        # Profile specific benchmark
        benchmark_script = sys.argv[1]
        output_name = Path(benchmark_script).stem
        
        print(f"📊 Profiling single benchmark: {benchmark_script}\n")
        quick_profile_single(benchmark_script, output_name)
        
    else:
        # Profile all benchmarks
        print("📊 Profiling ALL benchmarks (this may take 10-30 minutes)...\n")
        print("💡 Tip: Profile one benchmark first to test:")
        print("   python automated_nsight_profiler.py benchmarks/bench_kernels.py\n")
        
        response = input("Continue with full profiling? [y/N]: ")
        
        if response.lower() == 'y':
            profiler = AutomatedNsightProfiler()
            profiler.profile_all_benchmarks()
            create_master_summary_table()
        else:
            print("\n⏸️  Profiling cancelled")
            print("   Run with specific benchmark: python automated_nsight_profiler.py <benchmark.py>")
            return
    
    print("\n" + "="*80)
    print("✅ PROFILING WORKFLOW COMPLETE!")
    print("="*80)
    print()
    print("📁 Generated Files:")
    print("   - profiling_results/*.ncu-rep (profile data)")
    print("   - profiling_results/*_metrics.csv (extracted metrics)")
    print("   - report/gpu_profiling_summary.csv (master table)")
    print()
    print("📊 Next Steps:")
    print("   1. Use gpu_profiling_summary.csv in your report")
    print("   2. Create visualization: python generate_all_visualizations.py")
    print("   3. Add charts to presentation")
    print()


if __name__ == "__main__":
    main()