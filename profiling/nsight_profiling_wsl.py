#!/usr/bin/env python3
"""
Automated Nsight Compute Profiler - WSL Optimized
Handles timeout issues and extracts GPU metrics efficiently
"""

import subprocess
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

class NsightProfilerWSL:
    """Optimized profiler for WSL environment with timeout handling"""
    
    def __init__(self, output_dir="profiling_results", report_dir="report"):
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.report_dir.mkdir(exist_ok=True, parents=True)
        
        # Required metrics for grading (Page 3 rubric)
        self.required_metrics = [
            "gpu__time_duration.avg",           # Kernel Time
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # Memory Throughput
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",    # SM Usage
            "launch__occupancy_per_block_size.avg.pct_of_peak_sustained_active",  # Occupancy
        ]
    
    def check_ncu_available(self):
        """Check if ncu is available"""
        try:
            result = subprocess.run(
                ["ncu", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ Nsight Compute found: {result.stdout.strip()}\n")
                return True
            else:
                print("❌ ncu command failed")
                return False
        except Exception as e:
            print(f"❌ Nsight Compute not available: {e}")
            return False
    
    def profile_benchmark(self, benchmark_script, output_name=None, timeout_seconds=120):
        """
        Profile a benchmark with optimized timeout
        
        Args:
            benchmark_script: Path to benchmark script
            output_name: Output file name (without extension)
            timeout_seconds: Timeout in seconds (default 2 minutes)
        """
        
        if output_name is None:
            output_name = Path(benchmark_script).stem
        
        output_path = self.output_dir / output_name
        
        print("="*80)
        print(f"PROFILING: {benchmark_script}")
        print("="*80)
        
        # Step 1: Run profiling
        print(f"\n⏱️  Timeout set to {timeout_seconds} seconds")
        print(f"📊 Running Nsight Compute profiler...\n")
        
        cmd = [
            "ncu",
            "--set", "full",
            "--export", str(output_path),
            "--target-processes", "all",
            "--force-overwrite",
            "python", benchmark_script
        ]
        
        print(f"   Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            if result.returncode != 0:
                print(f"⚠️  Profiling had warnings (return code: {result.returncode})")
                if result.stderr:
                    print(f"   {result.stderr[:500]}")  # First 500 chars
            else:
                print("✅ Profiling complete!")
            
            # Check if .ncu-rep file was created
            ncu_file = Path(f"{output_path}.ncu-rep")
            if not ncu_file.exists():
                print(f"❌ Profile file not created: {ncu_file}")
                return None
            
            print(f"✅ Profile saved: {ncu_file}")
            
        except subprocess.TimeoutExpired:
            print(f"⚠️  Profiling timed out after {timeout_seconds} seconds")
            print("   💡 Try using a lightweight profiling benchmark")
            print("      (fewer iterations, smaller batch sizes)")
            return None
        
        except Exception as e:
            print(f"❌ Profiling failed: {e}")
            return None
        
        # Step 2: Extract metrics
        print(f"\n📊 Extracting metrics...\n")
        
        csv_output = self.output_dir / f"{output_name}_metrics.csv"
        
        metrics_str = ",".join(self.required_metrics)
        
        extract_cmd = [
            "ncu",
            "--import", f"{output_path}.ncu-rep",
            "--csv",
            "--page", "details",
            "--metrics", metrics_str
        ]
        
        try:
            result = subprocess.run(
                extract_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                # Save CSV
                with open(csv_output, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ Metrics extracted: {csv_output}\n")
                
                # Display summary
                self.display_summary(csv_output)
                
                return csv_output
            else:
                print(f"⚠️  Metric extraction had issues")
                if result.stderr:
                    print(f"   {result.stderr[:300]}")
                return None
                
        except Exception as e:
            print(f"❌ Metric extraction failed: {e}")
            return None
    
    def display_summary(self, csv_file):
        """Display formatted summary of profiling results"""
        
        try:
            df = pd.read_csv(csv_file)
            
            print("="*80)
            print("GPU PROFILING SUMMARY")
            print("="*80 + "\n")
            
            # Group by kernel name
            if 'Kernel Name' in df.columns:
                kernels = df['Kernel Name'].unique()
                
                for kernel in kernels[:5]:  # Show first 5 kernels
                    kernel_data = df[df['Kernel Name'] == kernel]
                    
                    print(f"🔷 {kernel}")
                    print("-" * 60)
                    
                    # Extract key metrics
                    for col in df.columns:
                        if 'time_duration' in col.lower():
                            val = kernel_data[col].iloc[0]
                            # Convert nanoseconds to milliseconds
                            if isinstance(val, (int, float)):
                                ms = val / 1_000_000
                                print(f"   ⏱️  Kernel Time: {ms:.4f} ms")
                        
                        elif 'dram__throughput' in col.lower() and 'pct' in col.lower():
                            val = kernel_data[col].iloc[0]
                            if isinstance(val, (int, float)):
                                status = "✅" if val > 80 else "⚠️" if val > 50 else "❌"
                                print(f"   📊 Memory Throughput: {val:.2f}% {status}")
                        
                        elif 'sm__throughput' in col.lower() and 'pct' in col.lower():
                            val = kernel_data[col].iloc[0]
                            if isinstance(val, (int, float)):
                                status = "✅" if val > 70 else "⚠️" if val > 40 else "❌"
                                print(f"   ⚡ SM Throughput: {val:.2f}% {status}")
                        
                        elif 'occupancy' in col.lower() and 'pct' in col.lower():
                            val = kernel_data[col].iloc[0]
                            if isinstance(val, (int, float)):
                                status = "✅" if val > 60 else "⚠️" if val > 40 else "❌"
                                print(f"   🔧 Occupancy: {val:.2f}% {status}")
                    
                    print()
            else:
                print("📊 Metric columns:")
                for col in df.columns:
                    print(f"   - {col}")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"⚠️  Could not display summary: {e}\n")
    
    def quick_profile(self, script_name="bench_baseline_kernels_profile.py"):
        """
        Quick profile with optimized settings
        
        Use lightweight profiling benchmarks for best results
        """
        
        print("\n" + "="*80)
        print("QUICK PROFILE MODE")
        print("="*80 + "\n")
        
        print("💡 Using lightweight profiling benchmark")
        print("   (10 iterations per kernel, optimized for ncu)\n")
        
        benchmarks_dir = Path("benchmarks")
        profile_script = benchmarks_dir / script_name
        
        if not profile_script.exists():
            print(f"❌ Script not found: {profile_script}")
            print("\n📝 Available lightweight profiling scripts:")
            print("   - bench_baseline_kernels_profile.py")
            print("   - bench_fusion3_profile.py")
            return None
        
        return self.profile_benchmark(
            str(profile_script),
            timeout_seconds=120  # 2 minutes
        )


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         NSIGHT COMPUTE PROFILER - WSL OPTIMIZED                              ║
║              Lightweight profiling for GPU metrics                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("🎯 Required Metrics (Grading Rubric):")
    print("   1. ⏱️  Kernel Time")
    print("   2. 📊 Memory Throughput")
    print("   3. 🔧 GPU Occupancy")
    print("   4. ⚡ SM Usage (Compute)")
    print("   5. 📈 Inference Time (from CNN benchmarks)\n")
    
    profiler = NsightProfilerWSL()
    
    # Check if ncu is available
    if not profiler.check_ncu_available():
        print("\n❌ Nsight Compute not found!")
        print("\n💡 On WSL, you may need to:")
        print("   1. Install CUDA Toolkit")
        print("   2. Add ncu to PATH")
        print("   3. Ensure NVIDIA driver is installed on Windows")
        sys.exit(1)
    
    # Quick profile mode
    if len(sys.argv) > 1:
        script = sys.argv[1]
        print(f"📊 Profiling: {script}\n")
        
        # Determine timeout based on script
        if "profile" in script.lower():
            timeout = 120  # Lightweight scripts: 2 min
        else:
            timeout = 300  # Regular benchmarks: 5 min
        
        profiler.profile_benchmark(script, timeout_seconds=timeout)
    else:
        # Default: profile lightweight baseline
        print("📊 No script specified, using default lightweight profile\n")
        profiler.quick_profile("bench_baseline_kernels_profile.py")
    
    print("\n" + "="*80)
    print("✅ PROFILING SESSION COMPLETE")
    print("="*80)
    print("\n📁 Generated files:")
    print("   - profiling_results/*.ncu-rep (profile data)")
    print("   - profiling_results/*_metrics.csv (extracted metrics)")
    print("\n💡 Next steps:")
    print("   1. Profile your best fusion: python nsight_profiling.py benchmarks/bench_fusion3_profile.py")
    print("   2. Create master summary table")
    print("   3. Add results to report\n")


if __name__ == "__main__":
    main()