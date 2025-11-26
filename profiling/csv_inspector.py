"""
CSV Inspector - See what's actually in your benchmark files
Run this first to understand your data structure
"""

import pandas as pd
from pathlib import Path

def inspect_csv(filepath):
    """
    Inspect a single CSV file
    """
    if not Path(filepath).exists():
        print(f"❌ Not found: {filepath}\n")
        return None
    
    print(f"\n{'='*70}")
    print(f"📄 FILE: {filepath}")
    print('='*70)
    
    try:
        df = pd.read_csv(filepath)
        
        print(f"\n📊 Basic Info:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        print(f"\n📋 Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n👀 First 3 rows:")
        print(df.head(3).to_string())
        
        print(f"\n📈 Summary Statistics:")
        print(df.describe().to_string())
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading file: {e}\n")
        return None


def inspect_all_csvs(report_dir="report"):
    """
    Inspect all CSV files in report directory
    """
    report_path = Path(report_dir)
    
    if not report_path.exists():
        print(f"❌ Directory not found: {report_dir}")
        return
    
    csv_files = sorted(report_path.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {report_dir}")
        return
    
    print(f"\n🔍 INSPECTING ALL CSV FILES IN: {report_dir}")
    print(f"Found {len(csv_files)} CSV files\n")
    
    results = {}
    for csv_file in csv_files:
        df = inspect_csv(csv_file)
        if df is not None:
            results[csv_file.name] = df
    
    # Create summary
    print(f"\n\n{'='*70}")
    print("📝 SUMMARY OF ALL FILES")
    print('='*70)
    
    for filename, df in results.items():
        print(f"\n{filename}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")


def quick_peek(report_dir="report"):
    """
    Quick peek at key files
    """
    key_files = [
        "baseline_kernels_benchmarks.csv",
        "fusion_ln_swiss_gelu_kernels_benchmarks.csv",
        "fusion_swiss_gelu_kernels_benchmarks.csv",
        "fusion_ln_swish_dropout_kernel_benchmarks.csv",
        "fusion_ln_swiss_gelu_cnn_inference.csv",
        "gpu_profiling_summary.csv",
        "workload_variations.csv",
    ]
    
    print("🔍 QUICK PEEK AT KEY FILES")
    print("="*70)
    
    for filename in key_files:
        filepath = Path(report_dir) / filename
        if filepath.exists():
            print(f"\n✅ {filename}")
            df = pd.read_csv(filepath)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
        else:
            # Try alternate spellings
            alt_names = [
                filename.replace("swiss", "swish"),
                filename.replace("swish", "swiss")
            ]
            found = False
            for alt_name in alt_names:
                alt_path = Path(report_dir) / alt_name
                if alt_path.exists():
                    print(f"\n✅ {alt_name} (alternate spelling)")
                    df = pd.read_csv(alt_path)
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    found = True
                    break
            
            if not found:
                print(f"\n❌ {filename} - NOT FOUND")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Inspect specific file
        inspect_csv(sys.argv[1])
    else:
        # Quick peek at all files
        quick_peek()
        
        print("\n\n" + "="*70)
        print("💡 TIP: To see detailed info for a specific file, run:")
        print("   python csv_inspector.py report/your_file.csv")
        print("="*70)