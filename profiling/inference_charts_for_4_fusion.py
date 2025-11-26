#!/usr/bin/env python3
"""
Complete Fusion CNN Inference Chart Generator
Generates all 4 fusion charts including Fusion 0 (LN + GELU)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

class CompleteFusionChartGenerator:
    def __init__(self, report_dir="report", figures_dir="figures"):
        self.report_dir = Path(report_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
    
    def create_inference_chart(self, df, fusion_name, output_filename, fusion_type="cnn"):
        """
        Create CNN inference or kernel benchmark bar chart
        
        Args:
            df: DataFrame with benchmark data
            fusion_name: Name for the chart title
            output_filename: Output file name
            fusion_type: "cnn" for CNN inference, "kernel" for kernel benchmarks
        """
        # Get unique batch sizes or shapes
        if 'batch_size' in df.columns:
            x_values = sorted(df['batch_size'].unique())
            x_label = 'Batch Size'
        elif 'shape' in df.columns:
            x_values = df['shape'].unique().tolist()
            x_label = 'Tensor Shape'
        else:
            print(f"⚠️  Cannot determine x-axis values for {fusion_name}")
            return None
        
        # Extract times for each implementation
        pytorch_times = []
        cuda_times = []
        triton_times = []
        
        for x_val in x_values:
            if 'batch_size' in df.columns:
                x_data = df[df['batch_size'] == x_val]
            else:
                x_data = df[df['shape'] == x_val]
            
            # PyTorch unfused
            pytorch_mask = (
                x_data['implementation'].str.contains('pytorch', case=False, na=False) |
                x_data['implementation'].str.contains('unfused', case=False, na=False)
            )
            pytorch_time = x_data[pytorch_mask]['mean_ms'].values
            pytorch_times.append(pytorch_time[0] if len(pytorch_time) > 0 else 0)
            
            # CUDA fused
            cuda_time = x_data[x_data['implementation'].str.contains('cuda', case=False, na=False)]['mean_ms'].values
            cuda_times.append(cuda_time[0] if len(cuda_time) > 0 else 0)
            
            # Triton fused
            triton_time = x_data[x_data['implementation'].str.contains('triton', case=False, na=False)]['mean_ms'].values
            triton_times.append(triton_time[0] if len(triton_time) > 0 else 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(x_values))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, pytorch_times, width, 
                      label='PyTorch Unfused', color='#95A5A6', alpha=0.8)
        bars2 = ax.bar(x, cuda_times, width, 
                      label='CUDA Fused', color='#3498DB', alpha=0.8)
        bars3 = ax.bar(x + width, triton_times, width, 
                      label='Triton Fused', color='#E74C3C', alpha=0.8)
        
        # Labels and title
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Inference Time (ms/batch)' if fusion_type == 'cnn' else 'Execution Time (ms)', 
                     fontsize=12, fontweight='bold')
        ax.set_title(fusion_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_values, rotation=0 if len(str(x_values[0])) < 5 else 15)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.figures_dir / output_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate speedups
        valid_speedups_cuda = [p/c for p, c in zip(pytorch_times, cuda_times) if c > 0 and p > 0]
        valid_speedups_triton = [p/t for p, t in zip(pytorch_times, triton_times) if t > 0 and p > 0]
        
        max_speedup_cuda = max(valid_speedups_cuda) if valid_speedups_cuda else 0
        max_speedup_triton = max(valid_speedups_triton) if valid_speedups_triton else 0
        best_speedup = max(max_speedup_cuda, max_speedup_triton)
        
        print(f"✅ {fusion_name}")
        print(f"   Chart saved: {output_filename}")
        print(f"   Best speedup: {best_speedup:.2f}×")
        print(f"   CUDA max: {max_speedup_cuda:.2f}×, Triton max: {max_speedup_triton:.2f}×\n")
        
        return save_path
    
    def generate_fusion0_from_kernel_data(self):
        """
        Generate Fusion 0 chart from kernel_benchmarks.csv
        Since Fusion 0 has no CNN inference, use kernel-level data
        """
        print("="*80)
        print("FUSION 0: LN + GELU")
        print("="*80)
        
        csv_path = self.report_dir / "baseline_kernels_benchmarks.csv"
        
        if not csv_path.exists():
            print(f"⚠️  {csv_path.name} not found\n")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Filter for fused_ln_gelu operations
        fusion0_data = df[df['kernel'] == 'fused_ln_gelu'].copy()
        
        if fusion0_data.empty:
            print(f"⚠️  No fused_ln_gelu data found in kernel_benchmarks.csv\n")
            return None
        
        # Reshape data for chart generation
        # Add a comparison to unfused (ln + gelu separately)
        ln_data = df[df['kernel'] == 'layernorm']
        gelu_data = df[df['kernel'] == 'gelu']
        
        if not ln_data.empty and not gelu_data.empty:
            # Calculate unfused time (LN + GELU separately)
            for impl in ['pytorch', 'cuda', 'triton']:
                ln_time = ln_data[ln_data['implementation'] == impl]['mean_ms'].values
                gelu_time = gelu_data[gelu_data['implementation'] == impl]['mean_ms'].values
                
                if len(ln_time) > 0 and len(gelu_time) > 0:
                    unfused_time = ln_time[0] + gelu_time[0]
                    
                    # Add unfused row to fusion0_data
                    new_row = fusion0_data[fusion0_data['implementation'] == impl].iloc[0].copy()
                    new_row['implementation'] = f'{impl}_unfused'
                    new_row['mean_ms'] = unfused_time
                    fusion0_data = pd.concat([fusion0_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Rename for consistency
        fusion0_data['batch_size'] = fusion0_data['shape']  # Use shape as pseudo batch size
        
        chart_path = self.create_inference_chart(
            fusion0_data,
            "Fusion 0 Performance (LayerNorm + GELU)",
            "fusion0_kernel_performance.png",
            fusion_type="kernel"
        )
        
        return chart_path
    
    def generate_all_fusion_charts(self):
        """
        Generate CNN inference charts for all 4 fusions
        """
        print(f"\n{'='*80}")
        print(f"  GENERATING ALL 4 FUSION CHARTS")
        print(f"{'='*80}\n")
        
        charts_generated = []
        
        # ========== FUSION 0: LN + GELU ==========
        # Use kernel-level data since no CNN inference exists
        chart = self.generate_fusion0_from_kernel_data()
        if chart:
            charts_generated.append(chart)
        
        # ========== FUSION 1: GELU + SWISH ==========
        print("="*80)
        print("FUSION 1: GELU + SWISH")
        print("="*80)
        
        csv_candidates = [
            "mnist_cnn_fusion_swiss_gelu.csv",
            "fusion_swiss_gelu_cnn_inference.csv",
            "fusion1_cnn_inference.csv"
        ]
        
        csv_path = None
        for candidate in csv_candidates:
            test_path = self.report_dir / candidate
            if test_path.exists():
                csv_path = test_path
                break
        
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            chart_path = self.create_inference_chart(
                df, 
                "Fusion 1 CNN Inference Performance (GELU + Swish)",
                "fusion1_cnn_inference.png"
            )
            if chart_path:
                charts_generated.append(chart_path)
        else:
            print(f"⚠️  No CSV found for Fusion 1\n")
        
        # ========== FUSION 2: LN + SWISH + DROPOUT ==========
        print("="*80)
        print("FUSION 2: LN + SWISH + DROPOUT")
        print("="*80)
        
        csv_candidates = [
            "fusion_ln_swiss_dropout_cnn_inference.csv",
            "fusion_ln_swish_dropout_cnn_inference.csv",
            "fusion2_cnn_inference.csv"
        ]
        
        csv_path = None
        for candidate in csv_candidates:
            test_path = self.report_dir / candidate
            if test_path.exists():
                csv_path = test_path
                break
        
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            chart_path = self.create_inference_chart(
                df,
                "Fusion 2 CNN Inference Performance (LN + Swish + Dropout)",
                "fusion2_cnn_inference.png"
            )
            if chart_path:
                charts_generated.append(chart_path)
        else:
            print(f"⚠️  No CSV found for Fusion 2\n")
        
        # ========== FUSION 3: LN + GELU + SWISH ==========
        print("="*80)
        print("FUSION 3: LN + GELU + SWISH")
        print("="*80)
        
        csv_candidates = [
            "fusion_ln_swiss_gelu_cnn_inference.csv",
            "fusion_ln_swish_gelu_cnn_inference.csv",
            "fusion3_cnn_inference.csv"
        ]
        
        csv_path = None
        for candidate in csv_candidates:
            test_path = self.report_dir / candidate
            if test_path.exists():
                csv_path = test_path
                break
        
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            chart_path = self.create_inference_chart(
                df,
                "Fusion 3 CNN Inference Performance (LN + GELU + Swish)",
                "fusion3_cnn_inference.png"
            )
            if chart_path:
                charts_generated.append(chart_path)
        else:
            print(f"⚠️  No CSV found for Fusion 3\n")
        
        # ========== SUMMARY ==========
        print(f"{'='*80}")
        print(f"  SUMMARY")
        print(f"{'='*80}\n")
        print(f"✅ Generated {len(charts_generated)} fusion charts:")
        for chart in charts_generated:
            print(f"   - {chart.name}")
        print(f"\n📁 All charts saved to: {self.figures_dir}/\n")
        
        # Create a summary comparison table
        self.create_fusion_summary_table(charts_generated)
    
    def create_fusion_summary_table(self, charts):
        """
        Create a summary table of all fusions
        """
        print("="*80)
        print("CREATING FUSION SUMMARY TABLE")
        print("="*80 + "\n")
        
        fusion_summary = {
            'Fusion': ['Fusion 0', 'Fusion 1', 'Fusion 2', 'Fusion 3'],
            'Operations': [
                'LN + GELU',
                'GELU + Swish',
                'LN + Swish + Dropout',
                'LN + GELU + Swish'
            ],
            'Chart_File': [
                'fusion0_kernel_performance.png',
                'fusion1_cnn_inference.png',
                'fusion2_cnn_inference.png',
                'fusion3_cnn_inference.png'
            ],
            'Type': [
                'Kernel-level',
                'CNN Inference',
                'CNN Inference',
                'CNN Inference'
            ]
        }
        
        summary_df = pd.DataFrame(fusion_summary)
        summary_path = self.report_dir / "fusion_charts_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Fusion summary saved: {summary_path.name}")
        print(summary_df.to_string(index=False))
        print()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           COMPLETE FUSION CNN INFERENCE CHART GENERATOR                      ║
║                     All 4 Fusions Including Fusion 0                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    generator = CompleteFusionChartGenerator()
    generator.generate_all_fusion_charts()
    
    print("="*80)
    print("✅ ALL FUSION CHARTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print()
    print("📊 Use these charts in your presentation:")
    print("   Slide 6: fusion0_kernel_performance.png (Fusion 0 - kernel level)")
    print("   Slide 7: fusion1_cnn_inference.png (Fusion 1 - CNN)")
    print("   Slide 8: fusion2_cnn_inference.png (Fusion 2 - CNN)")
    print("   Slide 9: fusion3_cnn_inference.png (Fusion 3 - CNN) ⭐ BEST RESULT")
    print()


if __name__ == "__main__":
    main()