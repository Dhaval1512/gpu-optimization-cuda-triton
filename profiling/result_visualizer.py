"""
Results Visualization Generator
Creates all required charts and tables for hackathon submission
Based on pages 18-20 requirements
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class ResultsVisualizer:
    def __init__(self, results_dir="report", figures_dir="figures"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
    
    def create_operation_comparison_table(self, operation_name, cuda_metrics, triton_metrics):
        """
        Create comparison table for a single operation (GELU, LayerNorm, Swish, Loss)
        
        Args:
            operation_name: Name of operation
            cuda_metrics: dict with keys: time_ms, memory_mb, throughput_samples_sec, gpu_efficiency_pct
            triton_metrics: dict with same keys
        """
        data = {
            'Metric': ['Time (ms)', 'Memory Usage (MB)', 'Inference Speed (samples/sec)', 'GPU Efficiency (%)'],
            'CUDA': [
                f"{cuda_metrics['time_ms']:.3f}",
                f"{cuda_metrics['memory_mb']:.2f}",
                f"{cuda_metrics['throughput_samples_sec']:.1f}",
                f"{cuda_metrics['gpu_efficiency_pct']:.1f}"
            ],
            'Triton': [
                f"{triton_metrics['time_ms']:.3f}",
                f"{triton_metrics['memory_mb']:.2f}",
                f"{triton_metrics['throughput_samples_sec']:.1f}",
                f"{triton_metrics['gpu_efficiency_pct']:.1f}"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate speedup
        speedup = triton_metrics['time_ms'] / cuda_metrics['time_ms']
        
        # Save as LaTeX table
        latex_path = self.results_dir / f"{operation_name}_comparison_table.tex"
        df.to_latex(latex_path, index=False, caption=f"{operation_name} Performance Comparison")
        
        # Save as CSV
        csv_path = self.results_dir / f"{operation_name}_comparison_table.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Table saved: {csv_path}")
        print(f"   Speedup: {speedup:.2f}× ({'CUDA' if speedup < 1 else 'Triton'} faster)")
        
        return df
    
    def create_workload_scaling_chart(self, operation_name, workload_type, sizes, cuda_times, triton_times):
        """
        Create scaling chart showing performance across different workload sizes
        
        Args:
            operation_name: e.g., "GELU"
            workload_type: "batch_size", "sequence_length", or "tensor_dimension"
            sizes: list of sizes tested (e.g., [16, 32, 64, 128, 256])
            cuda_times: list of times in ms
            triton_times: list of times in ms
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Absolute performance
        ax1.plot(sizes, cuda_times, 'o-', label='CUDA', linewidth=2, markersize=8, color='#2E86AB')
        ax1.plot(sizes, triton_times, 's-', label='Triton', linewidth=2, markersize=8, color='#A23B72')
        ax1.set_xlabel(workload_type.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title(f'{operation_name} - {workload_type.replace("_", " ").title()} Scaling', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        speedups = [t / c if c > 0 else 1 for c, t in zip(cuda_times, triton_times)]
        colors = ['#27AE60' if s >= 1 else '#E74C3C' for s in speedups]
        ax2.bar(range(len(sizes)), speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No Speedup')
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels(sizes)
        ax2.set_xlabel(workload_type.replace('_', ' ').title(), fontsize=12)
        ax2.set_ylabel('Speedup (CUDA / Triton)', fontsize=12)
        ax2.set_title(f'{operation_name} - Speedup Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(speedups):
            ax2.text(i, v + 0.05, f'{v:.2f}×', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{operation_name}_{workload_type}_scaling.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Chart saved: {save_path}")
    
    def create_comprehensive_heatmap(self, operation_name, batch_sizes, tensor_dims, cuda_times_2d, triton_times_2d):
        """
        Create 2D heatmap showing performance across batch_size × tensor_dimension
        
        Args:
            operation_name: e.g., "LayerNorm"
            batch_sizes: list of batch sizes
            tensor_dims: list of tensor dimensions
            cuda_times_2d: 2D numpy array (batch_sizes × tensor_dims)
            triton_times_2d: 2D numpy array
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # CUDA heatmap
        sns.heatmap(cuda_times_2d, annot=True, fmt='.2f', cmap='YlOrRd', 
                    xticklabels=tensor_dims, yticklabels=batch_sizes, ax=ax1, cbar_kws={'label': 'Time (ms)'})
        ax1.set_xlabel('Tensor Dimension')
        ax1.set_ylabel('Batch Size')
        ax1.set_title(f'{operation_name} - CUDA Performance', fontweight='bold')
        
        # Triton heatmap
        sns.heatmap(triton_times_2d, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=tensor_dims, yticklabels=batch_sizes, ax=ax2, cbar_kws={'label': 'Time (ms)'})
        ax2.set_xlabel('Tensor Dimension')
        ax2.set_ylabel('Batch Size')
        ax2.set_title(f'{operation_name} - Triton Performance', fontweight='bold')
        
        # Speedup heatmap
        speedup_2d = triton_times_2d / cuda_times_2d
        sns.heatmap(speedup_2d, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                    xticklabels=tensor_dims, yticklabels=batch_sizes, ax=ax3, cbar_kws={'label': 'Speedup'})
        ax3.set_xlabel('Tensor Dimension')
        ax3.set_ylabel('Batch Size')
        ax3.set_title(f'{operation_name} - Speedup (CUDA/Triton)', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.figures_dir / f"{operation_name}_comprehensive_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap saved: {save_path}")
    
    def create_fusion_comparison_chart(self, fusion_results):
        """
        Create comparison chart for all fusions
        
        Args:
            fusion_results: dict with structure:
                {
                    "Fusion 0 (LN+GELU)": {"cuda": 1.23, "triton": 1.45},
                    "Fusion 1 (GELU+Swish)": {"cuda": 0.89, "triton": 1.12},
                    ...
                }
        """
        fusion_names = list(fusion_results.keys())
        cuda_times = [fusion_results[f]["cuda"] for f in fusion_names]
        triton_times = [fusion_results[f]["triton"] for f in fusion_names]
        
        x = np.arange(len(fusion_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, cuda_times, width, label='CUDA', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, triton_times, width, label='Triton', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('Fusion Type', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title('Kernel Fusion Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(fusion_names, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.figures_dir / "fusion_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Fusion comparison saved: {save_path}")
    
    def create_master_summary_table(self, all_operations_data):
        """
        Create master summary table with all operations
        
        Args:
            all_operations_data: dict with structure:
                {
                    "GELU": {"cuda_time": 1.23, "triton_time": 1.45, "speedup": 0.85},
                    ...
                }
        """
        data = []
        for op_name, metrics in all_operations_data.items():
            data.append({
                'Operation': op_name,
                'CUDA Time (ms)': f"{metrics['cuda_time']:.3f}",
                'Triton Time (ms)': f"{metrics['triton_time']:.3f}",
                'Speedup': f"{metrics['speedup']:.2f}×",
                'Winner': 'CUDA' if metrics['speedup'] < 1 else 'Triton'
            })
        
        df = pd.DataFrame(data)
        
        # Save outputs
        csv_path = self.results_dir / "master_summary_table.csv"
        df.to_csv(csv_path, index=False)
        
        latex_path = self.results_dir / "master_summary_table.tex"
        df.to_latex(latex_path, index=False, caption="Complete Performance Summary")
        
        print(f"✅ Master summary saved: {csv_path}")
        return df


# Example usage
def generate_example_visualizations():
    """
    Generate example visualizations with sample data
    """
    viz = ResultsVisualizer()
    
    # Example 1: Single operation comparison table
    gelu_cuda = {
        'time_ms': 0.245,
        'memory_mb': 12.5,
        'throughput_samples_sec': 4096.3,
        'gpu_efficiency_pct': 87.2
    }
    gelu_triton = {
        'time_ms': 0.312,
        'memory_mb': 11.8,
        'throughput_samples_sec': 3205.1,
        'gpu_efficiency_pct': 82.5
    }
    viz.create_operation_comparison_table("GELU", gelu_cuda, gelu_triton)
    
    # Example 2: Workload scaling
    batch_sizes = [16, 32, 64, 128, 256]
    cuda_times = [0.12, 0.24, 0.48, 0.95, 1.89]
    triton_times = [0.15, 0.28, 0.52, 1.01, 2.05]
    viz.create_workload_scaling_chart("LayerNorm", "batch_size", batch_sizes, cuda_times, triton_times)
    
    # Example 3: 2D heatmap
    batch_sizes = [16, 32, 64, 128]
    tensor_dims = [128, 256, 512, 1024]
    cuda_times_2d = np.random.uniform(0.5, 2.0, (4, 4))
    triton_times_2d = np.random.uniform(0.6, 2.2, (4, 4))
    viz.create_comprehensive_heatmap("Swish", batch_sizes, tensor_dims, cuda_times_2d, triton_times_2d)
    
    # Example 4: Fusion comparison
    fusion_results = {
        "Fusion 0\n(LN+GELU)": {"cuda": 1.57, "triton": 1.41},
        "Fusion 1\n(GELU+Swish)": {"cuda": 1.97, "triton": 1.83},
        "Fusion 2\n(LN+Swish+Dropout)": {"cuda": 2.21, "triton": 1.74},
        "Fusion 3\n(LN+GELU+Swish)": {"cuda": 2.94, "triton": 2.67}
    }
    viz.create_fusion_comparison_chart(fusion_results)
    
    # Example 5: Master summary
    all_ops = {
        "GELU": {"cuda_time": 0.245, "triton_time": 0.312, "speedup": 0.78},
        "LayerNorm": {"cuda_time": 0.198, "triton_time": 0.215, "speedup": 0.92},
        "Swish": {"cuda_time": 0.156, "triton_time": 0.142, "speedup": 1.10},
        "Focal Loss": {"cuda_time": 0.487, "triton_time": 0.523, "speedup": 0.93}
    }
    viz.create_master_summary_table(all_ops)
    
    print("\n✅ All example visualizations generated!")


if __name__ == "__main__":
    generate_example_visualizations()