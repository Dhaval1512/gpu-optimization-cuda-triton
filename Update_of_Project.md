# JLR GPU Optimization Project -- Progress Summary

## 1. CUDA Kernel Implementation

-   Implemented CUDA kernels for:
    -   GELU
    -   Swish
    -   LayerNorm
    -   Fused LayerNorm + GELU
-   Verified correctness vs PyTorch (max diff \~0).

## 2. Triton Kernel Implementation

-   Implemented Triton versions of:
    -   `triton_gelu`
    -   `triton_swish`
    -   `triton_layernorm`
    -   `triton_fused_ln_gelu`
-   Replaced unsupported `tl.tanh()` with accurate `erf`-based GELU.
-   Achieved high numerical accuracy (max diff \< 1e-6).

## 3. Triton Correctness Tests

-   Created `tests/test_triton_ops.py`.
-   All Triton kernels validated vs PyTorch.
-   Fused LN+GELU achieved excellent accuracy after GELU fix.

## 4. Benchmarking (CUDA vs Triton vs PyTorch)

-   Created `benchmarks/bench_kernels.py`.
-   Benchmarked:
    -   GELU
    -   Swish
    -   LayerNorm
    -   Fused LN+GELU
-   Exported results to `report/kernel_benchmarks.csv`.

### Key Results:

-   CUDA Swish is **1.87× faster** than PyTorch.
-   Triton fused LN+GELU is **1.57× faster** than PyTorch.
-   CUDA fused LN+GELU is **1.38× faster** than PyTorch.
-   PyTorch fastest for simple ops (GELU).
-   Triton performs best for fused complex ops.

## 5. Plotting and Visualization

-   Created `plots/plot_kernel_benchmarks.py`.
-   Generated:
    -   GELU runtime chart
    -   Swish runtime chart
    -   LayerNorm runtime chart
    -   Fused LN+GELU chart
    -   Speedup comparison chart
-   All plots saved in `/plots`.

## 6. Nsight Profiling Setup

-   Created `profiling/profile_single_kernel.py`.
-   Added commands for:
    -   Nsight Systems (`nsys`) -- timeline profiling
    -   Nsight Compute (`ncu`) -- low-level GPU metrics
-   Profiling to be executed on a machine with NVIDIA GPU.

## 7. Friend's Responsibility

Your teammate will: - Run Nsight profiling on Windows/NVIDIA GPU -
Generate `.nsys-rep`, `.ncu-rep`, CSVs - Take screenshots of: - Timeline
(CUDA, Triton, PyTorch) - SM utilization - Memory throughput - Warp
occupancy

## 8. What You Will Do Next

-   Use the data to create:
    -   Final report sections
    -   Performance analysis
    -   Fusion vs Unfused discussion
    -   Presentation slides
