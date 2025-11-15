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

## Repository Structure

```
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- run_train.sh
|-- setup.py
|-- Update_of_Project.md
|-- baseline_cnn_mnist
|   |-- activations_cuda.py
|   |-- activations_triton.py
|   |-- inference_benchmark.py
|   |-- model.py
|   |-- model_cuda.py
|   |-- model_triton.py
|   |-- train.py
|   |-- utils.py
|   \-- __init__.py
|-- benchmarks
|   |-- bench_kernels.py
|   \-- __init__.py
|-- cuda_kernels
|   |-- bindings.cpp
|   |-- cuda_ops_all.cu
|   |-- fused_layernorm_gelu.cu
|   |-- gelu.cu
|   |-- layernorm.cu
|   \-- swish.cu
|-- extensions
|   \-- cuda_ops.py
|-- plots
|   |-- plot_inference_results.py
|   |-- plot_kernel_benchmarks.py
|   \-- plot_more_graphs.py
|-- profiling
|   \-- nsight
|       \-- compute
|           \-- profile_single_kernel.py
|-- report
|   \-- kernel_benchmarks.csv
|-- tests
|   |-- test_triton_ops.py
|   \-- __init__.py
\-- triton_kernels
    |-- triton_ops.py
    \-- __init__.py
```

### File Details

- `.gitignore` — ignores Python caches, build artifacts, and dataset folders to keep the repo clean.
- `README.md` — full project narrative covering motivation, architecture, setup, benchmark tables, and partner information.
- `requirements.txt` — pin set of PyTorch/Triton/CuPy/Nsight-related Python dependencies plus optional notebook tooling.
- `run_train.sh` — convenience shell script that invokes the MNIST trainer with tuned defaults and CLI overrides.
- `setup.py` — builds the `cuda_ops` PyTorch extension by compiling `cuda_kernels/cuda_ops_all.cu`.
- `Update_of_Project.md` — this running progress log plus repo overview.
- `baseline_cnn_mnist/activations_cuda.py` — thin wrappers that expose CUDA GELU/Swish custom ops to PyTorch modules.
- `baseline_cnn_mnist/activations_triton.py` — Triton activation wrappers, including safety checks for fused LN+GELU.
- `baseline_cnn_mnist/inference_benchmark.py` — loads shared weights, times baseline/CUDA/Triton CNNs over MNIST batches, and prints latency per configuration.
- `baseline_cnn_mnist/model.py` — baseline PyTorch CNN that mixes standard ReLU activations with log-softmax output.
- `baseline_cnn_mnist/model_cuda.py` — CNN variant that swaps in CUDA Swish/GELU activations to showcase custom kernels.
- `baseline_cnn_mnist/model_triton.py` — CNN variant backed by Triton Swish/GELU activations for comparison.
- `baseline_cnn_mnist/train.py` — downloads MNIST, trains the baseline CNN, and saves `baseline_cnn_mnist.pth` for reuse across variants.
- `baseline_cnn_mnist/utils.py` — common helper that times model inference via CUDA synchronizations.
- `baseline_cnn_mnist/__init__.py` — empty marker so the folder can be imported as a package.
- `benchmarks/bench_kernels.py` — CUDA-event microbenchmark harness that times GELU/Swish/LayerNorm/Fused LN+GELU for PyTorch vs CUDA vs Triton and writes the CSV in `report/`.
- `benchmarks/__init__.py` — empty package initializer.
- `cuda_kernels/bindings.cpp` — pybind11 bridge wiring tensor arguments to raw CUDA kernel launches for each op.
- `cuda_kernels/cuda_ops_all.cu` — consolidated CUDA implementation: device kernels plus Torch wrappers and module registration.
- `cuda_kernels/fused_layernorm_gelu.cu` — standalone fused LN+GELU kernel using shared-memory reductions and erf-based activation.
- `cuda_kernels/gelu.cu` — lightweight GELU kernel that matches PyTorch’s erf formulation.
- `cuda_kernels/layernorm.cu` — LayerNorm kernel performing per-row reductions, normalization, and affine transform.
- `cuda_kernels/swish.cu` — Swish kernel with inlined sigmoid approximation.
- `extensions/cuda_ops.py` — Python-facing wrappers that call into the compiled `cuda_ops` extension with shape/epsilon defaults.
- `plots/plot_inference_results.py` — uses manually entered CNN latency data to produce latency and speedup line plots.
- `plots/plot_kernel_benchmarks.py` — reads `report/kernel_benchmarks.csv` and renders runtime bars plus a speedup comparison figure.
- `plots/plot_more_graphs.py` — generates additional latency, speedup, relative performance, efficiency, and scatter plots from the same inference data.
- `profiling/nsight/compute/profile_single_kernel.py` — placeholder script reserved for Nsight Compute profiling commands.
- `report/kernel_benchmarks.csv` — numeric output from `benchmarks/bench_kernels.py` listing mean/std latency per kernel implementation.
- `tests/test_triton_ops.py` — GPU correctness tests comparing every Triton kernel against PyTorch references and printing max diffs.
- `tests/__init__.py` — empty file to allow `tests` to be imported as a module.
- `triton_kernels/triton_ops.py` — Triton JIT kernels plus Python launch helpers for GELU, Swish, LayerNorm, and fused LN+GELU.
- `triton_kernels/__init__.py` — empty module initializer for Triton kernels.
