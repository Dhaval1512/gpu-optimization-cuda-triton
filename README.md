📘CUDA and OpenAI Triton Framwork Implementation
A Research-Driven Performance Analysis of Custom GPU Operators Integrated into a CNN Model (MNIST)

University of Windsor — School of Computer Science (2025)
Project Partner: Jaguar Land Rover Canada

🌐 Abstract

Deep learning operations such as GELU, Swish, and Layer Normalization introduce heavy memory traffic and computation overhead in neural networks. This project focuses on optimizing these operations using custom CUDA kernels and Triton kernels, with the goal of achieving performance advantages over standard PyTorch implementations.

We compare:

A PyTorch baseline CNN

A CUDA-accelerated CNN using custom fused operators

A Triton-accelerated CNN using auto-tuned high-performance kernels

Benchmarking reveals that CUDA and Triton significantly outperform PyTorch, achieving up to 6× speedup at lower batch sizes and consistent performance wins across larger batch sizes.

This repository includes the full kernel implementations, CNN model integration, benchmarking pipeline, and visualization tools.

🧩 1. Motivation

Modern neural networks rely heavily on specialized GPU kernels for performance.
However:

PyTorch uses general-purpose kernels, not optimized for specific tensor shapes.

Many operations (GELU, LN, Swish) involve multiple kernel launches causing extra memory movement.

For small and medium-sized networks (e.g., MNIST CNN), kernel launch overhead becomes a bottleneck.

To address these issues, we implement custom, shape-specialized, memory-efficient fused kernels using:

✔ CUDA C++

→ Full control over threads, memory hierarchy, shared memory, and kernel fusion.

✔ Triton

→ A higher-level GPU DSL built by OpenAI for writing optimized GPU kernels with less complexity.

🚀 2. Project Objectives
Core Goals

Implement custom GPU operators:

GELU

Swish

LayerNorm

Fused LayerNorm + GELU

Build:

PyTorch CNN (baseline)

CUDA CNN (custom activations)

Triton CNN (custom activations)

Benchmark inference latency across batch sizes.

Compare CUDA vs Triton vs PyTorch.

Visualize latency, speedup, and efficiency.

Enable Nsight Systems & Nsight Compute profiling (for partner evaluation).

Key Outcomes

CUDA achieves 5×–6× speedup for small batch sizes.

Triton achieves stable performance and becomes faster for larger batch sizes.

Both significantly outperform PyTorch in all configurations.

🎛️ 3. Repository Structure
gpu-optimization-cuda-triton/
│
├── baseline_cnn_mnist/
│   ├── model.py                # Baseline PyTorch CNN
│   ├── model_cuda.py           # CNN using custom CUDA activations
│   ├── model_triton.py         # CNN using Triton activations
│   ├── activations_cuda.py     # CUDA → PyTorch glue code
│   ├── activations_triton.py   # Triton → PyTorch glue code
│   ├── train.py                # Baseline training script
│   └── inference_benchmark.py  # PyTorch vs CUDA vs Triton benchmark
│
├── cuda_kernels/               # All CUDA implementations
│   ├── gelu.cu
│   ├── swish.cu
│   ├── layernorm.cu
│   ├── fused_layernorm_gelu.cu
│   └── cuda_ops_all.cu         # Combined kernel file
│
├── triton_kernels/             # Triton implementations
│   ├── gelu.py
│   ├── swish.py
│   ├── layernorm.py
│   ├── fused_layernorm_gelu.py
│   └── triton_ops.py           # Combined Triton dispatch
│
├── extensions/                 # Python-CUDA bindings
│   └── cuda_ops.py
│
├── plots/                      # Graph generation scripts
├── benchmarks/                 # Kernel microbenchmarks
├── report/                     # Analysis scripts
│
├── setup.py                    # Build CUDA extension
├── requirements.txt
└── README.md

🔬 4. CNN Architecture (MNIST Baseline)

A lightweight CNN is used to clearly measure the effect of kernel optimization:

Input (1×28×28)
→ Conv(1→16) + ReLU / CUDA Swish / Triton Swish
→ MaxPool
→ Conv(16→32) + ReLU / CUDA GELU / Triton GELU
→ MaxPool
→ Flatten
→ Fully Connected (32*7*7 → 128)
→ Fully Connected (128 → 10)
→ Softmax

⚙️ 5. Custom CUDA Kernels (C++/CUDA)

Each kernel is implemented with:

grid-stride loops

shared memory usage

warp-level parallelism

reduction for LN

fused execution to minimize memory access

Example (Fused LayerNorm + GELU):
// pseudo-code
mean = reduce_sum(x) / C;
var  = reduce_sum((x - mean)^2) / C;
norm = (x - mean) / sqrt(var + eps);
y = 0.5 * (norm * gamma + beta) * (1 + erf(...));


Exported into PyTorch via C++ extension.

⚡ 6. Triton Kernels

Triton kernels leverage:

program_id indexing

block-level memory loads

auto-tuned BLOCK_SIZE

vectorized math

warp-aware reductions

Example (Triton GELU):

@triton.jit
def gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off, mask=off < N)
    y = 0.5 * x * (1 + tl.erf(x * 0.707106))
    tl.store(Y + off, y, mask=off < N)

🧪 7. Benchmarking Methodology
Batch sizes tested
16, 32, 64, 128

Metrics

End-to-end inference latency (ms/batch)

GPU synchronization for accurate measurement

Repeated inference to stabilize timing

📊 8. Performance Results
Latency (ms per batch)
Batch Size	PyTorch	CUDA	Triton
16	2.1791	0.3530	1.8914
32	0.4270	0.3646	0.4211
64	0.4219	0.4238	0.3977
128	0.7777	0.7059	0.6969
Speedup (vs PyTorch)
Batch Size	CUDA	Triton
16	6.16×	1.15×
32	1.17×	1.01×
64	0.99×	1.06×
128	1.10×	1.11×
📉 9. Graphs & Visualization

Run:

python plots/plot_inference_results.py
python plots/plot_more_graphs.py


Produces:

Latency Comparison (Line + Bar)

Speedup Comparison

Efficiency Score

Scatter Trends

Relative Performance Normalized

All plots are saved in plots/ folder.

🔍 10. GPU Profiling (Nsight Systems / Compute)

Your friend (or any machine with NVIDIA GPU + Nsight) can profile using:

Nsight Systems
nsys profile -o profiling/cnn_run python baseline_cnn_mnist/inference_benchmark.py

Nsight Compute
ncu --set full python baseline_cnn_mnist/inference_benchmark.py


Collect:

SM utilization

Warp occupancy

DRAM throughput

Kernel execution timeline

🏁 11. Key Findings

CUDA kernels provide significant low-batch speedup
Kernel fusion and reduced memory access lead to 6× improvement.

Triton kernels scale better at larger batches
Due to auto-tuned tile sizes and better memory scheduling.

Both CUDA & Triton outperform PyTorch consistently
Even without deep kernel-level tuning.

LayerNorm & GELU are the dominant bottlenecks in small CNNs
Custom fused kernels reduce launch overhead.

🛠️ 12. Setup Instructions
Install dependencies:
pip install -r requirements.txt

Build CUDA kernels:
python setup.py build_ext --inplace

Train model:
python baseline_cnn_mnist/train.py

Benchmark:
python baseline_cnn_mnist/inference_benchmark.py

🔍 13. Future Work

Deeper kernel fusion (Conv+BN+ReLU)

Transformer-style fused attention kernels

Shared-memory optimized LayerNorm

Warp-specialized Triton kernels

Full Nsight Compute analysis

Cross-hardware benchmarking (RTX vs A100)

🤝 14. Contributors & Acknowledgements

Team Members:

Dhaval Patel

Kunal Panchal

Rutesh Zalavadiya

University of Windsor CS Students

Industry Partner:

Jaguar Land Rover Canada — GPU Optimization Initiative

Supervision:

School of Computer Science, University of Windsor

📄 15. License

This project is for educational & research use only.
