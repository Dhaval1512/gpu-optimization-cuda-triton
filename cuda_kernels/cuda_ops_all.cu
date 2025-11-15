// cuda_kernels/cuda_ops_all.cu

// Windows compatibility fixes
#ifdef _WIN32
#define NOMINMAX
#pragma warning(disable: 4067)
#pragma warning(disable: 4624)
#pragma warning(disable: 4819)
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


namespace {

// ================== GELU (erf version) ==================

__device__ __forceinline__ float gelu_erf(float x) {
    const float inv_sqrt2 = 0.7071067811865475f; // 1/sqrt(2)
    return 0.5f * x * (1.f + erff(x * inv_sqrt2));
}

__global__ void gelu_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = gelu_erf(x[i]);
    }
}

// ================== Swish ==================

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.f / (1.f + expf(-x));
}

__global__ void swish_kernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float s = sigmoidf_fast(x[i]);
        y[i] = x[i] * s;
    }
}

// ================== LayerNorm ==================
// x: [rows, cols]
// gamma, beta: [cols]

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows, int cols, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    float* s_sum   = sdata;                   // [blockDim.x]
    float* s_sumsq = sdata + blockDim.x;      // [blockDim.x]

    // 1) partial sums
    float sum = 0.f;
    float sumsq = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[r * cols + c];
        sum   += v;
        sumsq += v * v;
    }
    s_sum[threadIdx.x]   = sum;
    s_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    // 2) reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]   += s_sum[threadIdx.x + stride];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / cols;
    float var  = s_sumsq[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // 3) normalize + affine
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        int idx = r * cols + c;
        float xn = (x[idx] - mean) * inv_std;
        y[idx] = xn * gamma[c] + beta[c];
    }
}

// ================== Fused LN + GELU ==================

__global__ void fused_ln_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows, int cols, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    float* s_sum   = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // partials
    float sum = 0.f;
    float sumsq = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[r * cols + c];
        sum   += v;
        sumsq += v * v;
    }
    s_sum[threadIdx.x]   = sum;
    s_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    // reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]   += s_sum[threadIdx.x + stride];
            s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean    = s_sum[0] / cols;
    float var     = s_sumsq[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    const float inv_sqrt2 = 0.7071067811865475f;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        int idx = r * cols + c;
        float xn = (x[idx] - mean) * inv_std;
        float h  = xn * gamma[c] + beta[c];
        // GELU (erf)
        y[idx] = 0.5f * h * (1.f + erff(h * inv_sqrt2));
    }
}

} // namespace

// ================== Host wrappers (PyTorch API) ==================

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    int N = x_contig.numel();
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );

    return y.view_as(x);
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    int N = x_contig.numel();
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    swish_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );

    return y.view_as(x);
}

torch::Tensor layernorm_forward(torch::Tensor x,
                                torch::Tensor gamma,
                                torch::Tensor beta,
                                double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be [cols]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");

    auto x_contig = x.contiguous();
    auto g_contig = gamma.contiguous();
    auto b_contig = beta.contiguous();
    auto y = torch::empty_like(x_contig);

    int rows = x_contig.size(0);
    int cols = x_contig.size(1);
    int threads = 256;
    int blocks  = rows;
    size_t shared = 2 * threads * sizeof(float);

    layernorm_kernel<<<blocks, threads, shared>>>(
        x_contig.data_ptr<float>(),
        g_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps)
    );

    return y;
}

torch::Tensor fused_ln_gelu_forward(torch::Tensor x,
                                    torch::Tensor gamma,
                                    torch::Tensor beta,
                                    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be [cols]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type() == torch::kFloat32, "beta must be float32");

    auto x_contig = x.contiguous();
    auto g_contig = gamma.contiguous();
    auto b_contig = beta.contiguous();
    auto y = torch::empty_like(x_contig);

    int rows = x_contig.size(0);
    int cols = x_contig.size(1);
    int threads = 256;
    int blocks  = rows;
    size_t shared = 2 * threads * sizeof(float);

    fused_ln_gelu_kernel<<<blocks, threads, shared>>>(
        x_contig.data_ptr<float>(),
        g_contig.data_ptr<float>(),
        b_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols, static_cast<float>(eps)
    );

    return y;
}

// ================== Focal Loss ==================

__global__ void focal_loss_kernel(
    const float* __restrict__ log_probs,  // [N, C]
    const long* __restrict__ targets,      // [N]
    float* __restrict__ losses,            // [N]
    int N, int C,
    float alpha, float gamma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int target_class = targets[idx];
        float log_pt = log_probs[idx * C + target_class];
        float pt = expf(log_pt);
        float focal_weight = powf(1.0f - pt, gamma);
        losses[idx] = -alpha * focal_weight * log_pt;
    }
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ losses,
    float* __restrict__ output,
    int N)
{
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = (idx < N) ? losses[idx] : 0.0f;
    sdata[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0] / N);
    }
}

torch::Tensor focal_loss_forward(
    torch::Tensor log_probs,
    torch::Tensor targets,
    double alpha,
    double gamma)
{
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(log_probs.dim() == 2, "log_probs must be [N, C]");
    TORCH_CHECK(targets.dim() == 1, "targets must be [N]");
    TORCH_CHECK(log_probs.scalar_type() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kInt64, "targets must be int64");
    
    auto log_probs_contig = log_probs.contiguous();
    auto targets_contig = targets.contiguous();
    
    int N = log_probs_contig.size(0);
    int C = log_probs_contig.size(1);
    
    // Step 1: Compute per-sample losses
    auto losses = torch::empty({N}, log_probs.options());
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    focal_loss_kernel<<<blocks, threads>>>(
        log_probs_contig.data_ptr<float>(),
        targets_contig.data_ptr<long>(),
        losses.data_ptr<float>(),
        N, C,
        static_cast<float>(alpha),
        static_cast<float>(gamma)
    );
    
    // Step 2: Reduce to mean
    auto output = torch::zeros({1}, log_probs.options());
    size_t shared = threads * sizeof(float);
    
    reduce_loss_kernel<<<blocks, threads, shared>>>(
        losses.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward", &gelu_forward, "GELU forward (CUDA)");
    m.def("swish_forward", &swish_forward, "Swish forward (CUDA)");
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("fused_ln_gelu_forward", &fused_ln_gelu_forward, "Fused LayerNorm + GELU forward (CUDA)");
    
    // ADD THIS LINE:
    m.def("focal_loss_forward", &focal_loss_forward, "Focal Loss forward (CUDA)");
}
