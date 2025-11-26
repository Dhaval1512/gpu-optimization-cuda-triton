#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction for sum (static to avoid multiple definition)
static __device__ float warp_reduce_sum_fusion3(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum using shared memory (static to avoid multiple definition)
static __device__ float block_reduce_sum_fusion3(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum_fusion3(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum_fusion3(val);
    }
    
    // Broadcast result to all threads
    if (wid == 0 && lane == 0) {
        shared[0] = val;
    }
    __syncthreads();
    
    return shared[0];
}

extern "C" __global__
void fused_ln_gelu_swish_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int M,  // Batch size
    int N,  // Feature dimension
    float eps)
{
    // Shared memory for reductions
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* x_row = x + row * N;
    float* y_row = y + row * N;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // === STEP 1: Compute mean ===
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += num_threads) {
        thread_sum += x_row[i];
    }
    
    float total_sum = block_reduce_sum_fusion3(thread_sum, shared);
    float mean = total_sum / N;
    
    // === STEP 2: Compute variance ===
    float thread_var_sum = 0.0f;
    for (int i = tid; i < N; i += num_threads) {
        float diff = x_row[i] - mean;
        thread_var_sum += diff * diff;
    }
    
    float total_var_sum = block_reduce_sum_fusion3(thread_var_sum, shared);
    float variance = total_var_sum / N;
    float rstd = rsqrtf(variance + eps);
    
    // Ensure all threads see mean and rstd
    __syncthreads();
    
    // === STEP 3-7: Normalize, Scale/Shift, GELU, Swish, Combine ===
    for (int i = tid; i < N; i += num_threads) {
        // LayerNorm
        float x_val = x_row[i];
        float x_norm = (x_val - mean) * rstd;
        float x_ln = x_norm * gamma[i] + beta[i];
        
        // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        const float inv_sqrt2 = 0.7071067811865475f;
        float gelu_out = 0.5f * x_ln * (1.0f + erff(x_ln * inv_sqrt2));
        
        // Swish: x * sigmoid(x) = x / (1 + exp(-x))
        float swish_out = x_ln / (1.0f + expf(-x_ln));
        
        // Combine
        y_row[i] = gelu_out + swish_out;
    }
}