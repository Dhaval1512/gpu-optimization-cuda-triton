#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum using shared memory
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    // Broadcast result to all threads
    if (wid == 0 && lane == 0) {
        shared[0] = val;
    }
    __syncthreads();
    
    return shared[0];
}

extern "C" __global__
void fused_ln_swish_dropout_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    unsigned char* __restrict__ mask,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int M,  // Batch size
    int N,  // Feature dimension
    float dropout_p,
    float eps,
    unsigned long long seed)
{
    // Shared memory for reductions
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* x_row = x + row * N;
    float* y_row = y + row * N;
    unsigned char* mask_row = mask + row * N;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float keep_prob = 1.0f - dropout_p;
    
    // === STEP 1: Compute mean ===
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += num_threads) {
        thread_sum += x_row[i];
    }
    
    float total_sum = block_reduce_sum(thread_sum, shared);
    float mean = total_sum / N;
    
    // === STEP 2: Compute variance ===
    float thread_var_sum = 0.0f;
    for (int i = tid; i < N; i += num_threads) {
        float diff = x_row[i] - mean;
        thread_var_sum += diff * diff;
    }
    
    float total_var_sum = block_reduce_sum(thread_var_sum, shared);
    float variance = total_var_sum / N;
    float rstd = rsqrtf(variance + eps);
    
    // Ensure all threads see the mean and rstd
    __syncthreads();
    
    // === STEP 3-6: Normalize, Scale/Shift, Swish, Dropout ===
    // Initialize random state for this thread
    curandState_t state;
    curand_init(seed, row * N + tid, 0, &state);
    
    for (int i = tid; i < N; i += num_threads) {
        // LayerNorm: normalize
        float x_val = x_row[i];
        float x_norm = (x_val - mean) * rstd;
        
        // LayerNorm: scale and shift
        float x_scaled = x_norm * gamma[i] + beta[i];
        
        // Swish: x * sigmoid(x)
        float swish_out = x_scaled / (1.0f + expf(-x_scaled));
        
        // Dropout
        float rand_val = curand_uniform(&state);
        bool keep = (rand_val < keep_prob);
        mask_row[i] = keep ? 1 : 0;
        
        // Apply dropout with scaling
        y_row[i] = keep ? (swish_out / keep_prob) : 0.0f;
    }
}