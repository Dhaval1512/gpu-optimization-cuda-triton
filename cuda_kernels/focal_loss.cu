// cuda_kernels/focal_loss.cu
// Focal Loss for addressing class imbalance in classification tasks
// Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void focal_loss_forward(
    const float* __restrict__ log_probs,  // [N, C] - log probabilities from log_softmax
    const long* __restrict__ targets,      // [N] - ground truth class indices
    float* __restrict__ losses,            // [N] - output per-sample loss
    int N,                                 // batch size
    int C,                                 // number of classes
    float alpha,                           // weighting factor (typically 0.25)
    float gamma)                           // focusing parameter (typically 2.0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int target_class = targets[idx];
        
        // Get log probability of the correct class
        float log_pt = log_probs[idx * C + target_class];
        
        // Convert to probability: pt = exp(log_pt)
        float pt = expf(log_pt);
        
        // Focal loss formula: -alpha * (1 - pt)^gamma * log(pt)
        float focal_weight = powf(1.0f - pt, gamma);
        
        losses[idx] = -alpha * focal_weight * log_pt;
    }
}

// Reduce losses to scalar (mean)
extern "C" __global__
void reduce_loss_mean(
    const float* __restrict__ losses,  // [N]
    float* __restrict__ output,        // [1]
    int N)
{
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    float sum = 0.0f;
    if (idx < N) {
        sum = losses[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0] / N);
    }
}