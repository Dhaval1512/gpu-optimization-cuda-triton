#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void fused_gelu_swish_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float val = x[idx];
    
    // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865475f;
    float gelu_out = 0.5f * val * (1.0f + erff(val * inv_sqrt2));
    
    // Swish: x * sigmoid(x) = x / (1 + exp(-x))
    float swish_out = val / (1.0f + expf(-val));
    
    // Combine both activations
    y[idx] = gelu_out + swish_out;
}