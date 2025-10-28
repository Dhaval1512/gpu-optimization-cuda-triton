__device__ __forceinline__ float gelu_erf(float x){
    const float inv_sqrt2 = 0.7071067811865475f; // 1/sqrt(2)
    // use CUDA's erff
    return 0.5f * x * (1.f + erff(x * inv_sqrt2));
}

extern "C" __global__
void gelu_forward(const float* __restrict__ x, float* __restrict__ y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = gelu_erf(x[i]);
}
