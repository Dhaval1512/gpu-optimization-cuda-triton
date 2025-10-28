__device__ __forceinline__ float sigmoidf_fast(float x){
    return 1.f / (1.f + expf(-x));
}
extern "C" __global__
void swish_forward(const float* __restrict__ x, float* __restrict__ y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float s = sigmoidf_fast(x[i]);
        y[i] = x[i] * s;
    }
}
