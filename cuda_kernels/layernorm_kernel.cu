extern "C" __global__
void layernorm_forward(
    const float* __restrict__ x,     // [rows, cols]
    const float* __restrict__ gamma, // [cols]
    const float* __restrict__ beta,  // [cols]
    float* __restrict__ y,           // [rows, cols]
    int rows, int cols, float eps)
{
    // one block per row; threads stride over cols
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];       // size = 2 * blockDim.x
    float* s_sum  = sdata;                  // partial sums
    float* s_sumsq= sdata + blockDim.x;     // partial squared sums

    // 1) compute sum and sumsq with strided loop
    float sum = 0.f;
    float sumsq = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[r*cols + c];
        sum   += v;
        sumsq += v * v;
    }
    s_sum[threadIdx.x]   = sum;
    s_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    // 2) parallel reduction
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
        int idx = r*cols + c;
        float xn = (x[idx] - mean) * inv_std;
        y[idx] = xn * gamma[c] + beta[c];
    }
}
