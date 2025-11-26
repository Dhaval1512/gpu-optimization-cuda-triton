extern "C" __global__
void fused_ln_gelu_forward(
    const float* __restrict__ x,     // [rows, cols]
    const float* __restrict__ gamma, // [cols]
    const float* __restrict__ beta,  // [cols]
    float* __restrict__ y,           // [rows, cols]
    int rows, int cols, float eps)
{
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];       // 2*blockDim.x
    float* s_sum   = sdata;
    float* s_sumsq = sdata + blockDim.x;

    // partials
    float sum = 0.f, sumsq = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[r*cols + c];
        sum   += v;
        sumsq += v*v;
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
    float mean = s_sum[0] / cols;
    float var  = s_sumsq[0] / cols - mean*mean;
    float inv_std = rsqrtf(var + eps);

    // erf-based GELU after affine
    const float inv_sqrt2 = 0.7071067811865475f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        int idx = r*cols + c;
        float xn = (x[idx] - mean) * inv_std;
        float h = xn * gamma[c] + beta[c];
        y[idx] = 0.5f * h * (1.f + erff(h * inv_sqrt2));
    }
}
