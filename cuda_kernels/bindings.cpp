#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare CUDA kernels (defined in .cu files)
extern void gelu_forward(const float* x, float* y, int N);
extern void swish_forward(const float* x, float* y, int N);
extern void layernorm_forward(const float* x, const float* gamma, const float* beta,
                              float* y, int rows, int cols, float eps);
extern void fused_ln_gelu_forward(const float* x, const float* gamma, const float* beta,
                                  float* y, int rows, int cols, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("gelu_forward", [](torch::Tensor x, torch::Tensor y, int N) {

        dim3 grid((N + 255) / 256);
        dim3 block(256);

        void* args[] = {
            (void*)&x.data_ptr<float>(),
            (void*)&y.data_ptr<float>(),
            (void*)&N
        };

        cudaLaunchKernel(
            (void*)gelu_forward,
            grid, block,
            args, 0, nullptr
        );
    });

    m.def("swish_forward", [](torch::Tensor x, torch::Tensor y, int N) {

        dim3 grid((N + 255) / 256);
        dim3 block(256);

        void* args[] = {
            (void*)&x.data_ptr<float>(),
            (void*)&y.data_ptr<float>(),
            (void*)&N
        };

        cudaLaunchKernel(
            (void*)swish_forward,
            grid, block,
            args, 0, nullptr
        );
    });

    m.def("layernorm_forward", [](torch::Tensor x,
                                  torch::Tensor gamma,
                                  torch::Tensor beta,
                                  torch::Tensor y,
                                  int rows, int cols, float eps) {

        dim3 grid(rows);
        dim3 block(256);
        size_t shared = 2 * 256 * sizeof(float);

        void* args[] = {
            (void*)&x.data_ptr<float>(),
            (void*)&gamma.data_ptr<float>(),
            (void*)&beta.data_ptr<float>(),
            (void*)&y.data_ptr<float>(),
            (void*)&rows,
            (void*)&cols,
            (void*)&eps
        };

        cudaLaunchKernel(
            (void*)layernorm_forward,
            grid, block,
            args, shared, nullptr
        );
    });

    m.def("fused_ln_gelu_forward", [](torch::Tensor x,
                                      torch::Tensor gamma,
                                      torch::Tensor beta,
                                      torch::Tensor y,
                                      int rows, int cols, float eps) {

        dim3 grid(rows);
        dim3 block(256);
        size_t shared = 2 * 256 * sizeof(float);

        void* args[] = {
            (void*)&x.data_ptr<float>(),
            (void*)&gamma.data_ptr<float>(),
            (void*)&beta.data_ptr<float>(),
            (void*)&y.data_ptr<float>(),
            (void*)&rows,
            (void*)&cols,
            (void*)&eps
        };

        cudaLaunchKernel(
            (void*)fused_ln_gelu_forward,
            grid, block,
            args, shared, nullptr
        );
    });
}
