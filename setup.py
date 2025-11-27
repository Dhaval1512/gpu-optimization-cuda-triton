from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

setup(
    name="cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="cuda_ops",
            sources=["cuda_kernels/cuda_ops_all.cu", 
                     "cuda_kernels/fused_gelu_swish_kernel.cu", 
                     "cuda_kernels/fused_ln_swish_dropout_kernel.cu",
                     "cuda_kernels/fused_ln_gelu_swish_kernel.cu"],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-use_fast_math',
                    # Suppress Windows-specific warnings that are blocking compilation
                    '-Xcudafe', '--diag_suppress=1581',  # _alloca warning
                    '-Xcudafe', '--diag_suppress=767',   # pointer conversion warnings
                    '-Xcudafe', '--diag_suppress=20012', # ASM operand warnings
                ]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)