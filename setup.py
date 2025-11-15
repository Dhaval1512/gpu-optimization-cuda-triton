from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="cuda_ops",
            sources=["cuda_kernels/cuda_ops_all.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
