from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_conv2d_cuda',
    ext_modules=[
        CUDAExtension('sparse_conv2d_cuda', [
            'sparse_conv2d_cuda.cpp',
            'sparse_conv2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })