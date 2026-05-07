import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

# Use g++-10 as host compiler (nvcc --ccbin).
# CUDA 11.x only supports GCC <= 10; GCC 11 headers are incompatible.
_NVCC_ARGS = ['-O3', '-allow-unsupported-compiler', '-ccbin', '/usr/bin/g++-10']


setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': _NVCC_ARGS,
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': _NVCC_ARGS,
            }),
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen-3.4.0'),
                '/usr/lib/cuda/include'],  # cuda.h for lietorch_gpu.h
            sources=[
                'dpvo/lietorch/src/lietorch.cpp',
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': _NVCC_ARGS,}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

