from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='reduce_cpp',
      ext_modules=[cpp_extension.CppExtension('reduce_cpp', ['reduce.cpp', 'bindings.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

