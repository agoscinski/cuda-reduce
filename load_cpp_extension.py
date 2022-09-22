import sys

import torch.utils.cpp_extension

if sys.platform.startswith("darwin"):
    extra_cflags = ["-O3", "-Xpreprocessor -fopenmp"]
    extra_ldflags = ["-L/opt/homebrew/lib", "-lomp"]
elif sys.platform.startswith("linux"):
    extra_cflags = ["-O3", "-fopenmp"]
    extra_ldflags = ["-fopenmp"]

torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=[
        "src/bindings.cpp",
        "src/reduce.cpp",
        "src/reduce_autograd.cpp",
        "src/reduce_cpu.cpp",
        "src/reduce_cuda.cu",
    ],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=extra_ldflags,
    is_python_module=False,
    verbose=True,
)
