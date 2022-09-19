import time

import torch
import torch.utils.cpp_extension

from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark

import reduce_python

torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=["reduce.cpp", "bindings.cpp"],
    extra_cflags=["-O3"],
    is_python_module=False,
)


if __name__ == "__main__":
    n_samples = 10000
    n_features = 1000

    torch.manual_seed(0xDEADBEEF)
    X = torch.rand((n_samples, 5, 6, n_features), requires_grad=True, dtype=torch.float32, device='cuda')
    X_keys = torch.randint(13, (n_samples, 3), dtype=torch.int32, device='cuda')
    print(X.shape)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with record_function("reduce_python"):
            reduce_python.reduce(X, X_keys, 0).mean().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    prof.export_chrome_trace("reduce_python.trace.json")
