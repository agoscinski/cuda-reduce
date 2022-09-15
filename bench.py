import time

import torch
import torch.utils.cpp_extension

import reduce_python

torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=["reduce.cpp", "bindings.cpp"],
    extra_cflags=["-O3"],
    is_python_module=False,
)


def bench(function, input, input_keys, dim, n_iters=10):
    start = time.time()
    for _ in range(n_iters):
        function(input, input_keys, dim)

    elapsed = time.time() - start
    direct = elapsed / n_iters

    elapsed = 0
    for _ in range(n_iters):
        result = function(input, input_keys, dim)
        summed = result.sum()

        start = time.time()
        summed.backward()
        elapsed += time.time() - start

    backward = elapsed / n_iters

    return direct, backward


if __name__ == "__main__":
    n_samples = 10000
    n_features = 1000

    torch.manual_seed(0xDEADBEEF)
    X = torch.rand((n_samples, 7, n_features), requires_grad=True, dtype=torch.float64)
    X_keys = torch.randint(4, (n_samples, 3), dtype=torch.int32)

    print("implementation  | forward pass | backward pass")

    forward, backward = bench(reduce_python.reduce, X, X_keys, 0)
    print(f"python function =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(reduce_python.reduce_custom_autograd, X, X_keys, 0)
    print(f"python autograd =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(torch.ops.reduce_cpp.reduce, X, X_keys, 0)
    print(f"C++ function    =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(torch.ops.reduce_cpp.reduce_custom_autograd, X, X_keys, 0)
    print(f"C++ function    =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")
