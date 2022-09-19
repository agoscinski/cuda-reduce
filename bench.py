import sys
import time

import numpy as np
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
        result, _ = function(input, input_keys, dim)
        summed = result.sum()

        start = time.time()
        summed.backward()
        elapsed += time.time() - start

    backward = elapsed / n_iters

    return direct, backward


def bench_descriptor(function, descriptor, n_iters=10):
    start = time.time()

    for _ in range(n_iters):
        for _, block in descriptor:
            keys = torch.from_numpy(block.samples.view(dtype=np.int32))
            function(block.values, keys.reshape(block.samples.shape[0], -1), 0)

    elapsed = time.time() - start
    direct = elapsed / n_iters

    elapsed = 0
    for _ in range(n_iters):
        for _, block in descriptor:
            keys = torch.from_numpy(block.samples.view(dtype=np.int32))
            result, _ = function(
                block.values, keys.reshape(block.samples.shape[0], -1), 0
            )
            summed = result.sum()

            start = time.time()
            summed.backward()
            elapsed += time.time() - start

    backward = elapsed / n_iters

    return direct, backward


def create_real_data(file, subset):
    try:
        import ase.io
        import rascaline
        from equistore import TensorBlock, TensorMap
    except ImportError:
        sys.exit(0)
    HYPERS = {
        "cutoff": 1.5,
        "max_radial": 20,
        "max_angular": 15,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.2}},
    }
    calculator = rascaline.SoapPowerSpectrum(**HYPERS)
    frames = ase.io.read(file, subset)
    descriptor = calculator.compute(frames, gradients=["positions", "cell"])

    blocks = []
    for _, block in descriptor:
        block = TensorBlock(
            values=torch.tensor(block.values, requires_grad=True),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            block.add_gradient(
                parameter,
                data=torch.tensor(gradient.data, requires_grad=True),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(block)

    return TensorMap(descriptor.keys, blocks)


if __name__ == "__main__":
    n_samples = 10000
    n_features = 1000

    # Random data
    torch.manual_seed(0xDEADBEEF)
    X = torch.rand((n_samples, 7, n_features), requires_grad=True, dtype=torch.float64)
    X_keys = torch.randint(4, (n_samples, 3), dtype=torch.int32)

    print("RANDOM DATA")
    print("implementation  | forward pass | backward pass")

    forward, backward = bench(reduce_python.reduce, X, X_keys, 0)
    print(f"python function =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(reduce_python.reduce_custom_autograd, X, X_keys, 0)
    print(f"python autograd =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(torch.ops.reduce_cpp.reduce, X, X_keys, 0)
    print(f"C++ function    =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench(torch.ops.reduce_cpp.reduce_custom_autograd, X, X_keys, 0)
    print(f"C++ autograd    =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    # Real data
    descriptor = create_real_data("random-methane-10k.extxyz", ":1000")

    print("\nREAL DATA")
    print("implementation  | forward pass | backward pass")

    forward, backward = bench_descriptor(reduce_python.reduce, descriptor)
    print(f"python function =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    forward, backward = bench_descriptor(torch.ops.reduce_cpp.reduce, descriptor)
    print(f"C++ function =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")
