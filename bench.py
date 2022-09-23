import sys
import time

import numpy as np
import torch
import torch.profiler
from torch.profiler import ProfilerActivity

import load_cpp_extension
import reduce_python


def bench(function, input, input_keys, dim, n_iters=10):
    start = time.time()
    for _ in range(n_iters):
        function(input, input_keys, dim)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    elapsed = time.time() - start
    direct = elapsed / n_iters

    elapsed = 0
    for _ in range(n_iters):
        values, _, _ = function(input, input_keys, dim)
        summed = values[0].sum()

        start = time.time()
        summed.backward()
        torch.cuda.synchronize()
        elapsed += time.time() - start

    backward = elapsed / n_iters

    return direct, backward


def extract_from_equistore(block):
    samples = torch.from_numpy(block.samples.copy().view(dtype=np.int32))
    samples = samples.reshape(block.samples.shape[0], -1)

    if "positions" in block.gradients_list():
        positions_grad = block.gradient("positions")
        positions_grad_samples = torch.from_numpy(
            positions_grad.samples.copy().view(dtype=np.int32)
        )
        positions_grad_samples = positions_grad_samples.reshape(
            positions_grad.samples.shape[0], -1
        )
        positions_grad = positions_grad.data
    else:
        positions_grad = None
        positions_grad_samples = None

    if "cell" in block.gradients_list():
        cell_grad = block.gradient("cell")
        cell_grad_samples = torch.from_numpy(
            cell_grad.samples.copy().view(dtype=np.int32)
        )
        cell_grad_samples = cell_grad_samples.reshape(cell_grad.samples.shape[0], -1)
        cell_grad = cell_grad.data

    else:
        cell_grad = None
        cell_grad_samples = None

    values = (block.values, samples)
    positions_grad = (positions_grad, positions_grad_samples)
    cell_grad = (cell_grad, cell_grad_samples)
    return values, positions_grad, cell_grad


def descriptor_to_cuda(descriptor):
    from equistore import TensorBlock, TensorMap

    blocks = []
    for _, block in descriptor:
        new_block = TensorBlock(
            values=block.values.detach().cuda().requires_grad_(True),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            new_block.add_gradient(
                parameter,
                data=gradient.data.detach().cuda().requires_grad_(True),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def descriptor_sizes(descriptor):
    sizeof_double = 8

    values_size = 0.0
    position_gradient_size = 0.0
    cell_gradient_size = 0.0

    for _, block in descriptor:
        values_size += len(block.values) * sizeof_double

        if "positions" in block.gradients_list():
            position_gradient_size += (
                len(block.gradient("positions").data) * sizeof_double
            )

        if "cell" in block.gradients_list():
            position_gradient_size += len(block.gradient("cell").data) * sizeof_double

    return values_size, position_gradient_size, cell_gradient_size


def reduced_sizes(descriptor):
    sizeof_double = 8

    values_size = 0.0
    position_gradient_size = 0.0
    cell_gradient_size = 0.0

    for _, block in descriptor:
        values, positions_grad, cell_grad = extract_from_equistore(block)
        values, positions_grad, cell_grad = reduce_python.reduce(
            *values, 0, *positions_grad, *cell_grad
        )

        values_size += len(values[0]) * sizeof_double
        if positions_grad[0] is not None:
            position_gradient_size += len(positions_grad[0]) * sizeof_double

        if cell_grad[0] is not None:
            cell_gradient_size += len(cell_grad[0]) * sizeof_double

    return values_size, position_gradient_size, cell_gradient_size


def bench_descriptor(function, descriptor, n_iters=10):
    start = time.time()

    for _ in range(n_iters):
        for _, block in descriptor:
            values, positions_grad, cell_grad = extract_from_equistore(block)
            function(*values, 0, *positions_grad, *cell_grad)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    elapsed = time.time() - start
    direct = elapsed / n_iters

    elapsed_values = 0
    elapsed_grad = 0
    for _ in range(n_iters):
        zero_grad_descriptor(descriptor)
        for _, block in descriptor:
            values, positions_grad, cell_grad = extract_from_equistore(block)
            values, positions_grad, cell_grad = function(
                *values, 0, *positions_grad, *cell_grad
            )

            summed = values[0].sum()
            start = time.time()

            summed.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_values += time.time() - start

            if positions_grad[0] is not None:
                summed = positions_grad[0].sum() + cell_grad[0].sum()
                start = time.time()

                summed.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_grad += time.time() - start

    backward_values = elapsed_values / n_iters
    backward_grad = elapsed_grad / n_iters

    return direct, backward_values, backward_grad


def create_real_data(file, subset, gradients):
    try:
        import ase.io
        import rascaline
        from equistore import TensorBlock, TensorMap
    except ImportError:
        print("rascaline not found, exiting")
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

    if gradients:
        descriptor = calculator.compute(frames, gradients=["positions", "cell"])
    else:
        descriptor = calculator.compute(frames)

    # XXX: remove
    descriptor.keys_to_samples("species_center")
    descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

    blocks = []
    for _, block in descriptor:
        new_block = TensorBlock(
            values=torch.tensor(block.values, requires_grad=True),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            new_block.add_gradient(
                parameter,
                data=torch.tensor(gradient.data, requires_grad=True),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def zero_grad_descriptor(descriptor):
    for _, block in descriptor:
        block.values.grad = None

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            gradient.data.grad = None


def format_throughput(value):
    value /= 1024
    if value > 1024 * 1024:
        return f"{value/(1024 * 1024):.4} GiB/s"
    elif value > 1024:
        return f"{value/1024:.4} MiB/s"
    else:
        return f"{value:.4} KiB/s"


def bench_random():
    print("RANDOM DATA")

    torch.manual_seed(0xDEADBEEF)
    n_samples = 10000
    n_features = 1000
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
    print(f"C++ autograd    =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms")

    if torch.cuda.is_available():
        X = X.to(device="cuda")
        X_keys = X_keys.to(device="cuda")

        forward, backward = bench(
            torch.ops.reduce_cpp.reduce,
            X,
            X_keys,
            0,
        )
        print(
            f"CUDA function (Cxx)=   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms"  # noqa
        )

        forward, backward = bench(
            torch.ops.reduce_cpp.reduce_custom_autograd,
            X,
            X_keys,
            0,
        )
        print(
            f"CUDA autograd   =   {1e3 * forward:.3} ms    -   {1e3 * backward:.5} ms"  # noqa
        )


def bench_real_data():
    print("REAL DATA -- no forward gradients")
    descriptor = create_real_data("random-methane-10k.extxyz", ":300", gradients=False)
    print("implementation  | forward pass | backward pass")

    forward, backward_v, _ = bench_descriptor(reduce_python.reduce, descriptor)
    print(f"python function =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")

    forward, backward_v, _ = bench_descriptor(
        reduce_python.reduce_custom_autograd, descriptor
    )
    print(f"python autograd =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")

    forward, backward_v, _ = bench_descriptor(torch.ops.reduce_cpp.reduce, descriptor)
    print(f"c++ function    =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")

    forward, backward_v, _ = bench_descriptor(
        torch.ops.reduce_cpp.reduce_custom_autograd, descriptor
    )
    print(f"c++ autograd    =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")

    if torch.cuda.is_available():
        values_size, _, _ = descriptor_sizes(descriptor)
        reduced_values_size, _, _ = reduced_sizes(descriptor)
        descriptor = descriptor_to_cuda(descriptor)

        forward, backward_v, _ = bench_descriptor(
            torch.ops.reduce_cpp.reduce, descriptor
        )
        print(f"CUDA function   =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")
        forward = format_throughput(values_size / forward)
        backward_v = format_throughput(reduced_values_size / backward_v)
        print(f"                  {forward}  -  {backward_v}")

        forward, backward_v, _ = bench_descriptor(
            torch.ops.reduce_cpp.reduce_custom_autograd, descriptor
        )
        print(f"CUDA autograd   =   {1e3 * forward:.5} ms  -  {1e3 * backward_v:.5} ms")
        forward = format_throughput(values_size / forward)
        backward_v = format_throughput(reduced_values_size / backward_v)
        print(f"                  {forward}  -  {backward_v}")


def bench_real_data_w_grad():
    descriptor = create_real_data("random-methane-10k.extxyz", ":100", gradients=True)
    print("REAL DATA -- with forward gradients")
    print("implementation  | forward pass | backward values | backward grad")

    forward, backward_v, backward_g = bench_descriptor(reduce_python.reduce, descriptor)
    forward = f"{1e3 * forward:.5} ms"
    backward_v = f"{1e3 * backward_v:.5} ms"
    backward_g = f"{1e3 * backward_g:.5} ms"
    print(f"python function =   {forward}  -    {backward_v}    -   {backward_g}")

    forward, backward_v, backward_g = bench_descriptor(
        reduce_python.reduce_custom_autograd, descriptor
    )
    forward = f"{1e3 * forward:.5} ms"
    backward_v = f"{1e3 * backward_v:.5} ms"
    backward_g = f"{1e3 * backward_g:.5} ms"
    print(f"python autograd =   {forward}   -    {backward_v}     -   {backward_g}")

    forward, backward_v, backward_g = bench_descriptor(
        torch.ops.reduce_cpp.reduce, descriptor
    )
    forward = f"{1e3 * forward:.5} ms"
    backward_v = f"{1e3 * backward_v:.5} ms"
    backward_g = f"{1e3 * backward_g:.5} ms"
    print(f"C++ function    =   {forward}  -    {backward_v}     -   {backward_g}")

    forward, backward_v, backward_g = bench_descriptor(
        torch.ops.reduce_cpp.reduce_custom_autograd, descriptor
    )
    forward = f"{1e3 * forward:.5} ms"
    backward_v = f"{1e3 * backward_v:.5} ms"
    backward_g = f"{1e3 * backward_g:.5} ms"
    print(f"C++ autograd    =   {forward}  -    {backward_v}     -   {backward_g}")

    if torch.cuda.is_available():
        values_size, pos_grad_s, cell_grad_s = descriptor_sizes(descriptor)
        red_values_size, red_pos_grad_s, red_cell_grad_s = reduced_sizes(descriptor)

        descriptor = descriptor_to_cuda(descriptor)
        forward, backward_v, backward_g = bench_descriptor(
            torch.ops.reduce_cpp.reduce, descriptor
        )
        forward_time = f"{1e3 * forward:.5} ms"
        backward_v_time = f"{1e3 * backward_v:.5} ms"
        backward_g_time = f"{1e3 * backward_g:.5} ms"
        print(
            f"CUDA function   =   {forward_time}  -    {backward_v_time}      -   {backward_g_time}"
        )
        forward = format_throughput((values_size + pos_grad_s + cell_grad_s) / forward)
        backward_v = format_throughput(red_values_size / backward_v)
        backward_g = format_throughput((red_pos_grad_s + red_cell_grad_s) / backward_g)
        print(f"                  {forward}  -   {backward_v}    -  {backward_g}")

        forward, backward_v, backward_g = bench_descriptor(
            torch.ops.reduce_cpp.reduce_custom_autograd, descriptor
        )
        forward_time = f"{1e3 * forward:.5} ms"
        backward_v_time = f"{1e3 * backward_v:.5} ms"
        backward_g_time = f"{1e3 * backward_g:.5} ms"
        print(
            f"CUDA autograd   =   {forward_time}  -    {backward_v_time}      -   {backward_g_time}"
        )
        forward = format_throughput((values_size + pos_grad_s + cell_grad_s) / forward)
        backward_v = format_throughput(red_values_size / backward_v)
        backward_g = format_throughput((red_pos_grad_s + red_cell_grad_s) / backward_g)
        print(f"                  {forward}  -   {backward_v}    -  {backward_g}")


if __name__ == "__main__":
    do_random = "--random" in sys.argv
    do_real = "--real" in sys.argv
    do_real_grad = "--real-grad" in sys.argv

    if not do_random and not do_real and not do_real_grad:
        print("add a flag for the benchmark you want (multiple flags are supported):")
        print("    --random      fully random data")
        print("    --real        actual data")
        print("    --real-grad   actual data, including forward gradients")
        sys.exit(1)

    if do_random:
        bench_random()
        print()

    if do_real:
        bench_real_data()
        print()

    if do_real_grad:
        bench_real_data_w_grad()
        print()
