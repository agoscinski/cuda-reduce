import torch
import torch.utils.cpp_extension

import reduce_python

torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=["reduce.cpp", "bindings.cpp"],
    extra_cflags=["-O3"],
    is_python_module=False,
)

torch.utils.cpp_extension.load(
    name="reduce_cuda_cpp",
    sources=["reduce_cuda.cu"],
    extra_cflags=["-O3"],
    is_python_module=False,
)

def test_same_result(
    context,
    function_1,
    function_2,
    args,
    verbose,
):
    (input, input_keys, dim) = args

    (reduced_1, reduced_keys_1), _, _ = function_1(input, input_keys, dim)
    (reduced_2, reduced_keys_2), _, _ = function_2(input, input_keys, dim)

    same_keys = torch.all(reduced_keys_1 == reduced_keys_2)
    same_shape = reduced_1.shape == reduced_2.shape
    all_close = torch.allclose(reduced_1, reduced_2)
    test_fails = not same_keys or not same_shape or not all_close

    if test_fails and verbose:
        print(f"tests failed for {context}")
        print("    input\n", input)
        print("    input_keys\n", input_keys)
        print("    dim", dim)
        print("    reduced_1\n", reduced_1)
        print("    reduced_2\n", reduced_2)

    assert same_keys, f"{context}: keys error {reduced_keys_1} vs {reduced_keys_2}"
    assert same_shape, f"{context}: shape error {reduced_1.shape} vs {reduced_2.shape}"

    error = torch.linalg.norm(reduced_1 - reduced_2)
    assert all_close, f"{context}: different values, absolute error {error}"


def test_right_values():
    X = torch.tensor(
        [
            [1.0, 11.0],
            [2.0, 22.0],
            [3.0, 33.0],
        ]
    )
    keys = torch.tensor(
        [
            [0, 0],
            [5, 0],
            [0, 2],
        ]
    )
    (reduced, reduced_keys), _, _ = reduce_python.reduce(X, keys, dim=0)
    expected = torch.tensor(
        [
            [4.0, 44.0],
            [2.0, 22.0],
        ]
    )
    if not torch.all(expected == reduced):
        raise Exception("wrong values")

    expected_keys = torch.tensor(
        [
            [0],
            [5],
        ]
    )
    if not torch.all(expected_keys == reduced_keys):
        raise Exception("wrong keys")

    gradient = torch.tensor(
        [
            [1.0, 11.0],
            [2.0, 22.0],
            [3.0, 33.0],
            [4.0, 44.0],
            [5.0, 55.0],
        ]
    )
    gradient_keys = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 2],
            [2, 0],
        ]
    )
    (
        (reduced, reduced_keys),
        (reduced_grad, reduced_grad_keys),
        _,
    ) = reduce_python.reduce(X, keys, 0, gradient, gradient_keys)

    expected = torch.tensor(
        [
            [6.0, 66.0],
            [3.0, 33.0],
            [2.0, 22.0],
            [4.0, 44.0],
        ]
    )
    if not torch.all(expected == reduced_grad):
        raise Exception("wrong gradients")

    expected_keys = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 2],
        ]
    )
    if not torch.all(expected_keys == reduced_grad_keys):
        raise Exception("wrong gradients keys")


if __name__ == "__main__":
    # very rudimentary test for sanity checks, can be extended if needed
    torch.manual_seed(0)

    # check that we get what we want
    test_right_values()

    # small test for debugging
    n = 10
    X = torch.rand((n, 3))
    X_keys = torch.randint(2, (n, 2))
    for dim in range(2):
        test_same_result(
            "python / C++",
            reduce_python.reduce,
            torch.ops.reduce_cpp.reduce,
            (X, X_keys, dim),
            verbose=True,
        )
        test_same_result(
            "python / py autograd",
            reduce_python.reduce,
            reduce_python.reduce_custom_autograd,
            (X, X_keys, dim),
            verbose=True,
        )
        test_same_result(
            "python / C++ autograd",
            reduce_python.reduce,
            torch.ops.reduce_cpp.reduce_custom_autograd,
            (X, X_keys, dim),
            verbose=True,
        )

    # large tests
    n = 100
    X = torch.rand((n, 10, 6, 6))
    X_keys = torch.randint(10, (n, 4))
    for dim in range(4):
        test_same_result(
            "python / C++",
            reduce_python.reduce,
            torch.ops.reduce_cpp.reduce,
            (X, X_keys, dim),
            verbose=False,
        )
        test_same_result(
            "python / py autograd",
            reduce_python.reduce,
            reduce_python.reduce_custom_autograd,
            (X, X_keys, dim),
            verbose=False,
        )
        test_same_result(
            "python / C++ autograd",
            reduce_python.reduce,
            torch.ops.reduce_cpp.reduce_custom_autograd,
            (X, X_keys, dim),
            verbose=False,
        )

        # TODO(Guillaume): add more tests for gradients here

    # custom autograd checks
    X = torch.rand((n, 60), requires_grad=True, dtype=torch.float64)
    X_keys = torch.randint(2, (n, 4))
    dim = 2
    torch.autograd.gradcheck(
        lambda *args: reduce_python.reduce_custom_autograd(*args)[0][0],
        (X, X_keys, dim),
        fast_mode=True,
    )

    torch.autograd.gradcheck(
        lambda *args: torch.ops.reduce_cpp.reduce_custom_autograd(*args)[0][0],
        (X, X_keys, dim),
        fast_mode=True,
    )

    # check gradients of gradients
    pos_grad = torch.rand((3 * n, 3, 60), requires_grad=True, dtype=torch.float64)
    pos_grad_keys = torch.randint(n, (3 * n, 3), dtype=torch.int32)
    cell_grad = torch.rand((n, 3, 3, 60), requires_grad=True, dtype=torch.float64)
    cell_grad_keys = torch.randint(n, (n, 1), dtype=torch.int32)
    torch.autograd.gradcheck(
        lambda *args: reduce_python.reduce_custom_autograd(*args)[1][0],
        (
            X,
            X_keys,
            dim,
            pos_grad,
            pos_grad_keys,
            None,
            None,
        ),
        fast_mode=True,
    )

    torch.autograd.gradcheck(
        lambda *args: reduce_python.reduce_custom_autograd(*args)[2][0],
        (X, X_keys, dim, None, None, cell_grad, cell_grad_keys),
        fast_mode=True,
    )

    torch.autograd.gradcheck(
        lambda *args: torch.ops.reduce_cpp.reduce_custom_autograd(*args)[1][0],
        (
            X,
            X_keys,
            dim,
            pos_grad,
            pos_grad_keys,
            None,
            None,
        ),
        fast_mode=True,
    )

    torch.autograd.gradcheck(
        lambda *args: torch.ops.reduce_cpp.reduce_custom_autograd(*args)[2][0],
        (X, X_keys, dim, None, None, cell_grad, cell_grad_keys),
        fast_mode=True,
    )

    X = X.to(device="cuda")
    X_keys = X_keys.to(device="cuda")
    torch.autograd.gradcheck(
        lambda *args: torch.ops.reduce_cuda_cpp.reduce_custom_autograd(*args)[2][0],
        (X, X_keys, dim, None, None, cell_grad, cell_grad_keys),
        fast_mode=True,
    )
    X.to(device="cpu")
    X_keys.to(device="cpu")

    print("All tests passed!")
