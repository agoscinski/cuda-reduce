import torch
import torch.utils.cpp_extension

import reduce_python

reduce_cpp = torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=["reduce.cpp", "bindings.cpp"],
)


def test_same_result(
    context,
    function_1,
    function_2,
    args,
    verbose,
):
    (input, input_keys, dim) = args

    reduced_1 = function_1(input, input_keys, dim)
    reduced_2 = function_2(input, input_keys, dim)

    same_shape = reduced_1.shape == reduced_2.shape
    all_close = torch.allclose(reduced_1, reduced_2)
    test_fails = not same_shape or not all_close

    if test_fails and verbose:
        print(f"tests failed for {context}")
        print("    input\n", input)
        print("    input_keys\n", input_keys)
        print("    dim", dim)
        print("    reduced_1\n", reduced_1)
        print("    reduced_2\n", reduced_2)

    assert same_shape, f"{context}: shape error {reduced_1.shape} vs {reduced_2.shape}"

    error = torch.linalg.norm(reduced_1 - reduced_2)
    assert all_close, f"{context}: different values, absolute error {error}"


if __name__ == "__main__":
    # very rudimentary test for sanity checks, can be extended if needed
    torch.manual_seed(0)

    # small test for debugging
    n = 10
    X = torch.rand((n, 3))
    X_keys = torch.randint(2, (n, 2))
    for dim in range(2):
        test_same_result(
            "python / C++",
            reduce_python.reduce,
            reduce_cpp.reduce,
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
            reduce_cpp.reduce_custom_autograd,
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
            reduce_cpp.reduce,
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
            reduce_cpp.reduce_custom_autograd,
            (X, X_keys, dim),
            verbose=False,
        )

    # custom autograd checks
    # X = torch.rand((n, 10, 6, 6), requires_grad=True, dtype=torch.float64)
    X = torch.rand((4, 2), requires_grad=True, dtype=torch.float64)
    X_keys = torch.randint(2, (4, 4))
    dim = 2
    torch.autograd.gradcheck(
        reduce_python.reduce_custom_autograd,
        (X, X_keys, dim),
        fast_mode=True,
    )

    torch.autograd.gradcheck(
        reduce_cpp.reduce_custom_autograd,
        (X, X_keys, dim),
        fast_mode=True,
    )

    print("All tests passed!")
