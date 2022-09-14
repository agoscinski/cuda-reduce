import reduce_python

import torch
from torch.utils.cpp_extension import load

reduce_cpp = load(name="reduce_cpp", sources=["reduce.cpp", "bindings.cpp"])

def test(input, input_keys, dim, verbose=False):
    input_python_reduced = reduce_python.reduce(input, input_keys, dim)
    input_cpp_reduced = reduce_cpp.reduce(input, input_keys, dim)
    condition_shape = (input_python_reduced.shape == input_cpp_reduced.shape) 
    condition_allclose = torch.allclose(input_python_reduced, input_cpp_reduced)
    test_fails = not(condition_shape) or not(condition_allclose)
    if test_fails and verbose:
        print("input\n", input)
        print("input_keys\n", input_keys)
        print("col", col)
        print("input_python_reduced\n", input_python_reduced)
        print("input_cpp_reduced\n", input_cpp_reduced)

    assert condition_shape , f"Dimension does not agree: python {input_input_python_reduced.shape}, cpp {input_cpp_reduced.shape}"
    assert condition_allclose, f"Values do not agree between python and cpp implementation: absolute error {torch.linalg.norm(input_python_reduced- input_cpp_reduced)}"

# very rudamentary test for sanity checks, can be extended if needed

# small test for debugging
n = 10
torch.manual_seed(0)
X = torch.rand((n, 3))
torch.manual_seed(0)
X_keys = torch.randint(2, (n, 2))
test(X, X_keys, 0, True)
test(X, X_keys, 1, True)

# large tests
n = 100
torch.manual_seed(0)
X = torch.rand((n, 10,6,6))
torch.manual_seed(0)
X_keys = torch.randint(10, (n, 4))
test(X, X_keys, 0)
test(X, X_keys, 1)
test(X, X_keys, 2)
test(X, X_keys, 3)

print("All tests passed!")
