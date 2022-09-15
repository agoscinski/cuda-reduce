# This file shows how to JIT compile the cpp reduce

import torch
import torch.utils.cpp_extension

reduce_cpp = torch.utils.cpp_extension.load(
    name="reduce_cpp", sources=["reduce.cpp", "bindings.cpp"]
)

torch.manual_seed(0)
n = 10
X = torch.rand((n, 3))
X_keys = torch.randint(2, (n, 2))
col = 1
X_reduced = reduce_cpp.reduce(X, X_keys, col)
print("X\n", X)
print("X_keys\n", X_keys)
print("col", col)
print("X_reduced\n", X_reduced)
