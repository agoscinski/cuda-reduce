# This file shows how to JIT compile the cpp reduce

import torch
import torch.utils.cpp_extension

import load_cpp_extension

torch.manual_seed(0)
n = 10
X = torch.rand((n, 3))
X_keys = torch.randint(2, (n, 2))
col = 1
X_reduced, X_reduced_keys = torch.ops.reduce_cpp.reduce(X, X_keys, col)
print("X\n", X)
print("X_keys\n", X_keys)
print("col", col)
print("X_reduced\n", X_reduced)
