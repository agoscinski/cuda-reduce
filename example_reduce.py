# This file shows how to install the cpp reduce
# Note you have to do `pip install .` before using this script

import torch

try:
    import reduce_cpp
except ModuleNotFoundError as e:
    print("!!! Please install module with `pip install -e .` !!!")
    raise e

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
