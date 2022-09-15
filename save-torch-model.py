import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="reduce_cpp",
    sources=["reduce.cpp", "bindings.cpp"],
    extra_cflags=["-O3"],
    is_python_module=False,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, keys):
        return torch.ops.reduce_cpp.reduce(X, keys, 1)


model = Model()

# train the model ...

# save the model to TorchScript code
model = torch.jit.script(model)
print(model.code)
torch.jit.save(model, "model.pt")
