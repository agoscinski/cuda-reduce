from setuptools import setup
from torch.utils import cpp_extension

if __name__ == "__main__":
    setup(
        name="reduce_cpp",
        ext_modules=[
            cpp_extension.CppExtension("reduce_cpp", ["reduce.cpp", "bindings.cpp"])
        ],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
