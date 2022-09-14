#include <torch/extension.h>
#include "reduce.hh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce", &reduce, "");
}
