#include <torch/torch.h>
#include "reduce.hh"

TORCH_LIBRARY(reduce_cpp, m) {
    m.def("reduce", reduce);
    m.def("reduce_custom_autograd", reduce_custom_autograd);
}

// TORCH_LIBRARY_IMPL(reduce_cpp, CPU, m) {

// }
