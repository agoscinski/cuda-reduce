#include <torch/torch.h>
#include "reduce.hh"

TORCH_LIBRARY(reduce_cpp, m) {
    m.def(R"(
        reduce(
            Tensor values,
            Tensor keys,
            int dim,
            Tensor? positions_grad = None,
            Tensor? positions_grad_keys = None,
            Tensor? cell_grad = None,
            Tensor? cell_grad_keys = None
        ) -> Tensor[][]
    )", reduce);

    m.def(R"(
        reduce_custom_autograd(
            Tensor values,
            Tensor keys,
            int dim,
            Tensor? positions_grad = None,
            Tensor? positions_grad_keys = None,
            Tensor? cell_grad = None,
            Tensor? cell_grad_keys = None
        ) -> Tensor[][]
    )", reduce_custom_autograd);
}

// TORCH_LIBRARY_IMPL(reduce_cpp, CPU, m) {

// }
