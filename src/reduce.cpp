#include <torch/torch.h>

#include "reduce.hh"

static std::vector<torch::Tensor> _reduce_grad(
    torch::Tensor gradient,
    torch::Tensor keys,
    torch::Tensor indexes
) {
    auto new_keys = keys.clone();
    auto new_keys_accessor = new_keys.accessor<int32_t, 2>();
    for (int grad_i=0; grad_i<keys.sizes()[0]; grad_i++) {
        auto sample = new_keys_accessor[grad_i][0];
        new_keys_accessor[grad_i][0] = indexes[sample].item<int32_t>();
    }

    auto unique_result = torch::unique_dim(new_keys, 0, /* sorted */ true, /*return_inverse*/ true);
    auto reduced_keys = std::get<0>(unique_result);
    // The mapping of indexes is the returned "inverse" from unique
    auto grad_indexes = std::get<1>(unique_result);

    std::vector<int64_t> reduced_shape = gradient.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_gradient = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(gradient.dtype())
            .device(gradient.device())
    );
    reduced_gradient.index_add_(0, grad_indexes.to(gradient.device()), gradient);

    return {reduced_gradient, reduced_keys};
}

std::vector<std::vector<torch::Tensor>> reduce(
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad,
    torch::optional<torch::Tensor> position_grad_keys,
    torch::optional<torch::Tensor> cell_grad,
    torch::optional<torch::Tensor> cell_grad_keys
) {
    torch::Tensor key = keys.index({"...", col});
    auto unique_result = at::_unique2(key, /* sorted */ true, /*return_inverse*/ true);
    auto reduced_keys = std::get<0>(unique_result);
    // The mapping of indexes is the returned "inverse" from unique
    auto indexes = std::get<1>(unique_result);

    std::vector<int64_t> reduced_shape = values.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    auto reduced_values = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(values.dtype())
            .device(values.device())
    );
    reduced_values.index_add_(0, indexes.to(values.device()), values);

    auto reduced_position_grad = torch::Tensor();
    auto reduced_position_grad_keys = torch::Tensor();
    if (position_grad) {
        assert(position_grad_keys);
        auto result = _reduce_grad(
            position_grad.value(),
            position_grad_keys.value(),
            indexes
        );

        reduced_position_grad = result[0];
        reduced_position_grad_keys = result[1];
    }

    auto reduced_cell_grad = torch::Tensor();
    auto reduced_cell_grad_keys = torch::Tensor();
    if (cell_grad) {
        assert(cell_grad_keys);
        auto result = _reduce_grad(
            cell_grad.value(),
            cell_grad_keys.value(),
            indexes
        );

        reduced_cell_grad = result[0];
        reduced_cell_grad_keys = result[1];
    }

    return {
        // values
        {reduced_values, reduced_keys.reshape({-1, 1})},
        // positions grad
        {reduced_position_grad, reduced_position_grad_keys},
        // cell grad
        {reduced_cell_grad, reduced_cell_grad_keys}
    };
}
