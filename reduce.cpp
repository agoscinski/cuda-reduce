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

    auto unique_result = torch::unique_dim(new_keys, 0);
    auto reduced_keys = std::get<0>(unique_result);

    torch::Tensor grad_indexes = torch::empty(
        {gradient.sizes()[0]},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(gradient.device())
    );

    for (int i = 0; i < reduced_keys.sizes()[0]; i++) {
        // FIXME: this might be slow
        auto mask = new_keys.eq(reduced_keys.index({torch::indexing::Slice(i, i+1)}));
        auto idx = torch::all(mask, /*dim=*/1);
        grad_indexes.index_put_({idx}, i);
    }

    std::vector<int64_t> reduced_shape = gradient.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_gradient = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(gradient.dtype())
            .device(gradient.device())
    );
    reduced_gradient.index_add_(0, grad_indexes, gradient);

    return {reduced_gradient, reduced_keys};
}

std::vector<std::vector<torch::Tensor>> reduce(
    torch::Tensor input,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad,
    torch::optional<torch::Tensor> position_grad_keys,
    torch::optional<torch::Tensor> cell_grad,
    torch::optional<torch::Tensor> cell_grad_keys
) {
    /* Accumulates the entries in the first dimensions of the input tensors
     * according to the keys in column col with the same value
     *
     * @param input The tensor to be reduced @param keys The meta information
     * about the first dimension of the input @param col The column number of
     * key in keys to be used for the reduction @return The input tensor with
     * the accumulated entries in the first dimension
     */

    // unique is used differently on the c++ frontend
    // see https://stackoverflow.com/a/70809901
    // https://pytorch.org/cppdocs/api/function_namespaceat_1a70a940329a0c5d01c1f3e651f7acec98.html
    torch::Tensor key = keys.index({"...", col});
    auto unique_result = at::_unique2(key);
    auto reduced_keys = std::get<0>(unique_result);

    torch::Tensor indexes = torch::empty(
        {input.sizes()[0]},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(input.device())
    );

    for (int i = 0; i < reduced_keys.sizes()[0]; i++) {
        auto idx = torch::where(key == reduced_keys[i])[0];
        indexes.index_put_({idx}, i);
    }

    std::vector<int64_t> reduced_shape = input.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_input = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(input.dtype())
            .device(input.device())
    );
    reduced_input.index_add_(0, indexes, input);

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
        {reduced_input, reduced_keys.reshape({-1, 1})},
        // positions grad
        {reduced_position_grad, reduced_position_grad_keys},
        // cell grad
        {reduced_cell_grad, reduced_cell_grad_keys}
    };
}

torch::autograd::variable_list ReduceValuesAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col
) {
    torch::Tensor key = keys.index({"...", col});

    auto unique_result = at::_unique2(key);
    auto reduced_keys = std::get<0>(unique_result);

    std::vector<int64_t> reduced_shape = values.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];

    torch::Tensor indexes = torch::empty(
        {values.sizes()[0]},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(values.device())
    );

    auto reduce_mapping = std::vector<torch::Tensor>();
    for (int i = 0; i < reduced_keys.sizes()[0]; i++) {
        auto idx = torch::where(key == reduced_keys[i])[0];
        indexes.index_put_({idx}, i);
        reduce_mapping.push_back(idx);
    }

    torch::Tensor reduced_values = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(values.dtype())
            .device(values.device())
    );
    reduced_values.index_add_(0, indexes, values);

    ctx->save_for_backward({values});
    ctx->saved_data["reduce_mapping"] = reduce_mapping;
    ctx->mark_non_differentiable({reduced_keys});

    return {reduced_values, reduced_keys.reshape({-1, 1}), indexes};
}

torch::autograd::variable_list ReduceValuesAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list outputs_grad
) {
    auto reduced_input_grad = outputs_grad[0];
    auto input = ctx->get_saved_variables()[0];
    auto reduce_mapping = ctx->saved_data["reduce_mapping"].toTensorVector();

    auto input_grad = torch::Tensor();
    if (input.requires_grad()) {
        input_grad = torch::zeros_like(input);

        for (int i=0; i<reduce_mapping.size(); i++) {
            input_grad.index_put_({reduce_mapping[i], "..."}, reduced_input_grad.index({i, "..."}));
        }
    }

    return {
        // values & keys
        input_grad,
        torch::Tensor(),
        // dim
        torch::Tensor(),
    };
}

torch::autograd::variable_list ReduceGradientAutograd::forward(
    torch::autograd::AutogradContext *ctx,
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

    auto unique_result = torch::unique_dim(new_keys, 0);
    auto reduced_keys = std::get<0>(unique_result);

    torch::Tensor grad_indexes = torch::empty(
        {gradient.sizes()[0]},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(gradient.device())
    );

    auto mapping = std::vector<torch::Tensor>();
    for (int i = 0; i < reduced_keys.sizes()[0]; i++) {
        // FIXME: this might be slow
        auto mask = new_keys.eq(reduced_keys.index({torch::indexing::Slice(i, i+1)}));
        auto idx = torch::all(mask, /*dim=*/1);
        grad_indexes.index_put_({idx}, i);
        mapping.push_back(idx);
    }

    std::vector<int64_t> reduced_shape = gradient.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_gradient = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(gradient.dtype())
            .device(gradient.device())
    );
    reduced_gradient.index_add_(0, grad_indexes, gradient);

    ctx->save_for_backward({gradient});
    ctx->saved_data["reduce_mapping"] = mapping;
    ctx->mark_non_differentiable({reduced_keys});

    return {reduced_gradient, reduced_keys};
}

torch::autograd::variable_list ReduceGradientAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list outputs_grad
) {
    auto reduced_gradient_grad = outputs_grad[0];
    auto gradient = ctx->get_saved_variables()[0];
    auto reduce_mapping = ctx->saved_data["reduce_mapping"].toTensorVector();

    auto gradient_grad = torch::Tensor();
    if (gradient.requires_grad()) {
        gradient_grad = torch::zeros_like(gradient);

        for (int i=0; i<reduce_mapping.size(); i++) {
            gradient_grad.index_put_({reduce_mapping[i], "..."}, reduced_gradient_grad.index({i, "..."}));
        }
    }

    return {
        // gradient & keys
        gradient_grad,
        torch::Tensor(),
        // indexes
        torch::Tensor(),
    };
}

std::vector<std::vector<torch::Tensor>> reduce_custom_autograd(
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad,
    torch::optional<torch::Tensor> position_grad_keys,
    torch::optional<torch::Tensor> cell_grad,
    torch::optional<torch::Tensor> cell_grad_keys
) {
    auto result = ReduceValuesAutograd::apply(
        values,
        keys,
        col
    );

    auto reduced_values = result[0];
    auto reduced_keys = result[1];
    auto indexes = result[2];

    auto reduced_position_grad = torch::Tensor();
    auto reduced_position_grad_keys = torch::Tensor();
    if (position_grad) {
        assert(position_grad_keys);
        auto result = ReduceGradientAutograd::apply(
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
        auto result = ReduceGradientAutograd::apply(
            cell_grad.value(),
            cell_grad_keys.value(),
            indexes
        );

        reduced_cell_grad = result[0];
        reduced_cell_grad_keys = result[1];
    }

    return {
        {reduced_values, reduced_keys},
        {reduced_position_grad, reduced_position_grad_keys},
        {reduced_cell_grad, reduced_cell_grad_keys}
    };
}
