#include <iostream>

#include "reduce.hh"

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
    torch::Tensor reduced_keys, _ue_idx, _ue_count;
    std::tie(reduced_keys, _ue_idx, _ue_count) = at::_unique2(key, true, false, false);

    std::vector<int64_t> reduced_shape = input.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];

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

    torch::Tensor reduced_input = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(input.dtype())
            .device(input.device())
    );
    reduced_input.index_add_(0, indexes, input);

    return {
        // values
        {reduced_input, reduced_keys},
        // positions grad
        {torch::Tensor(), torch::Tensor()},
        // cell grad
        {torch::Tensor(), torch::Tensor()}
    };
}

torch::autograd::variable_list ReduceAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad,
    torch::optional<torch::Tensor> position_grad_keys,
    torch::optional<torch::Tensor> cell_grad,
    torch::optional<torch::Tensor> cell_grad_keys
) {
    torch::Tensor key = keys.index({"...", col});

    torch::Tensor reduced_keys, _ue_idx, _ue_count;
    std::tie(reduced_keys, _ue_idx, _ue_count) = at::_unique2(key, true, false, false);

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

    return {reduced_values, reduced_keys}; // TODO: return the right thing
}

torch::autograd::variable_list ReduceAutograd::backward(
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
        // postion grad & keys
        torch::Tensor(),
        torch::Tensor(),
        // cell grad & keys
        torch::Tensor(),
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
    auto result = ReduceAutograd::apply(
        values,
        keys,
        col,
        std::move(position_grad),
        std::move(position_grad_keys),
        std::move(cell_grad),
        std::move(cell_grad_keys)
    );

    // return {{result[0], result[1]}, {result[2], result[3]}, {result[4], result[5]}};
    return {{result[0], result[1]}, {torch::Tensor(), torch::Tensor()}, {torch::Tensor(), torch::Tensor()}};
}
