#include <chrono>

#include <torch/torch.h>

#include "reduce.hh"

torch::autograd::variable_list ReduceValuesAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col
) {
    CHECK_CPU(keys);
    torch::Tensor key = keys.index({"...", col});

    auto unique_result = at::_unique2(key, /* sorted */ true, /*return_inverse*/ true);
    auto reduced_keys = std::get<0>(unique_result);

    // The mapping of indexes is the returned "inverse" from unique
    auto indexes = std::get<1>(unique_result);

    std::vector<int64_t> reduced_shape = values.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_values = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(values.dtype())
            .device(values.device())
    );

    auto n_samples = values.size(0);
    auto other_sizes = 1;
    for (int dim=1; dim<values.sizes().size(); dim++) {
        other_sizes *= values.sizes()[dim];
    }

    if (values.device().is_cpu()) {
        reduce_forward_cpu(
            reduced_values,
            values,
            indexes,
            n_samples,
            other_sizes
        );
    } else if (values.device().is_cuda()) {
        reduce_forward_cuda(
            reduced_values,
            values,
            // only move the indexes to CUDA if we need them here
            indexes.cuda(),
            n_samples,
            other_sizes
        );
    } else {
        throw std::runtime_error("ReduceValuesAutograd::backward is not implemented for this device");
    }

    ctx->save_for_backward({values, indexes});
    ctx->mark_non_differentiable({reduced_keys});

    return {reduced_values, reduced_keys.reshape({-1, 1}), indexes};
}

torch::autograd::variable_list ReduceValuesAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list outputs_grad
) {
    auto reduced_values_grad = outputs_grad[0].contiguous();
    auto values = ctx->get_saved_variables()[0];
    auto indexes = ctx->get_saved_variables()[1];

    auto values_grad = torch::Tensor();
    if (values.requires_grad()) {
        values_grad = torch::empty_like(values);

        CHECK_CONTIGUOUS(indexes);
        CHECK_CONTIGUOUS(values_grad);
        CHECK_CONTIGUOUS(reduced_values_grad);

        auto n_samples = values_grad.sizes()[0];
        auto other_sizes = 1;
        auto reduced_other_sizes = 1;
        for (int dim=1; dim<values_grad.sizes().size(); dim++) {
            other_sizes *= values_grad.sizes()[dim];
            reduced_other_sizes *= reduced_values_grad.sizes()[dim];
        }
        ASSERT_THROW(other_sizes == reduced_other_sizes, "wrong size");

        if (values.device().is_cpu()) {
            reduce_backward_cpu(
                values_grad,
                reduced_values_grad,
                indexes,
                n_samples,
                other_sizes
            );
        } else if (values.device().is_cuda()) {
            reduce_backward_cudamemcpy(
                values_grad,
                reduced_values_grad,
                indexes,
                n_samples,
                other_sizes
            );
        } else {
            throw std::runtime_error("ReduceValuesAutograd::backward is not implemented for this device");
        }
    }

    return {
        // values & keys
        values_grad,
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

    auto unique_result = torch::unique_dim(new_keys, 0, /*sorted*/ true, /*return_inverse*/ true);
    auto reduced_keys = std::get<0>(unique_result);

    // The mapping of indexes is the returned "inverse" from unique
    auto gradient_indexes = std::get<1>(unique_result);

    std::vector<int64_t> reduced_shape = gradient.sizes().vec();
    reduced_shape[0] = reduced_keys.sizes()[0];
    torch::Tensor reduced_gradient = torch::zeros(
        reduced_shape,
        torch::TensorOptions()
            .dtype(gradient.dtype())
            .device(gradient.device())
    );

    auto n_samples = gradient.sizes()[0];
    auto other_sizes = 1;
    auto reduced_other_sizes = 1;
    for (int dim=1; dim<gradient.sizes().size(); dim++) {
        other_sizes *= gradient.sizes()[dim];
    }

    if (gradient.device().is_cpu()) {
        reduce_forward_cpu(
            reduced_gradient,
            gradient,
            gradient_indexes,
            n_samples,
            other_sizes
        );
    } else if (gradient.device().is_cuda()) {
        reduce_forward_cuda(
            reduced_gradient,
            gradient,
            // only move the indexes to CUDA if we need them here
            gradient_indexes.cuda(),
            n_samples,
            other_sizes
        );
    } else {
        throw std::runtime_error("ReduceValuesAutograd::backward is not implemented for this device");
    }

    ctx->save_for_backward({gradient, gradient_indexes});
    ctx->mark_non_differentiable({reduced_keys});

    return {reduced_gradient, reduced_keys};
}

torch::autograd::variable_list ReduceGradientAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list outputs_grad
) {
    auto reduced_gradient_grad = outputs_grad[0].contiguous();
    auto gradient = ctx->get_saved_variables()[0];
    auto gradient_indexes = ctx->get_saved_variables()[1];

    auto gradient_grad = torch::Tensor();
    if (gradient.requires_grad()) {
        gradient_grad = torch::empty_like(gradient);

        CHECK_CONTIGUOUS(gradient_indexes);
        CHECK_CONTIGUOUS(gradient_grad);
        CHECK_CONTIGUOUS(reduced_gradient_grad);

        auto n_samples = gradient_grad.sizes()[0];

        auto other_sizes = 1;
        auto reduced_other_sizes = 1;
        for (int dim=1; dim<gradient_grad.sizes().size(); dim++) {
            other_sizes *= gradient_grad.sizes()[dim];
            reduced_other_sizes *= reduced_gradient_grad.sizes()[dim];
        }
        ASSERT_THROW(other_sizes == reduced_other_sizes, "wrong size");

        if (gradient.device().is_cpu()) {
            reduce_backward_cpu(
                gradient_grad,
                reduced_gradient_grad,
                gradient_indexes,
                n_samples,
                other_sizes
            );
        } else if (gradient.device().is_cuda()) {
            reduce_backward_cudamemcpy(
                gradient_grad,
                reduced_gradient_grad,
                gradient_indexes,
                n_samples,
                other_sizes
            );
        } else {
            throw std::runtime_error("ReduceGradientAutograd::backward is not implemented for this device");
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
