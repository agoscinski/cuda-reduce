#include <torch/torch.h>

std::vector<std::vector<torch::Tensor>> reduce(
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad = torch::nullopt,
    torch::optional<torch::Tensor> position_grad_keys = torch::nullopt,
    torch::optional<torch::Tensor> cell_grad = torch::nullopt,
    torch::optional<torch::Tensor> cell_grad_keys = torch::nullopt
);

class ReduceValuesAutograd : public torch::autograd::Function<ReduceValuesAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor values,
        torch::Tensor keys,
        int64_t col
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    );
};

class ReduceGradientAutograd : public torch::autograd::Function<ReduceGradientAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor gradients,
        torch::Tensor keys,
        torch::Tensor indexes
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    );
};

std::vector<std::vector<torch::Tensor>> reduce_custom_autograd(
    torch::Tensor values,
    torch::Tensor keys,
    int64_t col,
    torch::optional<torch::Tensor> position_grad = torch::nullopt,
    torch::optional<torch::Tensor> position_grad_keys = torch::nullopt,
    torch::optional<torch::Tensor> cell_grad = torch::nullopt,
    torch::optional<torch::Tensor> cell_grad_keys = torch::nullopt
);
