#include <torch/torch.h>

torch::Tensor reduce(torch::Tensor input, torch::Tensor keys, int64_t col);

class ReduceAutograd : public torch::autograd::Function<ReduceAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor keys,
        int64_t col
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output
    );
};

torch::Tensor reduce_custom_autograd(torch::Tensor input, torch::Tensor keys, int64_t col);
