#include <torch/torch.h>

#define ASSERT_THROW(x, message) do {if (!(x)) { throw std::runtime_error(message); } } while (false)
#define CHECK_CPU(x) ASSERT_THROW(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) ASSERT_THROW(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) ASSERT_THROW(x.is_contiguous(), #x " must be contiguous")

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

void reduce_backward_cpu(
    torch::Tensor& full,
    const torch::Tensor& reduced,
    const torch::Tensor& mapping,
    int n_samples,
    int other_sizes
);

void reduce_backward_cuda(
    torch::Tensor& full,
    const torch::Tensor& reduced,
    const torch::Tensor& mapping,
    int n_samples,
    int other_sizes
);
