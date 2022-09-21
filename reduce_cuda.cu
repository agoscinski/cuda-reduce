#include <torch/extension.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "reduce.hh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// implements parallelization over ...index_put_({reduce_mapping[i], "..."}, output_grad.index({i, "..."}));
template <typename scalar_t>
__global__ void reduce_cuda_kernel_backward_mapping(
    long int ** __restrict__ reduce_mapping, // should pointer of pointers or simplify life 
    scalar_t* __restrict__ output_grad,
    const int reduce_mapping_rows,
    const int * reduce_mapping_cols_per_row,
    scalar_t* __restrict__ input_grad,
    const int input_grad_tot_cols
   ) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < reduce_mapping_rows) {
    for (int j=0; j < reduce_mapping_cols_per_row[i]; j++) {
      for (int k=0; k<input_grad_tot_cols; k++) {
        input_grad[reduce_mapping[i][j]*input_grad_tot_cols+k] = output_grad[i*input_grad_tot_cols+k];
      }
    }
  }
}


void reduce_cuda_backward_mapping(
    std::vector<torch::Tensor> & reduce_mapping,
    torch::Tensor & input_grad,
    torch::Tensor & output_grad
    ) {

  if (not output_grad.is_contiguous()) { 
    output_grad = output_grad.contiguous();
  }
  for (int i=0; i<reduce_mapping.size(); i++) {
    CHECK_INPUT(reduce_mapping[i]);
  }
  CHECK_INPUT(input_grad);
  CHECK_INPUT(output_grad);


  std::vector<int> reduce_mapping_cols_per_row(reduce_mapping.size());
  for (int i=0; i<reduce_mapping.size(); i++) {
      reduce_mapping_cols_per_row[i] = reduce_mapping[i].sizes()[0];
  }
  int *d_reduce_mapping_cols_per_row;
  cudaMalloc(&d_reduce_mapping_cols_per_row,
      reduce_mapping.size()*sizeof(int)); 
  cudaMemcpy(d_reduce_mapping_cols_per_row,
      reduce_mapping_cols_per_row.data(),
      reduce_mapping.size()*sizeof(int),
      cudaMemcpyHostToDevice);


  std::vector<long int*> reduce_mapping_ptrs(reduce_mapping.size());
  for (int i=0; i<reduce_mapping.size(); i++) {
      reduce_mapping_ptrs.at(i) = reduce_mapping[i].data<long int>();
  }
  long int ** d_reduce_mapping_ptrs;
  cudaMalloc(&d_reduce_mapping_ptrs,
      reduce_mapping.size()*sizeof(long int*)); 
  cudaMemcpy(d_reduce_mapping_ptrs,
      reduce_mapping_ptrs.data(),
      reduce_mapping.size()*sizeof(long int*),
      cudaMemcpyHostToDevice);

  int input_grad_tot_cols{1};
  for (int i=1; i<input_grad.sizes().size(); i++) {
    input_grad_tot_cols *= input_grad.sizes()[i];
  }
  const int threads = 64;
  const int blocks = (reduce_mapping.size() + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(input_grad.type(), "reduce_cuda_kernel_backward_mapping", ([&] {
      reduce_cuda_kernel_backward_mapping<scalar_t><<<blocks, threads>>>(
          d_reduce_mapping_ptrs,
          output_grad.data<scalar_t>(),
          static_cast<int>(reduce_mapping.size()),
          d_reduce_mapping_cols_per_row,
          input_grad.data<scalar_t>(),
          input_grad_tot_cols 
          );
  }));

  cudaFree(d_reduce_mapping_ptrs);
  cudaFree(d_reduce_mapping_cols_per_row);
}

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
        reduce_cuda_backward_mapping(reduce_mapping, input_grad, reduced_input_grad);
        //for (int i=0; i<reduce_mapping.size(); i++) {
        //    input_grad.index_put_({reduce_mapping[i], "..."}, reduced_input_grad.index({i, "..."}));
        //}
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

TORCH_LIBRARY(reduce_cuda_cpp, m) {
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
