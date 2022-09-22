#include <torch/torch.h>

#include "reduce.hh"

template <typename scalar_t>
__global__ void reduce_backward_cuda_kernel(
    scalar_t* __restrict__ full,
    const scalar_t* __restrict__ reduced,
    const int32_t* __restrict__ mapping,
    int32_t n_samples,
    int32_t other_sizes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        auto reduce_id = mapping[i];
        for (int j=0; j<other_sizes; j++) {
            full[i * other_sizes + j] = reduced[reduce_id * other_sizes + j];
        }
    }
}

void reduce_backward_cuda(
    torch::Tensor& full,
    const torch::Tensor& reduced,
    const torch::Tensor& mapping,
    int n_samples,
    int other_sizes
) {
    CHECK_CUDA(full);
    CHECK_CUDA(reduced);
    CHECK_CUDA(mapping);

    int threads = 64;
    int blocks = (n_samples + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(full.type(), "reduce_backward_cpu", ([&] {
        reduce_backward_cuda_kernel<<<blocks, threads>>>(
            full.data_ptr<scalar_t>(),
            reduced.data_ptr<scalar_t>(),
            mapping.data_ptr<int32_t>(),
            n_samples,
            other_sizes
        );
    }));
}
