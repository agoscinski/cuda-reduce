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
    // // SOLUTION with for-loop in cuda kernel
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i < n_samples) {
    //    auto reduce_id = mapping[i];
    //    for (int j=0; j<other_sizes; j++) {
    //        full[i * other_sizes + j] = reduced[reduce_id * other_sizes + j];
    //    }
    // }

    // SOLUTION without for-loop in cuda kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n_samples && j < other_sizes) {
       auto reduce_id = mapping[i];
       full[i * other_sizes + j] = reduced[reduce_id * other_sizes + j];
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

    // // SOLUTION with for-loop in cuda kernel
    // int threads = 64;
    // int blocks = (n_samples + threads - 1) / threads;

    // SOLUTION without for-loop in cuda kernel
    //const dim3 blocks((n_samples + threads - 1) / threads, (other_sizes + threads - 1) / threads); // tried with one thread but this did not work
    const dim3 threads(1, 128);
    const dim3 blocks(1 + (n_samples - 1) / threads.x, 1 + (other_sizes - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(full.type(), "reduce_backward_cuda", ([&] {
        reduce_backward_cuda_kernel<<<blocks, threads>>>(
            full.data_ptr<scalar_t>(),
            reduced.data_ptr<scalar_t>(),
            mapping.data_ptr<int32_t>(),
            n_samples,
            other_sizes
        );
    }));
}
