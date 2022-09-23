#include <ATen/ATen.h>

#include "reduce.hh"

// template <typename scalar_t>
// __global__ void reduce_forward_cuda_kernel(
//     scalar_t* __restrict__ reduced,
//     const scalar_t* __restrict__ full,
//     const int32_t* __restrict__ mapping,
//     int32_t n_samples,
//     int32_t other_sizes
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < n_samples && j < other_sizes) {
//        auto reduce_id = mapping[i];
//        atomicAdd(buffer + reduce_id * other_sizes + j, full[i * other_sizes + j]);
//     //    reduced[reduce_id * other_sizes + j] += full[i * other_sizes + j];
//     }

//     atomicAdd(reduced, buffer)

// }

// // mapping = [0, 0, 1, 1, 2, 1, 0]
// // other_mapping = [
// //    [0, 1, 6] # 0,
// //    [2, 3, 5] # 1,
// //    [4] # 2,
// // ]


// void reduce_forward_cuda(
//     torch::Tensor& reduced,
//     const torch::Tensor& full,
//     const std::vector<torch::Tensor>& find_a_better_name_mapping,
//     int n_samples,
//     int other_sizes
// ) {
//     CHECK_CUDA(full);
//     CHECK_CUDA(reduced);
//     CHECK_CUDA(mapping);

//     const dim3 threads(1, 64);
//     const dim3 blocks(1 + (n_samples - 1) / threads.x, 1 + (other_sizes - 1) / threads.y);

//     AT_DISPATCH_FLOATING_TYPES(full.scalar_type(), "reduce_forward_cuda", ([&] {
//         reduce_forward_cuda_kernel<<<blocks, threads>>>(
//             reduced.data_ptr<scalar_t>(),
//             full.data_ptr<scalar_t>(),
//             mapping.data_ptr<int32_t>(),
//             n_samples,
//             other_sizes
//         );
//     }));
// }

template <typename scalar_t>
__global__ void reduce_forward_cuda_kernel(
    scalar_t* __restrict__ reduced,
    const scalar_t* __restrict__ full,
    const int64_t* __restrict__ mapping,
    int32_t n_samples,
    int32_t other_sizes,
    int32_t reduce_i,
    int32_t mapping_size
) {
    extern __shared__ double buffer[];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < other_sizes) {
        for (int mapping_i=0; mapping_i<mapping_size; mapping_i++) {
            buffer[j] += 0.0;
            // full[mapping[mapping_i] * other_sizes + j];
        }

        reduced[reduce_i * other_sizes + j] = buffer[j];
    }
}


void reduce_forward_cuda(
    at::Tensor& reduced,
    const at::Tensor& full,
    const std::vector<at::Tensor>& find_a_better_name_mapping,
    int n_samples,
    int other_sizes
) {
    CHECK_CUDA(full);
    CHECK_CUDA(reduced);

    ASSERT_THROW(find_a_better_name_mapping.size() == reduced.size(0), "reduce size error");

    AT_DISPATCH_FLOATING_TYPES(full.scalar_type(), "reduce_forward_cuda", ([&] {
        for (int reduce_i=0; reduce_i<find_a_better_name_mapping.size(); reduce_i++) {
            CHECK_CUDA(find_a_better_name_mapping[reduce_i]);

            int threads = 64;
            int blocks = 1 + (other_sizes - 1) / threads;

            reduce_forward_cuda_kernel<<<blocks, threads, other_sizes>>>(
                reduced.data_ptr<scalar_t>(),
                full.data_ptr<scalar_t>(),
                find_a_better_name_mapping[reduce_i].data_ptr<int64_t>(),
                n_samples,
                other_sizes,
                reduce_i,
                find_a_better_name_mapping[reduce_i].size(0)
            );
        }
    }));
}


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
    at::Tensor& full,
    const at::Tensor& reduced,
    const at::Tensor& mapping,
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
    const dim3 threads(1, 64);
    const dim3 blocks(1 + (n_samples - 1) / threads.x, 1 + (other_sizes - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(full.scalar_type(), "reduce_backward_cuda", ([&] {
        reduce_backward_cuda_kernel<<<blocks, threads>>>(
            full.data_ptr<scalar_t>(),
            reduced.data_ptr<scalar_t>(),
            mapping.data_ptr<int32_t>(),
            n_samples,
            other_sizes
        );
    }));
}

void reduce_backward_cudamemcpy(
    at::Tensor& full,
    const at::Tensor& reduced,
    const at::Tensor& mapping,
    int n_samples,
    int other_sizes
) {
    CHECK_CUDA(full);
    CHECK_CUDA(reduced);
    CHECK_CPU(mapping);

    auto to = full.data_ptr<double>();
    auto from = reduced.data_ptr<double>();
    int32_t* mapping_ptr = mapping.data_ptr<int32_t>();

    for (int i = 0; i < n_samples; i++)
    {
        auto to_i = to + other_sizes*i;
        auto reduce_id = mapping_ptr[i];
        auto from_i = from + reduce_id*other_sizes;
        cudaMemcpy(to_i, from_i, other_sizes*sizeof(double), cudaMemcpyDeviceToDevice);
    }
}
