#include <torch/torch.h>

#include "reduce.hh"

template <typename scalar_t>
void reduce_forwad_cpu_kernel(
    scalar_t* reduced,
    const scalar_t* full,
    const int64_t* mapping,
    int32_t n_samples,
    int32_t other_sizes
) {
    for (int i=0; i<n_samples; i++) {
        auto reduce_id = mapping[i];
        int row_start = i * other_sizes;
        for (int j=0; j<other_sizes; j++) {
            int reduced_row_start = reduce_id * other_sizes;
            reduced[reduced_row_start + j] += full[row_start + j];
        }
    }
}

void reduce_forward_cpu(
    torch::Tensor& reduced,
    const torch::Tensor& full,
    const torch::Tensor& mapping,
    int n_samples,
    int other_sizes
) {
    CHECK_CPU(full);
    CHECK_CPU(reduced);
    CHECK_CPU(mapping);

    AT_DISPATCH_FLOATING_TYPES(full.scalar_type(), "reduce_forward_cpu", ([&] {
        reduce_forwad_cpu_kernel(
            reduced.data_ptr<scalar_t>(),
            full.data_ptr<scalar_t>(),
            mapping.data_ptr<int64_t>(),
            n_samples,
            other_sizes
        );
    }));
}

template <typename scalar_t>
void reduce_backward_cpu_kernel(
    scalar_t* full,
    const scalar_t* reduced,
    const int64_t* mapping,
    int32_t n_samples,
    int32_t other_sizes
) {
    // #pragma omp parallel
    // {
        // WTF: This is faster than serial code even with a single thread
        // #pragma omp parallel for private(n_samples, other_sizes)
        for (int i=0; i<n_samples; i++) {
            auto reduce_id = mapping[i];
            for (int j=0; j<other_sizes; j++) {
                full[i * other_sizes + j] = reduced[reduce_id * other_sizes + j];
            }
        }
    // }
}

void reduce_backward_cpu(
    torch::Tensor& full,
    const torch::Tensor& reduced,
    const torch::Tensor& mapping,
    int n_samples,
    int other_sizes
) {
    CHECK_CPU(full);
    CHECK_CPU(reduced);
    CHECK_CPU(mapping);

    AT_DISPATCH_FLOATING_TYPES(full.scalar_type(), "reduce_backward_cpu", ([&] {
        reduce_backward_cpu_kernel(
            full.data_ptr<scalar_t>(),
            reduced.data_ptr<scalar_t>(),
            mapping.data_ptr<int64_t>(),
            n_samples,
            other_sizes
        );
    }));
}
