#include <iostream>

#include "reduce.hh"


torch::Tensor reduce(torch::Tensor input, torch::Tensor keys, int64_t col) {
  /* Accumulates the entries in the first dimensions of the input tensors
   * according to the keys in column col with the same value
   *
   * @param input The tensor to be reduced @param keys The meta information
   * about the first dimension of the input @param col The column number of key
   * in keys to be used for the reduction @return The input tensor with the
   * accumulated entries in the first dimension
   */

  // unique is used differently on the c++ frontend
  // see https://stackoverflow.com/a/70809901
  // https://pytorch.org/cppdocs/api/function_namespaceat_1a70a940329a0c5d01c1f3e651f7acec98.html
  torch::Tensor key = keys.index({"...", col});
  torch::Tensor unique_entries, _ue_idx, _ue_count;
  std::tie (unique_entries, _ue_idx, _ue_count) = at::unique_dim(key, false, false, false);
  // ArrayRef -> vector
  std::vector<int64_t> reduced_shape(std::begin(input.sizes()), std::end(input.sizes()));
  reduced_shape[0] = unique_entries.sizes()[0];
  torch::Tensor reduced_input = torch::zeros(reduced_shape);
  torch::Tensor idx;
  auto res = torch::where(key == unique_entries[0]);
  for (int i=0; i<unique_entries.sizes()[0]; i++) {
    idx = torch::where(key == unique_entries[i])[0];
    reduced_input.index({i, "..."}) = torch::sum(input.index({idx, "..."}), 0);
  }
  // TODO replace for loop with cuda stuff
  //AT_DISPATCH_FLOATING_TYPES(gates.type(), "reduce_cuda_kernel", ([&] {
  //  reduce_cuda_kernel<scalar_t><<<blocks, threads>>>(
  //      input.data<scalar_t>(),
  //      reduced_input.data<scalar_t>(),
  //      state_size);
  //}));
  return reduced_input;
}
