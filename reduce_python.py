import torch


def reduce(input, keys, dim):
    assert keys.dim() == 2, "keys should have only two dimensions"
    unique_entries = torch.unique(keys[:, dim])
    new_shape = torch.Size((len(unique_entries),) + input.shape[1:])
    reduced_input = torch.zeros(new_shape, dtype=input.dtype, device=input.device)
    for i, unique_entry in enumerate(unique_entries):
        idx = torch.where(keys[:, dim] == unique_entry)[0]
        reduced_input[i] = torch.sum(input[idx], axis=0)
    return reduced_input


class ReduceAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, keys, dim):
        assert keys.dim() == 2, "keys should have only two dimensions"
        unique_entries = torch.unique(keys[:, dim])
        new_shape = torch.Size((len(unique_entries),) + input.shape[1:])
        reduced_input = torch.zeros(new_shape, dtype=input.dtype, device=input.device)

        mapping = []
        for i, unique_entry in enumerate(unique_entries):
            idx = torch.where(keys[:, dim] == unique_entry)[0]
            mapping.append(idx)
            reduced_input[i] = torch.sum(input[idx], axis=0)

        ctx.save_for_backward(input)
        ctx.reduce_mapping = mapping

        # TODO: reduced_keys
        # ctx.mark_non_differentiable(reduced_keys)

        return reduced_input

    @staticmethod
    def backward(ctx, output_grad):

        (input,) = ctx.saved_tensors

        input_grad = None
        if input.requires_grad:
            input_grad = torch.zeros_like(input)

            for i, idx in enumerate(ctx.reduce_mapping):
                input_grad[idx] = output_grad[i]

        return input_grad, None, None


def reduce_custom_autograd(input, keys, dim):
    return ReduceAutograd.apply(input, keys, dim)
