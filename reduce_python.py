import torch


def reduce(input, keys, dim):
    assert keys.dim() == 2, "keys should have only two dimensions"
    reduced_keys = torch.unique(keys[:, dim])

    mapping = torch.empty(input.shape[0], dtype=torch.int32, device=input.device)
    for i, reduced_key in enumerate(reduced_keys):
        idx = torch.where(keys[:, dim] == reduced_key)[0]
        mapping.index_put_(
            (idx,), torch.tensor(i, dtype=torch.int32, device=input.device)
        )

    new_shape = (len(reduced_keys),) + input.shape[1:]
    reduced_input = torch.zeros(new_shape, dtype=input.dtype, device=input.device)
    reduced_input.index_add_(0, mapping, input)

    return (reduced_input, reduced_keys)


class ReduceAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, keys, dim):
        assert keys.dim() == 2, "keys should have only two dimensions"
        reduced_keys = torch.unique(keys[:, dim])

        indexes = torch.empty(input.shape[0], dtype=torch.int64, device=input.device)
        mapping = []
        for i, reduced_key in enumerate(reduced_keys):
            idx = torch.where(keys[:, dim] == reduced_key)[0]
            indexes.index_put_(
                (idx,), torch.tensor(i, dtype=torch.int64, device=input.device)
            )
            mapping.append(idx)

        new_shape = (len(reduced_keys),) + input.shape[1:]
        reduced_input = torch.zeros(new_shape, dtype=input.dtype, device=input.device)
        reduced_input.index_add_(0, indexes, input)

        ctx.save_for_backward(input)
        ctx.reduce_mapping = mapping

        ctx.mark_non_differentiable(reduced_keys)

        return reduced_input, reduced_keys

    @staticmethod
    def backward(ctx, reduced_input_grad, _reduced_keys_grad):

        (input,) = ctx.saved_tensors

        input_grad = None
        if input.requires_grad:
            input_grad = torch.zeros_like(input)
            for i, idx in enumerate(ctx.reduce_mapping):
                input_grad[idx] = reduced_input_grad[i]

        return input_grad, None, None


def reduce_custom_autograd(input, keys, dim):
    return ReduceAutograd.apply(input, keys, dim)
