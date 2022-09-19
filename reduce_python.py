from typing import Optional

import torch
from torch import Tensor


def reduce(
    values: Tensor,
    keys: Tensor,
    dim: int,
    positions_grad: Optional[Tensor] = None,
    positions_grad_keys: Optional[Tensor] = None,
    cell_grad: Optional[Tensor] = None,
    cell_grad_keys: Optional[Tensor] = None,
) -> (
    (Tensor, Tensor),
    (Optional[Tensor], Optional[Tensor]),
    (Optional[Tensor], Optional[Tensor]),
):
    # `keys` contains a description of the rows of `values`. Each row in `key`
    # correspond to a row in `values` (respectively for `reduced_values` /
    # `reduced_keys`)
    #
    # `positions_grad` contains the gradients w.r.t. positions of values, and
    # `positions_grad_keys` describes the rows of `positions_grad`. The first
    # column in `positions_grad_keys` is always the row in `values` we are
    # taking the gradient of.
    #
    # Similar considerations apply to `cell_grad` / `cell_grad_keys`

    device = values.device
    dtype = values.dtype

    assert keys.dim() == 2, "keys should have only two dimensions"
    reduced_keys = torch.unique(keys[:, dim])

    mapping = torch.empty(values.shape[0], dtype=torch.int32, device=device)
    all_indexes = []
    for i, reduced_key in enumerate(reduced_keys):
        idx = torch.where(keys[:, dim] == reduced_key)[0]
        mapping.index_put_((idx,), torch.tensor(i, dtype=torch.int32, device=device))
        all_indexes.append(idx)

    new_shape = (len(reduced_keys),) + values.shape[1:]
    reduced_values = torch.zeros(new_shape, dtype=dtype, device=device)
    reduced_values.index_add_(0, mapping, values)

    if positions_grad is not None:
        assert positions_grad_keys is not None
        assert positions_grad.device == device
        assert positions_grad.dtype == dtype

        new_positions_grad_keys = torch.empty_like(positions_grad_keys)
        for grad_i, grad_key in enumerate(positions_grad_keys):
            for new_i, summed_idx in enumerate(all_indexes):
                if grad_key[0] in summed_idx:
                    new_positions_grad_keys[grad_i, 0] = new_i
                    new_positions_grad_keys[grad_i, 1:] = grad_key[1:]
                    break

        reduced_positions_grad_keys = torch.unique(new_positions_grad_keys, dim=0)

        grad_mapping = torch.empty(
            positions_grad.shape[0], dtype=torch.int32, device=device
        )
        for i, reduced_key in enumerate(reduced_positions_grad_keys):
            # FIXME: this might be slow?
            idx = torch.all(new_positions_grad_keys == reduced_key[None, :], axis=1)
            grad_mapping.index_put_(
                (idx,), torch.tensor(i, dtype=torch.int32, device=device)
            )

        new_shape = (len(reduced_positions_grad_keys),) + positions_grad.shape[1:]
        reduced_positions_grad = torch.zeros(new_shape, dtype=dtype, device=device)
        reduced_positions_grad.index_add_(0, grad_mapping, positions_grad)

    else:
        reduced_positions_grad = None
        reduced_positions_grad_keys = None

    if cell_grad is not None:
        assert cell_grad_keys is not None
        assert cell_grad.device == device

        raise Exception("not implemented yet")
    else:
        reduced_cell_grad = None
        reduced_cell_grad_keys = None

    reduced_values = (reduced_values, reduced_keys)
    reduced_positions_grad = (reduced_positions_grad, reduced_positions_grad_keys)
    reduced_cell_grad = (reduced_cell_grad, reduced_cell_grad_keys)
    return (reduced_values, reduced_positions_grad, reduced_cell_grad)


class ReduceAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: Tensor,
        keys: Tensor,
        dim: int,
        positions_grad: Optional[Tensor] = None,
        positions_grad_keys: Optional[Tensor] = None,
        cell_grad: Optional[Tensor] = None,
        cell_grad_keys: Optional[Tensor] = None,
    ) -> (
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ):
        device = values.device
        dtype = values.dtype

        assert keys.dim() == 2, "keys should have only two dimensions"
        reduced_keys = torch.unique(keys[:, dim])

        indexes = torch.empty(values.shape[0], dtype=torch.int64, device=device)
        mapping = []
        for i, reduced_key in enumerate(reduced_keys):
            idx = torch.where(keys[:, dim] == reduced_key)[0]
            indexes.index_put_(
                (idx,), torch.tensor(i, dtype=torch.int64, device=device)
            )
            mapping.append(idx)

        new_shape = (len(reduced_keys),) + values.shape[1:]
        reduced_values = torch.zeros(new_shape, dtype=dtype, device=device)
        reduced_values.index_add_(0, indexes, values)

        if positions_grad is not None:
            assert positions_grad_keys is not None
            assert positions_grad.device == device
            assert positions_grad.dtype == dtype

        ctx.save_for_backward(values)
        ctx.reduce_mapping = mapping

        ctx.mark_non_differentiable(reduced_keys)

        return reduced_values, reduced_keys, None, None, None, None

    @staticmethod
    def backward(
        ctx,
        grad_reduced_values,
        _grad_reduced_keys,
        grad_reduced_positions_grad,
        _grad_reduced_positions_grad_keys,
        grad_reduced_cell_grad,
        _grad_reduced_cell_grad_keys,
    ):

        (values,) = ctx.saved_tensors

        input_grad = None
        if values.requires_grad:
            input_grad = torch.zeros_like(values)
            for i, idx in enumerate(ctx.reduce_mapping):
                input_grad[idx] = grad_reduced_values[i]

        return (
            # values & keys
            input_grad,
            None,
            # dim
            None,
            # positions grad & keys
            None,
            None,
            # cell grad & keys
            None,
            None,
        )


def reduce_custom_autograd(
    values: Tensor,
    keys: Tensor,
    dim: int,
    positions_grad: Optional[Tensor] = None,
    positions_grad_keys: Optional[Tensor] = None,
    cell_grad: Optional[Tensor] = None,
    cell_grad_keys: Optional[Tensor] = None,
) -> (
    (Tensor, Tensor),
    (Optional[Tensor], Optional[Tensor]),
    (Optional[Tensor], Optional[Tensor]),
):
    v, v_k, p, p_k, c, c_k = ReduceAutograd.apply(
        values,
        keys,
        dim,
        positions_grad,
        positions_grad_keys,
        cell_grad,
        cell_grad_keys,
    )
    return (v, v_k), (p, p_k), (c, c_k)
