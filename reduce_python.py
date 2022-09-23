from typing import Optional

import torch
from torch import Tensor


def _reduce_grad(gradient: Tensor, keys: Tensor, indexes: Tensor) -> (Tensor, Tensor):
    device = gradient.device
    dtype = gradient.dtype

    new_keys = keys.clone()
    for grad_i, grad_key in enumerate(keys):
        new_keys[grad_i, 0] = indexes[grad_key[0]]

    reduced_keys, grad_indexes = torch.unique(new_keys, dim=0, return_inverse=True)

    new_shape = (len(reduced_keys),) + gradient.shape[1:]
    reduced_gradient = torch.zeros(new_shape, dtype=dtype, device=device)
    reduced_gradient.index_add_(0, grad_indexes.to(gradient.device), gradient)

    return reduced_gradient, reduced_keys


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
    reduced_keys, indexes = torch.unique(keys[:, dim], return_inverse=True)

    new_shape = (len(reduced_keys),) + values.shape[1:]
    reduced_values = torch.zeros(new_shape, dtype=dtype, device=device)
    reduced_values.index_add_(0, indexes.to(values.device), values)

    if positions_grad is not None:
        assert positions_grad_keys is not None
        assert positions_grad.device == device
        assert positions_grad.dtype == dtype

        result = _reduce_grad(positions_grad, positions_grad_keys, indexes)
        reduced_positions_grad, reduced_positions_grad_keys = result
    else:
        reduced_positions_grad = None
        reduced_positions_grad_keys = None

    if cell_grad is not None:
        assert cell_grad_keys is not None
        assert cell_grad.device == device

        result = _reduce_grad(cell_grad, cell_grad_keys, indexes)
        reduced_cell_grad, reduced_cell_grad_keys = result
    else:
        reduced_cell_grad = None
        reduced_cell_grad_keys = None

    reduced_values = (reduced_values, reduced_keys.reshape(-1, 1))
    reduced_positions_grad = (reduced_positions_grad, reduced_positions_grad_keys)
    reduced_cell_grad = (reduced_cell_grad, reduced_cell_grad_keys)
    return (reduced_values, reduced_positions_grad, reduced_cell_grad)


class ReduceValuesAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, values: Tensor, keys: Tensor, dim: int
    ) -> (Tensor, Tensor, Tensor):
        device = values.device
        dtype = values.dtype

        assert keys.dim() == 2, "keys should have only two dimensions"
        reduced_keys = torch.unique(keys[:, dim])

        indexes = torch.empty(values.shape[0], dtype=torch.int32, device=device)
        mapping = []
        for i, reduced_key in enumerate(reduced_keys):
            idx = torch.where(keys[:, dim] == reduced_key)[0]
            indexes.index_put_(
                (idx,), torch.tensor(i, dtype=torch.int32, device=device)
            )
            mapping.append(idx)

        new_shape = (len(reduced_keys),) + values.shape[1:]
        reduced_values = torch.zeros(new_shape, dtype=dtype, device=device)
        reduced_values.index_add_(0, indexes, values)

        ctx.save_for_backward(values)
        ctx.reduce_mapping = mapping

        ctx.mark_non_differentiable(reduced_keys)

        return reduced_values, reduced_keys.reshape(-1, 1), indexes

    @staticmethod
    def backward(ctx, grad_reduced_values, _grad_reduced_keys, _grad_indexes):
        (values,) = ctx.saved_tensors

        values_grad = None
        if values.requires_grad:
            values_grad = torch.zeros_like(values)
            for i, idx in enumerate(ctx.reduce_mapping):
                values_grad[idx] = grad_reduced_values[i]

        return values_grad, None, None


class ReduceGradientAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, gradient: Tensor, keys: Tensor, indexes: Tensor
    ) -> (Tensor, Tensor):
        device = gradient.device
        dtype = gradient.dtype

        new_keys = keys.clone()
        for grad_i, grad_key in enumerate(keys):
            new_keys[grad_i, 0] = indexes[grad_key[0]]

        reduced_keys = torch.unique(new_keys, dim=0)

        indexes = torch.empty(gradient.shape[0], dtype=torch.int32, device=device)
        mapping = []
        for i, reduced_key in enumerate(reduced_keys):
            # FIXME: this might be slow?
            idx = torch.all(new_keys == reduced_key[None, :], axis=1)
            mapping.append(idx)
            indexes.index_put_(
                (idx,), torch.tensor(i, dtype=torch.int32, device=device)
            )

        new_shape = (len(reduced_keys),) + gradient.shape[1:]
        reduced_gradient = torch.zeros(new_shape, dtype=dtype, device=device)
        reduced_gradient.index_add_(0, indexes, gradient)

        ctx.save_for_backward(gradient)
        ctx.reduce_mapping = mapping
        ctx.mark_non_differentiable(reduced_keys)

        return reduced_gradient, reduced_keys

    @staticmethod
    def backward(ctx, grad_reduced_gradient, _grad_reduced_keys):
        (gradient,) = ctx.saved_tensors

        gradient_grad = None
        if gradient.requires_grad:
            gradient_grad = torch.zeros_like(gradient)
            for i, idx in enumerate(ctx.reduce_mapping):
                # TODO: use index_put here as well?
                gradient_grad[idx] = grad_reduced_gradient[i]

        return gradient_grad, None, None


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
    device = values.device
    dtype = values.dtype

    values, keys, indexes = ReduceValuesAutograd.apply(values, keys, dim)

    if positions_grad is not None:
        assert positions_grad_keys is not None
        assert positions_grad.device == device
        assert positions_grad.dtype == dtype

        results = ReduceGradientAutograd.apply(
            positions_grad, positions_grad_keys, indexes
        )

        positions_grad, positions_grad_keys = results

    if cell_grad is not None:
        assert cell_grad_keys is not None
        assert cell_grad.device == device
        assert cell_grad.dtype == dtype

        results = ReduceGradientAutograd.apply(cell_grad, cell_grad_keys, indexes)

        cell_grad, cell_grad_keys = results

    reduced_values = (values, keys)
    reduced_positions_grad = (positions_grad, positions_grad_keys)
    reduced_cell_grad = (cell_grad, keys)
    return (reduced_values, reduced_positions_grad, reduced_cell_grad)
