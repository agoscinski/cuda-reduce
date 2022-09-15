import torch


def reduce(input, keys, dim):
    assert keys.dim() == 2, "keys should have only two dimensions"
    unique_entries = torch.unique(keys[:, dim])
    new_shape = torch.Size((len(unique_entries),) + input.shape[1:])
    reduced_input = torch.zeros(new_shape)
    for i, unique_entry in enumerate(unique_entries):
        idx = torch.where(keys[:, dim] == unique_entry)[0]
        reduced_input[i : i + 1] = torch.sum(input[idx], axis=0)
    return reduced_input
