from tensordict import TensorDictBase


def zero_tensordict_rows_(tensordict: TensorDictBase, row_ids) -> None:
    """Clear selected rows in place while preserving every leaf's dtype."""
    for value in tensordict.values(include_nested=True, leaves_only=True):
        value.index_fill_(0, row_ids, 0)
