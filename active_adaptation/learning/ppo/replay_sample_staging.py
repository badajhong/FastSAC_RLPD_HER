"""Lossless CPU replay gathering into one transferable byte buffer.

Each field keeps its dtype and shape. Byte offsets respect dtype alignment so
one device copy can be unpacked into ordinary contiguous typed tensor views.
No replay indices are generated here, and replay storage is never modified.
"""

import math

import numpy as np
import torch


_NUMPY_DTYPES = frozenset((
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128,
))


def _packed_replay_layout(data, fields, row_count):
    """Return (field layout, byte count), retaining first-occurrence key order."""
    layout = []
    offset = 0
    for key in dict.fromkeys(fields):
        value = data[key]
        shape = (row_count, *value.shape[1:])
        element_size = value.element_size()
        offset = (offset + element_size - 1) // element_size * element_size
        size = math.prod(shape) * element_size
        layout.append((key, value.dtype, shape, offset, size))
        offset += size
    return tuple(layout), offset


def _unpack_replay_sample(packed, layout):
    """Create dtype-preserving views; the packed tensor owns their storage."""
    return {
        key: packed.narrow(0, offset, size).view(dtype).view(shape)
        for key, dtype, shape, offset, size in layout
    }


@torch.no_grad()
def _gather_packed_replay(data, indices, layout, staging):
    """Gather physical rows into preallocated CPU byte storage without casts.

    NumPy avoids starting a large intra-op Torch thread team for each CPU field.
    Bounds are checked before ``take(mode='clip')``: that mode avoids NumPy's
    protective output temporary, but must not clamp an invalid replay index.
    Unusual dtypes and noncontiguous storage retain the Torch gather path.
    """
    if indices.device.type != "cpu" or indices.dtype != torch.long or indices.ndim != 1:
        raise TypeError("Packed replay gather requires one-dimensional CPU int64 indices")
    if staging.device.type != "cpu" or staging.dtype != torch.uint8 or staging.ndim != 1:
        raise TypeError("Packed replay staging must be one-dimensional CPU uint8 storage")
    if not layout:
        return
    # Match index_select bounds against allocated storage, not the ring's
    # logical size. Source rows may include the unused capacity in test seams.
    index_array = indices.detach().numpy()
    if index_array.size:
        minimum, maximum = int(index_array.min()), int(index_array.max())
        if minimum < 0 or any(maximum >= data[key].shape[0] for key, *_ in layout):
            raise IndexError("Replay index is outside a source storage field")
    destinations = _unpack_replay_sample(staging, layout)
    # Benchmarks at 1/4 threads favor Torch's vectorized gather; a large
    # intra-op team (24 threads in the training run) is costly per field.
    use_numpy = torch.get_num_threads() > 4
    for key, dtype, shape, _offset, _size in layout:
        source = data[key]
        if source.device.type != "cpu":
            raise ValueError("Packed replay source fields must be on CPU")
        if shape[0] != indices.numel() or source.dtype != dtype or tuple(source.shape[1:]) != shape[1:]:
            raise ValueError("Packed replay layout does not match the source fields and indices")
        destination = destinations[key]
        if (
            use_numpy
            and source.is_contiguous()
            and dtype in _NUMPY_DTYPES
            and not source.is_conj()
            and not source.is_neg()
        ):
            np.take(source.detach().numpy(), index_array, axis=0,
                    out=destination.numpy(), mode="clip")
        else:
            torch.index_select(source, 0, indices, out=destination)
