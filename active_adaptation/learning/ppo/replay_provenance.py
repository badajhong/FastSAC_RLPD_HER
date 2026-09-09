"""Batch-local host snapshots of sampled replay provenance.

Public batch entries remain unchanged device tensors. Only prepared batches
carry the optional snapshot, so it cannot outlive a batch or enter replay.
"""

from __future__ import annotations

import torch


class _ReplayProvenanceBatch(dict):
    """A regular tensor mapping with a private, mutation-checked host cache."""

    _provenance_snapshot = None


def _provenance_signatures(values):
    # Inference tensors do not expose version counters. Re-read their contents
    # instead of assuming that their storage is immutable.
    if any(value.is_inference() or value.layout != torch.strided for value in values):
        return None
    # Tensor.data replacement can retain object identity and version. Track
    # storage and layout too. In-place writes through .data bypass PyTorch's
    # version tracking and are outside the sampled-metadata mutation contract.
    return tuple(
        (value._version, value.data_ptr(), value.dtype, value.device,
         tuple(value.shape), tuple(value.stride()))
        for value in values
    )


def _packed_provenance_cpu(values):
    """Use one device-to-host transfer, retaining the physical-index dtype."""
    index_dtype = values[2].dtype
    packed = torch.stack([value.to(dtype=index_dtype) for value in values]).cpu()
    return packed[0].bool(), packed[1].bool(), packed[2]


def replay_provenance_cpu(batch, keys):
    """Read masks and physical indices without changing public batch fields."""
    keys = tuple(keys)
    values = tuple(batch[key] for key in keys)
    signatures = _provenance_signatures(values)
    snapshot = getattr(batch, "_provenance_snapshot", None)
    if snapshot is not None and signatures is not None:
        old_keys, old_values, old_signatures, host_values = snapshot
        if (
            old_keys == keys
            and old_signatures == signatures
            and all(old is value for old, value in zip(old_values, values))
        ):
            return host_values

    flat = tuple(value.detach().reshape(-1) for value in values)
    if (
        len(flat) == 3
        and flat[0].device.type == "cuda"
        and all(value.device == flat[0].device for value in flat)
        and flat[0].dtype == flat[1].dtype == torch.bool
        and flat[2].dtype in (torch.int32, torch.int64)
        and all(value.numel() == flat[0].numel() for value in flat)
    ):
        host_values = _packed_provenance_cpu(flat)
    else:
        # Preserve dtype and malformed-input behavior for existing CPU, mixed
        # device and validation seams; callers retain their original checks.
        host_values = tuple(value.detach().cpu().reshape(-1) for value in values)

    if isinstance(batch, _ReplayProvenanceBatch):
        batch._provenance_snapshot = (
            (keys, values, signatures, host_values) if signatures is not None else None
        )
    return host_values


def prepared_with_provenance_snapshot(prepared, source, keys):
    """Share a snapshot only when normalization preserved metadata identities."""
    result = _ReplayProvenanceBatch(prepared)
    keys = tuple(keys)
    values = tuple(source[key] for key in keys)
    host_values = replay_provenance_cpu(source, keys)
    signatures = _provenance_signatures(values)
    if signatures is not None and all(result.get(key) is value for key, value in zip(keys, values)):
        result._provenance_snapshot = (keys, values, signatures, host_values)
    return result, host_values
