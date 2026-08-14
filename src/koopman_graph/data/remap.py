"""Growable node-universe remapping (open-world MVP).

Callers supply the index map. Silent merge of unrelated universes is refused.
"""

from __future__ import annotations

import torch
from torch import Tensor


def remap_node_features(
    features: Tensor,
    *,
    old_index: Tensor,
    new_capacity: int,
) -> Tensor:
    r"""Scatter node features into a larger union capacity.

    Parameters
    ----------
    features : Tensor
        Source features ``(N_old, F)``.
    old_index : Tensor
        Integer destinations in ``[0, new_capacity)`` with length ``N_old``.
    new_capacity : int
        Target union size \(N_{\max}\).

    Returns
    -------
    Tensor
        Features of shape ``(new_capacity, F)``.

    Raises
    ------
    ValueError
        If the map is not injective or indices are out of range.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must have shape (N, F), got {tuple(features.shape)}"
        )
    if old_index.ndim != 1 or int(old_index.shape[0]) != int(features.shape[0]):
        raise ValueError("old_index must be 1-D with length equal to N_old")
    if int(new_capacity) < int(features.shape[0]):
        raise ValueError("new_capacity must be at least N_old")
    if int(old_index.min()) < 0 or int(old_index.max()) >= int(new_capacity):
        raise ValueError("old_index contains IDs outside [0, new_capacity)")
    unique = torch.unique(old_index)
    if int(unique.numel()) != int(old_index.numel()):
        raise ValueError("old_index must be injective (no silent universe merge)")
    out = torch.zeros(
        int(new_capacity),
        features.shape[1],
        dtype=features.dtype,
        device=features.device,
    )
    out[old_index] = features
    return out
