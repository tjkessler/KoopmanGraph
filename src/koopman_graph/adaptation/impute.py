"""Heuristic graph diffusion imputation for masked node features."""

from __future__ import annotations

import torch
from torch import Tensor


def graph_diffuse_impute(
    x: Tensor,
    mask: Tensor,
    edge_index: Tensor,
    *,
    iterations: int = 8,
) -> Tensor:
    """Fill unobserved node features by iterative neighbor averaging.

    This is a **heuristic** warm-start for masked sensors, not a calibrated
    observation model. Observed rows are left unchanged.

    Parameters
    ----------
    x : Tensor
        Node features ``(num_nodes, feature_dim)``.
    mask : Tensor
        Boolean observation mask ``(num_nodes,)`` (``True`` = observed).
    edge_index : Tensor
        Edge index ``(2, E)``.
    iterations : int, optional
        Diffusion sweeps. Default is ``8``.

    Returns
    -------
    Tensor
        Imputed features with the same shape as ``x``.
    """
    if iterations < 1:
        msg = f"iterations must be positive, got {iterations}"
        raise ValueError(msg)
    out = x.clone()
    observed = mask.bool()
    if bool(observed.all()):
        return out
    num_nodes = x.size(0)
    src, dst = edge_index[0], edge_index[1]
    for _ in range(iterations):
        deg = torch.zeros(num_nodes, dtype=out.dtype, device=out.device)
        acc = torch.zeros_like(out)
        deg.index_add_(
            0, dst, torch.ones(dst.numel(), dtype=out.dtype, device=out.device)
        )
        acc.index_add_(0, dst, out[src])
        neighbor_mean = acc / deg.clamp_min(1.0).unsqueeze(1)
        out = torch.where(observed.unsqueeze(1), out, neighbor_mean)
    return out
