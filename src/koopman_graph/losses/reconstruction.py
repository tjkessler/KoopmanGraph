"""Masked MSE and worst-case reconstruction losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute mean squared error over observed nodes only.

    Averages squared errors over all feature channels of masked nodes::

        sum_{n in O, f} (pred_{n,f} - target_{n,f})^2 / (|O| * feature_dim)

    Parameters
    ----------
    prediction : Tensor
        Predicted node features with shape ``(num_nodes, feature_dim)``.
    target : Tensor
        Ground-truth node features with the same shape as ``prediction``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``. ``True`` marks an
        observed node included in the average.

    Returns
    -------
    Tensor
        Scalar masked mean squared error.
    """
    node_mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    expanded = node_mask.unsqueeze(-1)
    diff_sq = (prediction - target) ** 2
    denom = expanded.sum() * prediction.shape[-1]
    if denom <= 0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return (diff_sq * expanded).sum() / denom


class WorstCaseReconstructionLoss(nn.Module):
    """Batch :math:`L_\\infty`-style reconstruction penalty (robust training).

    For a prediction/target pair, computes per-node mean squared error over
    feature channels, then takes the maximum over nodes (optionally restricted
    to an observation mask). This is an **empirical robust training term** that
    emphasizes hard nodes within a batch/pair. It is **not** a PAC,
    :math:`L_\\infty`, or generalization-error certificate.

    Notes
    -----
    Prefer reporting this alongside mean MSE rather than claiming max-error
    training “bounds generalization error.”
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the max per-node MSE between ``prediction`` and ``target``.

        Parameters
        ----------
        prediction : Tensor
            Predicted node features with shape ``(num_nodes, feature_dim)``.
        target : Tensor
            Ground-truth node features with the same shape as ``prediction``.
        mask : Tensor or None, optional
            Boolean node mask with shape ``(num_nodes,)``. When provided, the
            max is taken only over observed nodes.

        Returns
        -------
        Tensor
            Scalar worst-case (max over nodes) mean-squared error.

        Raises
        ------
        ValueError
            If ``mask`` is provided but has the wrong length, or no nodes remain
            after masking.
        """
        node_mse = (prediction - target).square().mean(dim=-1)
        if mask is None:
            return node_mse.max()
        node_mask = mask.to(device=prediction.device, dtype=torch.bool)
        if node_mask.numel() != node_mse.numel():
            msg = (
                "mask length must match num_nodes, "
                f"got mask length {node_mask.numel()} and "
                f"num_nodes {node_mse.numel()}"
            )
            raise ValueError(msg)
        if not bool(node_mask.any()):
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        return node_mse[node_mask].max()
