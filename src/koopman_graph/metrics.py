"""Forecast evaluation metrics for graph snapshot sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.losses import masked_mse_loss
from koopman_graph.training import resolve_rollout_start_indices

_EPS = 1e-8


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    """Compute mean absolute error.

    Parameters
    ----------
    prediction : Tensor
        Predicted values.
    target : Tensor
        Ground-truth values with the same shape as ``prediction``.

    Returns
    -------
    Tensor
        Scalar mean absolute error.
    """
    return torch.mean(torch.abs(prediction - target))


def rmse(prediction: Tensor, target: Tensor) -> Tensor:
    """Compute root mean squared error.

    Parameters
    ----------
    prediction : Tensor
        Predicted values.
    target : Tensor
        Ground-truth values with the same shape as ``prediction``.

    Returns
    -------
    Tensor
        Scalar root mean squared error.
    """
    return torch.sqrt(nn.functional.mse_loss(prediction, target))


def mape(
    prediction: Tensor,
    target: Tensor,
    *,
    eps: float = _EPS,
) -> Tensor:
    """Compute mean absolute percentage error.

    Parameters
    ----------
    prediction : Tensor
        Predicted values.
    target : Tensor
        Ground-truth values with the same shape as ``prediction``.
    eps : float, optional
        Small constant added to the denominator for numerical stability.
        Default is ``1e-8``.

    Returns
    -------
    Tensor
        Scalar mean absolute percentage error (not scaled to 0–100).
    """
    return torch.mean(torch.abs((prediction - target) / (target.abs() + eps)))


def _masked_node_average(
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    """Average per-node scalar values over observed nodes.

    Parameters
    ----------
    values : Tensor
        Per-node values with shape ``(num_nodes,)``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``.

    Returns
    -------
    Tensor
        Scalar average over observed nodes.
    """
    node_mask = mask.to(device=values.device, dtype=values.dtype)
    denom = node_mask.sum()
    if denom <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values.abs() * node_mask).sum() / denom


def masked_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute mean absolute error over observed nodes.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features with shape ``(num_nodes, feature_dim)``.
    target : Tensor
        Ground-truth node features with the same shape as ``prediction``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``.

    Returns
    -------
    Tensor
        Scalar masked mean absolute error.
    """
    return _masked_node_average((prediction - target).abs().mean(dim=-1), mask)


def masked_rmse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute root mean squared error over observed nodes.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features with shape ``(num_nodes, feature_dim)``.
    target : Tensor
        Ground-truth node features with the same shape as ``prediction``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``.

    Returns
    -------
    Tensor
        Scalar masked root mean squared error.
    """
    return torch.sqrt(masked_mse_loss(prediction, target, mask))


def masked_mape(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    eps: float = _EPS,
) -> Tensor:
    """Compute mean absolute percentage error over observed nodes.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features with shape ``(num_nodes, feature_dim)``.
    target : Tensor
        Ground-truth node features with the same shape as ``prediction``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``.
    eps : float, optional
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    Tensor
        Scalar masked mean absolute percentage error.
    """
    per_node = ((prediction - target) / (target.abs() + eps)).abs().mean(dim=-1)
    return _masked_node_average(per_node, mask)


@dataclass(frozen=True)
class HorizonMetrics:
    """Forecast metrics at a single prediction horizon.

    Attributes
    ----------
    horizon : int
        Forecast horizon in steps.
    mae : float
        Mean absolute error averaged over evaluation origins.
    rmse : float
        Root mean squared error averaged over evaluation origins.
    mape : float
        Mean absolute percentage error averaged over evaluation origins.
    """

    horizon: int
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class EvaluationResult:
    """Multi-horizon forecast evaluation summary.

    Attributes
    ----------
    horizons : tuple of HorizonMetrics
        Per-horizon metrics in ascending horizon order.
    aggregate_mae : float
        Mean of per-horizon MAE values.
    aggregate_rmse : float
        Mean of per-horizon RMSE values.
    aggregate_mape : float
        Mean of per-horizon MAPE values.
    num_origins : int
        Number of forecast origins averaged over.
    """

    horizons: tuple[HorizonMetrics, ...]
    aggregate_mae: float
    aggregate_rmse: float
    aggregate_mape: float
    num_origins: int


def evaluate_forecast(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    *,
    horizons: Sequence[int] = (3, 6, 12),
    start_indices: Sequence[int] | None = None,
) -> EvaluationResult:
    """Evaluate autoregressive multi-horizon forecasts on a snapshot sequence.

    For each forecast origin, the model predicts up to ``max(horizons)`` steps
    ahead and metrics are averaged across origins at each requested horizon.

    Parameters
    ----------
    model : nn.Module
        Model implementing :meth:`~koopman_graph.model.GraphKoopmanModel.predict`.
    sequence : GraphSnapshotSequence
        Evaluation snapshots with shared topology.
    horizons : sequence of int, optional
        Forecast horizons to report. Default is ``(3, 6, 12)``.
    start_indices : sequence of int or None, optional
        Forecast-origin indices. When ``None``, uses every valid origin in
        ``sequence``.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate MAE, RMSE, and MAPE.

    Raises
    ------
    ValueError
        If ``horizons`` is empty, any horizon is invalid, or the sequence is
        too short.
    """
    if not horizons:
        msg = "horizons must contain at least one step"
        raise ValueError(msg)

    sorted_horizons = sorted(set(horizons))
    if any(horizon < 1 for horizon in sorted_horizons):
        msg = f"all horizons must be >= 1, got {sorted_horizons}"
        raise ValueError(msg)

    max_horizon = sorted_horizons[-1]
    origins = resolve_rollout_start_indices(
        sequence,
        horizon=max_horizon,
        rollout_start_indices="all" if start_indices is None else start_indices,
    )

    mae_sums = {horizon: 0.0 for horizon in sorted_horizons}
    rmse_sums = {horizon: 0.0 for horizon in sorted_horizons}
    mape_sums = {horizon: 0.0 for horizon in sorted_horizons}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in origins:
                initial_graph: Data = sequence[start]
                controls = None
                if getattr(model, "control_dim", 0) > 0:
                    controls = sequence.rollout_controls(start, max_horizon)
                future_topologies = None
                if sequence.is_dynamic_topology:
                    future_topologies = [
                        sequence[start + step] for step in range(1, max_horizon + 1)
                    ]
                predictions = model.predict(
                    initial_graph,
                    steps=max_horizon,
                    controls=controls,
                    future_topologies=future_topologies,
                )
                for horizon in sorted_horizons:
                    pred = predictions[horizon - 1].x
                    target = sequence[start + horizon].x
                    if sequence.has_observation_masks:
                        node_mask = sequence.observation_mask_at(start + horizon)
                        mae_sums[horizon] += float(
                            masked_mae(pred, target, node_mask).cpu()
                        )
                        rmse_sums[horizon] += float(
                            masked_rmse(pred, target, node_mask).cpu()
                        )
                        mape_sums[horizon] += float(
                            masked_mape(pred, target, node_mask).cpu()
                        )
                    else:
                        mae_sums[horizon] += float(mae(pred, target).cpu())
                        rmse_sums[horizon] += float(rmse(pred, target).cpu())
                        mape_sums[horizon] += float(mape(pred, target).cpu())
    finally:
        model.train(was_training)

    num_origins = len(origins)
    horizon_metrics = tuple(
        HorizonMetrics(
            horizon=horizon,
            mae=mae_sums[horizon] / num_origins,
            rmse=rmse_sums[horizon] / num_origins,
            mape=mape_sums[horizon] / num_origins,
        )
        for horizon in sorted_horizons
    )
    return EvaluationResult(
        horizons=horizon_metrics,
        aggregate_mae=sum(metric.mae for metric in horizon_metrics)
        / len(horizon_metrics),
        aggregate_rmse=sum(metric.rmse for metric in horizon_metrics)
        / len(horizon_metrics),
        aggregate_mape=sum(metric.mape for metric in horizon_metrics)
        / len(horizon_metrics),
        num_origins=num_origins,
    )
