"""Forecast evaluation metrics for graph snapshot sequences."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
    resolve_hetero_sequence,
    resolve_rollout_start_indices,
    resolve_sequence,
)
from koopman_graph.losses import masked_mse_loss
from koopman_graph.protocols import TrainableKoopmanModel

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

    Averages absolute error over the feature dimension per node, then averages
    those per-node values over masked (observed) nodes.

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
    resdmd : object or None
        Optional finite-dictionary :class:`~koopman_graph.analysis.ResDMDReport`
        when ``include_resdmd=True``.
    invariance : object or None
        Optional :class:`~koopman_graph.identification.SubspaceInvarianceReport`
        when ``include_invariance=True``. Typed as ``object`` so this module
        does not import ``identification`` at load.
    """

    horizons: tuple[HorizonMetrics, ...]
    aggregate_mae: float
    aggregate_rmse: float
    aggregate_mape: float
    num_origins: int
    resdmd: object | None = None
    invariance: object | None = None


def _hetero_eval_feature_vector(
    snapshot: HeteroData,
    node_types: Sequence[str],
) -> Tensor:
    """Flatten and concatenate per-type ``x`` in ``node_types`` order.

    Unequal trailing widths ``F_τ`` are allowed: each type is reshaped to a
    1-D vector before concatenation. This is a stacked aggregate over all
    physical channels, not a certified per-type metric report.

    Parameters
    ----------
    snapshot : HeteroData
        Multiplex or typed snapshot.
    node_types : sequence of str
        Ordered node-type names (typically ``sequence.node_type_names``).

    Returns
    -------
    Tensor
        Concatenated feature vector with shape ``(Σ_τ N_τ · F_τ,)``.

    Raises
    ------
    ValueError
        If a node type is missing or lacks ``x``.
    """
    blocks: list[Tensor] = []
    present = set(snapshot.node_types)
    for name in node_types:
        if name not in present:
            msg = (
                f"HeteroData snapshot is missing node type {name!r}; "
                f"present types are {sorted(present)!r}"
            )
            raise ValueError(msg)
        features = snapshot[name].x
        if features is None:
            msg = f"HeteroData node type {name!r} is missing feature matrix x"
            raise ValueError(msg)
        blocks.append(features.reshape(-1))
    return torch.cat(blocks, dim=0)


def _validate_hetero_evaluate_surface(sequence: HeteroGraphSnapshotSequence) -> None:
    """Reject hetero evaluate options that would silently use wrong topology.

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence
        Evaluation trajectory.

    Raises
    ------
    ValueError
        If unsupported hetero evaluate options are requested.
    """
    if sequence.is_dynamic_topology:
        msg = (
            "dynamic-topology HeteroGraphSnapshotSequence evaluate is "
            "unsupported; use static relation banks"
        )
        raise ValueError(msg)
    if sequence.has_controls:
        msg = (
            "controlled HeteroGraphSnapshotSequence evaluate is unsupported; "
            "omit control_inputs"
        )
        raise ValueError(msg)
    if sequence.has_observation_masks:
        msg = (
            "observation-masked HeteroGraphSnapshotSequence evaluate is "
            "unsupported; omit observation_masks"
        )
        raise ValueError(msg)


def _resolve_evaluate_sequence(
    sequence: SnapshotSequence | Sequence[Data] | Sequence[HeteroData],
) -> SnapshotSequence:
    """Normalize evaluate input into a homogeneous or hetero sequence.

    Parameters
    ----------
    sequence : SnapshotSequence or sequence of Data or HeteroData
        Raw evaluation input.

    Returns
    -------
    GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Validated sequence container.

    Raises
    ------
    TypeError
        If the input kind cannot be resolved.
    ValueError
        If a plain sequence is empty.
    """
    if isinstance(sequence, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)):
        return sequence
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
        if not sequence:
            msg = "evaluate sequence must contain at least one snapshot"
            raise ValueError(msg)
        first = sequence[0]
        if isinstance(first, HeteroData):
            return resolve_hetero_sequence(sequence)  # type: ignore[arg-type]
        if isinstance(first, Data):
            return resolve_sequence(sequence)  # type: ignore[arg-type]
    msg = (
        "evaluate_forecast expects GraphSnapshotSequence, "
        "HeteroGraphSnapshotSequence, or a non-empty sequence of Data / "
        f"HeteroData; got {type(sequence).__name__}"
    )
    raise TypeError(msg)


def _identity_resdmd(model: TrainableKoopmanModel, sequence: GraphSnapshotSequence):
    """Finite-dictionary ResDMD on frozen encodings (identity dictionary).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``encode``.
    sequence : GraphSnapshotSequence
        Homogeneous snapshots.

    Returns
    -------
    object or None
        :class:`~koopman_graph.analysis.ResDMDReport` when encodings are
        available and ``T >= 3``; otherwise ``None``.
    """
    encode = getattr(model, "encode", None)
    if encode is None or sequence.num_timesteps < 3:
        return None
    rows = []
    for index in range(sequence.num_timesteps):
        latent = encode(sequence[index])
        rows.append(latent.reshape(1, -1).detach())
    stacked = torch.cat(rows, dim=0)
    from koopman_graph.analysis.resdmd import resdmd

    return resdmd(stacked[:-1], stacked[1:])


def _dense_pernode_discrete_k(model: TrainableKoopmanModel) -> Tensor:
    """Return dense per-node ``K`` or refuse unsupported layouts.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``koopman`` and layout metadata.

    Returns
    -------
    Tensor
        Dense discrete map with shape ``(d, d)``.

    Raises
    ------
    ValueError
        If the operator is not an uncontrolled dense discrete
        :class:`~koopman_graph.operators.KoopmanOperator`.
    """
    from koopman_graph.operators import KoopmanOperator

    kind = getattr(model, "koopman_kind", "pernode")
    if kind != "pernode":
        msg = (
            "subspace invariance currently supports discrete per-node "
            f"KoopmanOperator only, got koopman_kind={kind!r}"
        )
        raise ValueError(msg)
    if getattr(model, "dynamics_mode", "discrete") != "discrete":
        msg = "subspace invariance requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if getattr(model, "n_delays", 1) != 1:
        msg = (
            "subspace invariance does not support delay embeddings "
            f"(n_delays={getattr(model, 'n_delays', None)})"
        )
        raise ValueError(msg)
    if int(getattr(model, "control_dim", 0)) != 0:
        msg = (
            "subspace invariance does not support controlled models "
            f"(control_dim={getattr(model, 'control_dim', None)})"
        )
        raise ValueError(msg)
    koopman = model.koopman
    if not isinstance(koopman, KoopmanOperator):
        msg = (
            "subspace invariance currently supports discrete per-node "
            f"KoopmanOperator only, got {type(koopman).__name__}"
        )
        raise ValueError(msg)
    if koopman.parameterization != "dense":
        msg = (
            "subspace invariance requires parameterization='dense', "
            f"got {koopman.parameterization!r}"
        )
        raise ValueError(msg)
    if koopman.control_dim != 0:
        msg = (
            "subspace invariance does not support controlled operators "
            f"(control_dim={koopman.control_dim})"
        )
        raise ValueError(msg)
    return koopman.K


def _encode_invariance_latents(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Stack per-snapshot encodings as ``(T, N, d)``.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``encode_at`` or ``encode``.
    sequence : GraphSnapshotSequence
        Homogeneous snapshots.

    Returns
    -------
    Tensor
        Time-major latents.

    Raises
    ------
    ValueError
        If the model has no encoder façade.
    """
    encode_at = getattr(model, "encode_at", None)
    encode = getattr(model, "encode", None)
    if encode_at is None and encode is None:
        msg = "subspace invariance requires model.encode or model.encode_at"
        raise ValueError(msg)
    stacked = []
    for index in range(sequence.num_timesteps):
        if encode_at is not None:
            stacked.append(encode_at(sequence, index))
        else:
            stacked.append(encode(sequence[index]))
    return torch.stack(stacked, dim=0)


def _identity_invariance(model: TrainableKoopmanModel, sequence: GraphSnapshotSequence):
    """Finite-sample subspace leakage on frozen encodings.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Homogeneous discrete per-node model.
    sequence : GraphSnapshotSequence
        Evaluation snapshots.

    Returns
    -------
    object
        :class:`~koopman_graph.identification.SubspaceInvarianceReport`.

    Raises
    ------
    ValueError
        If the operator layout is unsupported or the leakage denominator
        is degenerate.
    """
    matrix = _dense_pernode_discrete_k(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encodings = _encode_invariance_latents(model, sequence)
    finally:
        model.train(was_training)
    from koopman_graph.identification import subspace_invariance_report

    return subspace_invariance_report(encodings, matrix, held_out=True)


def _pack_evaluation_result(
    horizons: Sequence[int],
    mae_sums: dict[int, float],
    rmse_sums: dict[int, float],
    mape_sums: dict[int, float],
    origins: Sequence[int],
    resdmd: object | None = None,
    invariance: object | None = None,
) -> EvaluationResult:
    """Assemble per-horizon averages into :class:`EvaluationResult`.

    Parameters
    ----------
    horizons : sequence of int
        Sorted unique horizons.
    mae_sums, rmse_sums, mape_sums : dict of int to float
        Accumulated metric sums keyed by horizon.
    origins : sequence of int
        Forecast origins used for averaging.
    resdmd : object or None, optional
        Optional finite-dictionary ResDMD report.
    invariance : object or None, optional
        Optional finite-sample subspace-invariance report.

    Returns
    -------
    EvaluationResult
        Packed evaluation summary.
    """
    num_origins = len(origins)
    horizon_metrics = tuple(
        HorizonMetrics(
            horizon=horizon,
            mae=mae_sums[horizon] / num_origins,
            rmse=rmse_sums[horizon] / num_origins,
            mape=mape_sums[horizon] / num_origins,
        )
        for horizon in horizons
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
        resdmd=resdmd,
        invariance=invariance,
    )


def evaluate_forecast(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence | Sequence[Data] | Sequence[HeteroData],
    *,
    horizons: Sequence[int] = (3, 6, 12),
    start_indices: Sequence[int] | None = None,
    include_resdmd: bool = False,
    include_invariance: bool = False,
    topology_policy: str = "auto",
) -> EvaluationResult:
    """Evaluate autoregressive multi-horizon forecasts on a snapshot sequence.

    For each forecast origin, the model predicts up to ``max(horizons)`` steps
    ahead and metrics are averaged across origins at each requested horizon.

    Homogeneous sequences compare ``Data.x``. When presence masks are attached,
    ``predict`` receives the corresponding ``future_presence`` schedule (hold
    last active state for inactive entities; matvecs still use ``N_max``) and
    metrics use :meth:`~koopman_graph.data.GraphSnapshotSequence.loss_mask_at`
    so absent or unobserved rows do not contribute. Heterogeneous sequences
    compare concatenated flattened per-type features in
    :attr:`~koopman_graph.data.HeteroGraphSnapshotSequence.node_type_names`
    order (stacked aggregate over all physical channels; not a certified
    per-type coverage or metric report). Hetero evaluate requires static
    relation banks and rejects controls / observation masks.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable model implementing
        :meth:`~koopman_graph.protocols.TrainableKoopmanModel.predict` and the
        Module train/eval façade.
    sequence : GraphSnapshotSequence, HeteroGraphSnapshotSequence, or sequence
        Evaluation snapshots. Plain ``Data`` / ``HeteroData`` sequences are
        wrapped in the matching container.
    horizons : sequence of int, optional
        Forecast horizons to report. Default is ``(3, 6, 12)``.
    start_indices : sequence of int or None, optional
        Forecast-origin indices. When ``None``, uses every valid origin in
        ``sequence``.
    include_resdmd : bool, optional
        When ``True``, attach a finite-dictionary ResDMD report from frozen
        encodings. Default is ``False``.
    include_invariance : bool, optional
        When ``True``, attach finite-sample subspace leakage :math:`\\eta`
        from frozen encodings (truncated-SVD projector; last half of the
        time axis). Default is ``False``. Discrete dense per-node
        :math:`K` only; hetero / graph / continuous layouts raise.
        This is not a Haseli–Cortés certificate.
    topology_policy : {"auto", "recursive", "hold_last"}, optional
        Homogeneous topology schedule forwarded to ``predict``. When the
        resolved policy is recursive, dynamic sequences do **not** inject
        oracle future edges. Default ``"auto"`` keeps 0.14 oracle
        injection unless a recursive graph-state head is attached.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate MAE, RMSE, and MAPE.

    Raises
    ------
    ValueError
        If ``horizons`` is empty, any horizon is invalid, the sequence is too
        short, unsupported hetero evaluate options are requested, or
        ``include_invariance=True`` is used on an unsupported layout.
    TypeError
        If ``sequence`` is neither homogeneous nor hetero.
    """
    if not horizons:
        msg = "horizons must contain at least one step"
        raise ValueError(msg)

    sorted_horizons = sorted(set(horizons))
    if any(horizon < 1 for horizon in sorted_horizons):
        msg = f"all horizons must be >= 1, got {sorted_horizons}"
        raise ValueError(msg)

    resolved = _resolve_evaluate_sequence(sequence)
    if isinstance(resolved, HeteroGraphSnapshotSequence):
        if include_invariance:
            msg = (
                "include_invariance is unsupported for hetero evaluate; "
                "use a discrete dense per-node KoopmanOperator"
            )
            raise ValueError(msg)
        return _evaluate_hetero_forecast(
            model,
            resolved,
            horizons=sorted_horizons,
            start_indices=start_indices,
        )
    return _evaluate_homogeneous_forecast(
        model,
        resolved,
        horizons=sorted_horizons,
        start_indices=start_indices,
        include_resdmd=include_resdmd,
        include_invariance=include_invariance,
        topology_policy=topology_policy,
    )


def _evaluate_homogeneous_forecast(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizons: Sequence[int],
    start_indices: Sequence[int] | None,
    include_resdmd: bool = False,
    include_invariance: bool = False,
    topology_policy: str = "auto",
) -> EvaluationResult:
    """Homogeneous multi-horizon evaluate path.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Forecast model.
    sequence : GraphSnapshotSequence
        Homogeneous evaluation trajectory.
    horizons : sequence of int
        Sorted unique horizons.
    start_indices : sequence of int or None
        Explicit origins, or ``None`` for all valid origins.
    include_resdmd : bool, optional
        Attach finite-dictionary ResDMD when ``True``.
    include_invariance : bool, optional
        Attach finite-sample subspace leakage when ``True``.
    topology_policy : {"auto", "recursive", "hold_last"}, optional
        Topology schedule forwarded to ``predict``.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate metrics.
    """
    if include_invariance:
        _dense_pernode_discrete_k(model)
    max_horizon = horizons[-1]
    origins = resolve_rollout_start_indices(
        sequence,
        horizon=max_horizon,
        rollout_start_indices="all" if start_indices is None else start_indices,
    )

    mae_sums = {horizon: 0.0 for horizon in horizons}
    rmse_sums = {horizon: 0.0 for horizon in horizons}
    mape_sums = {horizon: 0.0 for horizon in horizons}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in origins:
                initial_graph: Data = sequence[start]
                controls = None
                if model.control_dim > 0:
                    controls = sequence.rollout_controls(start, max_horizon)
                future_topologies = None
                inject_oracle = sequence.is_dynamic_topology
                if inject_oracle:
                    from koopman_graph.nn.predicted_topology import (
                        resolve_topology_policy,
                    )

                    resolved_policy = resolve_topology_policy(model, topology_policy)
                    if topology_policy == "hold_last" or resolved_policy == "recursive":
                        inject_oracle = False
                if inject_oracle:
                    future_topologies = [
                        sequence[start + step] for step in range(1, max_horizon + 1)
                    ]
                history = None
                n_delays = int(getattr(model, "n_delays", 1))
                future_presence = None
                if sequence.has_presence_masks:
                    assert sequence.presence_masks is not None
                    future_presence = sequence.presence_masks[
                        start + 1 : start + max_horizon + 1
                    ]
                predict_kwargs: dict[str, object] = {
                    "controls": controls,
                    "future_topologies": future_topologies,
                }
                # Only forward when present so models without the kwarg (or with
                # call-site parity that rejects non-None) keep working under
                # evaluate_forecast on all-present sequences.
                if future_presence is not None:
                    predict_kwargs["future_presence"] = future_presence
                if "topology_policy" in inspect.signature(model.predict).parameters:
                    predict_kwargs["topology_policy"] = topology_policy
                predict_params = inspect.signature(model.predict).parameters
                trajectory = sequence.parameter_trajectory
                if trajectory is not None and "parameters" in predict_params:
                    predict_kwargs["parameters"] = [
                        trajectory[start + step] for step in range(max_horizon)
                    ]
                if n_delays > 1:
                    history = [
                        sequence[t] for t in range(max(0, start - n_delays + 1), start)
                    ]
                    predictions = model.predict(
                        initial_graph,
                        steps=max_horizon,
                        history=history,
                        **predict_kwargs,  # type: ignore[arg-type]
                    )
                else:
                    predictions = model.predict(
                        initial_graph,
                        steps=max_horizon,
                        **predict_kwargs,  # type: ignore[arg-type]
                    )
                for horizon in horizons:
                    pred = predictions[horizon - 1].x
                    target = sequence[start + horizon].x
                    node_mask = sequence.loss_mask_at(start + horizon)
                    if node_mask is not None:
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

    report = _identity_resdmd(model, sequence) if include_resdmd else None
    invariance = _identity_invariance(model, sequence) if include_invariance else None
    return _pack_evaluation_result(
        horizons,
        mae_sums,
        rmse_sums,
        mape_sums,
        origins,
        resdmd=report,
        invariance=invariance,
    )


def _evaluate_hetero_forecast(
    model: TrainableKoopmanModel,
    sequence: HeteroGraphSnapshotSequence,
    *,
    horizons: Sequence[int],
    start_indices: Sequence[int] | None,
) -> EvaluationResult:
    """Hetero multi-horizon evaluate path (stacked flattened features).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Hetero forecast model (``koopman='hetero_graph'``).
    sequence : HeteroGraphSnapshotSequence
        Multiplex or typed evaluation trajectory.
    horizons : sequence of int
        Sorted unique horizons.
    start_indices : sequence of int or None
        Explicit origins, or ``None`` for all valid origins.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate metrics on concatenated features.
    """
    _validate_hetero_evaluate_surface(sequence)
    n_delays = int(getattr(model, "n_delays", 1))
    if n_delays > 1:
        msg = (
            "delay embedding (n_delays > 1) is unsupported for hetero evaluate_forecast"
        )
        raise ValueError(msg)
    if not bool(getattr(model, "uses_hetero_koopman", False)):
        msg = (
            "HeteroGraphSnapshotSequence evaluate requires "
            "koopman='hetero_graph' (uses_hetero_koopman)"
        )
        raise TypeError(msg)

    max_horizon = horizons[-1]
    origins = resolve_rollout_start_indices(
        sequence,
        horizon=max_horizon,
        rollout_start_indices="all" if start_indices is None else start_indices,
    )
    node_types = sequence.node_type_names

    mae_sums = {horizon: 0.0 for horizon in horizons}
    rmse_sums = {horizon: 0.0 for horizon in horizons}
    mape_sums = {horizon: 0.0 for horizon in horizons}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in origins:
                initial_graph = sequence[start]
                predictions = model.predict(initial_graph, steps=max_horizon)
                for horizon in horizons:
                    pred_snap = predictions[horizon - 1]
                    target_snap = sequence[start + horizon]
                    if not isinstance(pred_snap, HeteroData):
                        msg = (
                            "hetero evaluate_forecast expects predict() to "
                            f"return HeteroData; got {type(pred_snap).__name__}"
                        )
                        raise TypeError(msg)
                    pred = _hetero_eval_feature_vector(pred_snap, node_types)
                    target = _hetero_eval_feature_vector(target_snap, node_types)
                    mae_sums[horizon] += float(mae(pred, target).cpu())
                    rmse_sums[horizon] += float(rmse(pred, target).cpu())
                    mape_sums[horizon] += float(mape(pred, target).cpu())
    finally:
        model.train(was_training)

    return _pack_evaluation_result(horizons, mae_sums, rmse_sums, mape_sums, origins)
