"""Inference helpers for GraphKoopmanModel (spectrum / predict / evaluate).

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.analysis import (
    compute_generator_spectrum,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    hold_last_topology_at,
    pack_rollout_snapshots,
)
from koopman_graph.metrics import EvaluationResult, evaluate_forecast
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.operators.contract import KoopmanOperatorContract
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

from .timing import resolve_time_increments, validate_uniform_discrete_increments
from .validation import validate_controls

EncodeRolloutOriginFn = Callable[
    ...,
    tuple[Tensor, Tensor, Tensor | None],
]


def compute_model_spectrum(
    koopman: KoopmanOperatorContract,
    *,
    uses_graph_koopman: bool,
    is_continuous: bool,
    time_step: float,
    delta_t: float | None = None,
    edge_index: Tensor | None = None,
    num_nodes: int | None = None,
    edge_weight: Tensor | None = None,
) -> KoopmanSpectrum:
    """Analyze the active Koopman operator spectrum for a model configuration.

    Parameters
    ----------
    koopman
        Active operator contract module.
    uses_graph_koopman : bool
        Whether ``koopman`` is a networked :class:`GraphKoopmanOperator`.
    is_continuous : bool
        Whether the model uses continuous-time dynamics.
    time_step : float
        Discrete model time step (used for discrete spectrum scaling).
    delta_t : float or None, optional
        Continuous integration horizon for generator → discrete spectrum.
    edge_index, num_nodes, edge_weight
        Topology arguments required for networked graph operators.

    Returns
    -------
    KoopmanSpectrum
        Magnitude-sorted eigenvalues, eigenvectors, and time scales.

    Raises
    ------
    ValueError
        If graph-operator topology is missing or continuous
        ``auxiliary_spectral`` spectrum is requested.
    """
    if uses_graph_koopman:
        if edge_index is None or num_nodes is None:
            msg = (
                "edge_index and num_nodes are required for "
                "GraphKoopmanModel.spectrum when koopman='graph' "
                "(topology-coupled effective operator); the per-node "
                "contract matrix K_self is not a substitute"
            )
            raise ValueError(msg)
        assert isinstance(koopman, GraphKoopmanOperator)
        return koopman.spectrum(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            time_step=time_step,
        )
    if is_continuous:
        if koopman.parameterization == "auxiliary_spectral":
            msg = (
                "GraphKoopmanModel.spectrum is unavailable for "
                "koopman_parameterization='auxiliary_spectral' (state-dependent "
                "generator); use model.koopman.generator_at(z) / "
                "instantaneous_spectrum(z), or compute_generator_spectrum on "
                "a frozen L(z)"
            )
            raise ValueError(msg)
        if delta_t is None:
            return compute_generator_spectrum(koopman.matrix)
        return discrete_spectrum_at_delta_t(koopman.matrix, delta_t)
    return compute_spectrum(koopman.matrix, time_step)


def latent_decode_rollout(
    koopman: KoopmanOperatorContract,
    decoder: nn.Module,
    encode_rollout_origin: EncodeRolloutOriginFn,
    *,
    x_or_data: Tensor | Data,
    steps: int,
    control_dim: int,
    default_delta_t: float,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    controls: Sequence[Tensor] | None = None,
    future_topologies: Sequence[Data] | None = None,
    step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
    history: Sequence[Data] | None = None,
) -> list[tuple[Tensor, Tensor, Tensor | None]]:
    """Autoregressively advance latent state and decode for multiple steps.

    Parameters
    ----------
    koopman, decoder
        Active propagator and reconstruction modules.
    encode_rollout_origin
        Callable matching :meth:`GraphKoopmanModel.encode_rollout_origin`.
    x_or_data : Tensor or Data
        Initial graph features or snapshot.
    steps : int
        Number of rollout steps (must be >= 1).
    control_dim : int
        Model control dimension (for control validation).
    default_delta_t : float
        Soft-default integration interval when ``step_deltas`` is omitted.
    edge_index, edge_weight, controls, future_topologies, step_deltas, history
        Same semantics as :meth:`GraphKoopmanModel.predict` / ``_rollout``.

    Returns
    -------
    list of tuple[Tensor, Tensor, Tensor or None]
        For each step, decoded prediction, ``edge_index``, and optional
        ``edge_weight`` used for decoding.

    Raises
    ------
    ValueError
        If ``steps < 1`` or controls / step deltas are invalid.
    """
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)

    validate_controls(control_dim=control_dim, controls=controls, steps=steps)
    if step_deltas is not None and len(step_deltas) != steps:
        msg = f"expected {steps} step_deltas for rollout, got {len(step_deltas)}"
        raise ValueError(msg)

    z, edge_index, edge_weight = encode_rollout_origin(
        x_or_data,
        edge_index=edge_index,
        edge_weight=edge_weight,
        history=history,
    )

    control_at = None if controls is None else (lambda step: controls[step])
    delta_t_at = None if step_deltas is None else (lambda step: step_deltas[step])
    return autoregressive_latent_rollout(
        koopman,
        decoder,
        z,
        steps=steps,
        topology_at=hold_last_topology_at(
            edge_index,
            edge_weight,
            future_topologies,
        ),
        control_at=control_at,
        delta_t_at=delta_t_at,
        default_delta_t=default_delta_t,
    )


def predict_snapshots(
    model: nn.Module,
    rollout_fn: Callable[..., list[tuple[Tensor, Tensor, Tensor | None]]],
    initial_graph: Tensor | Data,
    steps: int,
    *,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    controls: Sequence[Tensor] | None = None,
    future_topologies: Sequence[Data] | None = None,
    history: Sequence[Data] | None = None,
) -> list[Data]:
    """Run an eval-mode discrete-step rollout and pack ``Data`` snapshots.

    Parameters
    ----------
    model
        Module whose ``training`` flag is toggled around the rollout.
    rollout_fn
        Callable matching :meth:`GraphKoopmanModel._rollout`.
    initial_graph, steps, edge_index, edge_weight, controls, future_topologies,
    history
        Forwarded to ``rollout_fn``.

    Returns
    -------
    list of Data
        Packed forecast snapshots.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            rollout = rollout_fn(
                initial_graph,
                steps,
                edge_index,
                edge_weight,
                controls=controls,
                future_topologies=future_topologies,
                history=history,
            )
    finally:
        model.train(was_training)
    return pack_rollout_snapshots(rollout)


def predict_at_snapshots(
    model: nn.Module,
    rollout_fn: Callable[..., list[tuple[Tensor, Tensor, Tensor | None]]],
    initial_graph: Tensor | Data,
    *,
    is_continuous: bool,
    time_step: float,
    query_times: Sequence[float] | Sequence[Tensor] | None = None,
    step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    controls: Sequence[Tensor] | None = None,
    future_topologies: Sequence[Data] | None = None,
) -> list[Data]:
    """Forecast snapshots at arbitrary query times / step deltas.

    Parameters
    ----------
    model
        Module whose ``training`` flag is toggled around the rollout.
    rollout_fn
        Callable matching :meth:`GraphKoopmanModel._rollout`.
    initial_graph
        Initial graph snapshot at ``t = 0``.
    is_continuous : bool
        Whether continuous irregular increments are allowed.
    time_step : float
        Discrete model time step for uniformity checks.
    query_times, step_deltas, edge_index, edge_weight, controls,
    future_topologies
        Same semantics as :meth:`GraphKoopmanModel.predict_at`.

    Returns
    -------
    list of Data
        Predicted snapshots, one per query interval.
    """
    increments = resolve_time_increments(
        query_times=query_times,
        step_deltas=step_deltas,
    )
    if not is_continuous:
        validate_uniform_discrete_increments(
            time_step=time_step,
            increments=increments,
        )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            rollout = rollout_fn(
                initial_graph,
                len(increments),
                edge_index,
                edge_weight,
                controls=controls,
                future_topologies=future_topologies,
                step_deltas=increments,
            )
    finally:
        model.train(was_training)
    return pack_rollout_snapshots(rollout)


def evaluate_sequence(
    model: object,
    sequence: GraphSnapshotSequence | Sequence[Data],
    *,
    horizons: Sequence[int] = (3, 6, 12),
    start_indices: Sequence[int] | None = None,
) -> EvaluationResult:
    """Evaluate multi-horizon forecast accuracy on a snapshot sequence.

    Parameters
    ----------
    model
        Forecast model accepted by :func:`~koopman_graph.metrics.evaluate_forecast`.
    sequence : GraphSnapshotSequence or sequence of Data
        Evaluation snapshots with shared topology.
    horizons : sequence of int, optional
        Forecast horizons to report. Default is ``(3, 6, 12)``.
    start_indices : sequence of int or None, optional
        Forecast-origin indices. When ``None``, uses every valid origin.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate MAE, RMSE, and MAPE.
    """
    return evaluate_forecast(
        model,
        resolve_sequence(sequence),
        horizons=horizons,
        start_indices=start_indices,
    )
