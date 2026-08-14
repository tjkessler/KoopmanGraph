"""Inference helpers for GraphKoopmanModel (spectrum / predict / evaluate).

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import SnapshotSequence
from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    hold_last_topology_at,
    pack_rollout_snapshots,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)
from koopman_graph.metrics import EvaluationResult, evaluate_forecast
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.operators.contract import KoopmanOperatorContract
from koopman_graph.spectrum_types import (
    KoopmanSpectrum,
    compute_generator_spectrum,
    compute_spectrum,
    discrete_spectrum_at_delta_t,
)

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
    uses_hypergraph_koopman: bool = False,
    uses_continuous_graph_koopman: bool = False,
    uses_hetero_koopman: bool = False,
    uses_continuous_hetero_koopman: bool = False,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
    edge_indices: Sequence[Tensor] | None = None,
    edge_weights: Sequence[Tensor | None] | None = None,
    num_nodes_dict: Mapping[str, int] | None = None,
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
    uses_hypergraph_koopman : bool, optional
        Whether ``koopman`` is a :class:`HypergraphKoopmanOperator`.
    uses_continuous_graph_koopman : bool, optional
        Whether ``koopman`` is a :class:`ContinuousGraphKoopmanOperator`.
    uses_hetero_koopman : bool, optional
        Whether ``koopman`` is a :class:`HeteroGraphKoopmanOperator` or a
        :class:`ContinuousHeteroGraphKoopmanOperator`.
    uses_continuous_hetero_koopman : bool, optional
        Whether ``koopman`` is specifically a
        :class:`ContinuousHeteroGraphKoopmanOperator` (checked before the
        generic ``uses_hetero_koopman`` branch).
    hyperedge_index, hyperedge_weight
        Topology arguments required for hypergraph operators.
    edge_indices, edge_weights, num_nodes_dict
        Topology arguments required for hetero / multiplex operators.

    Returns
    -------
    KoopmanSpectrum
        Magnitude-sorted eigenvalues, eigenvectors, and time scales.

    Raises
    ------
    ValueError
        If networked-operator topology is missing or continuous
        ``auxiliary_spectral`` spectrum is requested.
    """
    if uses_continuous_hetero_koopman:
        if edge_indices is None or num_nodes is None:
            msg = (
                "edge_indices and num_nodes are required for "
                "GraphKoopmanModel.spectrum when koopman='hetero_graph' with "
                "dynamics_mode='continuous' (topology-coupled effective "
                "generator); the per-node contract matrix L_self is not a "
                "substitute"
            )
            raise ValueError(msg)
        assert isinstance(koopman, ContinuousHeteroGraphKoopmanOperator)
        return koopman.spectrum(
            edge_indices,
            num_nodes,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        )
    if uses_hetero_koopman:
        if edge_indices is None or num_nodes is None:
            msg = (
                "edge_indices and num_nodes are required for "
                "GraphKoopmanModel.spectrum when koopman='hetero_graph' "
                "(topology-coupled effective operator); the per-node "
                "contract matrix K_self is not a substitute"
            )
            raise ValueError(msg)
        assert isinstance(koopman, HeteroGraphKoopmanOperator)
        return koopman.spectrum(
            edge_indices,
            num_nodes,
            edge_weights=edge_weights,
            time_step=time_step,
            num_nodes_dict=num_nodes_dict,
        )
    if uses_continuous_graph_koopman:
        if edge_index is None or num_nodes is None:
            msg = (
                "edge_index and num_nodes are required for "
                "GraphKoopmanModel.spectrum when koopman='continuous_graph' "
                "(topology-coupled effective generator); the per-node "
                "contract matrix L_self is not a substitute"
            )
            raise ValueError(msg)
        assert isinstance(koopman, ContinuousGraphKoopmanOperator)
        return koopman.spectrum(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
    if uses_hypergraph_koopman:
        if hyperedge_index is None or num_nodes is None:
            msg = (
                "hyperedge_index and num_nodes are required for "
                "GraphKoopmanModel.spectrum when koopman='hypergraph' "
                "(topology-coupled effective operator); the per-node "
                "contract matrix K_self is not a substitute"
            )
            raise ValueError(msg)
        assert isinstance(koopman, HypergraphKoopmanOperator)
        return koopman.spectrum(
            hyperedge_index,
            num_nodes,
            hyperedge_weight=hyperedge_weight,
            time_step=time_step,
        )
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


def resolve_future_presence_at(
    future_presence: Tensor | Sequence[Tensor] | None,
    *,
    steps: int,
    num_nodes: int,
) -> Callable[[int], Tensor | None] | None:
    """Build a ``presence_at(step)`` callable from a forecast presence schedule.

    Parameters
    ----------
    future_presence : Tensor, sequence of Tensor, or None
        Per-step presence for rollout decode steps ``0 .. steps - 1``. A tensor
        must have shape ``(steps, num_nodes)``; a sequence must have length
        ``steps`` with each entry shape ``(num_nodes,)``. ``None`` disables the
        inactive-entity hold policy.
    steps : int
        Expected rollout length.
    num_nodes : int
        Fixed union size ``N_max``.

    Returns
    -------
    callable or None
        ``presence_at`` schedule, or ``None`` when no schedule is provided.

    Raises
    ------
    ValueError
        If shapes or lengths are inconsistent.
    """
    if future_presence is None:
        return None
    if isinstance(future_presence, Tensor):
        if future_presence.ndim != 2 or future_presence.shape != (steps, num_nodes):
            msg = (
                "future_presence tensor must have shape "
                f"(steps={steps}, num_nodes={num_nodes}), "
                f"got {tuple(future_presence.shape)}"
            )
            raise ValueError(msg)
        schedule = future_presence.bool()

        def _presence_at(step: int) -> Tensor:
            return schedule[step]

        return _presence_at

    if len(future_presence) != steps:
        msg = (
            f"future_presence length {len(future_presence)} does not match "
            f"steps={steps}"
        )
        raise ValueError(msg)
    masks = []
    for index, mask in enumerate(future_presence):
        if mask.ndim != 1 or int(mask.shape[0]) != num_nodes:
            msg = (
                f"future_presence[{index}] must have shape ({num_nodes},), "
                f"got {tuple(mask.shape)}"
            )
            raise ValueError(msg)
        masks.append(mask.bool())

    def _presence_at_seq(step: int) -> Tensor:
        """Return the presence mask for rollout step ``step``.

        Parameters
        ----------
        step : int
            Zero-based rollout step index.

        Returns
        -------
        Tensor
            Boolean mask of shape ``(num_nodes,)``.
        """
        return masks[step]

    return _presence_at_seq


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
    future_presence: Tensor | Sequence[Tensor] | None = None,
    step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
    history: Sequence[Data] | None = None,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
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
    future_presence : Tensor, sequence of Tensor, or None, optional
        Per-step entity presence for the inactive-node **hold** policy. See
        :func:`resolve_future_presence_at`. When omitted, all entities are
        treated as present (0.10 behavior). Operator matvecs still use
        ``N_max`` capacity.
    hyperedge_index, hyperedge_weight
        Static hyperedge incidence for hypergraph operators. When omitted and
        ``x_or_data`` is a ``Data`` snapshot, incidence is read from the
        snapshot fields.

    Returns
    -------
    list of tuple[Tensor, Tensor, Tensor | None]
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
    tail_index = None
    head_index = None
    if isinstance(x_or_data, Data):
        if hyperedge_index is None:
            hyperedge_index = snapshot_hyperedge_index(x_or_data)
            if hyperedge_weight is None:
                hyperedge_weight = snapshot_hyperedge_weight(x_or_data)
        from koopman_graph.graph_utils import (
            snapshot_head_index,
            snapshot_tail_index,
        )

        tail_index = snapshot_tail_index(x_or_data)
        head_index = snapshot_head_index(x_or_data)

    initial_features = None
    if isinstance(x_or_data, Data):
        initial_features = x_or_data.x
    elif isinstance(x_or_data, Tensor) and x_or_data.ndim == 2:
        initial_features = x_or_data
    presence_at = resolve_future_presence_at(
        future_presence,
        steps=steps,
        num_nodes=int(z.shape[0]),
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
        presence_at=presence_at,
        initial_features=initial_features,
        default_delta_t=default_delta_t,
        hyperedge_index=hyperedge_index,
        hyperedge_weight=hyperedge_weight,
        tail_index=tail_index,
        head_index=head_index,
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
    future_presence: Tensor | Sequence[Tensor] | None = None,
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
    future_presence, history
        Forwarded to ``rollout_fn``.
    initial_graph : Tensor | Data
        See the function signature / summary for ``initial_graph``.
    steps : int
        See the function signature / summary for ``steps``.
    edge_index : Tensor | None
        See the function signature / summary for ``edge_index``.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.
    controls : Sequence[Tensor] | None
        See the function signature / summary for ``controls``.
    future_topologies : Sequence[Data] | None
        See the function signature / summary for ``future_topologies``.
    future_presence : Tensor | Sequence[Tensor] | None
        Optional per-step presence schedule for the inactive-node hold policy.

    Returns
    -------

    list of Data
        Packed forecast snapshots."""
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
                future_presence=future_presence,
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
    future_presence: Tensor | Sequence[Tensor] | None = None,
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
    future_topologies, future_presence
        Same semantics as :meth:`GraphKoopmanModel.predict_at`.
    query_times : Sequence[float] | Sequence[Tensor] | None
        See the function signature / summary for ``query_times``.
    step_deltas : Sequence[float] | Sequence[Tensor] | None
        See the function signature / summary for ``step_deltas``.
    edge_index : Tensor | None
        See the function signature / summary for ``edge_index``.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.
    controls : Sequence[Tensor] | None
        See the function signature / summary for ``controls``.
    future_presence : Tensor | Sequence[Tensor] | None
        Optional per-step presence schedule for the inactive-node hold policy.

    Returns
    -------

    list of Data
        Predicted snapshots, one per query interval."""
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
                future_presence=future_presence,
                step_deltas=increments,
            )
    finally:
        model.train(was_training)
    return pack_rollout_snapshots(rollout)


def evaluate_sequence(
    model: object,
    sequence: SnapshotSequence | Sequence[Data] | Sequence[HeteroData],
    *,
    horizons: Sequence[int] = (3, 6, 12),
    start_indices: Sequence[int] | None = None,
    include_resdmd: bool = False,
) -> EvaluationResult:
    """Evaluate multi-horizon forecast accuracy on a snapshot sequence.

    Parameters
    ----------
    model
        Forecast model accepted by :func:`~koopman_graph.metrics.evaluate_forecast`.
    sequence : GraphSnapshotSequence, HeteroGraphSnapshotSequence, or sequence
        Homogeneous or hetero evaluation snapshots. Container normalization is
        handled inside :func:`~koopman_graph.metrics.evaluate_forecast`.
    horizons : sequence of int, optional
        Forecast horizons to report. Default is ``(3, 6, 12)``.
    start_indices : sequence of int or None, optional
        Forecast-origin indices. When ``None``, uses every valid origin.
    include_resdmd : bool, optional
        Forwarded to :func:`~koopman_graph.metrics.evaluate_forecast`.

    Returns
    -------
    EvaluationResult
        Per-horizon and aggregate MAE, RMSE, and MAPE.
    """
    return evaluate_forecast(
        model,  # type: ignore[arg-type]
        sequence,
        horizons=horizons,
        start_indices=start_indices,
        include_resdmd=include_resdmd,
    )
