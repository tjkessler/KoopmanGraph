"""Latent propagation and autoregressive rollout helpers (power-user).

Discrete/continuous latent advance, decode, shared rollout loops, and
decoded-rollout → :class:`~torch_geometric.data.Data` snapshot packing.
Import from :mod:`koopman_graph.graph_utils` (package re-exports) unless you
need the peer module explicitly.

``delta_t`` default policy
--------------------------
Continuous operators require an integration interval. Use
:func:`resolve_delta_t` (also used inside :func:`propagate_latent` /
:func:`inverse_propagate_latent`) so every call site agrees:

* **Explicit wins:** when ``delta_t`` is not ``None``, that value is used.
* **Model-backed paths:** :class:`~koopman_graph.model.GraphKoopmanModel`
  resolves missing intervals to :attr:`~koopman_graph.model.GraphKoopmanModel.time_step`
  via :meth:`~koopman_graph.model.GraphKoopmanModel.resolve_delta_t`. Training,
  losses, eval, and :class:`~koopman_graph.env.GraphKoopmanEnv` follow the same
  policy (pair intervals still come from
  :func:`~koopman_graph.data.resolve_pair_delta_t` when timestamps exist).
* **Standalone operators / bare helpers:** when no model ``time_step`` is
  available, ``default_delta_t`` defaults to ``1.0`` (same soft default as
  :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.forward`). Prefer
  passing an explicit interval for continuous operators used outside a model.

Autoregressive rollout topology policy
--------------------------------------
:func:`autoregressive_latent_rollout` is the shared encode-once → advance →
decode loop used by inference (``GraphKoopmanModel._rollout`` / ``predict``)
and training (``rollout_sequence_loss``). Call sites supply a ``topology_at``
callable so intentional decode-topology differences stay explicit:

* **Hold-last (inference):** :func:`hold_last_topology_at` — start from the
  initial graph topology; when ``future_topologies[step]`` is provided, update
  and retain that topology for later steps.
* **Per-step / teacher targets (training):** :func:`snapshot_topology_at` —
  decode step ``i`` uses the topology of snapshot ``i`` (typically the
  observed target at ``start + i + 1``).

Controls and ``delta_t`` are likewise supplied per step via optional
callables; missing controls mean uncontrolled advance, and missing
``delta_t`` falls through to ``default_delta_t`` via :func:`propagate_latent`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence

from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.graph_utils.topology import snapshot_edge_weight
from koopman_graph.operators.contract import KoopmanOperatorContract

# Single definition reused by losses, adaptation, and propagation helpers.
KoopmanPropagator = KoopmanOperatorContract

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]
RelationDecoderFn = Callable[
    [Tensor, Sequence[Tensor], Sequence[Tensor | None] | None],
    Tensor | dict[str, Tensor],
]
TopologyAtFn = Callable[[int], tuple[Tensor, Tensor | None]]
RelationTopologyAtFn = Callable[[int], tuple[list[Tensor], list[Tensor | None]]]
ControlAtFn = Callable[[int], Tensor | None]
DeltaTAtFn = Callable[[int], float | Tensor | None]


def resolve_delta_t(
    delta_t: float | Tensor | None,
    *,
    default_delta_t: float | Tensor = 1.0,
) -> float | Tensor:
    """Resolve a continuous integration interval.

    Parameters
    ----------
    delta_t : float, Tensor, or None
        Explicit interval. When ``None``, ``default_delta_t`` is returned.
    default_delta_t : float or Tensor, optional
        Fallback interval. Model-backed callers should pass ``time_step``;
        bare helpers default to ``1.0``.

    Returns
    -------
    float or Tensor
        Resolved positive integration interval (caller validates positivity
        where required).
    """
    if delta_t is None:
        return default_delta_t
    return delta_t


def _topology_kwargs_for(
    method: Callable[..., Tensor],
    *,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
    latent_window: Tensor | None = None,
    edge_indices: Sequence[Tensor] | None = None,
    edge_weights: Sequence[Tensor | None] | None = None,
    num_nodes_dict: Mapping[str, int] | None = None,
) -> dict[
    str,
    Tensor | None | Sequence[Tensor] | Sequence[Tensor | None] | Mapping[str, int],
]:
    """Return topology / window kwargs accepted by ``method``.

    Built-in operators accept optional ``edge_index`` / ``edge_weight`` and/or
    ``hyperedge_index`` / ``hyperedge_weight``, relational ``edge_indices`` /
    ``edge_weights``, and global/local operators accept ``latent_window``.
    Older custom injected operators may omit those parameters; skip them so
    Protocol injection keeps working without forcing every stub to update.

    Parameters
    ----------
    method : callable
        ``advance`` or ``inverse_advance`` bound method.
    edge_index : Tensor or None
        Pairwise topology to forward when supported.
    edge_weight : Tensor or None
        Optional edge weights to forward when supported.
    hyperedge_index : Tensor or None, optional
        Hyperedge incidence to forward when supported.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights to forward when supported.
    latent_window : Tensor or None, optional
        Latent history window for global/local operators.
    edge_indices : sequence of Tensor or None, optional
        Per-relation edge banks for hetero / multiplex operators.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation edge weights.
    num_nodes_dict : mapping of str to int or None, optional
        Per-type node counts for typed hetero operators.

    Returns
    -------
    dict
        Keyword arguments safe to pass to ``method``.
    """
    params = inspect.signature(method).parameters
    accepts_var_kw = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
    )
    kwargs: dict[
        str,
        Tensor | None | Sequence[Tensor] | Sequence[Tensor | None] | Mapping[str, int],
    ] = {}
    if accepts_var_kw or "edge_index" in params:
        kwargs["edge_index"] = edge_index
        if accepts_var_kw or "edge_weight" in params:
            kwargs["edge_weight"] = edge_weight
    if accepts_var_kw or "hyperedge_index" in params:
        kwargs["hyperedge_index"] = hyperedge_index
        if accepts_var_kw or "hyperedge_weight" in params:
            kwargs["hyperedge_weight"] = hyperedge_weight
    if accepts_var_kw or "edge_indices" in params:
        kwargs["edge_indices"] = edge_indices
        if accepts_var_kw or "edge_weights" in params:
            kwargs["edge_weights"] = edge_weights
    if accepts_var_kw or "latent_window" in params:
        kwargs["latent_window"] = latent_window
    if "num_nodes_dict" in params:
        kwargs["num_nodes_dict"] = num_nodes_dict
    return kwargs


def propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    default_delta_t: float | Tensor = 1.0,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
    latent_window: Tensor | None = None,
    edge_indices: Sequence[Tensor] | None = None,
    edge_weights: Sequence[Tensor | None] | None = None,
    num_nodes_dict: Mapping[str, int] | None = None,
) -> Tensor:
    """Advance latent states via the unified Koopman contract.

    Always resolves ``delta_t`` and calls
    :meth:`~koopman_graph.operators.KoopmanOperatorContract.advance`. Discrete
    operators ignore the interval; continuous operators (built-in or custom
    injected) require it. Dispatch does **not** use concrete ``isinstance``
    checks. Topology kwargs are forwarded for networked operators and ignored
    by per-node operators.

    Parameters
    ----------
    koopman : KoopmanOperatorContract
        Operator used for one forward step.
    z : Tensor
        Latent states to advance.
    control : Tensor or None, optional
        Optional control input for controlled operators.
    delta_t : float, Tensor, or None, optional
        Explicit continuous-time integration interval. Ignored for discrete
        operators.
    default_delta_t : float or Tensor, optional
        Fallback interval when ``delta_t is None``. Callers with a model
        ``time_step`` should pass it here (see module ``delta_t`` default
        policy).
    edge_index : Tensor or None, optional
        Graph topology for networked graph operators.
    edge_weight : Tensor or None, optional
        Optional edge weights for networked graph operators.
    hyperedge_index : Tensor or None, optional
        Hyperedge incidence for hypergraph operators.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights for hypergraph operators.
    latent_window : Tensor or None, optional
        Latent history for global/local operators (cold-start when omitted).
    edge_indices : sequence of Tensor or None, optional
        Per-relation edge banks for multiplex / hetero operators.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation edge weights.
    num_nodes_dict : mapping of str to int or None, optional
        Per-type node counts for typed hetero operators (forwarded only when
        the operator's ``advance`` declares the parameter).

    Returns
    -------
    Tensor
        Advanced latent states.
    """
    interval = resolve_delta_t(delta_t, default_delta_t=default_delta_t)
    return koopman.advance(
        z,
        interval,
        control=control,
        **_topology_kwargs_for(
            koopman.advance,
            edge_index=edge_index,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            latent_window=latent_window,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        ),
    )


def inverse_propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    default_delta_t: float | Tensor = 1.0,
    inverse_matrix: Tensor | None = None,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
    latent_window: Tensor | None = None,
    edge_indices: Sequence[Tensor] | None = None,
    edge_weights: Sequence[Tensor | None] | None = None,
    num_nodes_dict: Mapping[str, int] | None = None,
) -> Tensor:
    """Apply one inverse Koopman step via the unified contract.

    Always resolves ``delta_t`` and calls
    :meth:`~koopman_graph.operators.KoopmanOperatorContract.inverse_advance`.
    Discrete operators ignore the interval; continuous operators use it.
    Dispatch does **not** use concrete ``isinstance`` checks.

    Parameters
    ----------

    koopman : KoopmanOperatorContract
        Operator used for one inverse step.
    z : Tensor
        Latent states to recover.
    control : Tensor or None, optional
        Optional control input for controlled operators.
    delta_t : float, Tensor, or None, optional
        Explicit continuous-time integration interval. Ignored for discrete
        operators.
    default_delta_t : float or Tensor, optional
        Fallback interval when ``delta_t is None`` in continuous mode.
    inverse_matrix : Tensor or None, optional
        Optional precomputed discrete inverse matrix.
    edge_index : Tensor or None, optional
        Graph topology for networked operators.
    edge_weight : Tensor or None, optional
        Optional edge weights for networked operators.
    hyperedge_index : Tensor | None
        See the function signature / summary for ``hyperedge_index``.
    hyperedge_weight : Tensor | None
        See the function signature / summary for ``hyperedge_weight``.
    latent_window : Tensor | None
        See the function signature / summary for ``latent_window``.
    edge_indices : sequence of Tensor or None, optional
        Per-relation edge banks for multiplex / hetero operators.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation edge weights.
    num_nodes_dict : mapping of str to int or None, optional
        Per-type node counts for typed hetero operators.

    Returns
    -------

    Tensor
        Recovered latent states."""
    interval = resolve_delta_t(delta_t, default_delta_t=default_delta_t)
    return koopman.inverse_advance(
        z,
        interval,
        control=control,
        inverse_matrix=inverse_matrix,
        **_topology_kwargs_for(
            koopman.inverse_advance,
            edge_index=edge_index,
            edge_weight=edge_weight,
            latent_window=latent_window,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            num_nodes_dict=num_nodes_dict,
        ),
    )


def advance_and_decode(
    koopman: KoopmanPropagator,
    decoder: DecoderFn,
    z: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    default_delta_t: float | Tensor = 1.0,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
    latent_window: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Advance latent state once and decode to physical node features.

    Pairwise topology is passed to both the linear advance (networked graph
    operators) and the decoder. Static hyperedge incidence is forwarded to
    hypergraph operators; decoders that need incidence should be bound by the
    caller (see :func:`~koopman_graph.nn.hypergraph.bind_hypergraph_decoder`).

    Parameters
    ----------
    koopman : KoopmanOperator or ContinuousKoopmanOperator
        Operator used for one forward step.
    decoder : callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` mapping.
    z : Tensor
        Current latent states.
    edge_index : Tensor
        Topology with shape ``(2, num_edges)`` for advance and decode.
    edge_weight : Tensor or None, optional
        Optional scalar edge weights with shape ``(num_edges,)``.
    control : Tensor or None, optional
        Optional control input for controlled operators.
    delta_t : float, Tensor, or None, optional
        Explicit continuous-time integration interval.
    default_delta_t : float or Tensor, optional
        Fallback interval when ``delta_t is None`` in continuous mode.
    hyperedge_index : Tensor or None, optional
        Static bipartite incidence for hypergraph operators.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights for hypergraph operators.
    latent_window : Tensor or None, optional
        Latent history for global/local operators.

    Returns
    -------
    tuple[Tensor, Tensor]
        Advanced latent states and decoded node features.
    """
    z_next = propagate_latent(
        koopman,
        z,
        control=control,
        delta_t=delta_t,
        default_delta_t=default_delta_t,
        edge_index=edge_index,
        edge_weight=edge_weight,
        hyperedge_index=hyperedge_index,
        hyperedge_weight=hyperedge_weight,
        latent_window=latent_window,
    )
    prediction = decoder(z_next, edge_index, edge_weight)
    return z_next, prediction


def hold_last_topology_at(
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
    future_topologies: Sequence[Data] | None = None,
) -> TopologyAtFn:
    """Build a hold-last topology schedule for autoregressive rollouts.

    Step ``i`` uses ``future_topologies[i]`` when present, otherwise the most
    recently observed topology (starting from ``edge_index`` /
    ``edge_weight``). This is the inference policy used by
    :meth:`~koopman_graph.model.GraphKoopmanModel.predict`.

    Parameters
    ----------
    edge_index : Tensor
        Initial graph topology.
    edge_weight : Tensor or None, optional
        Initial optional edge weights.
    future_topologies : sequence of Data or None, optional
        Optional known future topologies (node features ignored).

    Returns
    -------
    callable
        ``topology_at(step) -> (edge_index, edge_weight)``.
    """
    current_edge_index = edge_index
    current_edge_weight = edge_weight

    def topology_at(step: int) -> tuple[Tensor, Tensor | None]:
        """Return the hold-last topology for one rollout step.

        Parameters
        ----------
        step : int
            Zero-based rollout step index.

        Returns
        -------
        tuple[Tensor, Tensor or None]
            Decode ``edge_index`` and optional ``edge_weight``.
        """
        nonlocal current_edge_index, current_edge_weight
        if future_topologies is not None and step < len(future_topologies):
            snapshot = future_topologies[step]
            current_edge_index = snapshot.edge_index
            current_edge_weight = snapshot_edge_weight(snapshot)
        return current_edge_index, current_edge_weight

    return topology_at


def snapshot_topology_at(snapshots: Sequence[Data]) -> TopologyAtFn:
    """Build a per-step topology schedule from a fixed snapshot list.

    Step ``i`` decodes with ``snapshots[i]`` topology. Training rollouts pass
    the observed target snapshots so decode edges match teacher targets.

    Parameters
    ----------
    snapshots : sequence of Data
        One topology-bearing snapshot per rollout step.

    Returns
    -------
    callable
        ``topology_at(step) -> (edge_index, edge_weight)``.
    """

    def topology_at(step: int) -> tuple[Tensor, Tensor | None]:
        """Return the topology of the snapshot at ``step``.

        Parameters
        ----------
        step : int
            Zero-based index into ``snapshots``.

        Returns
        -------
        tuple[Tensor, Tensor or None]
            Decode ``edge_index`` and optional ``edge_weight``.
        """
        snapshot = snapshots[step]
        return snapshot.edge_index, snapshot_edge_weight(snapshot)

    return topology_at


def autoregressive_latent_rollout(
    koopman: KoopmanPropagator,
    decoder: DecoderFn,
    z: Tensor,
    *,
    steps: int,
    topology_at: TopologyAtFn,
    control_at: ControlAtFn | None = None,
    delta_t_at: DeltaTAtFn | None = None,
    default_delta_t: float | Tensor = 1.0,
    hyperedge_index: Tensor | None = None,
    hyperedge_weight: Tensor | None = None,
) -> list[tuple[Tensor, Tensor, Tensor | None]]:
    """Run an autoregressive latent advance/decode loop.

    Shared primitive for inference and training rollouts. Callers encode the
    initial graph once, then supply per-step topology / control / ``delta_t``
    policies. See the module docstring for topology policy guidance.

    Hyperedge incidence is static (not scheduled by ``topology_at``) and is
    forwarded unchanged to each advance for hypergraph operators.

    Parameters
    ----------
    koopman : KoopmanOperator or ContinuousKoopmanOperator
        Operator used for each forward step.
    decoder : callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` mapping.
    z : Tensor
        Encoded latent state at the rollout origin.
    steps : int
        Number of advance/decode steps (must be >= 1).
    topology_at : callable
        ``topology_at(step) -> (edge_index, edge_weight)`` for decode step
        ``step`` in ``0 .. steps - 1``.
    control_at : callable or None, optional
        ``control_at(step) -> Tensor | None``. When omitted, every step is
        uncontrolled.
    delta_t_at : callable or None, optional
        ``delta_t_at(step) -> float | Tensor | None``. When omitted (or when a
        step returns ``None``), ``default_delta_t`` is used.
    default_delta_t : float or Tensor, optional
        Fallback continuous integration interval.
    hyperedge_index : Tensor or None, optional
        Static bipartite incidence for hypergraph operators.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights for hypergraph operators.

    Returns
    -------
    list of tuple[Tensor, Tensor, Tensor or None]
        For each step: decoded prediction, decode ``edge_index``, and optional
        decode ``edge_weight``.

    Raises
    ------
    ValueError
        If ``steps < 1``.
    """
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)

    outputs: list[tuple[Tensor, Tensor, Tensor | None]] = []
    latent = z
    history: list[Tensor] = []
    accepts_window = "latent_window" in inspect.signature(koopman.advance).parameters
    local_window = int(getattr(koopman, "local_window", 0) or 0)
    for step in range(steps):
        edge_index, edge_weight = topology_at(step)
        control = None if control_at is None else control_at(step)
        delta_t = None if delta_t_at is None else delta_t_at(step)
        latent_window = None
        if accepts_window and local_window > 0:
            from koopman_graph.operators.global_local import stack_latent_window

            latent_window = stack_latent_window(
                history,
                window=local_window,
                current=latent,
            )
        next_latent, prediction = advance_and_decode(
            koopman,
            decoder,
            latent,
            edge_index,
            edge_weight,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            latent_window=latent_window,
        )
        if accepts_window and local_window > 1:
            history.append(latent)
            history = history[-(local_window - 1) :]
        latent = next_latent
        outputs.append((prediction, edge_index, edge_weight))
    return outputs


def pack_rollout_snapshots(
    rollout: Sequence[tuple[Tensor, Tensor, Tensor | None]],
) -> list[Data]:
    """Convert decoded rollout tuples into PyG ``Data`` snapshots.

    Shared by model ``predict`` / ``evaluate`` packing and UQ ensemble member
    forecasts so peers outside :mod:`koopman_graph.model` need not import
    ``model.inference``.

    Parameters
    ----------
    rollout : sequence of tuple
        Each entry is ``(prediction, edge_index, edge_weight_or_None)``.

    Returns
    -------
    list of Data
        Snapshot list with ``x`` / ``edge_index`` / optional ``edge_weight``.
    """
    output_snapshots: list[Data] = []
    for prediction, step_edge_index, step_edge_weight in rollout:
        fields: dict[str, Tensor] = {
            "x": prediction,
            "edge_index": step_edge_index,
        }
        if step_edge_weight is not None:
            fields["edge_weight"] = step_edge_weight
        output_snapshots.append(Data(**fields))
    return output_snapshots


def hold_last_relation_topology_at(
    edge_indices: Sequence[Tensor],
    edge_weights: Sequence[Tensor | None] | None = None,
    future_topologies: Sequence[HeteroData] | None = None,
    *,
    num_relations: int,
    node_types: Sequence[str] | None = None,
    edge_types: Sequence[tuple[str, str, str]] | None = None,
) -> RelationTopologyAtFn:
    """Build a hold-last relation-bank schedule for hetero rollouts.

    Parameters
    ----------
    edge_indices : sequence of Tensor
        Initial ordered relation edge banks.
    edge_weights : sequence of Tensor or None, optional
        Initial optional per-relation weights.
    future_topologies : sequence of HeteroData or None, optional
        Optional known future topologies (node features ignored).
    num_relations : int
        Expected ``|R|`` (must match encoder / operator).
    node_types : sequence of str or None, optional
        Stack order for typed graphs. When given with more than one entry,
        future topologies are resolved with global (offset) edge numbering.
    edge_types : sequence of tuple of str or None, optional
        Explicit relation order for typed graphs.

    Returns
    -------
    callable
        ``topology_at(step) -> (edge_indices, edge_weights)``.
    """
    from koopman_graph.nn.heterogeneous import (
        resolve_multiplex_relation_inputs,
        resolve_typed_relation_inputs,
    )

    typed = node_types is not None and len(tuple(node_types)) > 1
    current_indices = list(edge_indices)
    current_weights = (
        [None] * num_relations if edge_weights is None else list(edge_weights)
    )

    def topology_at(step: int) -> tuple[list[Tensor], list[Tensor | None]]:
        """Return the hold-last relation banks for one rollout step.

        Parameters
        ----------
        step : int
            Zero-based rollout step index.

        Returns
        -------
        tuple[list[Tensor], list[Tensor | None]]
            Ordered relation edge banks and optional weights.
        """
        nonlocal current_indices, current_weights
        if future_topologies is not None and step < len(future_topologies):
            if typed:
                assert node_types is not None
                _, current_indices, current_weights, _ = resolve_typed_relation_inputs(
                    future_topologies[step],
                    node_types=node_types,
                    edge_types=edge_types,
                    num_relations=num_relations,
                )
            else:
                _, current_indices, current_weights = resolve_multiplex_relation_inputs(
                    future_topologies[step],
                    num_relations=num_relations,
                )
        return current_indices, current_weights

    return topology_at


def autoregressive_hetero_latent_rollout(
    koopman: KoopmanPropagator,
    decoder: RelationDecoderFn,
    z: Tensor,
    *,
    steps: int,
    topology_at: RelationTopologyAtFn,
    control_at: ControlAtFn | None = None,
    delta_t_at: DeltaTAtFn | None = None,
    default_delta_t: float | Tensor = 1.0,
    num_nodes_dict: Mapping[str, int] | None = None,
) -> list[tuple[Tensor | dict[str, Tensor], list[Tensor], list[Tensor | None]]]:
    """Run an autoregressive hetero advance/decode loop.

    Parameters
    ----------
    koopman : KoopmanOperatorContract
        Relational / multiplex operator.
    decoder : callable
        ``decoder(z, edge_indices, edge_weights) -> Tensor``.
    z : Tensor
        Encoded latent state at the rollout origin.
    steps : int
        Number of advance/decode steps (must be >= 1).
    topology_at : callable
        ``topology_at(step) -> (edge_indices, edge_weights)``.
    control_at : callable or None, optional
        Optional per-step control schedule.
    delta_t_at : callable or None, optional
        Optional per-step integration intervals.
    default_delta_t : float or Tensor, optional
        Fallback continuous integration interval.
    num_nodes_dict : mapping of str to int or None, optional
        Per-type node counts for typed operators.

    Returns
    -------
    list of tuple
        For each step: decoded prediction (a tensor for multiplex, a per-type
        mapping for typed decoders), relation edge banks, and optional
        relation weights.

    Raises
    ------
    ValueError
        If ``steps < 1``.
    """
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)

    outputs: list[tuple[Tensor | dict[str, Tensor], list[Tensor], list[Tensor | None]]]
    outputs = []
    latent = z
    for step in range(steps):
        edge_indices, edge_weights = topology_at(step)
        control = None if control_at is None else control_at(step)
        delta_t = None if delta_t_at is None else delta_t_at(step)
        next_latent = propagate_latent(
            koopman,
            latent,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        )
        prediction = decoder(next_latent, edge_indices, edge_weights)
        latent = next_latent
        outputs.append((prediction, edge_indices, edge_weights))
    return outputs


def pack_hetero_rollout_snapshots(
    rollout: Sequence[
        tuple[Tensor | Mapping[str, Tensor], list[Tensor], list[Tensor | None]]
    ],
    *,
    template: HeteroData,
    node_types: Sequence[str] | None = None,
    edge_types: Sequence[tuple[str, str, str]] | None = None,
) -> list[HeteroData]:
    """Pack hetero rollout tuples into ``HeteroData`` snapshots.

    Multiplex templates (exactly one node type) accept tensor predictions.
    Typed templates require per-type mappings keyed by node type; the
    relation banks recorded on each snapshot are written verbatim.

    Parameters
    ----------
    rollout : sequence of tuple
        Each entry is ``(prediction, edge_indices, edge_weights)`` where
        ``prediction`` is a tensor (multiplex) or a per-type mapping (typed).
    template : HeteroData
        Schema template (node types + ordered edge types).
    node_types : sequence of str or None, optional
        Explicit node-type order. Defaults to ``template.node_types``.
    edge_types : sequence of tuple of str or None, optional
        Explicit relation order. Defaults to ``template`` edge types sorted
        by ``repr``.

    Returns
    -------
    list of HeteroData
        Forecast snapshots preserving the template schema.

    Raises
    ------
    ValueError
        If relation counts mismatch, a typed template receives a tensor
        prediction, or a typed prediction omits a node type.
    """
    resolved_node_types = (
        tuple(template.node_types) if node_types is None else tuple(node_types)
    )
    if not resolved_node_types:
        msg = "pack_hetero_rollout_snapshots requires at least one node type"
        raise ValueError(msg)
    resolved_edge_types = (
        tuple(sorted(template.edge_types, key=repr))
        if edge_types is None
        else tuple(tuple(edge_type) for edge_type in edge_types)
    )
    packed: list[HeteroData] = []
    for prediction, step_edge_index, step_edge_weight in rollout:
        if len(step_edge_index) != len(resolved_edge_types):
            msg = (
                f"Expected {len(resolved_edge_types)} relation banks, "
                f"got {len(step_edge_index)}"
            )
            raise ValueError(msg)
        snap = HeteroData()
        if isinstance(prediction, Tensor):
            if len(resolved_node_types) != 1:
                msg = (
                    "Typed templates require per-type predictions keyed by node "
                    f"type; got a tensor for node types {resolved_node_types!r}"
                )
                raise ValueError(msg)
            snap[resolved_node_types[0]].x = prediction
        else:
            missing = [name for name in resolved_node_types if name not in prediction]
            if missing:
                msg = f"Prediction is missing node types {missing!r}"
                raise ValueError(msg)
            for name in resolved_node_types:
                snap[name].x = prediction[name]
        for edge_type, edges, weights in zip(
            resolved_edge_types, step_edge_index, step_edge_weight, strict=True
        ):
            snap[edge_type].edge_index = edges
            if weights is not None:
                snap[edge_type].edge_weight = weights
        packed.append(snap)
    return packed
