"""Shared graph-input and latent-propagation helpers (power-user layer).

This module is the documented internal utility surface for helpers that were
previously duplicated or imported as leading-underscore symbols across modules.
It is importable as ``koopman_graph.graph_utils`` but is **not** part of the
stable public façade (not listed in ``koopman_graph.__all__``).

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

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.operators import KoopmanOperatorContract

# Single definition reused by losses, adaptation, and propagation helpers.
KoopmanPropagator = KoopmanOperatorContract

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]
TopologyAtFn = Callable[[int], tuple[Tensor, Tensor | None]]
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


def snapshot_edge_weight(snapshot: Data) -> Tensor | None:
    """Return optional scalar edge weights attached to a snapshot.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot that may carry ``edge_weight``.

    Returns
    -------
    Tensor or None
        Edge weights with shape ``(num_edges,)``, or ``None`` when absent.
    """
    edge_weight = getattr(snapshot, "edge_weight", None)
    if edge_weight is None:
        return None
    return edge_weight


def symmetric_normalized_adjacency_edge_weights(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Return per-edge weights for ``D^{-1/2} A D^{-1/2}``.

    This is the shared core of the symmetric normalized graph Laplacian
    ``L_sym = I - D^{-1/2} A D^{-1/2}``. Sparse physics lifting
    (:func:`~koopman_graph.observables.graph_laplacian_features`) and dense
    benchmark diffusion
    (:func:`~koopman_graph.datasets.dynamics.normalized_step_operator`) both
    build on these weights; they differ only in sparse matvec vs dense
    assembly (duplicate edges accumulate in both paths).

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned weights.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.

    Returns
    -------
    Tensor
        Normalized edge weights with shape ``(num_edges,)``.
    """
    row, col = edge_index
    resolved_device = device if device is not None else edge_index.device
    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=resolved_device)
    else:
        weights = edge_weight.to(dtype=dtype, device=resolved_device)

    deg = torch.zeros(num_nodes, dtype=dtype, device=resolved_device)
    deg.index_add_(0, row, weights)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(
        torch.isfinite(deg_inv_sqrt),
        deg_inv_sqrt,
        torch.zeros_like(deg_inv_sqrt),
    )
    return deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]


def dense_symmetric_normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
) -> Tensor:
    """Assemble the dense symmetric normalized adjacency ``D^{-1/2} A D^{-1/2}``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the dense matrix.

    Returns
    -------
    Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``. Duplicate edges
        **accumulate** (sum), matching
        :func:`symmetric_normalized_adjacency_matvec` / ``index_add_``.
    """
    row, col = edge_index
    norm = symmetric_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
    )
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=edge_index.device)
    adj.index_put_((row, col), norm, accumulate=True)
    return adj


def symmetric_normalized_adjacency_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply ``D^{-1/2} A D^{-1/2}`` to node features without a dense matrix.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    num_nodes : int or None, optional
        Number of nodes. Inferred from ``x`` when omitted.

    Returns
    -------
    Tensor
        Smoothed node features with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D.
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)

    node_count = num_nodes if num_nodes is not None else x.size(0)
    row, col = edge_index
    norm_weights = symmetric_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=node_count,
        edge_weight=edge_weight,
        dtype=x.dtype,
        device=x.device,
    )
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * norm_weights.unsqueeze(-1))
    return out


def resolve_edge_index(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
) -> Tensor:
    """Extract or validate ``edge_index`` from input arguments.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Graph input; when a ``Data`` object, its ``edge_index`` is returned.
    edge_index : Tensor or None
        Explicit edge index for tensor input.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.

    Raises
    ------
    ValueError
        If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
    """
    if isinstance(x_or_data, Data):
        return x_or_data.edge_index
    if edge_index is None:
        msg = "edge_index is required when x_or_data is a tensor"
        raise ValueError(msg)
    return edge_index


def resolve_edge_weight(
    x_or_data: Tensor | Data,
    edge_weight: Tensor | None,
) -> Tensor | None:
    """Extract or validate optional ``edge_weight`` from input arguments.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Graph input; when a ``Data`` object, its ``edge_weight`` is returned
        when present.
    edge_weight : Tensor or None
        Explicit edge weights for tensor input.

    Returns
    -------
    Tensor or None
        Edge weights with shape ``(num_edges,)``, or ``None`` when unweighted.
    """
    if isinstance(x_or_data, Data):
        return snapshot_edge_weight(x_or_data)
    return edge_weight


def resolve_graph_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Extract node features, edge index, and optional weights from input.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Either a PyG ``Data`` object or node features ``x``.
    edge_index : Tensor or None
        Edge index required when ``x_or_data`` is a tensor.
    edge_weight : Tensor or None, optional
        Edge weights for tensor input. Ignored for ``Data`` input.

    Returns
    -------
    tuple of (Tensor, Tensor, Tensor or None)
        Node features, edge index, and optional edge weights.

    Raises
    ------
    ValueError
        If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
    """
    if isinstance(x_or_data, Data):
        return (
            x_or_data.x,
            x_or_data.edge_index,
            snapshot_edge_weight(x_or_data),
        )
    resolved_edge_index = resolve_edge_index(x_or_data, edge_index)
    return x_or_data, resolved_edge_index, edge_weight


def snapshot_to_device(snapshot: Data, device: torch.device) -> Data:
    """Move a graph snapshot to a target device, preserving edge weights.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot to transfer.
    device : torch.device
        Destination device.

    Returns
    -------
    Data
        Snapshot with tensors moved to ``device``.
    """
    fields: dict[str, Tensor] = {
        "x": snapshot.x.to(device),
        "edge_index": snapshot.edge_index.to(device),
    }
    edge_weight = snapshot_edge_weight(snapshot)
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight.to(device)
    return Data(**fields)


def propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    default_delta_t: float | Tensor = 1.0,
) -> Tensor:
    """Advance latent states via the unified Koopman contract.

    Always resolves ``delta_t`` and calls
    :meth:`~koopman_graph.operators.KoopmanOperatorContract.advance`. Discrete
    operators ignore the interval; continuous operators (built-in or custom
    injected) require it. Dispatch does **not** use concrete ``isinstance``
    checks.

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

    Returns
    -------
    Tensor
        Advanced latent states.
    """
    interval = resolve_delta_t(delta_t, default_delta_t=default_delta_t)
    return koopman.advance(z, interval, control=control)


def inverse_propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    default_delta_t: float | Tensor = 1.0,
    inverse_matrix: Tensor | None = None,
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

    Returns
    -------
    Tensor
        Recovered latent states.
    """
    interval = resolve_delta_t(delta_t, default_delta_t=default_delta_t)
    return koopman.inverse_advance(
        z,
        interval,
        control=control,
        inverse_matrix=inverse_matrix,
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
) -> tuple[Tensor, Tensor]:
    """Advance latent state once and decode to physical node features.

    Parameters
    ----------
    koopman : KoopmanOperator or ContinuousKoopmanOperator
        Operator used for one forward step.
    decoder : callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` mapping.
    z : Tensor
        Current latent states.
    edge_index : Tensor
        Decode topology with shape ``(2, num_edges)``.
    edge_weight : Tensor or None, optional
        Optional scalar edge weights with shape ``(num_edges,)``.
    control : Tensor or None, optional
        Optional control input for controlled operators.
    delta_t : float, Tensor, or None, optional
        Explicit continuous-time integration interval.
    default_delta_t : float or Tensor, optional
        Fallback interval when ``delta_t is None`` in continuous mode.

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
) -> list[tuple[Tensor, Tensor, Tensor | None]]:
    """Run an autoregressive latent advance/decode loop.

    Shared primitive for inference and training rollouts. Callers encode the
    initial graph once, then supply per-step topology / control / ``delta_t``
    policies. See the module docstring for topology policy guidance.

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
    for step in range(steps):
        edge_index, edge_weight = topology_at(step)
        control = None if control_at is None else control_at(step)
        delta_t = None if delta_t_at is None else delta_t_at(step)
        latent, prediction = advance_and_decode(
            koopman,
            decoder,
            latent,
            edge_index,
            edge_weight,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
        )
        outputs.append((prediction, edge_index, edge_weight))
    return outputs
