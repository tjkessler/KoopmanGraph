"""Encode-path helpers for GraphKoopmanModel (physics / delay / origin).

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.data.delay_windows import (
    apply_observation_mask_to_features,
    history_from_snapshots,
    stack_delay_features,
)
from koopman_graph.graph_utils import resolve_edge_index, resolve_edge_weight
from koopman_graph.observables import (
    PhysicsLiftingFn,
    PhysicsPosition,
    concatenate_observables,
    validate_physics_output,
)

from .validation import as_data

EncodeFn = Callable[
    [Tensor | Data, Tensor | None, Tensor | None],
    Tensor,
]


def encode_features(
    encoder: Callable[..., Tensor],
    x_or_data: Tensor | Data,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    *,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
    physics_dim: int = 0,
    physics_position: PhysicsPosition,
) -> Tensor:
    """Lift node features into the hybrid Koopman latent space.

    Parameters
    ----------
    encoder
        Topology-aware encoder callable ``(x, edge_index, edge_weight) -> z``.
    x_or_data : Tensor or Data
        Node features, delay window, or a PyG ``Data`` snapshot.
    edge_index : Tensor or None, optional
        Edge index required when ``x_or_data`` is a tensor.
    edge_weight : Tensor or None, optional
        Optional scalar edge weights for tensor input.
    physics_lifting_fn : callable or None, optional
        Optional physics feature map for hybrid observables.
    physics_dim : int, optional
        Expected physics feature width when lifting is enabled.
    physics_position : {"prepend"}
        Where physics features sit relative to GNN embeddings.

    Returns
    -------
    Tensor
        Latent node features with shape ``(num_nodes, latent_dim)``.

    Raises
    ------
    ValueError
        If delay-window tensor input lacks ``edge_index``, or physics lifting
        is requested with a raw delay-window tensor.
    """
    if isinstance(x_or_data, Tensor) and x_or_data.ndim == 3:
        if edge_index is None:
            msg = "edge_index is required for delay-window tensor input"
            raise ValueError(msg)
        z_gnn = encoder(x_or_data, edge_index, edge_weight)
        if physics_lifting_fn is None:
            return z_gnn
        msg = (
            "physics-informed observables with raw delay-window tensors are "
            "unsupported; pass a Data snapshot or use encode_at"
        )
        raise ValueError(msg)

    edge_index = resolve_edge_index(x_or_data, edge_index)
    edge_weight = resolve_edge_weight(x_or_data, edge_weight)
    z_gnn = encoder(x_or_data, edge_index, edge_weight)
    if physics_lifting_fn is None:
        return z_gnn

    snapshot = as_data(x_or_data, edge_index, edge_weight)
    physics_features = physics_lifting_fn(snapshot)
    validate_physics_output(
        physics_features,
        physics_dim=physics_dim,
        num_nodes=z_gnn.size(0),
    )
    return concatenate_observables(
        physics_features,
        z_gnn,
        position=physics_position,
    )


def encode_rollout_origin(
    encode: EncodeFn,
    *,
    n_delays: int,
    x_or_data: Tensor | Data,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    history: Sequence[Data] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Encode the initial state for an autoregressive rollout.

    Parameters
    ----------
    encode
        Encode callable matching :meth:`GraphKoopmanModel.encode`.
    n_delays : int
        Delay-embedding window length.
    x_or_data : Tensor or Data
        Initial node features or graph snapshot.
    edge_index : Tensor or None, optional
        Edge index when ``x_or_data`` is a tensor (or override).
    edge_weight : Tensor or None, optional
        Optional edge weights.
    history : sequence of Data or None, optional
        Past snapshots (oldest → newest) for delay embedding.

    Returns
    -------
    tuple of Tensor, Tensor, Tensor or None
        Encoded latent ``z``, resolved ``edge_index``, and optional
        ``edge_weight`` at the rollout origin.
    """
    if n_delays > 1 and isinstance(x_or_data, Data):
        past = list(history) if history is not None else []
        x_window, edge_resolved, weight_resolved, _ = history_from_snapshots(
            [*past, x_or_data],
            n_delays,
            pad=True,
        )
        z = encode(x_window, edge_resolved, weight_resolved)
        return z, edge_resolved, weight_resolved

    edge_resolved = resolve_edge_index(x_or_data, edge_index)
    weight_resolved = resolve_edge_weight(x_or_data, edge_weight)
    z = encode(x_or_data, edge_resolved, weight_resolved)
    return z, edge_resolved, weight_resolved


def encode_at_index(
    encoder: Callable[..., Tensor],
    encode: EncodeFn,
    sequence: GraphSnapshotSequence,
    index: int,
    *,
    n_delays: int,
    pad: bool = True,
    zero_unobserved: bool = True,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
    physics_dim: int = 0,
    physics_position: PhysicsPosition,
) -> Tensor:
    """Encode the delay window of ``sequence`` ending at ``index``.

    Parameters
    ----------
    encoder
        Topology-aware encoder callable.
    encode
        Single-snapshot / stacked encode callable (used when ``n_delays == 1``).
    sequence : GraphSnapshotSequence
        Source trajectory.
    index : int
        Inclusive end index of the delay window.
    n_delays : int
        Delay-embedding window length.
    pad : bool, optional
        Zero-pad missing history before the sequence start. Default is
        ``True``.
    zero_unobserved : bool, optional
        Zero unobserved node features inside the window when masks are
        present. Default is ``True``.
    physics_lifting_fn, physics_dim, physics_position
        Hybrid observable configuration (newest snapshot only when
        ``n_delays > 1``).

    Returns
    -------
    Tensor
        Latent node features with shape ``(num_nodes, latent_dim)``.
    """
    if n_delays == 1:
        snapshot = sequence[index]
        x = snapshot.x
        if zero_unobserved and sequence.has_observation_masks:
            x = apply_observation_mask_to_features(
                x,
                sequence.observation_mask_at(index),
            )
            snapshot = Data(
                x=x,
                edge_index=snapshot.edge_index,
                edge_weight=getattr(snapshot, "edge_weight", None),
            )
        return encode(snapshot, None, None)

    x_window, edge_index, edge_weight, _history_mask = stack_delay_features(
        sequence,
        index,
        n_delays,
        pad=pad,
        zero_unobserved=zero_unobserved,
    )
    z_gnn = encoder(x_window, edge_index, edge_weight)
    if physics_lifting_fn is None:
        return z_gnn

    snapshot = sequence[index]
    physics_features = physics_lifting_fn(snapshot)
    validate_physics_output(
        physics_features,
        physics_dim=physics_dim,
        num_nodes=z_gnn.size(0),
    )
    return concatenate_observables(
        physics_features,
        z_gnn,
        position=physics_position,
    )
