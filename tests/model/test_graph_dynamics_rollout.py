"""Rollout, evaluate, and permutation tests for graph-state topology."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphDynamicsConfig, GraphSnapshotSequence
from koopman_graph.nn.predicted_topology import (
    PredictedTopologyHead,
    build_supervision_index,
    candidate_edge_labels,
    decode_weighted_topology,
)
from koopman_graph.training import LossWeights, compute_batched_training_loss
from koopman_graph.uq import EnsembleGraphKoopmanModel


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index ``(2, 2*(N-1))``.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional cycle.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 3).

    Returns
    -------
    Tensor
        COO index.
    """
    forward_src = list(range(num_nodes))
    forward_dst = list(range(1, num_nodes)) + [0]
    src = forward_src + forward_dst
    dst = forward_dst + forward_src
    return torch.tensor([src, dst], dtype=torch.long)


def _directed_gnp_edge_index(num_nodes: int, *, seed: int) -> torch.Tensor:
    """Return a directed G(n,p) COO without self-loops.

    Parameters
    ----------
    num_nodes : int
        Node count.
    seed : int
        RNG seed.

    Returns
    -------
    Tensor
        COO index with at least one edge.
    """
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(num_nodes, num_nodes, generator=generator) < 0.35
    mask.fill_diagonal_(False)
    src, dst = mask.nonzero(as_tuple=True)
    if src.numel() == 0:
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
    return torch.stack([src, dst], dim=0)


def _tiny_graph_model(
    *,
    graph_dynamics: GraphDynamicsConfig | None,
    in_channels: int = 2,
    latent_dim: int = 2,
    koopman: str | None = "graph",
) -> GraphKoopmanModel:
    """Return a small homogeneous model.

    Parameters
    ----------
    graph_dynamics : GraphDynamicsConfig or None
        Opt-in graph-state config.
    in_channels : int, optional
        Feature width.
    latent_dim : int, optional
        Latent width.
    koopman : str or None, optional
        Operator kind.

    Returns
    -------
    GraphKoopmanModel
        Untrained model.
    """
    return GraphKoopmanModel(
        GNNEncoder(in_channels, 8, latent_dim, num_layers=1),
        GNNDecoder(latent_dim, 8, in_channels, num_layers=1),
        latent_dim=latent_dim,
        time_step=0.1,
        koopman=koopman,
        graph_dynamics=graph_dynamics,
    )


def test_default_predict_matches_hold_last() -> None:
    """``graph_dynamics=None`` stays on the 0.14 hold-last schedule."""
    torch.manual_seed(0)
    baseline = _tiny_graph_model(graph_dynamics=None)
    configured = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig())
    configured.encoder.load_state_dict(baseline.encoder.state_dict())
    configured.decoder.load_state_dict(baseline.decoder.state_dict())
    configured.koopman.load_state_dict(baseline.koopman.state_dict())
    origin = Data(x=torch.randn(6, 2), edge_index=_path_edge_index(6))
    baseline.eval()
    configured.eval()
    with torch.no_grad():
        hold = baseline.predict(origin, steps=3)
        auto = configured.predict(origin, steps=3, topology_policy="hold_last")
    for left, right in zip(hold, auto, strict=True):
        torch.testing.assert_close(left.x, right.x, atol=1e-6, rtol=0.0)


def test_recursive_predict_differs_from_hold_last() -> None:
    """A recursive head changes the forecast relative to hold-last."""
    torch.manual_seed(1)
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig(candidate_k=4))
    origin = Data(x=torch.randn(6, 2), edge_index=_path_edge_index(6))
    model.eval()
    with torch.no_grad():
        recursive = model.predict(origin, steps=2, topology_policy="recursive")
        hold = model.predict(origin, steps=2, topology_policy="hold_last")
    assert not torch.allclose(recursive[0].x, hold[0].x, atol=1e-6)
    assert recursive[0].edge_weight is not None
    assert hold[0].edge_weight is None or not torch.equal(
        recursive[0].edge_index, hold[0].edge_index
    )


def test_oracle_future_topologies_override_recursive() -> None:
    """Supplied ``future_topologies`` keep the oracle hold-last schedule."""
    torch.manual_seed(2)
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig())
    origin = Data(x=torch.randn(6, 2), edge_index=_path_edge_index(6))
    future = Data(x=torch.zeros(6, 2), edge_index=_cycle_edge_index(6))
    model.eval()
    with torch.no_grad():
        with_future = model.predict(origin, steps=1, future_topologies=[future])
        hold = model.predict(
            origin,
            steps=1,
            future_topologies=[future],
            topology_policy="hold_last",
        )
    torch.testing.assert_close(with_future[0].x, hold[0].x, atol=1e-6, rtol=0.0)
    assert torch.equal(with_future[0].edge_index, future.edge_index)


def test_evaluate_recursive_does_not_inject_oracle_futures() -> None:
    """Recursive evaluate does not pass oracle future edges into ``predict``."""
    torch.manual_seed(3)
    g0 = _path_edge_index(6)
    g1 = _cycle_edge_index(6)
    snapshots = [
        Data(x=torch.randn(6, 2), edge_index=g0 if t < 3 else g1) for t in range(8)
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig())
    seen: list[object] = []
    original = model.predict

    def _wrapped(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("future_topologies"))
        return original(*args, **kwargs)

    model.predict = _wrapped  # type: ignore[method-assign]
    model.evaluate(sequence, horizons=(1,), start_indices=(0,))
    assert seen
    assert all(item is None for item in seen)


def test_evaluate_default_still_injects_oracle_without_graph_dynamics() -> None:
    """0.14 evaluate on dynamic sequences still injects oracle futures."""
    torch.manual_seed(4)
    g0 = _path_edge_index(6)
    g1 = _cycle_edge_index(6)
    snapshots = [
        Data(x=torch.randn(6, 2), edge_index=g0 if t < 3 else g1) for t in range(8)
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    model = _tiny_graph_model(graph_dynamics=None)
    seen: list[object] = []
    original = model.predict

    def _wrapped(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("future_topologies"))
        return original(*args, **kwargs)

    model.predict = _wrapped  # type: ignore[method-assign]
    model.evaluate(sequence, horizons=(1,), start_indices=(0,))
    assert seen
    assert all(item is not None for item in seen)


def test_recursive_policy_without_head_raises() -> None:
    """Explicit recursive policy requires an attached topology head."""
    model = _tiny_graph_model(graph_dynamics=None)
    origin = Data(x=torch.randn(4, 2), edge_index=_path_edge_index(4))
    with pytest.raises(ValueError, match="recursive"):
        model.predict(origin, steps=1, topology_policy="recursive")


def test_batch_graphs_rejects_recursive_graph_state() -> None:
    """``batch_graphs=True`` refuses recursive predicted topology."""
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig())
    snapshots = [
        Data(x=torch.randn(4, 2), edge_index=_path_edge_index(4)) for _ in range(3)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="recursive graph-state"):
        compute_batched_training_loss(model, (sequence,), LossWeights())


def test_dense_logits_permute_with_nodes() -> None:
    """Permuting nodes permutes dense pairwise logits."""
    torch.manual_seed(5)
    head = PredictedTopologyHead(3, hidden_dim=8, max_nodes=8)
    z = torch.randn(5, 3)
    perm = torch.tensor([2, 0, 4, 1, 3])
    logits = head.pairwise_logits(z)
    logits_perm = head.pairwise_logits(z[perm])
    torch.testing.assert_close(
        logits_perm,
        logits[perm][:, perm],
        atol=1e-5,
        rtol=0.0,
    )


def test_supervision_index_keeps_union_over_cap() -> None:
    """Supervision keeps current ∪ next even when that exceeds ``k``."""
    current = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
    nxt = torch.tensor([[0, 0, 0], [3, 4, 5]], dtype=torch.long)
    index = build_supervision_index(
        6,
        candidate_k=2,
        current_edge_index=current,
        next_edge_index=nxt,
        generator=torch.Generator().manual_seed(0),
    )
    labels = candidate_edge_labels(index, nxt, 6)
    # All three next destinations of source 0 must remain.
    src0 = index[1][index[0] == 0]
    assert set(src0.tolist()) >= {3, 4, 5}
    assert float(labels.sum()) >= 3.0


def test_er_event_stub_head_beats_hold_last() -> None:
    """A correct next-graph head beats hold-last after a directed G(n,p) event.

    Truth is the model's own one-step oracle forecast on the post-event
    graph. A dense stub that scores those edges high should match the
    oracle more closely than hold-last on the pre-event path.
    """
    torch.manual_seed(6)
    num_nodes = 6
    g0 = _cycle_edge_index(num_nodes)
    g1 = _directed_gnp_edge_index(num_nodes, seed=11)
    origin = Data(x=torch.randn(num_nodes, 2), edge_index=g0)
    nxt = Data(x=torch.zeros(num_nodes, 2), edge_index=g1)
    model = _tiny_graph_model(
        graph_dynamics=GraphDynamicsConfig(topology_head="dense_mlp"),
        koopman="graph",
    )
    assert isinstance(model.predicted_topology, PredictedTopologyHead)

    def _oracle_logits(self: PredictedTopologyHead, z: torch.Tensor) -> torch.Tensor:
        del self
        logits = torch.full(
            (z.shape[0], z.shape[0]),
            -8.0,
            device=z.device,
            dtype=z.dtype,
        )
        logits[g1[0], g1[1]] = 8.0
        logits.fill_diagonal_(-1e9)
        return logits

    model.predicted_topology.pairwise_logits = types.MethodType(  # type: ignore[method-assign]
        _oracle_logits,
        model.predicted_topology,
    )
    model.eval()
    with torch.no_grad():
        truth = model.predict(origin, steps=1, future_topologies=[nxt])[0]
        recursive = model.predict(origin, steps=1, topology_policy="recursive")[0]
        hold = model.predict(origin, steps=1, topology_policy="hold_last")[0]
    mse_rec = float(nn.functional.mse_loss(recursive.x, truth.x))
    mse_hold = float(nn.functional.mse_loss(hold.x, truth.x))
    assert mse_rec < mse_hold


def test_uq_ensemble_respects_topology_policy() -> None:
    """Ensemble rollout uses the same recursive vs hold-last split as predict."""
    torch.manual_seed(8)
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig(candidate_k=4))
    ensemble = EnsembleGraphKoopmanModel([model])
    origin = Data(x=torch.randn(6, 2), edge_index=_path_edge_index(6))
    recursive = ensemble.predict(origin, steps=1, topology_policy="recursive")
    hold = ensemble.predict(origin, steps=1, topology_policy="hold_last")
    assert not torch.allclose(recursive[0].x, hold[0].x, atol=1e-6)


def test_decode_weighted_topology_uses_sigmoid_not_threshold() -> None:
    """Operator topology keeps all candidates with sigmoid weights."""
    torch.manual_seed(7)
    model = _tiny_graph_model(graph_dynamics=GraphDynamicsConfig(candidate_k=3))
    z = torch.randn(5, 2)
    current = _path_edge_index(5)
    assert model.predicted_topology is not None
    index, weight = decode_weighted_topology(model.predicted_topology, z, current)
    assert index.shape[1] == weight.shape[0]
    assert torch.all(weight > 0)
    assert torch.all(weight < 1)
