"""Tests for AdaptiveAdjacency and learn_topology (TASK-1308)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.nn import (
    DEFAULT_TOPOLOGY_EMBEDDING_DIM,
    AdaptiveAdjacency,
    HypergraphDecoder,
    HypergraphEncoder,
)
from koopman_graph.nn import adaptive_topology as adaptive_topology_mod


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_adaptive_adjacency_shape_row_stochastic_and_grad() -> None:
    """Dense Â = softmax(E_s E_t^⊤) is (N,N), row-stochastic (rows sum to 1)."""
    module = AdaptiveAdjacency(embedding_dim=4, num_nodes=5)
    adjacency = module.dense_adjacency()
    assert adjacency.shape == (5, 5)
    assert torch.allclose(adjacency.sum(dim=1), torch.ones(5), atol=1e-5)
    assert (adjacency >= 0).all()
    # Row sums are constant, so use a non-uniform objective for grad flow.
    loss = (adjacency * torch.arange(5, dtype=adjacency.dtype).view(1, -1)).sum()
    loss.backward()
    assert module.source_embedding.grad is not None
    assert module.target_embedding.grad is not None
    assert module.source_embedding.grad.abs().sum() > 0
    assert module.target_embedding.grad.abs().sum() > 0
    edge_index, edge_weight = module.materialize()
    assert edge_index.shape == (2, 25)
    assert edge_weight.shape == (25,)
    assert adaptive_topology_mod.AdaptiveAdjacency is AdaptiveAdjacency
    assert DEFAULT_TOPOLOGY_EMBEDDING_DIM == 8


def test_num_nodes_static_binding() -> None:
    """Changing N after the first bind raises."""
    module = AdaptiveAdjacency(num_nodes=3)
    module.set_num_nodes(3)
    with pytest.raises(ValueError, match="static per fit"):
        module.set_num_nodes(4)


def test_learn_topology_none_is_noop() -> None:
    """Default learn_topology=None matches a model without the flag."""
    torch.manual_seed(0)
    encoder = GNNEncoder(2, 8, 4, num_layers=1)
    decoder = GNNDecoder(4, 8, 2, num_layers=1)
    baseline = GraphKoopmanModel(
        encoder, decoder, latent_dim=4, time_step=1.0, koopman="graph"
    )
    torch.manual_seed(0)
    encoder2 = GNNEncoder(2, 8, 4, num_layers=1)
    decoder2 = GNNDecoder(4, 8, 2, num_layers=1)
    with_flag = GraphKoopmanModel(
        encoder2,
        decoder2,
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
        learn_topology=None,
    )
    data = Data(x=torch.randn(4, 2), edge_index=_path_edge_index(4))
    baseline.eval()
    with_flag.eval()
    with torch.no_grad():
        assert torch.allclose(baseline(data), with_flag(data), atol=1e-6)
    assert with_flag.adaptive_topology is None


def test_seeded_recovery_ranks_true_edges() -> None:
    """After fitting to a planted Â, true edges outrank spurious ones."""
    torch.manual_seed(1)
    num_nodes = 4
    true_edges = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    target = torch.zeros(num_nodes, num_nodes)
    for i, j in true_edges:
        target[i, j] = 1.0
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)

    module = AdaptiveAdjacency(embedding_dim=8, num_nodes=num_nodes)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.2)
    for _ in range(250):
        optimizer.zero_grad()
        loss = (module.dense_adjacency() - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    adjacency = module.dense_adjacency().detach()
    true_scores = [adjacency[i, j].item() for i, j in true_edges]
    spurious = [
        adjacency[i, j].item()
        for i in range(num_nodes)
        for j in range(num_nodes)
        if (i, j) not in true_edges
    ]
    wins = sum(t > s for t in true_scores for s in spurious)
    total = len(true_scores) * len(spurious)
    auc = wins / total
    assert auc >= 0.8


def test_factory_self_adaptive_and_invalid() -> None:
    """Factory accepts self_adaptive; rejects unknown modes."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        learn_topology="self_adaptive",
        topology_embedding_dim=6,
    )
    assert model.learn_topology == "self_adaptive"
    assert model.topology_embedding_dim == 6
    assert isinstance(model.adaptive_topology, AdaptiveAdjacency)
    with pytest.raises(ValueError, match="self_adaptive"):
        GraphKoopmanModel(
            GNNEncoder(2, 8, 3, num_layers=1),
            GNNDecoder(3, 8, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            learn_topology="nri",
        )


def test_pernode_graph_replace_topology() -> None:
    """Enabled learning replaces pairwise edges for pernode and graph encode."""
    torch.manual_seed(2)
    data = Data(x=torch.randn(3, 2), edge_index=_path_edge_index(3))
    for kind in ("pernode", "graph"):
        model = GraphKoopmanModel(
            GNNEncoder(2, 8, 3, num_layers=1),
            GNNDecoder(3, 8, 2, num_layers=1),
            latent_dim=3,
            time_step=1.0,
            koopman=kind,  # type: ignore[arg-type]
            learn_topology="self_adaptive",
        )
        model.eval()
        with torch.no_grad():
            _ = model.encode(data)
        assert model.adaptive_topology is not None
        assert model.adaptive_topology.num_nodes == 3
        learned_index, learned_weight = model.adaptive_topology.materialize()
        resolved_index, resolved_weight = model.resolve_pairwise_topology(data)
        assert torch.equal(resolved_index, learned_index)
        assert torch.allclose(resolved_weight, learned_weight)


def test_graph_spectrum_uses_learned_adjacency() -> None:
    """Graph spectrum materializes learned Â when learn_topology is enabled."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 2, num_layers=1),
        GNNDecoder(2, 8, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    with pytest.raises(ValueError, match="num_nodes"):
        model.spectrum()
    spectrum = model.spectrum(num_nodes=3)
    assert spectrum.eigenvalues.shape == (6,)


def test_hypergraph_encode_keeps_incidence() -> None:
    """Hypergraph encode still uses hyperedge_index when learning is enabled."""
    hyperedge_index = torch.tensor(
        [[0, 1, 1, 2], [0, 0, 1, 1]],
        dtype=torch.long,
    )
    data = Data(
        x=torch.randn(3, 2),
        edge_index=_path_edge_index(3),
        hyperedge_index=hyperedge_index,
    )
    model = GraphKoopmanModel(
        HypergraphEncoder(2, 8, 3, num_layers=1),
        HypergraphDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        koopman="hypergraph",
        learn_topology="self_adaptive",
    )
    model.eval()
    with torch.no_grad():
        z = model.encode(data)
        pred = model(data)
    assert z.shape == (3, 3)
    assert pred.shape == (3, 2)
    assert model.adaptive_topology is not None
    assert model.adaptive_topology.num_nodes == 3


def test_format1_round_trip(tmp_path: Path) -> None:
    """Format-1 save/load preserves learn_topology config and embeddings."""
    torch.manual_seed(3)
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        learn_topology="self_adaptive",
        topology_embedding_dim=5,
    )
    data = Data(x=torch.randn(4, 2), edge_index=_path_edge_index(4))
    model.eval()
    with torch.no_grad():
        before = model(data)
    path = tmp_path / "adaptive.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.learn_topology == "self_adaptive"
    assert loaded.topology_embedding_dim == 5
    assert isinstance(loaded.adaptive_topology, AdaptiveAdjacency)
    assert loaded.adaptive_topology.num_nodes == 4
    loaded.eval()
    with torch.no_grad():
        after = loaded(data)
    assert torch.allclose(before, after, atol=1e-5)


def test_topology_embedding_dim_must_be_positive() -> None:
    """Factory rejects non-positive topology embedding width."""
    with pytest.raises(ValueError, match="topology_embedding_dim"):
        GraphKoopmanModel(
            GNNEncoder(2, 8, 3, num_layers=1),
            GNNDecoder(3, 8, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            learn_topology="self_adaptive",
            topology_embedding_dim=0,
        )


def test_adaptive_adjacency_validation_and_forward() -> None:
    """AdaptiveAdjacency guards embedding width, node count, and binding."""
    with pytest.raises(ValueError, match="embedding_dim"):
        AdaptiveAdjacency(embedding_dim=0)
    module = AdaptiveAdjacency(embedding_dim=4)
    with pytest.raises(ValueError, match="num_nodes"):
        module.set_num_nodes(0)
    with pytest.raises(RuntimeError, match="set_num_nodes"):
        module.dense_adjacency()
    bound = AdaptiveAdjacency(embedding_dim=3, num_nodes=2)
    edge_index, edge_weight = bound()
    assert edge_index.shape == (2, 4)
    assert edge_weight.shape == (4,)


def test_materialize_learned_topology_paths() -> None:
    """Learned topology materialization infers N and rejects bad inputs."""
    data = Data(x=torch.randn(3, 2), edge_index=_path_edge_index(3))
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    with pytest.raises(RuntimeError, match="self_adaptive"):
        GraphKoopmanModel(
            GNNEncoder(2, 8, 3, num_layers=1),
            GNNDecoder(3, 8, 2, num_layers=1),
            latent_dim=3,
            time_step=1.0,
        ).materialize_learned_topology(data)
    learned_from_data = model.materialize_learned_topology(data)
    learned_from_tensor = model.materialize_learned_topology(data.x)
    assert learned_from_data[0].shape == learned_from_tensor[0].shape
    with pytest.raises(ValueError, match="num_nodes is required"):
        model.materialize_learned_topology(42)  # type: ignore[arg-type]


def test_encode_rollout_origin_and_rollout_materialize_topology() -> None:
    """Rollout origin and predict materialize static learned adjacency."""
    data = Data(x=torch.randn(3, 2), edge_index=_path_edge_index(3))
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    model.eval()
    with torch.no_grad():
        z, edge_index, edge_weight = model.encode_rollout_origin(data)
    assert z.shape == (3, 3)
    assert edge_index.shape[0] == 2
    assert edge_weight is not None
    with torch.no_grad():
        preds = model.predict(data, steps=2)
    assert len(preds) == 2


def test_spectrum_requires_adaptive_module_when_learning() -> None:
    """Spectrum on learned topology fails if the module is missing."""
    from unittest.mock import PropertyMock, patch

    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 2, num_layers=1),
        GNNDecoder(2, 8, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with (
        patch.object(
            type(model),
            "learns_pairwise_topology",
            new_callable=PropertyMock,
        ),
        patch.object(model, "adaptive_topology", None),
    ):
        type(model).learns_pairwise_topology.return_value = True  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="adaptive_topology module missing"):
            model.spectrum(num_nodes=3, edge_index=edge_index)


def test_materialize_learned_topology_explicit_num_nodes() -> None:
    """Explicit num_nodes resolves device from Data, Tensor, or neither."""
    data = Data(x=torch.randn(3, 2), edge_index=_path_edge_index(3))
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    from_data = model.materialize_learned_topology(data, num_nodes=3)
    from_tensor = model.materialize_learned_topology(data.x, num_nodes=3)
    from_scalar = model.materialize_learned_topology(0, num_nodes=3)  # type: ignore[arg-type]
    assert from_data[0].shape == from_tensor[0].shape == from_scalar[0].shape
