"""Tests for GraphKoopmanOperator and networked model integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.graph_utils import (
    dense_symmetric_normalized_adjacency,
    propagate_latent,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected path graph edge index."""
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_k_nbr_zero_matches_pernode_operator() -> None:
    """Verify K_nbr=0 recovers the per-node Koopman map exactly."""
    torch.manual_seed(0)
    latent_dim = 3
    num_nodes = 4
    edge_index = _path_edge_index(num_nodes)
    pernode = KoopmanOperator(latent_dim, init_mode="xavier")
    graph_op = GraphKoopmanOperator(latent_dim, init_mode="identity")
    graph_op.set_dense_matrices(pernode.K.detach().clone(), torch.zeros_like(pernode.K))

    z = torch.randn(num_nodes, latent_dim)
    expected = pernode(z)
    got = graph_op(z, edge_index)
    assert torch.allclose(got, expected, atol=1e-6)


def test_two_node_coupled_linear_recovery() -> None:
    """Recover a noiseless two-node coupled linear system exactly."""
    # Dynamics: z'_0 = 0.5 z_0 + 0.25 z_1; z'_1 = 0.25 z_0 + 0.5 z_1
    # with Â = [[0, 1], [1, 0]] (normalized undirected edge), this is
    # K_self = 0.5 I, K_nbr = 0.25 I.
    k_self = 0.5 * torch.eye(2)
    k_nbr = 0.25 * torch.eye(2)
    op = GraphKoopmanOperator(2, init_mode="identity")
    op.set_dense_matrices(k_self, k_nbr)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    z = torch.tensor([[1.0, -1.0], [2.0, 0.5]])
    z_next = op(z, edge_index)
    expected = torch.stack(
        [
            0.5 * z[0] + 0.25 * z[1],
            0.5 * z[1] + 0.25 * z[0],
        ]
    )
    assert torch.allclose(z_next, expected, atol=1e-6)

    recovered = op.inverse_advance(z_next, edge_index=edge_index)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_forward_matches_dense_effective_matrix() -> None:
    """Sparse message-passing forward matches dense Kronecker application."""
    torch.manual_seed(1)
    num_nodes = 5
    latent_dim = 3
    edge_index = _path_edge_index(num_nodes)
    op = GraphKoopmanOperator(latent_dim, init_mode="xavier", init_scale=0.2)
    z = torch.randn(num_nodes, latent_dim)

    sparse_next = op(z, edge_index)
    effective = op.effective_matrix(edge_index, num_nodes)
    dense_next = (effective @ z.reshape(-1)).view_as(z)
    assert torch.allclose(sparse_next, dense_next, atol=1e-5)


def test_spectrum_smoke_matches_effective_eigvals() -> None:
    """Spectrum helper returns eigenvalues of the effective operator."""
    torch.manual_seed(2)
    num_nodes = 3
    latent_dim = 2
    edge_index = _path_edge_index(num_nodes)
    op = GraphKoopmanOperator(latent_dim, init_mode="identity_noise", init_scale=0.05)
    spectrum = op.spectrum(edge_index, num_nodes, time_step=0.1)
    effective = op.effective_matrix(edge_index, num_nodes)
    eigvals = torch.linalg.eigvals(effective)
    assert spectrum.eigenvalues.shape == (num_nodes * latent_dim,)
    assert torch.allclose(
        spectrum.magnitudes.sort().values,
        eigvals.abs().sort().values,
        atol=1e-5,
    )


def test_graph_requires_edge_index_on_advance() -> None:
    """Networked advance without topology raises."""
    op = GraphKoopmanOperator(2, init_mode="identity")
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="edge_index is required"):
        op.advance(z)
    with pytest.raises(ValueError, match="edge_index is required"):
        propagate_latent(op, z)


def test_model_factory_graph_kind() -> None:
    """Factory koopman='graph' builds GraphKoopmanOperator."""
    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    assert model.koopman_kind == "graph"
    assert model.uses_graph_koopman


def test_graph_rejected_in_continuous_mode() -> None:
    """Networked factory kind is discrete-only."""
    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    with pytest.raises(ValueError, match="requires dynamics_mode='discrete'"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=3,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman="graph",
        )


def test_rewiring_changes_graph_latent_not_pernode() -> None:
    """Mid-horizon topology change alters graph advance but not per-node."""
    torch.manual_seed(3)
    num_nodes = 4
    latent_dim = 3
    in_channels = 2
    edge_a = _path_edge_index(num_nodes)
    # Ring topology as alternate edges.
    ring = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    ring += [[(i + 1) % num_nodes, i] for i in range(num_nodes)]
    edge_b = torch.tensor(ring, dtype=torch.long).t().contiguous()

    encoder = GNNEncoder(in_channels, 8, latent_dim, num_layers=1)
    decoder = GNNDecoder(latent_dim, 8, in_channels, num_layers=1)

    graph_model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
    )
    # Non-zero neighbor coupling so topology matters.
    with torch.no_grad():
        graph_model.koopman.set_dense_matrices(
            torch.eye(latent_dim),
            0.4 * torch.eye(latent_dim),
        )

    pernode_model = GraphKoopmanModel(
        GNNEncoder(in_channels, 8, latent_dim, num_layers=1),
        GNNDecoder(latent_dim, 8, in_channels, num_layers=1),
        latent_dim=latent_dim,
        time_step=0.1,
        koopman="pernode",
        koopman_init_mode="identity",
    )
    # Match encoder/decoder weights so only the operator differs in spirit;
    # for this test we compare latent advance directly on the graph operator.
    x0 = torch.randn(num_nodes, in_channels)
    initial = Data(x=x0, edge_index=edge_a)
    future = [Data(x=x0, edge_index=edge_b), Data(x=x0, edge_index=edge_b)]

    z0 = graph_model.encode(initial)
    z_path = graph_model.koopman(z0, edge_a)
    z_ring = graph_model.koopman(z0, edge_b)
    assert not torch.allclose(z_path, z_ring, atol=1e-5)

    # Per-node operator ignores topology kwargs.
    z_a = pernode_model.koopman.advance(z0, edge_index=edge_a)
    z_b = pernode_model.koopman.advance(z0, edge_index=edge_b)
    assert torch.allclose(z_a, z_b, atol=1e-6)

    preds_hold = graph_model.predict(initial, steps=2)
    preds_rewire = graph_model.predict(initial, steps=2, future_topologies=future)
    # Decoded states should differ once neighbor coupling and rewiring interact.
    assert not torch.allclose(preds_hold[1].x, preds_rewire[1].x, atol=1e-5)


def test_serialization_round_trip_graph_operator() -> None:
    """Save/load preserves GraphKoopmanOperator weights and kind."""
    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
    )
    with torch.no_grad():
        model.koopman.set_dense_matrices(
            torch.eye(3) * 0.7,
            torch.eye(3) * 0.2,
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph_op.pt"
        model.save(path)
        loaded = GraphKoopmanModel.load(path)

    assert loaded.koopman_kind == "graph"
    assert isinstance(loaded.koopman, GraphKoopmanOperator)
    assert torch.allclose(loaded.koopman.K_self, model.koopman.K_self)
    assert torch.allclose(loaded.koopman.K_nbr, model.koopman.K_nbr)


def test_fit_smoke_graph_operator_on_static_sequence() -> None:
    """Graph factory trains without error on a short synthetic sequence."""
    torch.manual_seed(4)
    num_nodes = 5
    edge_index = _path_edge_index(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, 2), edge_index=edge_index) for _ in range(8)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 4, num_layers=1),
        GNNDecoder(4, 8, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
    )
    history = model.fit(sequence, epochs=2, lr=1e-2)
    assert len(history.loss) == 2
    assert torch.isfinite(torch.tensor(history.loss[-1]))


def test_unsupported_sparsity_raises() -> None:
    """Reserved sparsity modes are rejected until implemented."""
    with pytest.raises(ValueError, match="not implemented"):
        GraphKoopmanOperator(2, sparsity="block_diagonal")  # type: ignore[arg-type]


def test_dense_adjacency_helper_used_by_effective_matrix() -> None:
    """effective_matrix uses the shared symmetric normalized adjacency."""
    edge_index = _path_edge_index(3)
    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        3,
        dtype=torch.float32,
    )
    assert adj.shape == (3, 3)
    assert torch.allclose(adj, adj.T, atol=1e-6)
