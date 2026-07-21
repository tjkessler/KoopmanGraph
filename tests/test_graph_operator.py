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


def test_serialization_round_trip_graph_operator_controls_delays_bilinear(
    tmp_path: Path,
) -> None:
    """Graph checkpoints round-trip kind, delays, controls, and bilinear factors."""
    from koopman_graph.nn import DelayEmbeddingEncoder
    from koopman_graph.serialization import build_model_config

    model = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=6, hidden_channels=8, latent_dim=4, num_layers=1
        ),
        decoder=GNNDecoder(
            latent_dim=4, hidden_channels=8, out_channels=2, num_layers=1
        ),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
        control_dim=2,
        control_mode="bilinear",
        bilinear_rank=1,
        n_delays=3,
    )
    assert model.koopman_kind == "graph"
    assert isinstance(model.encoder, DelayEmbeddingEncoder)
    with torch.no_grad():
        state = model.state_dict()
        state["koopman._self.K"].copy_(torch.eye(4) * 0.8)
        state["koopman._nbr.K"].copy_(torch.eye(4) * 0.1)
        state["koopman._self.B"].fill_(0.15)
        state["koopman._self.P"].fill_(0.05)
        state["koopman._self.Q"].fill_(-0.02)
        model.load_state_dict(state)

    config = build_model_config(model)
    assert config["koopman_kind"] == "graph"
    assert config["control_mode"] == "bilinear"
    assert config["bilinear_rank"] == 1
    assert config["control_dim"] == 2
    assert config["n_delays"] == 3

    path = tmp_path / "graph_bilinear_delay.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)

    assert loaded.koopman_kind == "graph"
    assert isinstance(loaded.koopman, GraphKoopmanOperator)
    assert loaded.control_mode == "bilinear"
    assert loaded.bilinear_rank == 1
    assert loaded.control_dim == 2
    assert loaded.n_delays == 3
    assert isinstance(loaded.encoder, DelayEmbeddingEncoder)
    assert loaded.encoder.base_encoder.in_channels == 6
    assert torch.allclose(loaded.koopman.K_self, model.koopman.K_self)
    assert torch.allclose(loaded.koopman.K_nbr, model.koopman.K_nbr)
    original = model.state_dict()
    restored = loaded.state_dict()
    for key in ("koopman._self.B", "koopman._self.P", "koopman._self.Q"):
        assert torch.allclose(restored[key], original[key])


def test_serialization_round_trip_graph_operator_additive_control(
    tmp_path: Path,
) -> None:
    """Graph additive control_dim / B survive save/load."""
    from koopman_graph.serialization import build_model_config

    model = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=2, hidden_channels=8, latent_dim=3, num_layers=1
        ),
        decoder=GNNDecoder(
            latent_dim=3, hidden_channels=8, out_channels=2, num_layers=1
        ),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
        control_dim=1,
        control_mode="additive",
    )
    with torch.no_grad():
        model.koopman.set_dense_matrices(
            torch.eye(3) * 0.9,
            torch.eye(3) * 0.05,
            control_matrix=torch.tensor([[0.2, -0.1, 0.3]]),
        )

    config = build_model_config(model)
    assert config["koopman_kind"] == "graph"
    assert config["control_mode"] == "additive"
    assert config["control_dim"] == 1
    assert config["bilinear_rank"] is None

    path = tmp_path / "graph_additive.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "graph"
    assert loaded.control_mode == "additive"
    assert loaded.control_dim == 1
    assert loaded.bilinear_rank is None
    assert torch.allclose(
        loaded.state_dict()["koopman._self.B"],
        model.state_dict()["koopman._self.B"],
    )


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


def test_graph_operator_factorized_reset_and_monitoring() -> None:
    """Factorized neighbor reset, monitoring helpers, and dense writes."""
    op = GraphKoopmanOperator(
        3,
        parameterization="odo",
        init_mode="identity_noise",
        init_scale=0.05,
        control_dim=1,
        control_mode="additive",
    )
    op.reset_parameters()
    assert op.K_self.shape == (3, 3)
    assert op.K_nbr.shape == (3, 3)
    assert torch.equal(op.K, op.matrix)
    assert op.bound_metric().ndim == 0
    assert op.spectral_radius().ndim == 0
    assert op.stability_certificate() is None

    dense = GraphKoopmanOperator(
        2,
        init_mode="identity_noise",
        init_scale=0.0,
        control_dim=1,
        control_mode="bilinear",
    )
    assert torch.allclose(dense.K_nbr, torch.zeros(2, 2))
    dense.reset_parameters()
    assert torch.allclose(dense.K_nbr, torch.zeros(2, 2))
    state = dense.state_dict()
    assert "_self.K" in state
    assert "_nbr.K" in state
    restored = GraphKoopmanOperator(2, control_dim=1, control_mode="bilinear")
    restored.load_state_dict(state)
    assert torch.allclose(restored.K_self, dense.K_self)
    assert torch.allclose(restored.K_nbr, dense.K_nbr)

    k_self = torch.diag(torch.tensor([0.8, 0.7]))
    k_nbr = 0.05 * torch.eye(2)
    b = torch.tensor([[0.1, -0.05]])
    n_stack = torch.zeros(1, 2, 2)
    n_stack[0] = 0.02 * torch.eye(2)
    dense.set_dense_matrices(
        k_self,
        k_nbr,
        control_matrix=b,
        bilinear_matrices=n_stack,
    )
    assert torch.allclose(dense.K_self, k_self)
    assert torch.allclose(dense.K_nbr, k_nbr)


def test_graph_operator_control_forward_inverse_and_errors() -> None:
    """Controlled advance/inverse plus shape and control validation."""
    edge_index = _path_edge_index(3)
    op = GraphKoopmanOperator(
        2,
        init_mode="identity",
        control_dim=1,
        control_mode="additive",
    )
    k_self = torch.diag(torch.tensor([0.9, 0.8]))
    k_nbr = 0.1 * torch.eye(2)
    op.set_dense_matrices(k_self, k_nbr, control_matrix=torch.tensor([[0.2, -0.1]]))
    z = torch.randn(3, 2)
    control = torch.tensor([0.5])
    z_next = op(z, edge_index, control=control)
    recovered = op.inverse_advance(z_next, control=control, edge_index=edge_index)
    assert torch.allclose(recovered, z, atol=1e-4)

    per_node = torch.ones(3, 1) * 0.25
    z_next_pn = op(z, edge_index, control=per_node)
    recovered_pn = op.inverse_advance(
        z_next_pn, control=per_node, edge_index=edge_index
    )
    assert torch.allclose(recovered_pn, z, atol=1e-4)

    bilinear = GraphKoopmanOperator(
        2,
        init_mode="identity",
        control_dim=1,
        control_mode="bilinear",
    )
    bilinear.set_dense_matrices(
        k_self,
        torch.zeros_like(k_self),
        control_matrix=torch.tensor([[0.1, 0.0]]),
        bilinear_matrices=0.05 * torch.eye(2).unsqueeze(0),
    )
    z_b = bilinear(z, edge_index, control=control)
    recovered_b = bilinear.inverse_advance(z_b, control=control, edge_index=edge_index)
    assert torch.allclose(recovered_b, z, atol=1e-4)

    z_b_pn = bilinear(z, edge_index, control=per_node)
    recovered_b_pn = bilinear.inverse_advance(
        z_b_pn, control=per_node, edge_index=edge_index
    )
    assert torch.allclose(recovered_b_pn, z, atol=1e-4)

    with pytest.raises(ValueError, match="expects z with shape"):
        op(torch.randn(3, 2, 1), edge_index, control=control)
    with pytest.raises(ValueError, match="trailing dimension"):
        op(torch.randn(3, 4), edge_index, control=control)
    with pytest.raises(ValueError, match="uncontrolled"):
        GraphKoopmanOperator(2)(z, edge_index, control=control)
    with pytest.raises(ValueError, match="control input is required"):
        op(z, edge_index, control=None)
    with pytest.raises(ValueError, match="edge_index is required"):
        op.inverse_advance(z, control=control)
    with pytest.raises(ValueError, match="inverse_advance expects"):
        op.inverse_advance(torch.randn(3), control=control, edge_index=edge_index)
    with pytest.raises(ValueError, match="control input is required"):
        op.inverse_advance(z_next, edge_index=edge_index)
    with pytest.raises(ValueError, match="Per-node control has"):
        bilinear.inverse_advance(
            z_b_pn, control=torch.ones(2, 1) * 0.25, edge_index=edge_index
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        bilinear.inverse_advance(
            z_b, control=torch.ones(1, 1, 1), edge_index=edge_index
        )


def test_graph_bilinear_inverse_round_trip_weighted_and_low_rank() -> None:
    """Per-node bilinear inverse honors weights, neighbor coupling, and low-rank N."""
    torch.manual_seed(1)
    edge_index = _path_edge_index(3)
    edge_weight = torch.tensor([1.0, 1.0, 0.5, 0.5])
    k_self = torch.diag(torch.tensor([0.85, 0.7]))
    k_nbr = 0.08 * torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    z = torch.randn(3, 2)
    global_u = torch.tensor([0.3, -0.2])
    per_node_u = torch.tensor([[0.4, -0.1], [0.0, 0.2], [-0.3, 0.5]])

    full_rank = GraphKoopmanOperator(
        2,
        init_mode="identity",
        control_dim=2,
        control_mode="bilinear",
    )
    full_rank.set_dense_matrices(
        k_self,
        k_nbr,
        control_matrix=torch.tensor([[0.1, 0.0], [0.0, -0.05]]),
        bilinear_matrices=0.05
        * torch.stack([torch.eye(2), torch.tensor([[0.0, 1.0], [1.0, 0.0]])]),
    )
    for control in (global_u, per_node_u):
        advanced = full_rank.advance(
            z, control=control, edge_index=edge_index, edge_weight=edge_weight
        )
        recovered = full_rank.inverse_advance(
            advanced,
            control=control,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        assert torch.allclose(recovered, z, atol=1e-4)

    low_rank = GraphKoopmanOperator(
        2,
        init_mode="identity",
        control_dim=2,
        control_mode="bilinear",
        bilinear_rank=1,
    )
    with torch.no_grad():
        low_rank._self._parameters["K"].copy_(k_self)
        low_rank._nbr._parameters["K"].copy_(k_nbr)
        low_rank._self.B.copy_(torch.tensor([[0.1, 0.0], [0.0, -0.05]]))
        low_rank._self.P.copy_(0.2 * torch.ones(2, 2, 1))
        low_rank._self.Q.copy_(0.1 * torch.ones(2, 2, 1))
    for control in (global_u, per_node_u):
        advanced = low_rank.advance(
            z, control=control, edge_index=edge_index, edge_weight=edge_weight
        )
        recovered = low_rank.inverse_advance(
            advanced,
            control=control,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        assert torch.allclose(recovered, z, atol=1e-4)

    # Shared vs per-node block assembly must disagree when controls differ by node.
    blocks = full_rank.effective_matrix(
        edge_index,
        3,
        edge_weight=edge_weight,
        k_self_blocks=torch.stack(
            [
                k_self + 0.1 * torch.eye(2),
                k_self,
                k_self - 0.05 * torch.eye(2),
            ]
        ),
    )
    shared = full_rank.effective_matrix(
        edge_index, 3, edge_weight=edge_weight, k_self=k_self
    )
    assert blocks.shape == shared.shape == (6, 6)
    assert not torch.allclose(blocks, shared)
    with pytest.raises(ValueError, match="at most one"):
        full_rank.effective_matrix(
            edge_index, 3, k_self=k_self, k_self_blocks=torch.stack([k_self] * 3)
        )


def test_graph_operator_singular_effective_uses_pinv() -> None:
    """Singular effective matrices fall back to a pseudoinverse inverse."""
    edge_index = _path_edge_index(2)
    op = GraphKoopmanOperator(2, init_mode="identity")
    # Rank-deficient self term with zero neighbor coupling → singular effective.
    op.set_dense_matrices(torch.zeros(2, 2), torch.zeros(2, 2))
    z = torch.randn(2, 2)
    recovered = op.inverse_advance(z, edge_index=edge_index)
    assert recovered.shape == z.shape
    assert torch.isfinite(recovered).all()


def _graph_model(latent_dim: int = 2, in_channels: int = 2) -> GraphKoopmanModel:
    """Build a small discrete graph Koopman model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels, 4, latent_dim, num_layers=1),
        decoder=GNNDecoder(latent_dim, 4, in_channels, num_layers=1),
        latent_dim=latent_dim,
        time_step=0.1,
        koopman="graph",
    )


def test_model_spectrum_requires_topology_for_graph() -> None:
    """Graph-model spectrum never silently falls back to K_self."""
    model = _graph_model()
    with pytest.raises(ValueError, match="edge_index and num_nodes are required"):
        model.spectrum()
    with pytest.raises(ValueError, match="edge_index and num_nodes are required"):
        model.spectrum(edge_index=_path_edge_index(3))
    with pytest.raises(ValueError, match="edge_index and num_nodes are required"):
        model.spectrum(num_nodes=3)


def test_model_spectrum_matches_operator_effective() -> None:
    """Model spectrum delegates to the topology-coupled operator spectrum."""
    torch.manual_seed(3)
    num_nodes = 4
    latent_dim = 2
    edge_index = _path_edge_index(num_nodes)
    model = _graph_model(latent_dim=latent_dim)
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(
        0.6 * torch.eye(latent_dim),
        0.15 * torch.eye(latent_dim),
    )
    model_spec = model.spectrum(edge_index=edge_index, num_nodes=num_nodes)
    op_spec = model.koopman.spectrum(edge_index, num_nodes, time_step=model.time_step)
    assert model_spec.eigenvalues.shape == (num_nodes * latent_dim,)
    assert torch.allclose(model_spec.eigenvalues, op_spec.eigenvalues, atol=1e-6)
    # Distinct from the misleading K_self-only spectrum (wrong size + values).
    self_mags = torch.linalg.eigvals(model.koopman.matrix).abs().sort().values
    assert model_spec.magnitudes.numel() != self_mags.numel()
    assert not torch.allclose(
        model_spec.magnitudes[: self_mags.numel()].sort().values,
        self_mags,
        atol=1e-4,
    )


def test_model_spectrum_sensitive_to_topology_and_neighbor() -> None:
    """Changing topology, weights, or K_nbr changes spectrum with fixed K_self."""
    torch.manual_seed(4)
    num_nodes = 4
    latent_dim = 2
    k_self = 0.7 * torch.eye(latent_dim)
    k_nbr = 0.2 * torch.eye(latent_dim)
    model = _graph_model(latent_dim=latent_dim)
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(k_self, k_nbr)

    path = _path_edge_index(num_nodes)
    # Star-like: node 0 connected to all others (undirected).
    star_pairs = []
    for node in range(1, num_nodes):
        star_pairs.extend([[0, node], [node, 0]])
    star = torch.tensor(star_pairs, dtype=torch.long).t().contiguous()

    path_spec = model.spectrum(edge_index=path, num_nodes=num_nodes)
    star_spec = model.spectrum(edge_index=star, num_nodes=num_nodes)
    assert not torch.allclose(path_spec.magnitudes, star_spec.magnitudes, atol=1e-5)

    weights = torch.ones(path.shape[1])
    weights[0] = 3.0
    weights[1] = 3.0
    weighted_spec = model.spectrum(
        edge_index=path,
        num_nodes=num_nodes,
        edge_weight=weights,
    )
    assert not torch.allclose(path_spec.magnitudes, weighted_spec.magnitudes, atol=1e-5)

    model.koopman.set_dense_matrices(k_self, 0.05 * torch.eye(latent_dim))
    nbr_spec = model.spectrum(edge_index=path, num_nodes=num_nodes)
    assert not torch.allclose(path_spec.magnitudes, nbr_spec.magnitudes, atol=1e-5)
    assert torch.allclose(model.koopman.K_self, k_self)


def test_pernode_model_spectrum_ignores_topology_kwargs() -> None:
    """Ordinary discrete models keep matrix-based spectrum without topology."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 3, num_layers=1),
        decoder=GNNDecoder(3, 4, 2, num_layers=1),
        latent_dim=3,
        time_step=0.2,
    )
    spectrum = model.spectrum()
    assert spectrum.eigenvalues.shape == (3,)
    # Topology kwargs must not be required (and are ignored for per-node).
    again = model.spectrum(edge_index=_path_edge_index(4), num_nodes=4)
    assert torch.allclose(spectrum.eigenvalues, again.eigenvalues)


def test_resolve_spectrum_forwards_graph_topology() -> None:
    """resolve_spectrum / dynamical_similarity forward topology for graph models."""
    from koopman_graph.analysis import dynamical_similarity, resolve_spectrum

    torch.manual_seed(5)
    num_nodes = 3
    edge_index = _path_edge_index(num_nodes)
    model = _graph_model()
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(0.5 * torch.eye(2), 0.1 * torch.eye(2))

    with pytest.raises(ValueError, match="edge_index and num_nodes"):
        resolve_spectrum(model)

    resolved = resolve_spectrum(model, edge_index=edge_index, num_nodes=num_nodes)
    direct = model.spectrum(edge_index=edge_index, num_nodes=num_nodes)
    assert torch.allclose(resolved.eigenvalues, direct.eigenvalues)

    distance = dynamical_similarity(
        model,
        model,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    assert distance.item() == pytest.approx(0.0)


def test_decode_mode_shapes_graph_uses_effective_spectrum() -> None:
    """Mode-shape decoding passes topology into graph-model spectrum."""
    from koopman_graph.analysis import decode_mode_shapes

    torch.manual_seed(6)
    num_nodes = 3
    edge_index = _path_edge_index(num_nodes)
    model = _graph_model()
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(0.8 * torch.eye(2), 0.05 * torch.eye(2))
    data = Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
    shapes = decode_mode_shapes(model, data, mode_indices=[0, 1])
    assert shapes.shape == (2, num_nodes, 2)
    assert torch.isfinite(shapes.real).all()


def test_graph_bound_metric_is_factor_level_not_effective() -> None:
    """bound_metric monitors factors, not the topology-coupled spectrum."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    op = GraphKoopmanOperator(2, init_mode="identity")
    op.set_dense_matrices(0.5 * torch.eye(2), 2.0 * torch.eye(2))
    factor_bound = op.bound_metric()
    effective_radius = (
        torch.linalg.eigvals(op.effective_matrix(edge_index, 2)).abs().max()
    )
    # Factor max(|λ(K_self)|, |λ(K_nbr)|) need not equal ρ(I⊗K_self+Â⊗K_nbr).
    assert factor_bound.item() != pytest.approx(effective_radius.item(), abs=1e-4)
    assert factor_bound.item() < effective_radius.item()
