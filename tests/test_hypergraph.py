"""Tests for hyperedge incidence and hypergraph encoder/decoder peers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.baselines import DMDBaseline
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.data.validation import require_no_hyperedges
from koopman_graph.graph_utils import (
    dense_hyperedge_normalized_adjacency,
    hyperedge_normalized_incidence_weights,
    snapshot_to_device,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    GNNDecoder,
    GNNEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
)
from koopman_graph.serialization import build_model_config


def _numpy_zhou_hat(
    hyperedge_index: np.ndarray,
    *,
    num_nodes: int,
    hyperedge_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Dense Zhou ``Ĥ`` reference implemented in NumPy."""
    if hyperedge_index.size == 0:
        return np.zeros((num_nodes, num_nodes), dtype=np.float64)
    node_idx = hyperedge_index[0]
    hedge_idx = hyperedge_index[1]
    num_hyperedges = int(hedge_idx.max()) + 1
    incidence = np.zeros((num_nodes, num_hyperedges), dtype=np.float64)
    for n, h in zip(node_idx, hedge_idx, strict=True):
        incidence[int(n), int(h)] = 1.0
    weights = (
        np.ones(num_hyperedges, dtype=np.float64)
        if hyperedge_weight is None
        else np.asarray(hyperedge_weight, dtype=np.float64)
    )
    node_degree = incidence @ weights
    hyperedge_degree = incidence.sum(axis=0)
    deg_v_inv_sqrt = np.zeros_like(node_degree)
    positive = node_degree > 0
    deg_v_inv_sqrt[positive] = node_degree[positive] ** -0.5
    deg_e_inv = np.zeros_like(hyperedge_degree)
    positive_e = hyperedge_degree > 0
    deg_e_inv[positive_e] = hyperedge_degree[positive_e] ** -1.0
    scaled = incidence * weights[np.newaxis, :] * deg_e_inv[np.newaxis, :]
    mid = scaled @ incidence.T
    return deg_v_inv_sqrt[:, np.newaxis] * mid * deg_v_inv_sqrt[np.newaxis, :]


def _hyper_sequence(
    hyperedge_index: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_timesteps: int = 3,
    num_nodes: int = 4,
    in_channels: int = 2,
    hyperedge_weight: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a static hyperedge-carrying sequence on a path graph."""
    return GraphSnapshotSequence.from_arrays(
        torch.randn(num_timesteps, num_nodes, in_channels),
        edge_index,
        hyperedge_index=hyperedge_index,
        hyperedge_weight=hyperedge_weight,
    )


def test_sequence_accepts_static_hyperedges(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Container exposes shared hyperedge fields when present."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
    )
    assert sequence.has_hyperedges
    assert sequence.hyperedge_index is not None
    assert torch.equal(sequence.hyperedge_index, synthetic_hyperedge_index)
    assert sequence.hyperedge_weight is None
    for snapshot in sequence:
        assert torch.equal(snapshot.hyperedge_index, synthetic_hyperedge_index)


def test_sequence_without_hyperedges_is_noop(
    make_snapshots,
) -> None:
    """Default sequences remain hyperedge-free."""
    sequence = GraphSnapshotSequence(make_snapshots(num_timesteps=3))
    assert not sequence.has_hyperedges
    assert sequence.hyperedge_index is None
    assert sequence.hyperedge_weight is None


def test_rejects_varying_hyperedges(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Time-varying hyperedge incidence raises at construction."""
    snapshots = [
        Data(
            x=torch.randn(4, 2),
            edge_index=synthetic_hypergraph_edge_index,
            hyperedge_index=synthetic_hyperedge_index,
        )
        for _ in range(2)
    ]
    snapshots[1].hyperedge_index = torch.tensor(
        [[0, 1], [0, 0]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="time-varying hyperedges"):
        GraphSnapshotSequence(snapshots)


def test_rejects_hyperedge_presence_mismatch(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Mixed hyperedge presence across snapshots is rejected."""
    snapshots = [
        Data(
            x=torch.randn(4, 2),
            edge_index=synthetic_hypergraph_edge_index,
            hyperedge_index=synthetic_hyperedge_index,
        ),
        Data(
            x=torch.randn(4, 2),
            edge_index=synthetic_hypergraph_edge_index,
        ),
    ]
    with pytest.raises(ValueError, match="hyperedge_index presence"):
        GraphSnapshotSequence(snapshots)


def test_dense_normalization_matches_numpy(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Zhou ``Ĥ`` matches an independent NumPy reference."""
    hat = dense_hyperedge_normalized_adjacency(
        synthetic_hyperedge_index,
        num_nodes=4,
        dtype=torch.float64,
    )
    expected = _numpy_zhou_hat(
        synthetic_hyperedge_index.numpy(),
        num_nodes=4,
    )
    assert hat.numpy() == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert (
        hyperedge_normalized_incidence_weights is dense_hyperedge_normalized_adjacency
    )


def test_isolated_node_zero_row() -> None:
    """Isolated nodes receive a zero row/column in ``Ĥ``."""
    # Nodes 0–2 in one hyperedge; node 3 isolated.
    hyperedge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 0]],
        dtype=torch.long,
    )
    hat = dense_hyperedge_normalized_adjacency(
        hyperedge_index,
        num_nodes=4,
        dtype=torch.float64,
    )
    assert torch.allclose(hat[3], torch.zeros(4, dtype=torch.float64))
    assert torch.allclose(hat[:, 3], torch.zeros(4, dtype=torch.float64))
    # Covered nodes keep positive self-coupling from the singleton group.
    assert hat[0, 0] > 0


def test_singleton_hyperedge_retained() -> None:
    """Singleton hyperedges contribute (degree 1), not zero rows."""
    hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long)
    hat = dense_hyperedge_normalized_adjacency(
        hyperedge_index,
        num_nodes=2,
        dtype=torch.float64,
    )
    assert hat[0, 0] == pytest.approx(1.0)
    assert hat[1, 1] == pytest.approx(0.0)


def test_normalization_rejects_invalid_inputs() -> None:
    """Normalization validates shape, node count, and weight length."""
    with pytest.raises(ValueError, match="num_nodes"):
        dense_hyperedge_normalized_adjacency(
            torch.zeros(2, 0, dtype=torch.long),
            num_nodes=-1,
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="shape \\(2, nnz\\)"):
        dense_hyperedge_normalized_adjacency(
            torch.zeros(3, 2, dtype=torch.long),
            num_nodes=2,
            dtype=torch.float32,
        )
    empty = dense_hyperedge_normalized_adjacency(
        torch.zeros(2, 0, dtype=torch.long),
        num_nodes=3,
        dtype=torch.float32,
    )
    assert empty.shape == (3, 3)
    assert torch.equal(empty, torch.zeros(3, 3))
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="hyperedge_weight"):
        dense_hyperedge_normalized_adjacency(
            hyperedge_index,
            num_nodes=2,
            hyperedge_weight=torch.tensor([1.0, 2.0]),
            dtype=torch.float32,
        )


def test_from_arrays_hyperedge_weight_requires_index(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """from_arrays rejects hyperedge_weight without hyperedge_index."""
    with pytest.raises(ValueError, match="hyperedge_weight requires"):
        GraphSnapshotSequence.from_arrays(
            torch.randn(2, 4, 2),
            synthetic_hypergraph_edge_index,
            hyperedge_weight=torch.tensor([1.0]),
        )


def test_weighted_normalization_matches_numpy() -> None:
    """Hyperedge weights enter the Zhou formula."""
    hyperedge_index = torch.tensor(
        [[0, 1, 1, 2], [0, 0, 1, 1]],
        dtype=torch.long,
    )
    weights = torch.tensor([2.0, 0.5], dtype=torch.float64)
    hat = dense_hyperedge_normalized_adjacency(
        hyperedge_index,
        num_nodes=3,
        hyperedge_weight=weights,
        dtype=torch.float64,
    )
    expected = _numpy_zhou_hat(
        hyperedge_index.numpy(),
        num_nodes=3,
        hyperedge_weight=weights.numpy(),
    )
    assert hat.numpy() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_snapshot_to_device_preserves_hyperedges(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Device transfer keeps hyperedge fields."""
    snapshot = Data(
        x=torch.randn(4, 2),
        edge_index=synthetic_hypergraph_edge_index,
        hyperedge_index=synthetic_hyperedge_index,
        hyperedge_weight=torch.tensor([1.0, 2.0]),
    )
    moved = snapshot_to_device(snapshot, torch.device("cpu"))
    assert torch.equal(moved.hyperedge_index, synthetic_hyperedge_index)
    assert torch.equal(moved.hyperedge_weight, snapshot.hyperedge_weight)


def test_require_no_hyperedges_guard(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
    make_snapshots,
) -> None:
    """Shared guard rejects hyperedge sequences and accepts plain ones."""
    require_no_hyperedges(GraphSnapshotSequence(make_snapshots(num_timesteps=2)))
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=2,
    )
    with pytest.raises(ValueError, match="hyperedge-carrying"):
        require_no_hyperedges(sequence)


def test_baseline_rejects_hyperedges(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Classical baselines reject hyperedge-carrying sequences at fit."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=4,
    )
    with pytest.raises(ValueError, match="hyperedge-carrying"):
        DMDBaseline().fit(sequence)


def test_model_fit_rejects_hyperedges_for_gnn_peers(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Non-hypergraph encoder/decoder models still reject hyperedge sequences."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=4,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=8),
        decoder=GNNDecoder(latent_dim=8, hidden_channels=8, out_channels=3),
        latent_dim=8,
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="hyperedge-carrying"):
        model.fit(sequence, epochs=1)


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_hypergraph_encoder_decoder_shapes(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
    activation: str,
) -> None:
    """Encode→decode round-trip preserves node/feature shapes."""
    in_channels = 3
    latent_dim = 5
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=2,
        in_channels=in_channels,
    )
    snapshot = sequence[0]
    encoder = HypergraphEncoder(
        in_channels=in_channels,
        hidden_channels=8,
        latent_dim=latent_dim,
        activation=activation,  # type: ignore[arg-type]
    )
    decoder = HypergraphDecoder(
        latent_dim=latent_dim,
        hidden_channels=8,
        out_channels=in_channels,
        activation=activation,  # type: ignore[arg-type]
    )
    z = encoder(snapshot)
    assert z.shape == (snapshot.num_nodes, latent_dim)
    reconstructed = decoder(z, sequence.hyperedge_index, sequence.hyperedge_weight)
    assert reconstructed.shape == (snapshot.num_nodes, in_channels)

    z_tensor = encoder(
        snapshot.x,
        sequence.hyperedge_index,
        sequence.hyperedge_weight,
    )
    assert z_tensor.shape == z.shape


def test_hypergraph_gradient_flow(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Gradients flow through hypergraph encode→decode."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=2,
        in_channels=3,
    )
    encoder = HypergraphEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = HypergraphDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    z = encoder(sequence[0])
    out = decoder(z, sequence.hyperedge_index, sequence.hyperedge_weight)
    out.sum().backward()
    for module in (encoder, decoder):
        for param in module.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_hypergraph_activation_validation() -> None:
    """Invalid activation identifiers raise like GNN peers."""
    with pytest.raises(ValueError, match="activation"):
        HypergraphEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=4,
            activation="swish",  # type: ignore[arg-type]
        )


def test_hypergraph_exported_from_package() -> None:
    """Hypergraph peers are root-stable and nn-exported."""
    assert "HypergraphEncoder" in koopman_graph.__all__
    assert "HypergraphDecoder" in koopman_graph.__all__
    assert koopman_graph.HypergraphEncoder is HypergraphEncoder
    assert koopman_graph.HypergraphDecoder is HypergraphDecoder


def test_hypergraph_model_fit_predict_smoke(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Fit/predict with hypergraph peers and a per-node operator."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=5,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(in_channels=3, hidden_channels=8, latent_dim=8),
        decoder=HypergraphDecoder(latent_dim=8, hidden_channels=8, out_channels=3),
        latent_dim=8,
        time_step=1.0,
        koopman="pernode",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    preds = model.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert preds[0].x.shape == (4, 3)


def test_hypergraph_mismatched_peers_raise(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Mixed hypergraph/GNN peers are rejected at fit."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=3,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(in_channels=3, hidden_channels=8, latent_dim=8),
        decoder=GNNDecoder(latent_dim=8, hidden_channels=8, out_channels=3),
        latent_dim=8,
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="must be used together"):
        model.fit(sequence, epochs=1)


def test_hypergraph_checkpoint_round_trip(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
    tmp_path,
) -> None:
    """Format-1 checkpoints round-trip hypergraph encoder/decoder peers."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=3,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(in_channels=3, hidden_channels=8, latent_dim=8),
        decoder=HypergraphDecoder(latent_dim=8, hidden_channels=8, out_channels=3),
        latent_dim=8,
        time_step=1.0,
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["encoder"]["type"] == "hyper_enc"
    assert config["decoder"]["type"] == "hyper_dec"
    path = tmp_path / "hyper_peers.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, HypergraphEncoder)
    assert isinstance(loaded.decoder, HypergraphDecoder)


def test_env_rejects_hyperedges(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Gymnasium env rejects hyperedge-carrying reference sequences."""
    gymnasium = pytest.importorskip("gymnasium")
    del gymnasium
    from koopman_graph.env import GraphKoopmanEnv

    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=4,
        in_channels=3,
    )
    # Build a controlled model on a plain sequence, then try env construction.
    plain = GraphSnapshotSequence.from_arrays(
        torch.randn(4, 4, 3),
        synthetic_hypergraph_edge_index,
        control_inputs=torch.randn(4, 1),
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=8),
        decoder=GNNDecoder(latent_dim=8, hidden_channels=8, out_channels=3),
        latent_dim=8,
        time_step=1.0,
        control_dim=1,
    )
    model.fit(plain, epochs=1)

    def _reward(_snapshot: Data, _step: int) -> float:
        return 0.0

    with pytest.raises(ValueError, match="hyperedge-carrying"):
        GraphKoopmanEnv(
            model=model,
            reference_sequence=sequence,
            reward_fn=_reward,
        )


def _undirected_path_hyperedges(
    num_nodes: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pairwise path ``edge_index`` and matching 2-uniform hyperedges."""
    edges: list[list[int]] = []
    hedges_n: list[int] = []
    hedges_e: list[int] = []
    for i, h in enumerate(range(num_nodes - 1)):
        edges.extend([[i, i + 1], [i + 1, i]])
        hedges_n.extend([i, i + 1])
        hedges_e.extend([h, h])
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    hyperedge_index = torch.tensor([hedges_n, hedges_e], dtype=torch.long)
    return edge_index, hyperedge_index


def test_hypergraph_operator_factory_and_continuous_reject() -> None:
    """Factory builds hypergraph kind and rejects continuous mode."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    model = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4),
        decoder=GNNDecoder(4, 8, 3),
        latent_dim=4,
        time_step=1.0,
        koopman="hypergraph",
    )
    assert isinstance(model.koopman, HypergraphKoopmanOperator)
    assert model.uses_hypergraph_koopman
    with pytest.raises(ValueError, match="hypergraph"):
        GraphKoopmanModel(
            encoder=GNNEncoder(3, 8, 4),
            decoder=GNNDecoder(4, 8, 3),
            latent_dim=4,
            time_step=1.0,
            dynamics_mode="continuous",
            koopman="hypergraph",
        )


def test_hypergraph_operator_requires_hyperedge_on_advance() -> None:
    """Advance without hyperedge_index raises."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=3, init_mode="identity")
    z = torch.randn(4, 3)
    with pytest.raises(ValueError, match="hyperedge_index is required"):
        op.advance(z)


def test_hypergraph_operator_forward_matches_effective_matrix(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Sparse-style forward matches dense effective matvec."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(
        latent_dim=3, init_mode="identity_noise", init_scale=0.05
    )
    z = torch.randn(4, 3)
    z_next = op(z, synthetic_hyperedge_index)
    effective = op.effective_matrix(synthetic_hyperedge_index, num_nodes=4)
    expected = (effective @ z.reshape(-1)).view_as(z)
    assert torch.allclose(z_next, expected, atol=1e-5)


def test_hypergraph_operator_spectrum_numpy_reference(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Effective-matrix eigenvalues match a NumPy Kronecker reference."""
    from koopman_graph.graph_utils import dense_hyperedge_normalized_adjacency
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=2, init_mode="identity")
    k_self = torch.diag(torch.tensor([0.5, 0.8]))
    k_hedge = torch.diag(torch.tensor([0.1, -0.2]))
    op.set_dense_matrices(k_self, k_hedge)
    hat = dense_hyperedge_normalized_adjacency(
        synthetic_hyperedge_index,
        num_nodes=4,
        dtype=torch.float32,
    )
    eye = torch.eye(4)
    ref = torch.kron(eye, k_self) + torch.kron(hat, k_hedge)
    eff = op.effective_matrix(synthetic_hyperedge_index, num_nodes=4)
    assert torch.allclose(eff, ref, atol=1e-6)
    spectrum = op.spectrum(synthetic_hyperedge_index, num_nodes=4)
    eigvals = torch.linalg.eigvals(ref)
    assert torch.allclose(
        spectrum.eigenvalues.abs().sort().values,
        eigvals.abs().sort().values,
        atol=1e-5,
    )


def test_two_uniform_equivalence_vs_graph_operator() -> None:
    """2-uniform Zhou Ĥ relates to Â by Ĥ=½(I+Â); matching factors agree."""
    from koopman_graph.graph_utils import (
        dense_hyperedge_normalized_adjacency,
        dense_symmetric_normalized_adjacency,
    )
    from koopman_graph.operators import GraphKoopmanOperator, HypergraphKoopmanOperator

    edge_index, hyperedge_index = _undirected_path_hyperedges(4)
    k_self_h = torch.diag(torch.tensor([0.7, 0.4, -0.1]))
    k_hedge = torch.diag(torch.tensor([0.2, -0.3, 0.15]))
    # Ĥ = ½(I + Â) ⇒ K_self_g = K_self_h + ½ K_hedge, K_nbr = ½ K_hedge
    k_self_g = k_self_h + 0.5 * k_hedge
    k_nbr = 0.5 * k_hedge

    hyp = HypergraphKoopmanOperator(latent_dim=3, init_mode="identity")
    graph = GraphKoopmanOperator(latent_dim=3, init_mode="identity")
    hyp.set_dense_matrices(k_self_h, k_hedge)
    graph.set_dense_matrices(k_self_g, k_nbr)

    hat = dense_hyperedge_normalized_adjacency(
        hyperedge_index, num_nodes=4, dtype=torch.float32
    )
    adj = dense_symmetric_normalized_adjacency(edge_index, 4, dtype=torch.float32)
    assert torch.allclose(hat, 0.5 * (torch.eye(4) + adj), atol=1e-5)

    z = torch.randn(4, 3)
    assert torch.allclose(
        hyp(z, hyperedge_index),
        graph(z, edge_index),
        atol=1e-5,
    )
    assert torch.allclose(
        hyp.effective_matrix(hyperedge_index, 4),
        graph.effective_matrix(edge_index, 4),
        atol=1e-5,
    )


def test_model_spectrum_requires_hyperedge_topology() -> None:
    """Model spectrum with koopman=hypergraph requires hyperedge_index."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4),
        decoder=GNNDecoder(4, 8, 3),
        latent_dim=4,
        time_step=1.0,
        koopman="hypergraph",
    )
    with pytest.raises(ValueError, match="hyperedge_index"):
        model.spectrum(num_nodes=4)
    edge_index, hyperedge_index = _undirected_path_hyperedges(4)
    spectrum = model.spectrum(hyperedge_index=hyperedge_index, num_nodes=4)
    assert spectrum.eigenvalues.numel() == 4 * 4


def test_hypergraph_operator_model_fit_smoke(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Fit/predict with hypergraph encoder/decoder/operator on hyperedges."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=5,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(3, 8, 8),
        decoder=HypergraphDecoder(8, 8, 3),
        latent_dim=8,
        time_step=1.0,
        koopman="hypergraph",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    preds = model.predict(sequence[0], steps=2)
    assert len(preds) == 2


def test_hypergraph_operator_checkpoint_round_trip(
    synthetic_hyperedge_index: torch.Tensor,
    synthetic_hypergraph_edge_index: torch.Tensor,
    tmp_path,
) -> None:
    """Format-1 checkpoints round-trip hypergraph operators."""
    sequence = _hyper_sequence(
        synthetic_hyperedge_index,
        synthetic_hypergraph_edge_index,
        num_timesteps=3,
        in_channels=3,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(3, 8, 4),
        decoder=HypergraphDecoder(4, 8, 3),
        latent_dim=4,
        time_step=1.0,
        koopman="hypergraph",
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["koopman_kind"] == "hypergraph"
    path = tmp_path / "hyper_op.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "hypergraph"


def test_hypergraph_k_hedge_zero_matches_pernode() -> None:
    """K_hedge=0 recovers the per-node Koopman map exactly."""
    from koopman_graph.operators import HypergraphKoopmanOperator, KoopmanOperator

    torch.manual_seed(0)
    hyperedge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2, 3], [0, 0, 0, 1, 1, 1, 2]],
        dtype=torch.long,
    )
    pernode = KoopmanOperator(3, init_mode="xavier")
    hyp = HypergraphKoopmanOperator(3, init_mode="identity")
    hyp.set_dense_matrices(pernode.K.detach().clone(), torch.zeros_like(pernode.K))
    z = torch.randn(4, 3)
    assert torch.allclose(hyp(z, hyperedge_index), pernode(z), atol=1e-6)


def test_hypergraph_inverse_advance_roundtrip(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """inverse_advance recovers latents for a well-conditioned map."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=2, init_mode="identity")
    op.set_dense_matrices(
        torch.diag(torch.tensor([0.6, 0.8])),
        torch.diag(torch.tensor([0.05, -0.05])),
    )
    z = torch.randn(4, 2)
    z_next = op.advance(z, hyperedge_index=synthetic_hyperedge_index)
    recovered = op.inverse_advance(z_next, hyperedge_index=synthetic_hyperedge_index)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_hypergraph_bound_metric_is_factor_level(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """bound_metric monitors factors, not the topology-coupled spectrum."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=2, init_mode="identity")
    op.set_dense_matrices(0.9 * torch.eye(2), 2.0 * torch.eye(2))
    factor_bound = op.bound_metric()
    assert factor_bound.ndim == 0
    # Effective spectral radius can exceed the factor-level surrogate.
    spectrum = op.spectrum(synthetic_hyperedge_index, num_nodes=4)
    assert spectrum.magnitudes.max() >= factor_bound - 1e-5


def test_eigenvalue_loss_hypergraph_dense_requires_topology() -> None:
    """Hypergraph dense eig-reg never falls back to K_self alone."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(2, init_mode="identity")
    loss_fn = EigenvalueRegularizationLoss()
    with pytest.raises(ValueError, match="hyperedge_index"):
        loss_fn(op)


def test_eigenvalue_loss_hypergraph_coupling_affects_penalty(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Larger K_hedge increases the effective-operator hinge."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import HypergraphKoopmanOperator

    k_self = 0.5 * torch.eye(2)
    loss_fn = EigenvalueRegularizationLoss()
    mild = HypergraphKoopmanOperator(2, init_mode="identity")
    mild.set_dense_matrices(k_self, 0.1 * torch.eye(2))
    strong = HypergraphKoopmanOperator(2, init_mode="identity")
    strong.set_dense_matrices(k_self.clone(), 2.0 * torch.eye(2))
    mild_loss = loss_fn(mild, hyperedge_index=synthetic_hyperedge_index, num_nodes=4)
    strong_loss = loss_fn(
        strong, hyperedge_index=synthetic_hyperedge_index, num_nodes=4
    )
    assert mild_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert strong_loss.item() > mild_loss.item()


def test_eigenvalue_loss_hypergraph_structural_uses_factor_bound() -> None:
    """Hypergraph structural modes keep factor bound_metric without topology."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(3, parameterization="dissipative")
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(op).item() == pytest.approx(0.0, abs=1e-8)


def test_hypergraph_operator_rejects_non_dense_sparsity() -> None:
    """Only sparsity='dense' is implemented for hypergraph operators."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    with pytest.raises(ValueError, match="sparsity"):
        HypergraphKoopmanOperator(2, sparsity="block_diagonal")  # type: ignore[arg-type]


def test_hypergraph_operator_factorized_reset_and_aliases(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Factorized init, matrix aliases, and reset_parameters stay consistent."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity_noise",
        init_scale=0.01,
        parameterization="dissipative",
    )
    assert torch.equal(op.matrix, op.K_self)
    assert torch.equal(op.K, op.K_self)
    assert op.spectral_radius().ndim == 0
    assert op.stability_certificate() is not None
    op.reset_parameters()
    z = torch.randn(4, 2)
    _ = op.advance(z, hyperedge_index=synthetic_hyperedge_index)


def test_hypergraph_operator_forward_validation_and_control(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Forward rejects bad shapes and enforces control presence rules."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=2, init_mode="identity")
    with pytest.raises(ValueError, match="num_nodes, latent_dim"):
        op(torch.randn(2, 2, 2), synthetic_hyperedge_index)
    with pytest.raises(ValueError, match="trailing dimension"):
        op(torch.randn(4, 3), synthetic_hyperedge_index)
    with pytest.raises(ValueError, match="uncontrolled"):
        op(torch.randn(4, 2), synthetic_hyperedge_index, control=torch.ones(1))

    controlled = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity",
        control_dim=1,
        control_mode="additive",
    )
    with pytest.raises(ValueError, match="control input is required"):
        controlled(torch.randn(4, 2), synthetic_hyperedge_index)
    z_next = controlled(
        torch.randn(4, 2),
        synthetic_hyperedge_index,
        control=torch.tensor([0.1]),
    )
    assert z_next.shape == (4, 2)

    bilinear = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity",
        control_dim=1,
        control_mode="bilinear",
        bilinear_rank=1,
    )
    z = torch.randn(4, 2)
    control = torch.tensor([0.2])
    advanced = bilinear.advance(
        z,
        hyperedge_index=synthetic_hyperedge_index,
        control=control,
    )
    recovered = bilinear.inverse_advance(
        advanced,
        hyperedge_index=synthetic_hyperedge_index,
        control=control,
    )
    assert torch.allclose(recovered, z, atol=1e-4)


def test_hypergraph_effective_matrix_argument_guards(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """effective_matrix rejects conflicting / misshapen self overrides."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    op = HypergraphKoopmanOperator(latent_dim=2, init_mode="identity")
    bad_blocks = torch.zeros(3, 2, 2)
    with pytest.raises(ValueError, match="at most one"):
        op.effective_matrix(
            synthetic_hyperedge_index,
            num_nodes=4,
            k_self=torch.eye(2),
            k_self_blocks=bad_blocks,
        )
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        op.effective_matrix(
            synthetic_hyperedge_index,
            num_nodes=4,
            k_self_blocks=bad_blocks,
        )


def test_hypergraph_orbit_tied_set_dense_and_inverse_guards(
    synthetic_hyperedge_index: torch.Tensor,
) -> None:
    """Orbit-tied writeback and inverse_advance validation paths."""
    from koopman_graph.operators import HypergraphKoopmanOperator

    partition = ((0, 1), (2, 3))
    op = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity",
        orbit_partition=partition,
    )
    k_self = torch.diag(torch.tensor([0.7, 0.4]))
    k_hedge = 0.05 * torch.eye(2)
    op.set_dense_matrices(k_self, k_hedge)
    z = torch.randn(4, 2)
    z_next = op.advance(z, hyperedge_index=synthetic_hyperedge_index)
    assert z_next.shape == (4, 2)

    with pytest.raises(ValueError, match="hyperedge_index is required"):
        op.inverse_advance(z_next)
    with pytest.raises(ValueError, match="expects z with shape"):
        op.inverse_advance(
            torch.randn(4, 3),
            hyperedge_index=synthetic_hyperedge_index,
        )

    controlled = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity",
        control_dim=1,
    )
    with pytest.raises(ValueError, match="control input is required"):
        controlled.inverse_advance(
            z_next,
            hyperedge_index=synthetic_hyperedge_index,
        )
    control = torch.tensor([0.05])
    advanced = controlled.advance(
        z,
        hyperedge_index=synthetic_hyperedge_index,
        control=control,
    )
    recovered = controlled.inverse_advance(
        advanced,
        hyperedge_index=synthetic_hyperedge_index,
        control=control,
    )
    assert torch.allclose(recovered, z, atol=1e-4)

    bilinear = HypergraphKoopmanOperator(
        latent_dim=2,
        init_mode="identity",
        control_dim=1,
        control_mode="bilinear",
        bilinear_rank=1,
    )
    per_node = torch.full((4, 1), 0.05)
    advanced_pn = bilinear.advance(
        z,
        hyperedge_index=synthetic_hyperedge_index,
        control=per_node,
    )
    recovered_pn = bilinear.inverse_advance(
        advanced_pn,
        hyperedge_index=synthetic_hyperedge_index,
        control=per_node,
    )
    assert torch.allclose(recovered_pn, z, atol=1e-4)
    with pytest.raises(ValueError, match="Per-node control"):
        bilinear.inverse_advance(
            advanced_pn,
            hyperedge_index=synthetic_hyperedge_index,
            control=torch.zeros(3, 1),
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        bilinear.inverse_advance(
            advanced_pn,
            hyperedge_index=synthetic_hyperedge_index,
            control=torch.zeros(2, 2, 1),
        )
