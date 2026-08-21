"""Tests for GraphKoopmanOperator and networked model integration."""

from __future__ import annotations

import importlib.util
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
from koopman_graph.spectrum_types import compute_spectrum

_HAS_PYNAUTY = importlib.util.find_spec("pynauty") is not None


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected path graph edge index."""
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected ring graph edge index."""
    edges: list[list[int]] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        edges.extend([[node, nxt], [nxt, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _star_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected star with center 0 and leaves 1..N-1."""
    edges: list[list[int]] = []
    for leaf in range(1, num_nodes):
        edges.extend([[0, leaf], [leaf, 0]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _eigvals_match(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Greedy multiset match of complex eigenvalues (order-invariant)."""
    if left.shape != right.shape:
        return False
    remaining = right.detach().clone()
    for value in left.detach():
        diffs = (remaining - value).abs()
        index = int(torch.argmin(diffs))
        if not torch.isclose(value, remaining[index], rtol=rtol, atol=atol):
            return False
        remaining[index] = complex(float("inf"), float("inf"))
    return True


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
    """Sparse message-passing forward matches dense Kronecker matrix application."""
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


def test_spectrum_eligible_default_uses_kronecker_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default shared-self symmetric spectrum routes through Kronecker helper."""
    from koopman_graph.operators import graph as graph_mod

    calls: list[dict[str, object]] = []
    original = graph_mod.spectrum_k_eff_kronecker_polynomial

    def _spy(**kwargs: object):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _spy)
    edge_index = _path_edge_index(3)
    op = GraphKoopmanOperator(2, init_mode="identity")
    spectrum = op.spectrum(edge_index, 3, time_step=0.25)
    assert len(calls) == 1
    assert spectrum.eigenvalues.shape == (6,)
    assert spectrum.eigenvectors.shape == (6, 6)


def test_spectrum_ineligible_paths_skip_kronecker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual, orbit banks, and distributed do not call the Kronecker helper."""
    from koopman_graph.operators import graph as graph_mod

    def _boom(**_kwargs: object):
        msg = "kronecker helper should not run on ineligible spectrum paths"
        raise AssertionError(msg)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _boom)
    edge_index = _path_edge_index(3)
    time_step = 0.1

    dual = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    dual_spec = dual.spectrum(edge_index, 3, time_step=time_step)
    dual_oracle = compute_spectrum(
        dual.effective_matrix(edge_index, 3),
        time_step,
    )
    assert dual_spec.eigenvalues.shape == (6,)
    assert _eigvals_match(
        dual_spec.eigenvalues,
        dual_oracle.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )

    orbit = GraphKoopmanOperator(
        2,
        init_mode="identity",
        orbit_partition=((0, 1), (2,)),
    )
    orbit_spec = orbit.spectrum(edge_index, 3, time_step=time_step)
    orbit_oracle = compute_spectrum(
        orbit.effective_matrix(edge_index, 3),
        time_step,
    )
    assert orbit_spec.eigenvalues.shape == (6,)
    assert _eigvals_match(
        orbit_spec.eigenvalues,
        orbit_oracle.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )

    distributed = GraphKoopmanOperator(2, init_mode="identity", sparsity="distributed")
    distributed.set_dense_matrices(
        torch.tensor([[0.45, 0.20], [0.05, 0.35]]),
        torch.tensor([[0.25, 0.10], [0.15, 0.05]]),
    )
    dist_spec = distributed.spectrum(edge_index, 3, num_modes=2)
    assert dist_spec.eigenvalues.shape == (2,)
    assert dist_spec.eigenvectors.shape == (2, 2)


@pytest.mark.skipif(not _HAS_PYNAUTY, reason="pynauty not installed")
def test_spectrum_isotypic_skips_kronecker_and_matches_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isotypic self banks dense-route (shared_self False) and match oracle."""
    from koopman_graph.operators import graph as graph_mod

    def _boom(**_kwargs: object):
        msg = "kronecker helper should not run for isotypic self banks"
        raise AssertionError(msg)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _boom)
    star = _star_edge_index(5)
    time_step = 0.1
    op = GraphKoopmanOperator(2, init_mode="identity", isotypic_symmetry=True)
    got = op.spectrum(star, 5, time_step=time_step)
    oracle = compute_spectrum(op.effective_matrix(star, 5), time_step)
    assert op.uses_orbit_selves
    assert got.eigenvalues.shape == (10,)
    assert _eigvals_match(
        got.eigenvalues,
        oracle.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("topology", ["path", "ring"])
@pytest.mark.parametrize("adjacency", ["symmetric", "random_walk"])
@pytest.mark.parametrize(("num_nodes", "latent_dim"), [(4, 2), (6, 3)])
def test_operator_spectrum_parity_vs_dense_oracle(
    topology: str,
    adjacency: str,
    num_nodes: int,
    latent_dim: int,
) -> None:
    """GraphKoopmanOperator.spectrum matches dense compute_spectrum (small N).

    Tolerance rtol=atol=1e-5 justified by float32 eig accumulation on N·d ≤ 18
    (same contract as helper-level Kronecker parity).
    """
    torch.manual_seed(3)
    time_step = 0.1
    edge_index = (
        _path_edge_index(num_nodes)
        if topology == "path"
        else _ring_edge_index(num_nodes)
    )
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity_noise",
        init_scale=0.05,
        adjacency=adjacency,  # type: ignore[arg-type]
    )
    got = op.spectrum(edge_index, num_nodes, time_step=time_step)
    oracle = compute_spectrum(
        op.effective_matrix(edge_index, num_nodes),
        time_step,
    )
    ambient = num_nodes * latent_dim
    assert got.eigenvalues.shape == (ambient,)
    assert got.eigenvectors.shape == (ambient, ambient)
    assert got.time_step == time_step
    assert _eigvals_match(
        got.eigenvalues,
        oracle.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )


def test_operator_spectrum_block_diagonal_uses_kronecker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible block_diagonal sparsity still routes through Kronecker."""
    from koopman_graph.operators import graph as graph_mod

    calls: list[int] = []
    original = graph_mod.spectrum_k_eff_kronecker_polynomial

    def _spy(**kwargs: object):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _spy)
    edge_index = _path_edge_index(4)
    op = GraphKoopmanOperator(2, init_mode="identity", sparsity="block_diagonal")
    spectrum = op.spectrum(edge_index, 4, time_step=0.1)
    assert len(calls) == 1
    oracle = compute_spectrum(op.effective_matrix(edge_index, 4), 0.1)
    assert _eigvals_match(
        spectrum.eigenvalues,
        oracle.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )


def test_operator_spectrum_moderate_n_kronecker_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moderate-N eligible spectrum finishes via Kronecker (no tight dense oracle).

    N=100, d=2 → ambient 200. R7: assert finite shapes only; skip dense eigvals.
    """
    from koopman_graph.operators import graph as graph_mod

    calls: list[int] = []
    original = graph_mod.spectrum_k_eff_kronecker_polynomial

    def _spy(**kwargs: object):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _spy)
    num_nodes = 100
    latent_dim = 2
    ambient = num_nodes * latent_dim
    edge_index = _path_edge_index(num_nodes)
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity_noise",
        init_scale=0.05,
        adjacency="symmetric",
    )
    spectrum = op.spectrum(edge_index, num_nodes, time_step=1.0)
    assert len(calls) == 1
    assert spectrum.eigenvalues.shape == (ambient,)
    assert spectrum.eigenvectors.shape == (ambient, ambient)
    assert bool(torch.isfinite(spectrum.eigenvalues).all())
    assert bool(torch.isfinite(spectrum.eigenvectors).all())
    assert bool(torch.isfinite(spectrum.magnitudes).all())


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


def test_graph_in_continuous_mode_builds_continuous_graph() -> None:
    """Factory koopman='graph' + continuous builds ContinuousGraphKoopmanOperator."""
    from koopman_graph import ContinuousGraphKoopmanOperator

    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman="graph",
    )
    assert isinstance(model.koopman, ContinuousGraphKoopmanOperator)
    assert model.koopman_kind == "continuous_graph"
    assert model.uses_continuous_graph_koopman


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
    """Unknown sparsity strings are rejected with a clear error."""
    with pytest.raises(ValueError, match="must be 'dense'"):
        GraphKoopmanOperator(2, sparsity="bogus")  # type: ignore[arg-type]


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
    assert op.factor_stability_certificate() is None

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


def test_block_diagonal_constructs_and_matches_dense_advance() -> None:
    """block_diagonal advances identically to dense (same forward math)."""
    torch.manual_seed(0)
    edge_index = _path_edge_index(5)
    dense = GraphKoopmanOperator(3, init_mode="identity_noise", init_scale=0.05)
    block = GraphKoopmanOperator(3, init_mode="identity", sparsity="block_diagonal")
    block.set_dense_matrices(
        dense.K_self.detach().clone(), dense.K_nbr.detach().clone()
    )
    z = torch.randn(5, 3)
    assert torch.allclose(block(z, edge_index), dense(z, edge_index), atol=1e-6)


def test_block_diagonal_inverse_exact_when_decoupled() -> None:
    """Blockwise inverse is exact for empty edges or zero K_nbr."""
    k_self = torch.diag(torch.tensor([0.6, 0.8, -0.2]))
    z = torch.randn(4, 3)

    empty = GraphKoopmanOperator(3, init_mode="identity", sparsity="block_diagonal")
    empty.set_dense_matrices(k_self, 0.3 * torch.eye(3))
    empty_edges = torch.empty(2, 0, dtype=torch.long)
    z_next = empty.advance(z, edge_index=empty_edges)
    recovered = empty.inverse_advance(z_next, edge_index=empty_edges)
    assert torch.allclose(recovered, z, atol=1e-5)

    zero_nbr = GraphKoopmanOperator(3, init_mode="identity", sparsity="block_diagonal")
    zero_nbr.set_dense_matrices(k_self, torch.zeros(3, 3))
    edges = _path_edge_index(4)
    z_next = zero_nbr.advance(z, edge_index=edges)
    recovered = zero_nbr.inverse_advance(z_next, edge_index=edges)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_block_diagonal_inverse_approximation_bounded_vs_dense() -> None:
    """Coupled Jacobi inverse stays near the dense inverse on a small graph."""
    torch.manual_seed(1)
    edge_index = _path_edge_index(4)
    k_self = torch.diag(torch.tensor([0.7, 0.5]))
    k_nbr = 0.15 * torch.eye(2)
    dense = GraphKoopmanOperator(2, init_mode="identity")
    block = GraphKoopmanOperator(2, init_mode="identity", sparsity="block_diagonal")
    dense.set_dense_matrices(k_self, k_nbr)
    block.set_dense_matrices(k_self.clone(), k_nbr.clone())
    z = torch.randn(4, 2)
    z_next = dense.advance(z, edge_index=edge_index)
    dense_rec = dense.inverse_advance(z_next, edge_index=edge_index)
    block_rec = block.inverse_advance(z_next, edge_index=edge_index)
    # Exact dense recovery; Jacobi approximation error is moderate for mild K_nbr.
    assert torch.allclose(dense_rec, z, atol=1e-5)
    err = (block_rec - z).norm() / z.norm()
    assert err.item() < 0.25


def test_block_diagonal_inverse_large_n_smoke() -> None:
    """N>=2000 inverse_advance avoids materializing the Nd×Nd effective matrix."""
    torch.manual_seed(2)
    num_nodes = 2000
    latent_dim = 4
    # Path graph: O(N) edges.
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    op = GraphKoopmanOperator(
        latent_dim, init_mode="identity", sparsity="block_diagonal"
    )
    op.set_dense_matrices(0.8 * torch.eye(latent_dim), 0.05 * torch.eye(latent_dim))
    z = torch.randn(num_nodes, latent_dim)
    z_next = op.advance(z, edge_index=edge_index)
    recovered = op.inverse_advance(z_next, edge_index=edge_index)
    assert recovered.shape == z.shape
    assert torch.isfinite(recovered).all()
    # Dense effective would be (8000, 8000) ≈ 256MB float32; block path stays O(Nd).
    assert recovered.numel() * recovered.element_size() < 1_000_000


def test_distributed_sparsity_constructs() -> None:
    """sparsity='distributed' constructs and keeps the sparse forward path."""
    op = GraphKoopmanOperator(2, init_mode="identity", sparsity="distributed")
    assert op.sparsity == "distributed"
    edge_index = _path_edge_index(3)
    z = torch.randn(3, 2)
    advanced = op.advance(z, edge_index=edge_index)
    assert advanced.shape == z.shape


def test_distributed_checkpoint_round_trip(tmp_path) -> None:
    """Format-1 checkpoints round-trip koopman_sparsity=distributed."""
    from pathlib import Path

    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
        koopman_sparsity="distributed",
    )
    assert model.koopman.sparsity == "distributed"
    path = Path(tmp_path) / "dist.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.sparsity == "distributed"


def test_distributed_spectrum_and_inverse_match_dense_moduli() -> None:
    """Distributed spectrum / inverse_advance agree with dense on modest N·d."""
    num_nodes = 4
    latent_dim = 2
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]],
        dtype=torch.long,
    )
    dense = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency="random_walk",
        sparsity="dense",
    )
    distributed = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency="random_walk",
        sparsity="distributed",
    )
    k_self = torch.tensor([[0.45, 0.20], [0.05, 0.35]])
    k_nbr = torch.tensor([[0.25, 0.10], [0.15, 0.05]])
    dense.set_dense_matrices(k_self, k_nbr)
    distributed.set_dense_matrices(k_self, k_nbr)

    num_modes = 3
    dense_spec = dense.spectrum(edge_index, num_nodes)
    dist_spec = distributed.spectrum(edge_index, num_nodes, num_modes=num_modes)
    assert dist_spec.eigenvalues.shape == (num_modes,)
    dense_abs = dense_spec.eigenvalues.abs()[:num_modes]
    assert torch.allclose(dist_spec.eigenvalues.abs(), dense_abs, atol=1e-4)

    z = torch.randn(num_nodes, latent_dim)
    y = dense.advance(z, edge_index=edge_index)
    recovered = distributed.inverse_advance(y, edge_index=edge_index)
    assert torch.allclose(recovered, z, atol=1e-4)


def test_block_diagonal_rejects_inverse_matrix_kwarg() -> None:
    """Precomputed inverse_matrix is dense-only."""
    op = GraphKoopmanOperator(2, init_mode="identity", sparsity="block_diagonal")
    edge_index = _path_edge_index(3)
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="inverse_matrix"):
        op.inverse_advance(
            z,
            edge_index=edge_index,
            inverse_matrix=torch.eye(6),
        )


def test_block_diagonal_checkpoint_round_trip(tmp_path) -> None:
    """Format-1 checkpoints round-trip koopman_sparsity=block_diagonal."""
    from pathlib import Path

    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
        koopman_sparsity="block_diagonal",
    )
    assert model.koopman.sparsity == "block_diagonal"
    path = Path(tmp_path) / "bd.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.sparsity == "block_diagonal"


def _directed_path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a one-way directed path ``0→1→…→N-1``."""
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_invalid_adjacency_raises() -> None:
    """Invalid ``adjacency`` values name the accepted set."""
    with pytest.raises(ValueError, match="dual_random_walk.*random_walk.*symmetric"):
        GraphKoopmanOperator(2, adjacency="bogus")  # type: ignore[arg-type]


def test_symmetric_adjacency_seeded_regression() -> None:
    """Default ``adjacency='symmetric'`` stays bit-stable vs pre-change fixture."""
    torch.manual_seed(0)
    op = GraphKoopmanOperator(2, init_mode="xavier", init_scale=0.2)
    assert op.adjacency == "symmetric"
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    z = torch.randn(3, 2)
    out = op(z, edge_index)
    expected = torch.tensor(
        [
            [0.7360891103744507, -0.971832811832428],
            [-0.11684554815292358, 0.9164610505104065],
            [0.3142688274383545, 0.627437949180603],
        ]
    )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_dual_equals_random_walk_when_k_bwd_zero() -> None:
    """``K_bwd`` is exactly zero at init so dual matches random_walk."""
    edge_index = _directed_path_edge_index(4)
    z = torch.randn(4, 2)
    rw = GraphKoopmanOperator(2, init_mode="identity", adjacency="random_walk")
    dual = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    assert torch.equal(dual.K_bwd, torch.zeros(2, 2))
    assert torch.allclose(dual.K_fwd, dual.K_nbr)
    k_self = 0.4 * torch.eye(2)
    k_nbr = 0.15 * torch.eye(2)
    rw.set_dense_matrices(k_self, k_nbr)
    dual.set_dense_matrices(k_self, k_nbr)
    assert torch.allclose(rw(z, edge_index), dual(z, edge_index), atol=1e-6)


def test_adjacency_modes_forward_match_effective_matrix() -> None:
    """Sparse forward matches dense Kronecker effective map for all modes."""
    edge_index = _directed_path_edge_index(4)
    z = torch.randn(4, 3)
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        torch.manual_seed(3)
        op = GraphKoopmanOperator(
            3,
            init_mode="xavier",
            init_scale=0.1,
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        if adjacency == "dual_random_walk":
            op.set_dense_matrices(
                op.K_self.detach().clone(),
                op.K_nbr.detach().clone(),
                k_bwd=0.05 * torch.eye(3),
            )
        sparse_next = op(z, edge_index)
        effective = op.effective_matrix(edge_index, 4)
        assert effective.shape == (12, 12)
        dense_next = (effective @ z.reshape(-1)).view_as(z)
        assert torch.allclose(sparse_next, dense_next, atol=1e-5), adjacency


def test_adjacency_modes_dense_inverse_round_trip() -> None:
    """Dense ``inverse_advance`` round-trips for all adjacency modes."""
    edge_index = _directed_path_edge_index(3)
    z = torch.randn(3, 2)
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        op = GraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        op.set_dense_matrices(
            0.6 * torch.eye(2),
            0.2 * torch.eye(2),
            k_bwd=None if adjacency != "dual_random_walk" else 0.1 * torch.eye(2),
        )
        z_next = op(z, edge_index)
        recovered = op.inverse_advance(z_next, edge_index=edge_index)
        assert torch.allclose(recovered, z, atol=1e-5), adjacency


def test_adjacency_modes_block_diagonal_jacobi_documented_approx() -> None:
    """Jacobi inverse is approximate (not exact) when neighbor factors are nonzero."""
    edge_index = _directed_path_edge_index(3)
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        dense = GraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        block = GraphKoopmanOperator(
            2,
            init_mode="identity",
            sparsity="block_diagonal",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        k_bwd = None if adjacency != "dual_random_walk" else 0.15 * torch.eye(2)
        for op in (dense, block):
            op.set_dense_matrices(0.7 * torch.eye(2), 0.25 * torch.eye(2), k_bwd=k_bwd)
        z_next = dense(z, edge_index)
        dense_rec = dense.inverse_advance(z_next, edge_index=edge_index)
        block_rec = block.inverse_advance(z_next, edge_index=edge_index)
        assert torch.allclose(dense_rec, z, atol=1e-5), adjacency
        # Jacobi is a documented approximation for all three modes.
        assert not torch.allclose(block_rec, z, atol=1e-3), adjacency


def test_directed_spectrum_allows_complex_eigenvalues() -> None:
    """Asymmetric coupling may yield complex spectra without real-dtype leaks."""
    edge_index = _directed_path_edge_index(4)
    op = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    op.set_dense_matrices(
        0.3 * torch.eye(2),
        0.4 * torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        k_bwd=0.2 * torch.eye(2),
    )
    spectrum = op.spectrum(edge_index, 4)
    assert spectrum.eigenvalues.dtype.is_complex
    assert spectrum.eigenvalues.shape == (8,)
    assert torch.isfinite(spectrum.eigenvalues.real).all()
    assert torch.isfinite(spectrum.eigenvalues.imag).all()
    assert torch.isfinite(spectrum.magnitudes).all()


def test_random_walk_modes_handle_sink_and_isolated_nodes() -> None:
    """Sink / isolated nodes produce finite advances (no NaN/Inf)."""
    # Nodes 0→1; node 2 isolated; node 1 is a sink.
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    z = torch.randn(3, 2)
    for adjacency in ("random_walk", "dual_random_walk"):
        op = GraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        op.set_dense_matrices(
            torch.eye(2),
            0.5 * torch.eye(2),
            k_bwd=None if adjacency != "dual_random_walk" else 0.25 * torch.eye(2),
        )
        out = op(z, edge_index)
        assert torch.isfinite(out).all(), adjacency
        effective = op.effective_matrix(edge_index, 3)
        assert torch.isfinite(effective).all(), adjacency


def test_orbit_ties_keep_shared_backward_and_tie_forward() -> None:
    """Orbit partitions tie ``K_self`` / ``K_fwd``; ``K_bwd`` stays shared."""
    op = GraphKoopmanOperator(
        2,
        init_mode="identity",
        adjacency="dual_random_walk",
        orbit_partition=((0, 1), (2,)),
    )
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    op.ensure_orbit_binding(3, edge_index=edge_index)
    assert op._orbit_selves is not None
    assert op._orbit_nbrs is not None
    assert len(op._orbit_selves) == 2
    assert len(op._orbit_nbrs) == 2
    assert op._bwd is not None
    assert op._orbit_selves[0] is not op._orbit_selves[1]
    assert op._nbr is op._orbit_nbrs[0]
    assert op._nbr is not op._orbit_nbrs[1]
    assert op._bwd is not op._orbit_selves[0]
    assert op._bwd is not op._orbit_nbrs[0]
    blocks = op.tied_self_blocks(3)
    assert torch.allclose(blocks[0], blocks[1])
    assert torch.allclose(op.K_fwd, op.K_nbr)
    with pytest.raises(AttributeError, match="K_bwd"):
        _ = GraphKoopmanOperator(2, adjacency="symmetric").K_bwd


def test_dual_random_walk_learns_directed_advection_better_than_symmetric() -> None:
    """Dual recovers a one-way advection teacher with a documented error gap.

    Tolerance contract: after a fixed-seed short Adam fit, dual one-step MSE
    is at most half the symmetric MSE, and dual MSE is below ``1e-3``.
    """
    torch.manual_seed(0)
    num_nodes = 5
    latent_dim = 2
    edge_index = _directed_path_edge_index(num_nodes)
    teacher = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency="dual_random_walk",
    )
    # Upstream advection along directed edges via the backward walk term.
    teacher.set_dense_matrices(
        0.25 * torch.eye(latent_dim),
        torch.zeros(latent_dim, latent_dim),
        k_bwd=0.65 * torch.eye(latent_dim),
    )
    states = torch.randn(48, num_nodes, latent_dim)
    with torch.no_grad():
        targets = torch.stack([teacher(state, edge_index) for state in states])

    def _fit_mse(adjacency: str, *, steps: int = 250) -> float:
        torch.manual_seed(1)
        student = GraphKoopmanOperator(
            latent_dim,
            init_mode="identity_noise",
            init_scale=1e-3,
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=0.08)
        for _ in range(steps):
            optimizer.zero_grad()
            pred = torch.stack([student(state, edge_index) for state in states])
            loss = torch.mean((pred - targets) ** 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = torch.stack([student(state, edge_index) for state in states])
            return float(torch.mean((pred - targets) ** 2).item())

    err_dual = _fit_mse("dual_random_walk")
    err_symmetric = _fit_mse("symmetric")
    assert err_dual < 1e-3
    assert err_dual <= 0.5 * err_symmetric


def test_dense_effective_inverse_matches_per_call_inverse_advance() -> None:
    """Precomputed dense inverse agrees with uncached inverse_advance (TASK-1503)."""
    torch.manual_seed(3)
    num_nodes = 4
    latent_dim = 3
    edge_index = _path_edge_index(num_nodes)
    op = GraphKoopmanOperator(latent_dim, init_mode="xavier", init_scale=0.15)
    z = torch.randn(num_nodes, latent_dim)
    z_next = op.advance(z, edge_index=edge_index)
    inverse = op.dense_effective_inverse(edge_index, num_nodes)
    recovered_shared = op.inverse_advance(
        z_next, edge_index=edge_index, inverse_matrix=inverse
    )
    recovered_fresh = op.inverse_advance(z_next, edge_index=edge_index)
    # Same effective map; float32 solve noise.
    torch.testing.assert_close(recovered_shared, recovered_fresh, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(recovered_shared, z, rtol=0.0, atol=1e-5)


def test_dense_effective_inverse_rejects_block_diagonal() -> None:
    """dense_effective_inverse is dense-sparsity only."""
    op = GraphKoopmanOperator(2, sparsity="block_diagonal")
    with pytest.raises(ValueError, match="sparsity='dense'"):
        op.dense_effective_inverse(_path_edge_index(3), 3)


def test_backward_sequence_precomputes_one_dense_graph_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static dense graph: one dense_inverse_or_pinv per backward sequence loss."""
    import koopman_graph.operators.graph as graph_mod
    from koopman_graph.training.pair_objectives import (
        compute_backward_consistency_sequence_loss,
    )

    calls = {"count": 0}
    original = graph_mod.dense_inverse_or_pinv

    def _counting(matrix: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return original(matrix)

    monkeypatch.setattr(graph_mod, "dense_inverse_or_pinv", _counting)

    torch.manual_seed(4)
    edge_index = _path_edge_index(5)
    snapshots = [Data(x=torch.randn(5, 3), edge_index=edge_index) for _ in range(4)]
    sequence = GraphSnapshotSequence(snapshots)
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    assert model.koopman.sparsity == "dense"
    model.eval()
    with torch.no_grad():
        loss = compute_backward_consistency_sequence_loss(model, sequence)
    assert torch.isfinite(loss)
    # One evaluation-scoped inverse for the whole sequence (not per pair).
    assert calls["count"] == 1


def test_backward_sequence_skips_precompute_for_block_diagonal_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """block_diagonal backward path does not call dense_effective_inverse."""
    from koopman_graph.training.pair_objectives import (
        compute_backward_consistency_sequence_loss,
    )

    torch.manual_seed(5)
    edge_index = _path_edge_index(4)
    snapshots = [Data(x=torch.randn(4, 3), edge_index=edge_index) for _ in range(3)]
    sequence = GraphSnapshotSequence(snapshots)
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=3),
        decoder=GNNDecoder(latent_dim=3, hidden_channels=8, out_channels=3),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_sparsity="block_diagonal",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    assert model.koopman.sparsity == "block_diagonal"
    hits = {"count": 0}
    original = model.koopman.dense_effective_inverse

    def _spy(*args, **kwargs):
        hits["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.koopman, "dense_effective_inverse", _spy)
    model.eval()
    with torch.no_grad():
        loss = compute_backward_consistency_sequence_loss(model, sequence)
    assert torch.isfinite(loss)
    assert hits["count"] == 0


def test_graph_forward_does_not_use_dense_effective_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse forward advance must not assemble/invert the dense map."""
    torch.manual_seed(6)
    edge_index = _path_edge_index(4)
    op = GraphKoopmanOperator(3, init_mode="xavier", init_scale=0.1)
    hits = {"count": 0}
    original = op.dense_effective_inverse

    def _spy(*args, **kwargs):
        hits["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(op, "dense_effective_inverse", _spy)
    z = torch.randn(4, 3)
    _ = op.advance(z, edge_index=edge_index)
    assert hits["count"] == 0


def test_topologies_equal_content_fingerprint() -> None:
    """Topology fingerprint is content equality, not storage pointer identity."""
    from koopman_graph.training.pair_objectives import topologies_equal

    edges_a = _path_edge_index(4)
    edges_b = edges_a.clone()
    assert edges_a.data_ptr() != edges_b.data_ptr()
    assert topologies_equal(edges_a, None, edges_b, None)
    weights = torch.ones(edges_a.shape[1])
    assert topologies_equal(edges_a, weights, edges_b, weights.clone())
    assert not topologies_equal(edges_a, weights, edges_b, 2.0 * weights)
    cycle = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )
    assert not topologies_equal(edges_a, None, cycle, None)


def test_backward_sequence_reuses_inverse_per_distinct_dynamic_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic topology: one inverse per distinct pair-target key (TASK-1504)."""
    import koopman_graph.operators.graph as graph_mod
    from koopman_graph.training.pair_objectives import (
        compute_backward_consistency_sequence_loss,
    )

    calls = {"count": 0}
    original = graph_mod.dense_inverse_or_pinv

    def _counting(matrix: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return original(matrix)

    monkeypatch.setattr(graph_mod, "dense_inverse_or_pinv", _counting)

    torch.manual_seed(8)
    num_nodes = 5
    edges_a = _path_edge_index(num_nodes)
    edges_b = torch.tensor(
        [
            [i for i in range(num_nodes)],
            [(i + 1) % num_nodes for i in range(num_nodes)],
        ],
        dtype=torch.long,
    )
    # Pair targets are sequence[t+1]: B, A, B → two distinct topologies.
    snapshots = [
        Data(
            x=torch.randn(num_nodes, 3),
            edge_index=(edges_a if t % 2 == 0 else edges_b).clone(),
        )
        for t in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    assert sequence.is_dynamic_topology
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
    )
    model.eval()
    with torch.no_grad():
        loss = compute_backward_consistency_sequence_loss(model, sequence)
    assert torch.isfinite(loss)
    assert calls["count"] == 2


def test_backward_sequence_bilinear_does_not_share_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bilinear control: inverse count equals num_pairs (no shared precompute)."""
    import koopman_graph.operators.graph as graph_mod
    from koopman_graph.training.pair_objectives import (
        compute_backward_consistency_sequence_loss,
    )

    calls = {"count": 0}
    original = graph_mod.dense_inverse_or_pinv

    def _counting(matrix: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return original(matrix)

    monkeypatch.setattr(graph_mod, "dense_inverse_or_pinv", _counting)

    torch.manual_seed(9)
    edge_index = _path_edge_index(4)
    snapshots = [Data(x=torch.randn(4, 3), edge_index=edge_index) for _ in range(4)]
    sequence = GraphSnapshotSequence(snapshots, control_inputs=torch.randn(4, 1))
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=3),
        decoder=GNNDecoder(latent_dim=3, hidden_channels=8, out_channels=3),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        control_dim=1,
        control_mode="bilinear",
    )
    assert model.koopman.control_mode == "bilinear"
    model.eval()
    with torch.no_grad():
        loss = compute_backward_consistency_sequence_loss(model, sequence)
    assert torch.isfinite(loss)
    num_pairs = sequence.num_timesteps - 1
    # Each pair folds control into K_self and inverts independently.
    assert calls["count"] == num_pairs
