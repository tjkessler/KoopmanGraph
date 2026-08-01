"""Tests for ContinuousGraphKoopmanOperator (TASK-1307)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    ContinuousGraphKoopmanOperator,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.spectrum_types import discrete_spectrum_at_delta_t


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected path graph edge index."""
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _tiny_continuous_graph_model(
    *,
    koopman: str = "continuous_graph",
    sparsity: str = "dense",
    seed: int = 0,
) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=3, num_layers=1)
    decoder = GNNDecoder(latent_dim=3, hidden_channels=8, out_channels=2, num_layers=1)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman=koopman,  # type: ignore[arg-type]
        koopman_sparsity=sparsity,  # type: ignore[arg-type]
    )


def test_exports_and_defaults() -> None:
    """Root export and constructor defaults match the approved plan."""
    operator = ContinuousGraphKoopmanOperator(4)
    assert operator.sparsity == "dense"
    assert operator.control_dim == 0
    assert ContinuousGraphKoopmanOperator is not None


def test_advance_matches_expm_effective_generator() -> None:
    """Dense advance matches vec⁻¹(exp(L_eff Δt) vec(Z))."""
    torch.manual_seed(0)
    num_nodes = 4
    latent_dim = 3
    delta_t = 0.25
    edge_index = _path_edge_index(num_nodes)
    operator = ContinuousGraphKoopmanOperator(
        latent_dim, init_mode="xavier", init_scale=0.2
    )
    z = torch.randn(num_nodes, latent_dim)

    advanced = operator.advance(z, delta_t, edge_index=edge_index)
    transition = operator.transition_matrix(delta_t, edge_index, num_nodes)
    expected = (transition @ z.reshape(-1)).view_as(z)
    assert torch.allclose(advanced, expected, atol=1e-5)


def test_matches_discrete_graph_at_fixed_delta_t() -> None:
    """With L_nbr=0, continuous advance matches discrete K_self=exp(L_self Δt)."""
    torch.manual_seed(1)
    num_nodes = 3
    latent_dim = 2
    delta_t = 0.5
    edge_index = _path_edge_index(num_nodes)

    l_self = torch.tensor([[-0.4, 0.1], [-0.2, -0.5]])
    l_nbr = torch.zeros(latent_dim, latent_dim)
    continuous = ContinuousGraphKoopmanOperator(latent_dim, init_mode="identity")
    continuous.set_dense_matrices(l_self, l_nbr)
    discrete = GraphKoopmanOperator(latent_dim, init_mode="identity")
    discrete.set_dense_matrices(
        torch.linalg.matrix_exp(l_self * delta_t),
        torch.zeros_like(l_nbr),
    )
    z = torch.randn(num_nodes, latent_dim)
    z_c = continuous.advance(z, delta_t, edge_index=edge_index)
    z_d = discrete.advance(z, edge_index=edge_index)
    assert torch.allclose(z_c, z_d, atol=1e-5)


def test_coupled_dense_matches_effective_expm_not_factorwise() -> None:
    """Coupled L_nbr path still equals the dense effective exponential."""
    torch.manual_seed(2)
    num_nodes = 3
    latent_dim = 2
    delta_t = 0.3
    edge_index = _path_edge_index(num_nodes)
    l_self = torch.tensor([[-0.4, 0.1], [-0.2, -0.5]])
    l_nbr = 0.1 * torch.eye(2)
    operator = ContinuousGraphKoopmanOperator(latent_dim, init_mode="identity")
    operator.set_dense_matrices(l_self, l_nbr)
    z = torch.randn(num_nodes, latent_dim)
    got = operator.advance(z, delta_t, edge_index=edge_index)
    expected = (
        operator.transition_matrix(delta_t, edge_index, num_nodes) @ z.reshape(-1)
    ).view_as(z)
    assert torch.allclose(got, expected, atol=1e-5)


def test_hurwitz_bound_metric_and_certificate() -> None:
    """Structural modes expose factor-level bound_metric / certificate."""
    operator = ContinuousGraphKoopmanOperator(3, parameterization="lyapunov")
    bound = operator.bound_metric()
    assert bound.ndim == 0
    assert torch.isfinite(bound)
    certificate = operator.stability_certificate()
    assert certificate is not None
    assert certificate.margin.item() > 0


def test_block_diagonal_ignores_neighbor_coupling() -> None:
    """BD sparsity advances with L_self only (self-dominated approximation)."""
    torch.manual_seed(3)
    num_nodes = 4
    latent_dim = 2
    delta_t = 0.2
    edge_index = _path_edge_index(num_nodes)
    l_self = torch.tensor([[-0.5, 0.1], [0.0, -0.4]])
    l_nbr = 0.5 * torch.eye(2)
    dense = ContinuousGraphKoopmanOperator(latent_dim, init_mode="identity")
    dense.set_dense_matrices(l_self, l_nbr)
    block = ContinuousGraphKoopmanOperator(
        latent_dim, init_mode="identity", sparsity="block_diagonal"
    )
    block.set_dense_matrices(l_self, l_nbr)
    z = torch.randn(num_nodes, latent_dim)
    bd_next = block.advance(z, delta_t, edge_index=edge_index)
    self_only = ContinuousGraphKoopmanOperator(latent_dim, init_mode="identity")
    self_only.set_dense_matrices(l_self, torch.zeros_like(l_nbr))
    expected = self_only.advance(z, delta_t, edge_index=edge_index)
    assert torch.allclose(bd_next, expected, atol=1e-5)
    dense_next = dense.advance(z, delta_t, edge_index=edge_index)
    assert not torch.allclose(dense_next, bd_next, atol=1e-3)


def test_factory_aliases_and_validation() -> None:
    """Factory accepts graph/continuous_graph continuous; rejects discrete alias."""
    model_alias = _tiny_continuous_graph_model(koopman="continuous_graph")
    model_graph = _tiny_continuous_graph_model(koopman="graph")
    assert model_alias.koopman_kind == "continuous_graph"
    assert model_graph.koopman_kind == "continuous_graph"
    assert isinstance(model_alias.koopman, ContinuousGraphKoopmanOperator)
    assert model_alias.uses_continuous_graph_koopman

    with pytest.raises(ValueError, match="requires dynamics_mode='continuous'"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=3),
            decoder=GNNDecoder(latent_dim=3, hidden_channels=8, out_channels=2),
            latent_dim=3,
            time_step=0.1,
            dynamics_mode="discrete",
            koopman="continuous_graph",
        )
    dist = ContinuousGraphKoopmanOperator(3, sparsity="distributed")
    assert dist.sparsity == "distributed"
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        ContinuousGraphKoopmanOperator(3, parameterization="auxiliary_spectral")
    with pytest.raises(ValueError, match="must be 'dense'"):
        ContinuousGraphKoopmanOperator(3, sparsity="bogus")  # type: ignore[arg-type]


def test_spectrum_requires_topology() -> None:
    """Model spectrum requires edge_index / num_nodes for continuous_graph."""
    model = _tiny_continuous_graph_model()
    edge_index = _path_edge_index(5)
    with pytest.raises(ValueError, match="edge_index and num_nodes"):
        model.spectrum()
    spectrum = model.spectrum(edge_index=edge_index, num_nodes=5)
    assert spectrum.eigenvalues.shape == (15,)


def test_predict_at_irregular_deltas() -> None:
    """Irregular step_deltas round-trip through GraphKoopmanModel.predict_at."""
    model = _tiny_continuous_graph_model(seed=4)
    model.eval()
    edge_index = _path_edge_index(4)
    data = Data(x=torch.randn(4, 2), edge_index=edge_index)
    deltas = [0.1, 0.25, 0.15]
    with torch.no_grad():
        preds = model.predict_at(data, step_deltas=deltas)
    assert len(preds) == len(deltas)
    for pred in preds:
        assert pred.x.shape == (4, 2)


def test_format1_checkpoint_round_trip(tmp_path: Path) -> None:
    """Format-1 save/load preserves continuous_graph kind and weights."""
    model = _tiny_continuous_graph_model(seed=5)
    edge_index = _path_edge_index(4)
    data = Data(x=torch.randn(4, 2), edge_index=edge_index)
    path = tmp_path / "cg.pt"
    model.eval()
    with torch.no_grad():
        before = model.predict_at(data, step_deltas=[0.1, 0.2])
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "continuous_graph"
    assert isinstance(loaded.koopman, ContinuousGraphKoopmanOperator)
    assert loaded.dynamics_mode == "continuous"
    loaded.eval()
    with torch.no_grad():
        after = loaded.predict_at(data, step_deltas=[0.1, 0.2])
    for left, right in zip(before, after, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-5)


def test_inverse_advance_uncontrolled_round_trip() -> None:
    """Dense uncontrolled inverse recovers the pre-image."""
    torch.manual_seed(6)
    num_nodes = 3
    latent_dim = 2
    delta_t = 0.2
    edge_index = _path_edge_index(num_nodes)
    operator = ContinuousGraphKoopmanOperator(
        latent_dim, init_mode="xavier", init_scale=0.1
    )
    z = torch.randn(num_nodes, latent_dim)
    advanced = operator.advance(z, delta_t, edge_index=edge_index)
    recovered = operator.inverse_advance(advanced, delta_t, edge_index=edge_index)
    assert torch.allclose(recovered, z, atol=1e-4)


def _directed_path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a one-way directed path ``0→1→…→N-1``."""
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_invalid_adjacency_raises() -> None:
    """Invalid ``adjacency`` values name the accepted set."""
    with pytest.raises(ValueError, match="dual_random_walk.*random_walk.*symmetric"):
        ContinuousGraphKoopmanOperator(2, adjacency="bogus")  # type: ignore[arg-type]


def test_symmetric_adjacency_seeded_regression() -> None:
    """Default ``adjacency='symmetric'`` stays bit-stable vs pre-change fixture."""
    torch.manual_seed(0)
    op = ContinuousGraphKoopmanOperator(2, init_mode="xavier", init_scale=0.2)
    assert op.adjacency == "symmetric"
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    z = torch.randn(3, 2)
    out = op.advance(z, 0.25, edge_index=edge_index)
    expected = torch.tensor(
        [
            [0.5703337788581848, 0.5958525538444519],
            [-0.7420508861541748, -0.19750848412513733],
            [-0.5039222240447998, 0.30766671895980835],
        ]
    )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_dual_equals_random_walk_when_l_bwd_zero() -> None:
    """``L_bwd`` is exactly zero at init so dual matches random_walk."""
    edge_index = _directed_path_edge_index(4)
    z = torch.randn(4, 2)
    delta_t = 0.3
    rw = ContinuousGraphKoopmanOperator(
        2, init_mode="identity", adjacency="random_walk"
    )
    dual = ContinuousGraphKoopmanOperator(
        2, init_mode="identity", adjacency="dual_random_walk"
    )
    assert torch.equal(dual.L_bwd, torch.zeros(2, 2))
    assert torch.allclose(dual.L_fwd, dual.L_nbr)
    l_self = torch.tensor([[-0.5, 0.1], [0.0, -0.4]])
    l_nbr = 0.15 * torch.eye(2)
    rw.set_dense_matrices(l_self, l_nbr)
    dual.set_dense_matrices(l_self, l_nbr)
    assert torch.allclose(
        rw.advance(z, delta_t, edge_index=edge_index),
        dual.advance(z, delta_t, edge_index=edge_index),
        atol=1e-6,
    )


def test_adjacency_modes_advance_match_expm_and_generator_spectrum() -> None:
    """Advance matches ``exp(L_eff Δt)``; generator spectrum is well-formed."""
    edge_index = _directed_path_edge_index(3)
    z = torch.randn(3, 2)
    delta_t = 0.2
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        op = ContinuousGraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        l_self = torch.tensor([[-0.4, 0.1], [-0.05, -0.5]])
        l_nbr = 0.12 * torch.eye(2)
        l_bwd = None if adjacency != "dual_random_walk" else 0.08 * torch.eye(2)
        op.set_dense_matrices(l_self, l_nbr, l_bwd=l_bwd)
        advanced = op.advance(z, delta_t, edge_index=edge_index)
        expected = (
            op.transition_matrix(delta_t, edge_index, 3) @ z.reshape(-1)
        ).view_as(z)
        assert torch.allclose(advanced, expected, atol=1e-5), adjacency
        spectrum = op.spectrum(edge_index, 3)
        assert spectrum.eigenvalues.shape == (6,)
        assert torch.isfinite(spectrum.eigenvalues.real).all()
        assert torch.isfinite(spectrum.eigenvalues.imag).all()
        assert torch.isfinite(spectrum.magnitudes).all()


def test_directed_discrete_spectrum_matches_transition_eigvals() -> None:
    """``discrete_spectrum_at_delta_t(L_eff)`` matches ``eigvals(exp(L_eff Δt))``.

    Tolerance: sorted magnitudes agree to ``atol=1e-5`` (documented contract for
    nonzero neighbor coupling, where factorwise ``exp(L_*)`` is not equivalent).
    """
    edge_index = _directed_path_edge_index(4)
    delta_t = 0.35
    op = ContinuousGraphKoopmanOperator(
        2, init_mode="identity", adjacency="dual_random_walk"
    )
    op.set_dense_matrices(
        torch.tensor([[-0.6, 0.2], [0.0, -0.5]]),
        0.1 * torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        l_bwd=0.05 * torch.eye(2),
    )
    generator = op.effective_generator(edge_index, 4)
    mapped = discrete_spectrum_at_delta_t(generator, delta_t)
    transition = op.transition_matrix(delta_t, edge_index, 4)
    eigvals = torch.linalg.eigvals(transition)
    assert torch.allclose(
        mapped.magnitudes.sort().values,
        eigvals.abs().sort().values,
        atol=1e-5,
    )


def test_neighbor_zero_continuous_matches_discrete_for_all_adjacency_modes() -> None:
    """With zero neighbor factors, continuous matches discrete at fixed ``Δt``."""
    edge_index = _directed_path_edge_index(3)
    delta_t = 0.4
    z = torch.randn(3, 2)
    l_self = torch.tensor([[-0.3, 0.05], [0.0, -0.45]])
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        continuous = ContinuousGraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        continuous.set_dense_matrices(l_self, torch.zeros(2, 2))
        discrete = GraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        discrete.set_dense_matrices(
            torch.linalg.matrix_exp(l_self * delta_t),
            torch.zeros(2, 2),
        )
        assert torch.allclose(
            continuous.advance(z, delta_t, edge_index=edge_index),
            discrete.advance(z, edge_index=edge_index),
            atol=1e-5,
        ), adjacency


def test_directed_modes_irregular_delta_t_sequence() -> None:
    """Varying ``delta_t`` advances remain finite for directed modes."""
    edge_index = _directed_path_edge_index(4)
    z = torch.randn(4, 2)
    deltas = [0.1, 0.25, 0.05, 0.4]
    for adjacency in ("random_walk", "dual_random_walk"):
        op = ContinuousGraphKoopmanOperator(
            2,
            init_mode="identity",
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        op.set_dense_matrices(
            torch.tensor([[-0.5, 0.0], [0.1, -0.4]]),
            0.2 * torch.eye(2),
            l_bwd=None if adjacency != "dual_random_walk" else 0.1 * torch.eye(2),
        )
        state = z
        for delta in deltas:
            state = op.advance(state, delta, edge_index=edge_index)
            assert torch.isfinite(state).all(), adjacency


def test_block_diagonal_ignores_adjacency_neighbor_coupling() -> None:
    """BD sparsity stays self-only under directed adjacency modes."""
    edge_index = _directed_path_edge_index(3)
    delta_t = 0.2
    z = torch.randn(3, 2)
    l_self = torch.tensor([[-0.5, 0.1], [0.0, -0.4]])
    l_nbr = 0.5 * torch.eye(2)
    block = ContinuousGraphKoopmanOperator(
        2,
        init_mode="identity",
        sparsity="block_diagonal",
        adjacency="dual_random_walk",
    )
    block.set_dense_matrices(l_self, l_nbr, l_bwd=0.3 * torch.eye(2))
    self_only = ContinuousGraphKoopmanOperator(
        2, init_mode="identity", adjacency="dual_random_walk"
    )
    self_only.set_dense_matrices(l_self, torch.zeros(2, 2))
    assert torch.allclose(
        block.advance(z, delta_t, edge_index=edge_index),
        self_only.advance(z, delta_t, edge_index=edge_index),
        atol=1e-5,
    )


def test_hypergraph_rejects_adjacency_kwarg() -> None:
    """Hypergraph stays Zhou-symmetric and does not expose ``adjacency``."""
    with pytest.raises(TypeError, match="adjacency"):
        HypergraphKoopmanOperator(2, adjacency="random_walk")  # type: ignore[call-arg]
    assert "adjacency" not in HypergraphKoopmanOperator.__init__.__code__.co_varnames


def test_continuous_graph_api_guards_and_properties() -> None:
    """Constructor guards, properties, generator overrides, and zero Δt."""
    with pytest.raises(ValueError, match="sparsity must be"):
        ContinuousGraphKoopmanOperator(2, sparsity="bogus")  # type: ignore[arg-type]

    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(
        2, init_mode="identity", adjacency="dual_random_walk", control_dim=1
    )
    op.reset_parameters()
    assert op.matrix.shape == (2, 2)
    assert torch.equal(op.L, op.matrix)
    assert op.B is not None and op.B.numel() == 2
    assert torch.isfinite(op.bound_metric())
    assert torch.isfinite(op.max_real_part())
    assert torch.isfinite(op.spectral_radius())
    _ = op.L_bwd

    plain = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(AttributeError, match="L_bwd"):
        _ = plain.L_bwd
    with pytest.raises(ValueError, match="l_bwd is only valid"):
        plain.set_dense_matrices(
            torch.eye(2) * -0.5, torch.zeros(2, 2), l_bwd=torch.eye(2)
        )
    with pytest.raises(ValueError, match="control matrix requested"):
        plain._networked_control_matrix(3)
    with pytest.raises(ValueError, match="at most one of l_self"):
        plain.effective_generator(
            edge_index,
            3,
            l_self=torch.eye(2) * -0.4,
            l_self_blocks=torch.eye(2).expand(3, 2, 2) * -0.4,
        )
    with pytest.raises(ValueError, match="l_self_blocks must have shape"):
        plain.effective_generator(
            edge_index, 3, l_self_blocks=torch.eye(2).expand(2, 2, 2)
        )
    blocks = torch.eye(2).expand(3, 2, 2) * -0.5
    gen = plain.effective_generator(edge_index, 3, l_self_blocks=blocks)
    assert gen.shape == (6, 6)

    z = torch.randn(3, 2)
    assert torch.equal(plain.advance(z, 0.0, edge_index=edge_index), z)
    with pytest.raises(ValueError, match="expects z with shape"):
        plain.advance(torch.randn(3, 4), 0.1, edge_index=edge_index)
    with pytest.raises(ValueError, match="control input provided"):
        plain.inverse_advance(z, 0.1, edge_index=edge_index, control=torch.ones(1))
    # Cached inverse_matrix path for uncontrolled dense inverse.
    advanced = plain.advance(z, 0.2, edge_index=edge_index)
    inv = torch.linalg.matrix_exp(plain.effective_generator(edge_index, 3) * (-0.2))
    recovered = plain.inverse_advance(
        advanced, 0.2, edge_index=edge_index, inverse_matrix=inv
    )
    assert torch.allclose(recovered, z, atol=1e-4)

    # Shared bilinear control (ndim==1) on the dense path.
    bilinear = ContinuousGraphKoopmanOperator(
        2,
        control_dim=1,
        control_mode="bilinear",
        init_mode="identity",
    )
    out = bilinear.advance(z, 0.1, edge_index=edge_index, control=torch.tensor([0.15]))
    assert out.shape == z.shape
    out_nodes = bilinear.advance(
        z, 0.1, edge_index=edge_index, control=torch.full((3, 1), 0.15)
    )
    assert out_nodes.shape == z.shape

    # Non-dense parameterization exercises neighbor reset with parameter tensors.
    soft = ContinuousGraphKoopmanOperator(
        2,
        init_mode="xavier",
        init_scale=0.05,
        parameterization="dissipative",
        adjacency="dual_random_walk",
        control_dim=1,
    )
    soft.reset_parameters()
    assert soft.L_nbr.shape == (2, 2)


def test_transition_matrix_cache_matches_fresh_matrix_exp() -> None:
    """Cached Φ matches a cleared rebuild within float32 tolerance (TASK-1507)."""
    torch.manual_seed(11)
    num_nodes = 4
    edge_index = _path_edge_index(num_nodes)
    op = ContinuousGraphKoopmanOperator(3, init_mode="xavier", init_scale=0.1)
    op.clear_transition_cache()
    phi_first = op.transition_matrix(0.25, edge_index, num_nodes)
    phi_cached = op.transition_matrix(0.25, edge_index, num_nodes)
    torch.testing.assert_close(phi_first, phi_cached, rtol=0.0, atol=0.0)
    op.clear_transition_cache()
    phi_rebuilt = op.transition_matrix(0.25, edge_index, num_nodes)
    # Same generator; float32 exp noise vs exact cache hit above.
    torch.testing.assert_close(phi_first, phi_rebuilt, rtol=1e-5, atol=1e-5)


def test_transition_matrix_cache_keys_distinct_delta_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct Δt values each call matrix_exp once."""
    calls = {"count": 0}
    original = torch.linalg.matrix_exp

    def _counting(matrix: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "matrix_exp", _counting)
    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    op.clear_transition_cache()
    op.transition_matrix(0.1, edge_index, 3)
    op.transition_matrix(0.1, edge_index, 3)
    assert calls["count"] == 1
    op.transition_matrix(0.2, edge_index, 3)
    assert calls["count"] == 2


def test_stale_phi_cleared_after_parameter_update() -> None:
    """clear_transition_cache prevents poisoned Φ after parameter edits."""
    torch.manual_seed(12)
    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    op.clear_transition_cache()
    phi_before = op.transition_matrix(0.1, edge_index, 3).detach().clone()
    with torch.no_grad():
        # Dense parameterization stores a leaf generator on the self factor.
        op._self.L.add_(0.5 * torch.eye(2))
    # Without clear, the evaluation cache still returns the old Φ.
    phi_stale = op.transition_matrix(0.1, edge_index, 3)
    assert torch.equal(phi_stale, phi_before)
    op.clear_transition_cache()
    phi_fresh = op.transition_matrix(0.1, edge_index, 3)
    assert not torch.allclose(phi_fresh, phi_before, atol=1e-5)


def test_cached_phi_advance_grads_match_cleared_path() -> None:
    """Reused Φ within one evaluation yields the same grads as cleared rebuilds."""
    torch.manual_seed(13)
    edge_index = _path_edge_index(3)
    z = torch.randn(3, 2)
    op = ContinuousGraphKoopmanOperator(2, init_mode="xavier", init_scale=0.05)
    op.clear_transition_cache()
    out_cached = op.advance(z, 0.15, edge_index=edge_index) + op.advance(
        z, 0.15, edge_index=edge_index
    )
    loss_cached = out_cached.pow(2).sum()
    grads_cached = torch.autograd.grad(loss_cached, list(op.parameters()))

    op.clear_transition_cache()
    out1 = op.advance(z, 0.15, edge_index=edge_index)
    op.clear_transition_cache()
    out2 = op.advance(z, 0.15, edge_index=edge_index)
    loss_fresh = (out1 + out2).pow(2).sum()
    grads_fresh = torch.autograd.grad(loss_fresh, list(op.parameters()))
    for left, right in zip(grads_cached, grads_fresh, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)


def test_compute_training_loss_clears_transition_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_training_loss clears continuous Φ cache at evaluation start."""
    from koopman_graph.data import GraphSnapshotSequence
    from koopman_graph.training import LossWeights, compute_training_loss

    model = _tiny_continuous_graph_model(seed=14)
    assert isinstance(model.koopman, ContinuousGraphKoopmanOperator)
    edge_index = _path_edge_index(5)
    model.koopman.transition_matrix(0.1, edge_index, 5)
    assert len(model.koopman._phi_cache) == 1
    hits = {"count": 0}
    original = model.koopman.clear_transition_cache

    def _spy() -> None:
        hits["count"] += 1
        original()

    monkeypatch.setattr(model.koopman, "clear_transition_cache", _spy)
    snapshots = [Data(x=torch.randn(5, 2), edge_index=edge_index) for _ in range(3)]
    sequence = GraphSnapshotSequence(snapshots)
    model.eval()
    with torch.no_grad():
        breakdown = compute_training_loss(
            model, sequence, LossWeights(reconstruction=1.0)
        )
    assert torch.isfinite(breakdown.total)
    assert hits["count"] == 1


def test_transition_cache_excluded_from_state_dict() -> None:
    """Φ / L_eff caches are ephemeral and absent from checkpoints."""
    op = ContinuousGraphKoopmanOperator(2)
    edge_index = _path_edge_index(3)
    op.transition_matrix(0.1, edge_index, 3)
    assert len(op._phi_cache) == 1
    assert len(op._leff_cache) == 1
    state = op.state_dict()
    assert all("phi" not in key and "leff" not in key for key in state)


def test_effective_generator_assembly_once_across_delta_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same topology / distinct Δt rebuild L_eff once (TASK-1511 / G7)."""
    calls = {"count": 0}
    original = ContinuousGraphKoopmanOperator._dense_neighbor_coupling

    def _counting(
        self: ContinuousGraphKoopmanOperator,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_weight: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        calls["count"] += 1
        return original(
            self,
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
        )

    monkeypatch.setattr(
        ContinuousGraphKoopmanOperator, "_dense_neighbor_coupling", _counting
    )
    edge_index = _path_edge_index(4)
    op = ContinuousGraphKoopmanOperator(3, init_mode="identity")
    op.clear_transition_cache()
    op.transition_matrix(0.1, edge_index, 4)
    op.transition_matrix(0.2, edge_index, 4)
    op.transition_matrix(0.1, edge_index, 4)  # Φ hit; must not rebuild L_eff
    assert calls["count"] == 1


def test_effective_generator_cache_matches_cleared_rebuild() -> None:
    """Cached L_eff matches a cleared rebuild within float32 tolerance."""
    torch.manual_seed(31)
    edge_index = _path_edge_index(4)
    op = ContinuousGraphKoopmanOperator(3, init_mode="xavier", init_scale=0.1)
    op.clear_transition_cache()
    first = op.effective_generator(edge_index, 4)
    second = op.effective_generator(edge_index, 4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    op.clear_transition_cache()
    rebuilt = op.effective_generator(edge_index, 4)
    torch.testing.assert_close(first, rebuilt, rtol=0.0, atol=0.0)


def test_stale_leff_cleared_after_parameter_update() -> None:
    """clear_transition_cache drops poisoned L_eff after parameter edits."""
    torch.manual_seed(32)
    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    op.clear_transition_cache()
    before = op.effective_generator(edge_index, 3).detach().clone()
    with torch.no_grad():
        op._self.L.add_(0.5 * torch.eye(2))
    stale = op.effective_generator(edge_index, 3)
    assert torch.equal(stale, before)
    op.clear_transition_cache()
    fresh = op.effective_generator(edge_index, 3)
    assert not torch.allclose(fresh, before, atol=1e-5)


def test_cached_leff_advance_grads_match_cleared_path() -> None:
    """Shared L_eff across Δt yields grads matching cleared rebuilds."""
    torch.manual_seed(33)
    edge_index = _path_edge_index(3)
    z = torch.randn(3, 2)
    op = ContinuousGraphKoopmanOperator(2, init_mode="xavier", init_scale=0.05)
    op.clear_transition_cache()
    out_cached = op.advance(z, 0.1, edge_index=edge_index) + op.advance(
        z, 0.2, edge_index=edge_index
    )
    loss_cached = out_cached.pow(2).sum()
    grads_cached = torch.autograd.grad(loss_cached, list(op.parameters()))

    op.clear_transition_cache()
    out1 = op.advance(z, 0.1, edge_index=edge_index)
    op.clear_transition_cache()
    out2 = op.advance(z, 0.2, edge_index=edge_index)
    loss_fresh = (out1 + out2).pow(2).sum()
    grads_fresh = torch.autograd.grad(loss_fresh, list(op.parameters()))
    for left, right in zip(grads_cached, grads_fresh, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)


def test_bilinear_override_skips_leff_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """l_self overrides do not populate or hit the L_eff cache."""
    calls = {"count": 0}
    original = ContinuousGraphKoopmanOperator._dense_neighbor_coupling

    def _counting(
        self: ContinuousGraphKoopmanOperator,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_weight: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        calls["count"] += 1
        return original(
            self,
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
        )

    monkeypatch.setattr(
        ContinuousGraphKoopmanOperator, "_dense_neighbor_coupling", _counting
    )
    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    op.clear_transition_cache()
    override = torch.eye(2)
    op.effective_generator(edge_index, 3, l_self=override)
    op.effective_generator(edge_index, 3, l_self=override)
    assert calls["count"] == 2
    assert len(op._leff_cache) == 0
