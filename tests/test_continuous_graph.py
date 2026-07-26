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
)


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
    with pytest.raises(ValueError, match="distributed"):
        ContinuousGraphKoopmanOperator(3, sparsity="distributed")
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        ContinuousGraphKoopmanOperator(3, parameterization="auxiliary_spectral")


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
