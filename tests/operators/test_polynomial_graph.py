"""Tests for monomial ``filter_degree`` on discrete graph operators."""

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
from koopman_graph.graph_utils.topology import (
    dense_random_walk_normalized_adjacency,
    dense_symmetric_normalized_adjacency,
)
from koopman_graph.operators.polynomial_graph import (
    apply_monomial_powers,
    dense_polynomial_kronecker,
    validate_filter_degree,
)
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_model_config,
    load_checkpoint,
)

# Independent Kronecker oracle vs operator forward (float64).
_POLY_RTOL = 1e-6
_POLY_ATOL = 1e-8


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected path graph edge index."""
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _tiny_graph_model(**kwargs: object) -> GraphKoopmanModel:
    """Build a small discrete graph model for factory tests."""
    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    return GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
        **kwargs,  # type: ignore[arg-type]
    )


def test_validate_filter_degree_rejects_invalid_values() -> None:
    """Non-integers, booleans, and negative degrees raise ``ValueError``."""
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_filter_degree(-1)
    with pytest.raises(ValueError, match="non-negative int"):
        validate_filter_degree(True)
    with pytest.raises(ValueError, match="non-negative int"):
        validate_filter_degree(1.5)
    assert validate_filter_degree(0) == 0
    assert validate_filter_degree(2) == 2


def test_dense_polynomial_kronecker_matches_explicit_sum() -> None:
    """Helper Kronecker sum matches an explicit ``I, A, A^2`` construction."""
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    identity = torch.eye(3, dtype=torch.float64)
    k0 = torch.tensor([[0.5, 0.1], [0.0, 0.4]], dtype=torch.float64)
    k1 = torch.tensor([[0.2, 0.0], [0.1, 0.3]], dtype=torch.float64)
    k2 = torch.tensor([[0.05, 0.02], [-0.01, 0.04]], dtype=torch.float64)
    assembled = dense_polynomial_kronecker(adjacency, (k0, k1, k2), start_power=0)
    expected = (
        torch.kron(identity, k0)
        + torch.kron(adjacency, k1)
        + torch.kron(adjacency @ adjacency, k2)
    )
    assert torch.allclose(assembled, expected, rtol=_POLY_RTOL, atol=_POLY_ATOL)


def test_dense_polynomial_kronecker_rejects_empty_or_invalid() -> None:
    """Assembly helper rejects empty banks and non-square adjacency."""
    adjacency = torch.eye(2)
    with pytest.raises(ValueError, match="non-empty"):
        dense_polynomial_kronecker(adjacency, ())
    with pytest.raises(ValueError, match="start_power"):
        dense_polynomial_kronecker(adjacency, (torch.eye(2),), start_power=-1)
    with pytest.raises(ValueError, match="square 2-D"):
        dense_polynomial_kronecker(torch.ones(2, 3), (torch.eye(2),))
    """Repeated matvec matches ``A^k Z @ K_k.T`` for ``k = 1, 2``."""
    adjacency = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float64)
    states = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float64)
    k1 = torch.tensor([[0.2, 0.1], [0.0, 0.3]], dtype=torch.float64)
    k2 = torch.tensor([[0.05, 0.0], [0.02, 0.04]], dtype=torch.float64)
    got = apply_monomial_powers(states, (k1, k2), adjacency.matmul, min_power=1)
    hop1 = adjacency @ states
    hop2 = adjacency @ hop1
    expected = hop1 @ k1.T + hop2 @ k2.T
    assert torch.allclose(got, expected, rtol=_POLY_RTOL, atol=_POLY_ATOL)


def test_default_filter_degree_is_one_and_state_dict_omits_hop_modules() -> None:
    """Default ``P=1`` matches an explicit ``filter_degree=1`` operator."""
    torch.manual_seed(0)
    implicit = GraphKoopmanOperator(3, init_mode="identity_noise", init_scale=0.05)
    torch.manual_seed(0)
    explicit = GraphKoopmanOperator(
        3,
        init_mode="identity_noise",
        init_scale=0.05,
        filter_degree=1,
    )
    assert implicit.filter_degree == 1
    assert implicit.receptive_field_hops() == 1
    assert set(implicit.state_dict()) == set(explicit.state_dict())
    for key, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    assert not any("hop_factors" in key for key in implicit.state_dict())
    edge_index = _path_edge_index(4)
    latents = torch.randn(4, 3)
    assert torch.equal(implicit(latents, edge_index), explicit(latents, edge_index))


def test_filter_degree_zero_matches_pernode_even_with_nonzero_neighbor() -> None:
    """``P=0`` ignores ``K_nbr`` and matches a per-node self map."""
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.4]])
    k_nbr = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    graph_op = GraphKoopmanOperator(2, init_mode="identity", filter_degree=0)
    graph_op.set_dense_matrices(k_self, k_nbr)
    pernode = KoopmanOperator(2, init_mode="identity")
    pernode.set_dense_matrix(k_self)
    edge_index = _path_edge_index(3)
    latents = torch.randn(3, 2)
    assert torch.allclose(graph_op(latents, edge_index), pernode(latents), atol=1e-6)


def test_filter_degree_two_forward_matches_independent_kronecker() -> None:
    """``P=2`` sparse forward matches dense ``sum_k A^k ⊗ K_k`` (float64)."""
    num_nodes = 3
    latent_dim = 2
    edge_index = _path_edge_index(num_nodes)
    k0 = torch.tensor([[0.6, 0.05], [-0.02, 0.55]], dtype=torch.float64)
    k1 = torch.tensor([[0.12, 0.03], [0.04, 0.08]], dtype=torch.float64)
    k2 = torch.tensor([[0.03, -0.01], [0.02, 0.05]], dtype=torch.float64)
    operator = GraphKoopmanOperator(latent_dim, init_mode="identity", filter_degree=2)
    operator = operator.double()
    operator.set_dense_matrices(k0, k1, hop_matrices=(k2,))
    latents = torch.tensor(
        [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]],
        dtype=torch.float64,
    )
    got = operator(latents, edge_index)
    adjacency = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float64,
    )
    identity = torch.eye(num_nodes, dtype=torch.float64)
    effective = (
        torch.kron(identity, k0)
        + torch.kron(adjacency, k1)
        + torch.kron(adjacency @ adjacency, k2)
    )
    expected = (effective @ latents.reshape(-1)).view_as(latents)
    assert torch.allclose(got, expected, rtol=_POLY_RTOL, atol=_POLY_ATOL)
    assembled = operator.effective_matrix(edge_index, num_nodes)
    assert torch.allclose(assembled, effective, rtol=_POLY_RTOL, atol=_POLY_ATOL)


def test_filter_degree_two_dual_matches_independent_kronecker() -> None:
    """Dual ``P=2`` uses independent powers of each random-walk shift."""
    num_nodes = 3
    latent_dim = 2
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long)
    k0 = torch.tensor([[0.5, 0.0], [0.1, 0.4]], dtype=torch.float64)
    k1 = torch.tensor([[0.1, 0.02], [0.0, 0.08]], dtype=torch.float64)
    k2 = torch.tensor([[0.04, 0.01], [-0.01, 0.03]], dtype=torch.float64)
    k_bwd = torch.tensor([[0.07, 0.0], [0.02, 0.06]], dtype=torch.float64)
    k_bwd2 = torch.tensor([[0.02, 0.0], [0.01, 0.03]], dtype=torch.float64)
    operator = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        filter_degree=2,
        adjacency="dual_random_walk",
    ).double()
    operator.set_dense_matrices(
        k0,
        k1,
        k_bwd=k_bwd,
        hop_matrices=(k2,),
        bwd_hop_matrices=(k_bwd2,),
    )
    latents = torch.randn(num_nodes, latent_dim, dtype=torch.float64)
    got = operator(latents, edge_index)
    adj_fwd = dense_random_walk_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float64,
        direction="forward",
    )
    adj_bwd = dense_random_walk_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float64,
        direction="backward",
    )
    identity = torch.eye(num_nodes, dtype=torch.float64)
    effective = (
        torch.kron(identity, k0)
        + torch.kron(adj_fwd, k1)
        + torch.kron(adj_fwd @ adj_fwd, k2)
        + torch.kron(adj_bwd, k_bwd)
        + torch.kron(adj_bwd @ adj_bwd, k_bwd2)
    )
    expected = (effective @ latents.reshape(-1)).view_as(latents)
    assert torch.allclose(got, expected, rtol=_POLY_RTOL, atol=_POLY_ATOL)


def test_filter_degree_two_dense_inverse_round_trip() -> None:
    """Dense inverse recovers the preimage for a well-conditioned ``P=2`` map."""
    num_nodes = 3
    latent_dim = 2
    edge_index = _path_edge_index(num_nodes)
    operator = GraphKoopmanOperator(latent_dim, init_mode="identity", filter_degree=2)
    operator = operator.double()
    identity = torch.eye(latent_dim, dtype=torch.float64)
    operator.set_dense_matrices(
        0.5 * identity,
        0.1 * identity,
        hop_matrices=(0.05 * identity,),
    )
    latents = torch.randn(num_nodes, latent_dim, dtype=torch.float64)
    advanced = operator(latents, edge_index)
    recovered = operator.inverse_advance(advanced, edge_index=edge_index)
    assert torch.allclose(recovered, latents, rtol=_POLY_RTOL, atol=_POLY_ATOL)


@pytest.mark.parametrize("filter_degree", [0, 2])
def test_non_unit_filter_degree_uses_kronecker_spectrum_helper(
    monkeypatch: pytest.MonkeyPatch,
    filter_degree: int,
) -> None:
    """Eligible ``P!=1`` spectrum routes through the polynomial Kronecker helper."""
    from koopman_graph.operators import graph as graph_mod

    calls: list[object] = []
    original = graph_mod.spectrum_k_eff_kronecker_polynomial

    def _spy(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _spy)
    edge_index = _path_edge_index(3)
    operator = GraphKoopmanOperator(
        2, init_mode="identity", filter_degree=filter_degree
    )
    if filter_degree == 2:
        identity = torch.eye(2)
        operator.set_dense_matrices(
            0.5 * identity,
            0.1 * identity,
            hop_matrices=(0.05 * identity,),
        )
    spectrum = operator.spectrum(edge_index, 3, time_step=0.1)
    assert len(calls) == 1
    effective = operator.effective_matrix(edge_index, 3)
    eigvals = torch.linalg.eigvals(effective)
    assert torch.allclose(
        spectrum.magnitudes.sort().values,
        eigvals.abs().sort().values,
        atol=1e-5,
    )


def test_filter_degree_two_block_diagonal_uses_kronecker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible block-diagonal ``P=2`` spectrum still Kronecker-routes."""
    from koopman_graph.operators import graph as graph_mod

    calls: list[int] = []
    original = graph_mod.spectrum_k_eff_kronecker_polynomial

    def _spy(**kwargs: object) -> object:
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_polynomial", _spy)
    edge_index = _path_edge_index(3)
    operator = GraphKoopmanOperator(
        2,
        init_mode="identity",
        filter_degree=2,
        sparsity="block_diagonal",
    )
    spectrum = operator.spectrum(edge_index, 3, time_step=0.1)
    assert len(calls) == 1
    oracle = torch.linalg.eigvals(operator.effective_matrix(edge_index, 3))
    assert torch.allclose(
        spectrum.magnitudes.sort().values,
        oracle.abs().sort().values,
        atol=1e-5,
    )


def test_non_unit_filter_degree_refuses_block_diagonal_and_distributed() -> None:
    """One-hop inverse / Arnoldi paths refuse ``P!=1``."""
    edge_index = _path_edge_index(3)
    latents = torch.randn(3, 2)
    block = GraphKoopmanOperator(
        2,
        init_mode="identity",
        filter_degree=2,
        sparsity="block_diagonal",
    )
    with pytest.raises(ValueError, match="filter_degree=1"):
        block.inverse_advance(latents, edge_index=edge_index)
    distributed = GraphKoopmanOperator(
        2,
        init_mode="identity",
        filter_degree=2,
        sparsity="distributed",
    )
    with pytest.raises(ValueError, match="filter_degree=1"):
        distributed.inverse_advance(latents, edge_index=edge_index)
    with pytest.raises(ValueError, match="filter_degree=1"):
        distributed.spectrum(edge_index, 3)


def test_factory_default_matches_explicit_filter_degree_one() -> None:
    """Omitting ``koopman_filter_degree`` matches an explicit ``1``."""
    torch.manual_seed(0)
    implicit = _tiny_graph_model()
    torch.manual_seed(0)
    explicit = _tiny_graph_model(koopman_filter_degree=1)
    assert implicit.koopman.filter_degree == 1
    assert set(implicit.koopman.state_dict()) == set(explicit.koopman.state_dict())
    for key, value in implicit.koopman.state_dict().items():
        assert torch.equal(value, explicit.koopman.state_dict()[key])


def test_factory_rejects_filter_degree_for_non_discrete_graph() -> None:
    """Non-default hop degree is rejected off the discrete graph kind."""
    with pytest.raises(ValueError, match="only meaningful for discrete"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 3, num_layers=1),
            decoder=GNNDecoder(3, 4, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            koopman="pernode",
            koopman_filter_degree=2,
        )
    with pytest.raises(ValueError, match="only meaningful for discrete"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 3, num_layers=1),
            decoder=GNNDecoder(3, 4, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman="graph",
            koopman_filter_degree=2,
        )


def test_factory_filter_degree_two_allocates_hop_modules() -> None:
    """Factory ``koopman_filter_degree=2`` builds extra hop parameters."""
    model = _tiny_graph_model(koopman_filter_degree=2)
    assert model.koopman.filter_degree == 2
    assert model.koopman.receptive_field_hops() == 2
    assert any("_hop_factors." in key for key in model.koopman.state_dict())


def test_checkpoint_without_filter_degree_loads_as_one(tmp_path: Path) -> None:
    """Additive ``filter_degree``: absent key reconstructs ``P=1``."""
    model = _tiny_graph_model()
    config = build_model_config(model)
    assert config["filter_degree"] == 1
    assert FORMAT_VERSION == 1
    del config["filter_degree"]
    path = tmp_path / "missing_filter_degree.pt"
    torch.save(
        {
            "format_version": 1,
            "package_version": "0.14.0",
            "config": config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = load_checkpoint(path)
    assert loaded.koopman.filter_degree == 1
    edge_index = _path_edge_index(3)
    data_x = torch.randn(3, 2)
    snapshot = Data(x=data_x, edge_index=edge_index)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        before = model.predict(snapshot, steps=1)
        after = loaded.predict(snapshot, steps=1)
    assert torch.allclose(before[0].x, after[0].x, atol=1e-5)


def test_filter_degree_two_checkpoint_round_trip() -> None:
    """``P=2`` save/load preserves hop factors and predictions."""
    model = _tiny_graph_model(koopman_filter_degree=2)
    identity = torch.eye(3)
    model.koopman.set_dense_matrices(
        0.6 * identity,
        0.1 * identity,
        hop_matrices=(0.05 * identity,),
    )
    edge_index = _path_edge_index(3)
    snapshot = Data(x=torch.randn(3, 2), edge_index=edge_index)
    model.eval()
    with torch.no_grad():
        before = model.predict(snapshot, steps=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "poly_graph.pt"
        model.save(path)
        loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.filter_degree == 2
    for key, value in model.koopman.state_dict().items():
        assert torch.allclose(value, loaded.koopman.state_dict()[key])
    loaded.eval()
    with torch.no_grad():
        after = loaded.predict(snapshot, steps=2)
    for left, right in zip(before, after, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-5)


def test_operator_rejects_invalid_filter_degree() -> None:
    """Constructor validation matches the shared helper."""
    with pytest.raises(ValueError, match="filter_degree"):
        GraphKoopmanOperator(2, filter_degree=-1)
    with pytest.raises(ValueError, match="filter_degree"):
        GraphKoopmanOperator(2, filter_degree=True)
