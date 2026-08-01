"""Tests for :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.graph_utils import dense_relation_normalized_adjacency
from koopman_graph.operators import GraphKoopmanOperator, HeteroGraphKoopmanOperator

# Match GraphKoopmanOperator sparse↔dense comparisons (float32 Kronecker path).
_ATOL = 1e-5
_INV_ATOL = 1e-4


def _two_relation_banks(num_nodes: int = 4) -> list[torch.Tensor]:
    """Build two directed relation edge banks on a small multiplex graph."""
    # r0: path 0→1→2→3
    r0 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    # r1: reverse-ish cycle edges
    r1 = torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long)
    assert num_nodes == 4
    return [r0, r1]


def _hand_built_effective(
    k_self: torch.Tensor,
    k_relations: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    num_nodes: int,
    *,
    normalization: str = "rgcn_in_degree",
) -> torch.Tensor:
    """Assemble oracle ``I ⊗ K_self + Σ_r Â_r ⊗ K_r``."""
    identity = torch.eye(num_nodes, dtype=k_self.dtype)
    effective = torch.kron(identity, k_self)
    for edge_index, k_rel in zip(edge_indices, k_relations, strict=True):
        adj = dense_relation_normalized_adjacency(
            edge_index,
            num_nodes,
            dtype=k_self.dtype,
            normalization=normalization,  # type: ignore[arg-type]
        )
        effective = effective + torch.kron(adj, k_rel)
    return effective


def test_construction_oracle_advance_matches_keff_matmul() -> None:
    """Hand-built K_eff: sparse advance matches vec matmul (N=4, |R|=2, d=2)."""
    torch.manual_seed(0)
    num_nodes = 4
    latent_dim = 2
    edge_indices = _two_relation_banks(num_nodes)
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]
    op = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
    )
    op.set_dense_matrices(k_self, k_relations)

    z = torch.randn(num_nodes, latent_dim)
    sparse_next = op(z, edge_indices)
    oracle = _hand_built_effective(k_self, k_relations, edge_indices, num_nodes)
    dense_next = (oracle @ z.reshape(-1)).view_as(z)
    assert torch.allclose(sparse_next, dense_next, atol=_ATOL)

    effective = op.effective_matrix(edge_indices, num_nodes)
    assert torch.allclose(effective, oracle, atol=_ATOL)
    assert effective.shape == (num_nodes * latent_dim, num_nodes * latent_dim)


def test_effective_matrix_matches_oracle_assembly() -> None:
    """Operator effective_matrix equals independent Kronecker assembly."""
    torch.manual_seed(1)
    num_nodes = 3
    edge_indices = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[2, 0], [0, 1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(
        2, num_relations=2, init_mode="xavier", init_scale=0.2
    )
    k_self = op.K_self.detach().clone()
    k_relations = [k.detach().clone() for k in op.K_relations]
    oracle = _hand_built_effective(k_self, k_relations, edge_indices, num_nodes)
    got = op.effective_matrix(edge_indices, num_nodes)
    assert torch.allclose(got, oracle, atol=_ATOL)


def test_advance_is_linear_in_z() -> None:
    """Advance is linear in z for fixed topology (no activations)."""
    torch.manual_seed(2)
    num_nodes = 4
    edge_indices = _two_relation_banks(num_nodes)
    op = HeteroGraphKoopmanOperator(
        2, num_relations=2, init_mode="xavier", init_scale=0.3
    )
    z1 = torch.randn(num_nodes, 2)
    z2 = torch.randn(num_nodes, 2)
    a, b = 0.3, -1.2
    left = op(a * z1 + b * z2, edge_indices)
    right = a * op(z1, edge_indices) + b * op(z2, edge_indices)
    assert torch.allclose(left, right, atol=_ATOL)


def test_forward_matches_dense_effective_matrix_random_factors() -> None:
    """Sparse forward matches effective @ vec(Z) for random factors."""
    torch.manual_seed(3)
    num_nodes = 4
    edge_indices = _two_relation_banks(num_nodes)
    op = HeteroGraphKoopmanOperator(
        3,
        num_relations=2,
        init_mode="identity_noise",
        init_scale=0.15,
    )
    z = torch.randn(num_nodes, 3)
    sparse_next = op.advance(z, edge_indices=edge_indices)
    dense_next = (op.effective_matrix(edge_indices, num_nodes) @ z.reshape(-1)).view_as(
        z
    )
    assert torch.allclose(sparse_next, dense_next, atol=_ATOL)


def test_block_diagonal_shares_dense_forward() -> None:
    """block_diagonal sparsity matches dense forward."""
    torch.manual_seed(4)
    edge_indices = _two_relation_banks()
    dense = HeteroGraphKoopmanOperator(2, num_relations=2, sparsity="dense")
    block = HeteroGraphKoopmanOperator(2, num_relations=2, sparsity="block_diagonal")
    k_self = torch.eye(2) * 0.8
    k_rels = [0.1 * torch.eye(2), 0.05 * torch.randn(2, 2)]
    dense.set_dense_matrices(k_self, k_rels)
    block.set_dense_matrices(k_self, k_rels)
    z = torch.randn(4, 2)
    assert torch.allclose(dense(z, edge_indices), block(z, edge_indices), atol=_ATOL)


def test_distributed_sparsity_constructs() -> None:
    """sparsity='distributed' constructs for multiplex hetero."""
    op = HeteroGraphKoopmanOperator(2, num_relations=1, sparsity="distributed")
    assert op.sparsity == "distributed"


def test_distributed_spectrum_and_inverse_smoke() -> None:
    """Distributed hetero spectrum / inverse agree with dense on modest N·d."""
    num_nodes = 4
    latent_dim = 2
    edge_indices = _two_relation_banks(num_nodes)
    dense = HeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity", sparsity="dense"
    )
    distributed = HeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity", sparsity="distributed"
    )
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]
    dense.set_dense_matrices(k_self, k_relations)
    distributed.set_dense_matrices(k_self, k_relations)

    num_modes = 3
    dense_abs = dense.spectrum(edge_indices, num_nodes).eigenvalues.abs()[:num_modes]
    dist_spec = distributed.spectrum(edge_indices, num_nodes, num_modes=num_modes)
    assert dist_spec.eigenvalues.shape == (num_modes,)
    assert torch.allclose(dist_spec.eigenvalues.abs(), dense_abs, atol=1e-4)

    z = torch.randn(num_nodes, latent_dim)
    y = dense.advance(z, edge_indices=edge_indices)
    recovered = distributed.inverse_advance(y, edge_indices=edge_indices)
    assert torch.allclose(recovered, z, atol=_INV_ATOL)


def test_advance_requires_edge_indices() -> None:
    """Missing relation banks raise on advance."""
    op = HeteroGraphKoopmanOperator(2, num_relations=2)
    with pytest.raises(ValueError, match="edge_indices is required"):
        op.advance(torch.randn(4, 2))
    with pytest.raises(ValueError, match="uncontrolled"):
        op.advance(
            torch.randn(4, 2),
            edge_indices=_two_relation_banks(),
            control=torch.ones(1),
        )


def test_relation_bank_count_mismatch() -> None:
    """Wrong number of relation banks raises."""
    op = HeteroGraphKoopmanOperator(2, num_relations=2)
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="Expected 2 relation edge banks"):
        op(z, [_two_relation_banks()[0]])
    with pytest.raises(ValueError, match="Expected 2 relation matrices"):
        op.set_dense_matrices(torch.eye(2), [torch.eye(2)])


def test_package_export_not_root() -> None:
    """Hetero operator is on operators.__all__ but not root __all__."""
    import koopman_graph
    import koopman_graph.operators as operators

    assert "HeteroGraphKoopmanOperator" in operators.__all__
    assert "HeteroGraphKoopmanOperator" not in koopman_graph.__all__
    assert operators.HeteroGraphKoopmanOperator is HeteroGraphKoopmanOperator


def test_dense_inverse_advance_round_trip() -> None:
    """Dense inverse recovers latents on a tiny multiplex graph."""
    torch.manual_seed(5)
    num_nodes = 4
    edge_indices = _two_relation_banks(num_nodes)
    op = HeteroGraphKoopmanOperator(2, num_relations=2, init_mode="identity")
    k_self = torch.diag(torch.tensor([0.85, 0.7]))
    k_rels = [
        0.1 * torch.eye(2),
        0.05 * torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    ]
    op.set_dense_matrices(k_self, k_rels)
    z = torch.randn(num_nodes, 2)
    z_next = op(z, edge_indices)
    recovered = op.inverse_advance(z_next, edge_indices=edge_indices)
    assert torch.allclose(recovered, z, atol=_INV_ATOL)


def test_block_diagonal_inverse_exact_when_relations_zero() -> None:
    """Self-dominated block_diagonal inverse is exact when K_r = 0."""
    edge_indices = _two_relation_banks()
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        sparsity="block_diagonal",
        init_mode="identity",
    )
    k_self = torch.diag(torch.tensor([0.9, 0.8]))
    op.set_dense_matrices(k_self, [torch.zeros(2, 2), torch.zeros(2, 2)])
    z = torch.randn(4, 2)
    z_next = op(z, edge_indices)
    recovered = op.inverse_advance(z_next, edge_indices=edge_indices)
    assert torch.allclose(recovered, z, atol=_INV_ATOL)


def test_block_diagonal_inverse_is_self_only_when_coupled() -> None:
    """Coupled block_diagonal inverse ignores relation terms (documented)."""
    edge_indices = _two_relation_banks()
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        sparsity="block_diagonal",
        init_mode="identity",
    )
    k_self = torch.diag(torch.tensor([0.9, 0.8]))
    k_rels = [0.2 * torch.eye(2), 0.1 * torch.eye(2)]
    op.set_dense_matrices(k_self, k_rels)
    z = torch.randn(4, 2)
    z_next = op(z, edge_indices)
    recovered = op.inverse_advance(z_next, edge_indices=edge_indices)
    # Self-only inverse of z_next would be z_next @ K_self^{-T}, not z.
    self_only = z_next @ torch.linalg.inv(k_self).T
    assert torch.allclose(recovered, self_only, atol=_ATOL)
    assert not torch.allclose(recovered, z, atol=1e-3)


def test_spectrum_and_assembled_spectral_radius() -> None:
    """spectrum / spectral_radius match eigvals of K_eff (Q6)."""
    torch.manual_seed(6)
    num_nodes = 4
    edge_indices = _two_relation_banks(num_nodes)
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        init_mode="identity_noise",
        init_scale=0.05,
    )
    effective = op.effective_matrix(edge_indices, num_nodes)
    eigvals = torch.linalg.eigvals(effective)
    spectrum = op.spectrum(edge_indices, num_nodes, time_step=0.1)
    assert spectrum.eigenvalues.shape == (num_nodes * 2,)
    assert torch.allclose(
        spectrum.magnitudes.sort().values,
        eigvals.abs().sort().values,
        atol=_ATOL,
    )
    rho = op.spectral_radius(edge_indices, num_nodes)
    assert torch.allclose(rho, eigvals.abs().max().real, atol=_ATOL)


def test_spectral_radius_requires_topology() -> None:
    """Missing topology raises (no silent K_self-only ρ)."""
    op = HeteroGraphKoopmanOperator(2, num_relations=2)
    with pytest.raises(ValueError, match="assembled"):
        op.spectral_radius()
    with pytest.raises(ValueError, match="assembled"):
        op.spectral_radius(edge_indices=_two_relation_banks())


def test_homogeneous_reduction_random_walk_one_relation() -> None:
    """|R|=1 + random_walk matches GraphKoopmanOperator(adjacency=random_walk)."""
    torch.manual_seed(7)
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    k_self = torch.tensor([[0.75, 0.1], [-0.05, 0.65]])
    k_nbr = torch.tensor([[0.12, 0.0], [0.03, 0.08]])

    hetero = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        normalization="random_walk",
        init_mode="identity",
    )
    graph = GraphKoopmanOperator(
        2,
        adjacency="random_walk",
        init_mode="identity",
    )
    hetero.set_dense_matrices(k_self, [k_nbr])
    graph.set_dense_matrices(k_self, k_nbr)

    z = torch.randn(num_nodes, 2)
    assert torch.allclose(
        hetero(z, [edge_index]),
        graph(z, edge_index),
        atol=_ATOL,
    )
    assert torch.allclose(
        hetero.effective_matrix([edge_index], num_nodes),
        graph.effective_matrix(edge_index, num_nodes),
        atol=_ATOL,
    )
    z_next = hetero(z, [edge_index])
    assert torch.allclose(
        hetero.inverse_advance(z_next, edge_indices=[edge_index]),
        graph.inverse_advance(z_next, edge_index=edge_index),
        atol=_INV_ATOL,
    )


def test_additive_control_forward_inverse_round_trip() -> None:
    """Additive self-term control round-trips through dense inverse."""
    torch.manual_seed(8)
    edge_indices = _two_relation_banks()
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        init_mode="identity",
        control_dim=1,
        control_mode="additive",
    )
    k_self = torch.diag(torch.tensor([0.9, 0.8]))
    k_rels = [0.05 * torch.eye(2), 0.04 * torch.eye(2)]
    op.set_dense_matrices(
        k_self,
        k_rels,
        control_matrix=torch.tensor([[0.2, -0.1]]),
    )
    z = torch.randn(4, 2)
    control = torch.tensor([0.5])
    z_next = op(z, edge_indices, control=control)
    recovered = op.inverse_advance(
        z_next,
        edge_indices=edge_indices,
        control=control,
    )
    assert torch.allclose(recovered, z, atol=_INV_ATOL)


def test_bilinear_control_forward_inverse_round_trip() -> None:
    """Bilinear self-block control round-trips (global and per-node)."""
    torch.manual_seed(9)
    edge_indices = _two_relation_banks()
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        init_mode="identity",
        control_dim=1,
        control_mode="bilinear",
    )
    k_self = torch.diag(torch.tensor([0.85, 0.75]))
    k_rels = [0.03 * torch.eye(2), torch.zeros(2, 2)]
    n_stack = 0.05 * torch.eye(2).unsqueeze(0)
    op.set_dense_matrices(
        k_self,
        k_rels,
        control_matrix=torch.tensor([[0.1, 0.0]]),
        bilinear_matrices=n_stack,
    )
    z = torch.randn(4, 2)
    global_u = torch.tensor([0.4])
    z_next = op(z, edge_indices, control=global_u)
    recovered = op.inverse_advance(
        z_next,
        edge_indices=edge_indices,
        control=global_u,
    )
    assert torch.allclose(recovered, z, atol=_INV_ATOL)

    per_node_u = torch.tensor([[0.3], [-0.1], [0.2], [0.0]])
    z_next_pn = op(z, edge_indices, control=per_node_u)
    recovered_pn = op.inverse_advance(
        z_next_pn,
        edge_indices=edge_indices,
        control=per_node_u,
    )
    assert torch.allclose(recovered_pn, z, atol=_INV_ATOL)


def test_control_validation_errors() -> None:
    """Missing / mismatched control inputs raise clearly."""
    edge_indices = _two_relation_banks()
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        control_dim=1,
        control_mode="additive",
    )
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="control input is required"):
        op(z, edge_indices, control=None)
    with pytest.raises(ValueError, match="control input is required"):
        op.inverse_advance(z, edge_indices=edge_indices)

    bilinear = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        control_dim=1,
        control_mode="bilinear",
    )
    with pytest.raises(ValueError, match="Per-node control"):
        bilinear.inverse_advance(
            z,
            edge_indices=[edge_indices[0]],
            control=torch.ones(2, 1),
        )


def test_bound_metric_stable_but_assembled_rho_unstable() -> None:
    """§3.5 fixture: factor bounds look stable while ρ(K_eff) > 1."""
    # Bidirectional edge → Â has eigenvalues ±1 under R-GCN in-degree.
    # With K_self = K_r = 0.9 I, K_eff = 0.9 (I + Â) ⊗ I has ρ = 1.8.
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        init_mode="identity",
        parameterization="dense",
    )
    k_self = 0.9 * torch.eye(2)
    k_rel = 0.9 * torch.eye(2)
    op.set_dense_matrices(k_self, [k_rel])
    assert op.bound_metric().item() <= 1.0 + 1e-6
    rho = op.spectral_radius(edge_indices, num_nodes=2)
    assert rho.item() > 1.0


def test_hetero_eigenvalue_loss_requires_topology_and_coupling_matters() -> None:
    """Hetero dense eig-reg never uses K_self alone; coupling raises penalty."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    edge_indices = _two_relation_banks()
    loss_fn = EigenvalueRegularizationLoss()
    op = HeteroGraphKoopmanOperator(2, num_relations=2, init_mode="identity")
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        loss_fn(op)

    k_self = 0.5 * torch.eye(2)
    mild = HeteroGraphKoopmanOperator(2, num_relations=2, init_mode="identity")
    mild.set_dense_matrices(k_self, [0.05 * torch.eye(2), 0.05 * torch.eye(2)])
    strong = HeteroGraphKoopmanOperator(2, num_relations=2, init_mode="identity")
    strong.set_dense_matrices(k_self.clone(), [1.5 * torch.eye(2), 1.5 * torch.eye(2)])
    mild_loss = loss_fn(mild, edge_indices=edge_indices, num_nodes=4)
    strong_loss = loss_fn(strong, edge_indices=edge_indices, num_nodes=4)
    assert mild_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert strong_loss.item() > mild_loss.item()


def test_assembled_eigreg_noops_above_ceiling() -> None:
    """Assembled networked eig-reg returns zero when N·d exceeds the ceiling."""
    from koopman_graph.losses import (
        MAX_ASSEMBLED_EIGREG_SIZE,
        EigenvalueRegularizationLoss,
    )

    # Choose N so N·d > ceiling with d=2.
    num_nodes = (MAX_ASSEMBLED_EIGREG_SIZE // 2) + 1
    # Sparse chain of edges (no need for a dense all-pairs bank).
    src = torch.arange(num_nodes - 1, dtype=torch.long)
    dst = src + 1
    edge_indices = [torch.stack([src, dst], dim=0)]
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    op.set_dense_matrices(2.0 * torch.eye(2), [2.0 * torch.eye(2)])
    loss_fn = EigenvalueRegularizationLoss()
    # Without ceiling this construction is strongly unstable; with ceiling it
    # must no-op rather than allocate an enormous eigendecomposition.
    assert num_nodes * op.latent_dim > MAX_ASSEMBLED_EIGREG_SIZE
    penalty = loss_fn(op, edge_indices=edge_indices, num_nodes=num_nodes)
    assert penalty.item() == pytest.approx(0.0, abs=1e-12)


def test_relation_tying_validation_errors() -> None:
    """Invalid (relation_tying, basis_size) combinations raise clearly."""
    with pytest.raises(ValueError, match="basis_size is required"):
        HeteroGraphKoopmanOperator(2, num_relations=2, relation_tying="basis")
    with pytest.raises(ValueError, match="basis_size must be None"):
        HeteroGraphKoopmanOperator(
            2,
            num_relations=2,
            relation_tying="independent",
            basis_size=1,
        )
    with pytest.raises(ValueError, match="must be <= num_relations"):
        HeteroGraphKoopmanOperator(
            2,
            num_relations=2,
            relation_tying="basis",
            basis_size=3,
        )
    with pytest.raises(ValueError, match="relation_tying must be one of"):
        HeteroGraphKoopmanOperator(
            2,
            num_relations=2,
            relation_tying="tied",  # type: ignore[arg-type]
        )


def test_basis_matches_independent_when_b_equals_r() -> None:
    """independent with |R|=B matches fused basis parameterization."""
    torch.manual_seed(11)
    num_nodes = 4
    latent_dim = 2
    edge_indices = _two_relation_banks(num_nodes)
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]

    independent = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
        relation_tying="independent",
    )
    independent.set_dense_matrices(k_self, k_relations)

    basis = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
        relation_tying="basis",
        basis_size=2,
    )
    basis.self_operator_for("node").set_dense_matrix(k_self.clone())
    basis.set_basis_factors(k_relations, torch.eye(2))

    z = torch.randn(num_nodes, latent_dim)
    next_ind = independent(z, edge_indices)
    next_basis = basis(z, edge_indices)
    assert torch.allclose(next_ind, next_basis, atol=_ATOL)
    for left, right in zip(independent.K_relations, basis.K_relations, strict=True):
        assert torch.allclose(left, right, atol=_ATOL)
    eff_ind = independent.effective_matrix(edge_indices, num_nodes)
    eff_basis = basis.effective_matrix(edge_indices, num_nodes)
    assert torch.allclose(eff_ind, eff_basis, atol=_ATOL)


def test_set_dense_matrices_rejects_basis_mode() -> None:
    """Basis operators require set_basis_factors, not set_dense_matrices."""
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        relation_tying="basis",
        basis_size=1,
    )
    with pytest.raises(ValueError, match="set_basis_factors"):
        op.set_dense_matrices(torch.eye(2), [torch.eye(2), torch.eye(2)])
