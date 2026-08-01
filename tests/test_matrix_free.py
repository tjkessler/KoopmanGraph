"""Matrix-free apply / invert / spectrum helpers (TASK-1827–1829).

These tests cover operator-math matvecs, Richardson inverses, and Arnoldi
spectra only — not trainer DDP / ``[distributed]`` process-group extras.
"""

from __future__ import annotations

import pytest
import torch

from koopman_graph.operators import (
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    MatrixFreeInverseResult,
    MatrixFreeSpectrumResult,
    apply_k_eff_graph,
    apply_k_eff_hetero,
    flatten_node_latents,
    invert_k_eff_graph,
    invert_k_eff_hetero,
    spectrum_k_eff_graph,
    spectrum_k_eff_hetero,
    unflatten_node_latents,
)
from koopman_graph.operators.matrix_free import (
    _arnoldi_eigs,
    _invert_square_factor,
    _richardson_invert,
)

_ATOL = 1e-5
_SPECTRUM_ATOL = 1e-4


def _sorted_by_modulus(eigvals: torch.Tensor) -> torch.Tensor:
    """Sort eigenvalues by |λ| ↓ (rounded), then Re ↓, then Im ↓."""
    order = sorted(
        range(eigvals.numel()),
        key=lambda i: (
            -round(float(eigvals[i].abs()), 10),
            -float(eigvals[i].real),
            -float(eigvals[i].imag),
        ),
    )
    return eigvals[torch.tensor(order, dtype=torch.long)]


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        edges.extend([[node, nxt], [nxt, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_flatten_unflatten_roundtrip() -> None:
    """C-order flatten / unflatten preserves node latent layout."""
    z = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    flat = flatten_node_latents(z)
    assert flat.shape == (12,)
    restored = unflatten_node_latents(flat, num_nodes=4, latent_dim=3)
    assert torch.equal(restored, z)


def test_unflatten_rejects_wrong_length() -> None:
    """Wrong flat length raises clearly."""
    with pytest.raises(ValueError, match="expects flat with shape"):
        unflatten_node_latents(torch.zeros(5), num_nodes=2, latent_dim=2)


def test_apply_k_eff_graph_matches_dense_and_advance() -> None:
    """Graph matvec matches dense K_eff @ v and sparse advance flatten."""
    torch.manual_seed(0)
    num_nodes = 5
    latent_dim = 2
    edge_index = _cycle_edge_index(num_nodes)
    op = GraphKoopmanOperator(latent_dim, init_mode="identity")
    k_self = torch.tensor([[0.6, 0.1], [0.0, 0.5]])
    k_nbr = torch.tensor([[0.05, 0.0], [0.0, 0.04]])
    op.set_dense_matrices(k_self, k_nbr)

    z = torch.randn(num_nodes, latent_dim)
    flat = flatten_node_latents(z)
    matvec = apply_k_eff_graph(
        flat,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency="symmetric",
    )
    dense = op.effective_matrix(edge_index, num_nodes) @ flat
    advanced = flatten_node_latents(op.advance(z, edge_index=edge_index))
    assert torch.allclose(matvec, dense, atol=_ATOL)
    assert torch.allclose(matvec, advanced, atol=_ATOL)


def test_apply_k_eff_graph_dual_random_walk_matches_dense() -> None:
    """Dual random-walk matvec matches the dense effective operator."""
    torch.manual_seed(1)
    num_nodes = 4
    latent_dim = 2
    edge_index = _cycle_edge_index(num_nodes)
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency="dual_random_walk",
    )
    k_self = 0.5 * torch.eye(latent_dim)
    k_nbr = 0.1 * torch.eye(latent_dim)
    k_bwd = 0.05 * torch.eye(latent_dim)
    op.set_dense_matrices(k_self, k_nbr, k_bwd=k_bwd)

    flat = flatten_node_latents(torch.randn(num_nodes, latent_dim))
    matvec = apply_k_eff_graph(
        flat,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency="dual_random_walk",
        k_bwd=k_bwd,
    )
    dense = op.effective_matrix(edge_index, num_nodes) @ flat
    assert torch.allclose(matvec, dense, atol=_ATOL)


def test_apply_k_eff_graph_dual_requires_k_bwd() -> None:
    """dual_random_walk without k_bwd raises."""
    with pytest.raises(ValueError, match="k_bwd is required"):
        apply_k_eff_graph(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_nbr=torch.eye(2),
            edge_index=_cycle_edge_index(2),
            num_nodes=2,
            adjacency="dual_random_walk",
        )


def test_apply_k_eff_hetero_matches_dense() -> None:
    """Multiplex hetero matvec matches dense K_eff @ v."""
    torch.manual_seed(2)
    num_nodes = 4
    latent_dim = 2
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
    )
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]
    op.set_dense_matrices(k_self, k_relations)

    flat = flatten_node_latents(torch.randn(num_nodes, latent_dim))
    matvec = apply_k_eff_hetero(
        flat,
        k_self=k_self,
        k_relations=k_relations,
        edge_indices=edge_indices,
        num_nodes=num_nodes,
    )
    dense = op.effective_matrix(edge_indices, num_nodes) @ flat
    assert torch.allclose(matvec, dense, atol=_ATOL)


def test_apply_k_eff_hetero_rejects_bank_mismatch() -> None:
    """Mismatched relation bank counts raise clearly."""
    with pytest.raises(ValueError, match="edge_indices length"):
        apply_k_eff_hetero(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_relations=[torch.eye(2), torch.eye(2)],
            edge_indices=[_cycle_edge_index(2)],
            num_nodes=2,
        )


def test_invert_k_eff_graph_recovers_true_solution() -> None:
    """Well-conditioned graph inverse recovers x from K_eff @ x."""
    torch.manual_seed(3)
    num_nodes = 5
    latent_dim = 2
    edge_index = _cycle_edge_index(num_nodes)
    k_self = torch.tensor([[0.7, 0.05], [0.0, 0.6]])
    k_nbr = torch.tensor([[0.04, 0.0], [0.0, 0.03]])
    x_true = flatten_node_latents(torch.randn(num_nodes, latent_dim))
    rhs = apply_k_eff_graph(
        x_true,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    result = invert_k_eff_graph(
        rhs,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    assert isinstance(result, MatrixFreeInverseResult)
    assert result.converged
    assert result.iterations >= 1
    assert torch.allclose(result.solution, x_true, atol=_ATOL)


def test_invert_k_eff_hetero_recovers_true_solution() -> None:
    """Well-conditioned multiplex hetero inverse recovers x from K_eff @ x."""
    torch.manual_seed(4)
    num_nodes = 4
    latent_dim = 2
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long),
    ]
    k_self = torch.tensor([[0.75, 0.05], [0.0, 0.65]])
    k_relations = [
        torch.tensor([[0.05, 0.0], [0.0, 0.04]]),
        torch.tensor([[0.0, 0.03], [0.02, 0.0]]),
    ]
    x_true = flatten_node_latents(torch.randn(num_nodes, latent_dim))
    rhs = apply_k_eff_hetero(
        x_true,
        k_self=k_self,
        k_relations=k_relations,
        edge_indices=edge_indices,
        num_nodes=num_nodes,
    )
    result = invert_k_eff_hetero(
        rhs,
        k_self=k_self,
        k_relations=k_relations,
        edge_indices=edge_indices,
        num_nodes=num_nodes,
    )
    assert result.converged
    assert torch.allclose(result.solution, x_true, atol=_ATOL)


def test_invert_k_eff_graph_raises_on_divergence() -> None:
    """Aggressive coupling + tiny budget raises a clear divergence error."""
    num_nodes = 4
    edge_index = _cycle_edge_index(num_nodes)
    k_self = 0.2 * torch.eye(2)
    # Large neighbor term makes self-preconditioned Richardson diverge.
    k_nbr = 3.0 * torch.eye(2)
    rhs = flatten_node_latents(torch.ones(num_nodes, 2))
    with pytest.raises(ValueError, match="failed to converge"):
        invert_k_eff_graph(
            rhs,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            max_iters=3,
            tol=1e-12,
        )


def test_invert_k_eff_graph_tol_affects_iteration_count() -> None:
    """Tighter tolerance requires at least as many iterations as a looser one."""
    torch.manual_seed(5)
    num_nodes = 5
    edge_index = _cycle_edge_index(num_nodes)
    k_self = torch.tensor([[0.8, 0.0], [0.0, 0.7]])
    k_nbr = torch.tensor([[0.08, 0.0], [0.0, 0.06]])
    x_true = flatten_node_latents(torch.randn(num_nodes, 2))
    rhs = apply_k_eff_graph(
        x_true,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    loose = invert_k_eff_graph(
        rhs,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        tol=1e-3,
    )
    tight = invert_k_eff_graph(
        rhs,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        tol=1e-8,
    )
    assert loose.converged and tight.converged
    assert tight.iterations >= loose.iterations


def test_spectrum_k_eff_graph_matches_dense_eigvals() -> None:
    """Graph Arnoldi top modes match dense eigvals (sorted by |λ|)."""
    # Small directed graph + nonsymmetric factors → simple distinct moduli.
    num_nodes = 4
    latent_dim = 2
    num_modes = 3
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]],
        dtype=torch.long,
    )
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency="random_walk",
    )
    k_self = torch.tensor([[0.45, 0.20], [0.05, 0.35]])
    k_nbr = torch.tensor([[0.25, 0.10], [0.15, 0.05]])
    op.set_dense_matrices(k_self, k_nbr)
    k_eff = op.effective_matrix(edge_index, num_nodes).detach()
    dense = _sorted_by_modulus(torch.linalg.eigvals(k_eff))[:num_modes]

    result = spectrum_k_eff_graph(
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_modes=num_modes,
        adjacency="random_walk",
        ncv=num_nodes * latent_dim,
    )
    assert isinstance(result, MatrixFreeSpectrumResult)
    assert result.converged
    assert result.num_modes == num_modes
    assert result.eigenvalues.shape == (num_modes,)
    assert result.residual_norms.shape == (num_modes,)
    assert torch.allclose(result.eigenvalues.abs(), dense.abs(), atol=_SPECTRUM_ATOL)
    assert torch.allclose(result.eigenvalues, dense, atol=_SPECTRUM_ATOL)


def test_spectrum_k_eff_hetero_matches_dense_eigvals() -> None:
    """Hetero multiplex Arnoldi top modes match dense eigvals."""
    num_nodes = 4
    latent_dim = 2
    num_modes = 3
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
    )
    k_self = torch.tensor([[0.7, 0.1], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]
    op.set_dense_matrices(k_self, k_relations)
    k_eff = op.effective_matrix(edge_indices, num_nodes).detach()
    dense = _sorted_by_modulus(torch.linalg.eigvals(k_eff))[:num_modes]

    result = spectrum_k_eff_hetero(
        k_self=k_self,
        k_relations=k_relations,
        edge_indices=edge_indices,
        num_nodes=num_nodes,
        num_modes=num_modes,
        ncv=num_nodes * latent_dim,
    )
    assert result.num_modes == num_modes
    assert result.eigenvalues.shape == (num_modes,)
    # Moduli match; conjugate pairs may flip sign under float noise.
    assert torch.allclose(result.eigenvalues.abs(), dense.abs(), atol=_SPECTRUM_ATOL)
    for lam in result.eigenvalues:
        dists = (dense - lam).abs()
        conj_dists = (dense - lam.conj()).abs()
        assert min(float(dists.min()), float(conj_dists.min())) <= _SPECTRUM_ATOL


def test_spectrum_k_eff_graph_rejects_invalid_num_modes() -> None:
    """num_modes outside [1, N·d] raises."""
    with pytest.raises(ValueError, match="num_modes must be in"):
        spectrum_k_eff_graph(
            k_self=torch.eye(2),
            k_nbr=0.1 * torch.eye(2),
            edge_index=_cycle_edge_index(3),
            num_nodes=3,
            num_modes=0,
        )


def test_flatten_rejects_non_2d() -> None:
    """``flatten_node_latents`` requires ``(N, d)`` input."""
    with pytest.raises(ValueError, match="flatten_node_latents expects"):
        flatten_node_latents(torch.zeros(6))


def test_unflatten_rejects_invalid_sizes() -> None:
    """``unflatten_node_latents`` rejects bad ``num_nodes`` / ``latent_dim``."""
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        unflatten_node_latents(torch.zeros(4), num_nodes=0, latent_dim=2)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        unflatten_node_latents(torch.zeros(4), num_nodes=2, latent_dim=0)
    with pytest.raises(ValueError, match="expects flat with shape"):
        unflatten_node_latents(torch.zeros(4, 1), num_nodes=2, latent_dim=2)


def test_apply_k_eff_graph_rejects_bad_adjacency() -> None:
    """Unknown ``adjacency`` mode raises clearly."""
    with pytest.raises(ValueError, match="adjacency must be one of"):
        apply_k_eff_graph(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_nbr=torch.eye(2),
            edge_index=_cycle_edge_index(2),
            num_nodes=2,
            adjacency="invalid",  # type: ignore[arg-type]
        )


def test_apply_k_eff_hetero_rejects_bad_normalization_and_empty_bank() -> None:
    """Bad hetero normalization or empty relation bank raises."""
    with pytest.raises(ValueError, match="normalization must be one of"):
        apply_k_eff_hetero(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_relations=[torch.eye(2)],
            edge_indices=[_cycle_edge_index(2)],
            num_nodes=2,
            normalization="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="k_relations must contain at least one"):
        apply_k_eff_hetero(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_relations=[],
            edge_indices=[],
            num_nodes=2,
        )


def test_richardson_invert_validation_via_graph_inverse() -> None:
    """Richardson helper validates ``max_iters``, ``tol``, and ``x0`` shape."""
    num_nodes = 3
    edge_index = _cycle_edge_index(num_nodes)
    k_self = torch.eye(2)
    k_nbr = 0.1 * torch.eye(2)
    rhs = torch.randn(num_nodes * 2)
    with pytest.raises(ValueError, match="max_iters must be positive"):
        invert_k_eff_graph(
            rhs,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            max_iters=0,
        )
    with pytest.raises(ValueError, match="tol must be positive"):
        invert_k_eff_graph(
            rhs,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            tol=0.0,
        )
    with pytest.raises(ValueError, match="rhs must be a 1-D"):
        invert_k_eff_graph(
            rhs.reshape(num_nodes, 2),
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
    with pytest.raises(ValueError, match="x0 must have shape"):
        invert_k_eff_graph(
            rhs,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            x0=torch.zeros(num_nodes * 2 + 1),
        )


def test_spectrum_k_eff_graph_float32_and_max_arnoldi_alias() -> None:
    """Float32 spectrum runs; ``max_arnoldi`` aliases ``ncv`` when consistent."""
    num_nodes = 4
    latent_dim = 2
    num_modes = 2
    edge_index = _cycle_edge_index(num_nodes)
    k_self = torch.tensor([[0.6, 0.1], [0.0, 0.5]], dtype=torch.float32)
    k_nbr = torch.tensor([[0.05, 0.0], [0.0, 0.04]], dtype=torch.float32)
    ncv = num_nodes * latent_dim
    result = spectrum_k_eff_graph(
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_modes=num_modes,
        ncv=ncv,
        max_arnoldi=ncv,
    )
    assert result.eigenvalues.dtype == torch.complex64
    assert result.residual_norms.dtype == torch.float32
    assert result.num_modes == num_modes

    with pytest.raises(ValueError, match="max_arnoldi and ncv must agree"):
        spectrum_k_eff_graph(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_modes=num_modes,
            ncv=ncv,
            max_arnoldi=ncv + 1,
        )


def test_spectrum_k_eff_graph_raises_when_tol_too_tight() -> None:
    """Absurdly tight tol with tiny ncv fails clearly."""
    num_nodes = 5
    with pytest.raises(ValueError, match="Arnoldi spectrum found only"):
        spectrum_k_eff_graph(
            k_self=torch.tensor([[0.6, 0.1], [0.0, 0.5]]),
            k_nbr=torch.tensor([[0.05, 0.0], [0.0, 0.04]]),
            edge_index=_cycle_edge_index(num_nodes),
            num_nodes=num_nodes,
            num_modes=4,
            ncv=4,
            tol=1e-30,
        )


def test_apply_k_eff_graph_rejects_mismatched_factor_shapes() -> None:
    """Square K factors must match latent_dim."""
    with pytest.raises(ValueError, match="k_self must have shape"):
        apply_k_eff_graph(
            torch.zeros(4),
            k_self=torch.ones(2, 3),
            k_nbr=torch.eye(2),
            edge_index=_cycle_edge_index(2),
            num_nodes=2,
        )
    with pytest.raises(ValueError, match="k_nbr must have shape"):
        apply_k_eff_graph(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_nbr=torch.eye(3),
            edge_index=_cycle_edge_index(2),
            num_nodes=2,
        )


def test_apply_k_eff_hetero_rejects_edge_weights_length_mismatch() -> None:
    """Per-relation edge_weights must align with k_relations."""
    with pytest.raises(ValueError, match="edge_weights length"):
        apply_k_eff_hetero(
            torch.zeros(4),
            k_self=torch.eye(2),
            k_relations=[torch.eye(2), torch.eye(2)],
            edge_indices=[_cycle_edge_index(2), _cycle_edge_index(2)],
            num_nodes=2,
            edge_weights=[None],
        )


def test_invert_square_factor_validation() -> None:
    """Preconditioner inversion rejects non-square and singular factors."""
    with pytest.raises(ValueError, match="square 2-D matrix"):
        _invert_square_factor(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="singular"):
        _invert_square_factor(torch.zeros(2, 2))


def test_richardson_invert_immediate_convergence() -> None:
    """Zero initial residual returns without Richardson updates."""
    rhs = torch.tensor([1.0, 2.0, 3.0])
    result = _richardson_invert(
        lambda x: x,
        lambda residual: residual,
        rhs,
        max_iters=8,
        tol=1e-6,
        x0=rhs.clone(),
    )
    assert isinstance(result, MatrixFreeInverseResult)
    assert result.converged
    assert result.iterations == 0


def test_arnoldi_eigs_rejects_invalid_parameters() -> None:
    """Arnoldi helper validates dim, tol, num_modes, and ncv."""

    def apply_k(flat: torch.Tensor) -> torch.Tensor:
        return flat

    with pytest.raises(ValueError, match="dim must be positive"):
        _arnoldi_eigs(
            apply_k,
            0,
            num_modes=1,
            tol=1e-6,
            ncv=1,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="tol must be positive"):
        _arnoldi_eigs(
            apply_k,
            4,
            num_modes=1,
            tol=0.0,
            ncv=2,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="ncv must be in"):
        _arnoldi_eigs(
            apply_k,
            4,
            num_modes=2,
            tol=1e-6,
            ncv=1,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="num_modes must be in"):
        _arnoldi_eigs(
            apply_k,
            4,
            num_modes=5,
            tol=1e-6,
            ncv=4,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
