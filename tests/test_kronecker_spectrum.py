"""Tests for Kronecker-sum eigenvalue reduction helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator
from koopman_graph.operators.graph import GraphKoopmanOperator
from koopman_graph.operators.kronecker_spectrum import (
    DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
    eigenvalues_k_eff_kronecker_sum,
    kronecker_sum_spectrum_eligible,
    spectrum_k_eff_kronecker_sum,
    spectrum_l_eff_kronecker_sum,
)
from koopman_graph.spectrum_types import compute_generator_spectrum, compute_spectrum


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected path edge index."""
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected ring edge index."""
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def _eigvals_match(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Greedy multiset match of complex eigenvalues (phase/order invariant).

    Sorting by real/imag flips conjugate-pair order under float noise; matching
    each left value to the nearest unused right value is stable for small N·d.
    """
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


@pytest.mark.parametrize(
    ("adjacency", "sparsity", "shared_self", "expected"),
    [
        ("symmetric", "dense", True, True),
        ("symmetric", "block_diagonal", True, True),
        ("random_walk", "dense", True, True),
        ("dual_random_walk", "dense", True, False),
        ("symmetric", "distributed", True, False),
        ("symmetric", "dense", False, False),
        ("random_walk", "block_diagonal", False, False),
    ],
)
def test_eligibility_appendix_c(
    adjacency: str,
    sparsity: str,
    shared_self: bool,
    expected: bool,
) -> None:
    """Eligibility matches design Appendix C."""
    assert (
        kronecker_sum_spectrum_eligible(
            adjacency=adjacency,
            sparsity=sparsity,
            shared_self=shared_self,
        )
        is expected
    )


@pytest.mark.parametrize("topology", ["path", "ring"])
@pytest.mark.parametrize("adjacency", ["symmetric", "random_walk"])
@pytest.mark.parametrize(("num_nodes", "latent_dim"), [(4, 2), (6, 3)])
def test_eigenvalues_match_dense_effective_matrix(
    topology: str,
    adjacency: str,
    num_nodes: int,
    latent_dim: int,
) -> None:
    """Kronecker eigvals match dense effective_matrix within float32 noise.

    Tolerance: relative 1e-5 with absolute floor 1e-5 — justified by float32
    ``eigh``/``eig`` accumulation on small teaching graphs (N·d ≤ 18).
    """
    torch.manual_seed(0)
    if topology == "path":
        edge_index = _path_edge_index(num_nodes)
    else:
        edge_index = _ring_edge_index(num_nodes)
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity_noise",
        init_scale=0.05,
        adjacency=adjacency,  # type: ignore[arg-type]
    )
    dense = torch.linalg.eigvals(op.effective_matrix(edge_index, num_nodes))
    reduced = eigenvalues_k_eff_kronecker_sum(
        k_self=op.K_self,
        k_nbr=op.K_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
    )
    assert reduced is not None
    assert reduced.shape == (num_nodes * latent_dim,)
    assert _eigvals_match(reduced, dense, rtol=1e-5, atol=1e-5)


def test_nonfinite_factors_return_none() -> None:
    """NaN factors refuse the reduction."""
    edge_index = _path_edge_index(3)
    k_self = torch.eye(2)
    k_nbr = torch.zeros(2, 2)
    k_self[0, 0] = torch.nan
    assert (
        eigenvalues_k_eff_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=3,
            adjacency="symmetric",
        )
        is None
    )


def test_dual_adjacency_returns_none() -> None:
    """Dual adjacency is unsupported by the eigenvalue helper."""
    edge_index = _path_edge_index(3)
    assert (
        eigenvalues_k_eff_kronecker_sum(
            k_self=torch.eye(2),
            k_nbr=torch.zeros(2, 2),
            edge_index=edge_index,
            num_nodes=3,
            adjacency="dual_random_walk",
        )
        is None
    )


def test_random_walk_residual_failure_returns_none() -> None:
    """Failed random-walk residual check returns None for dense fall-back."""
    edge_index = _path_edge_index(4)
    k_self = torch.eye(2)
    k_nbr = 0.1 * torch.randn(2, 2)

    def _bad_decomp(adj: torch.Tensor, *, residual_tol: float):
        _ = residual_tol
        # Valid shapes but deliberately failing residual via mock return of None
        # through a huge residual path: return eigenvectors that do not diagonalize.
        eigenvalues = torch.ones(adj.shape[0], dtype=torch.complex64)
        eigenvectors = torch.eye(adj.shape[0], dtype=torch.complex64)
        reconstructed = eigenvectors @ torch.diag(eigenvalues)
        target = adj.to(dtype=eigenvectors.dtype) @ eigenvectors
        # Force residual path by patching vector_norm ratio — instead return None
        # from the private helper by calling through with tiny tol after bad V.
        _ = reconstructed, target
        return None

    with patch(
        "koopman_graph.operators.kronecker_spectrum._random_walk_eigendecomposition",
        side_effect=_bad_decomp,
    ):
        result = eigenvalues_k_eff_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=4,
            adjacency="random_walk",
            residual_tol=DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
        )
    assert result is None


def test_random_walk_residual_tol_rejects_bad_basis() -> None:
    """Injected identity 'eigenbasis' fails residual and yields None."""
    edge_index = _path_edge_index(5)
    k_self = torch.eye(2)
    k_nbr = torch.zeros(2, 2)

    real_eig = torch.linalg.eig

    def _spoof_eig(matrix: torch.Tensor):
        # Preserve dtype/device but replace V with I so ÂV ≠ VΛ for RW Â.
        values, _vectors = real_eig(matrix)
        identity = torch.eye(
            matrix.shape[0],
            dtype=_vectors.dtype,
            device=matrix.device,
        )
        return values, identity

    with patch("torch.linalg.eig", side_effect=_spoof_eig):
        result = eigenvalues_k_eff_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=5,
            adjacency="random_walk",
            residual_tol=1e-6,
        )
    assert result is None


def test_orbit_shared_self_false_is_ineligible() -> None:
    """Orbit / isotypic self banks are gated by shared_self=False."""
    assert not kronecker_sum_spectrum_eligible(
        adjacency="symmetric",
        sparsity="dense",
        shared_self=False,
    )


@pytest.mark.parametrize("topology", ["path", "ring"])
@pytest.mark.parametrize("adjacency", ["symmetric", "random_walk"])
@pytest.mark.parametrize(("num_nodes", "latent_dim"), [(4, 2), (6, 3)])
def test_spectrum_matches_compute_spectrum_and_residuals(
    topology: str,
    adjacency: str,
    num_nodes: int,
    latent_dim: int,
) -> None:
    """Full Kronecker spectrum matches dense compute_spectrum; modes residual-small.

    Eigenvalue / metadata tolerance: rtol=atol=1e-5 (float32). Ambient mode
    residual ||K_eff u - μ u|| / ||u|| ≤ 1e-4 — phase-invariant subspace
    check justified by float32 kron + eig accumulation on N·d ≤ 18.
    """
    torch.manual_seed(1)
    time_step = 0.1
    if topology == "path":
        edge_index = _path_edge_index(num_nodes)
    else:
        edge_index = _ring_edge_index(num_nodes)
    op = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity_noise",
        init_scale=0.05,
        adjacency=adjacency,  # type: ignore[arg-type]
    )
    effective = op.effective_matrix(edge_index, num_nodes)
    dense = compute_spectrum(effective, time_step)
    reduced = spectrum_k_eff_kronecker_sum(
        k_self=op.K_self,
        k_nbr=op.K_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        time_step=time_step,
    )
    assert reduced is not None
    ambient = num_nodes * latent_dim
    assert reduced.eigenvalues.shape == (ambient,)
    assert reduced.eigenvectors.shape == (ambient, ambient)
    assert reduced.time_step == time_step
    assert _eigvals_match(
        reduced.eigenvalues,
        dense.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(reduced.magnitudes, reduced.eigenvalues.abs(), atol=0.0)
    assert torch.allclose(
        reduced.growth_rates,
        torch.log(reduced.magnitudes.clamp_min(1e-30)) / time_step,
        atol=1e-6,
    )
    assert torch.allclose(
        reduced.frequencies,
        torch.angle(reduced.eigenvalues) / (2 * torch.pi * time_step),
        atol=1e-6,
    )
    # Columns are unit-normalized.
    col_norms = torch.linalg.vector_norm(reduced.eigenvectors, dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    effective_c = effective.to(dtype=reduced.eigenvectors.dtype)
    for mode in range(ambient):
        vector = reduced.eigenvectors[:, mode]
        residual = torch.linalg.vector_norm(
            effective_c @ vector - reduced.eigenvalues[mode] * vector
        ) / torch.linalg.vector_norm(vector).clamp_min(1e-30)
        assert float(residual.detach()) <= 1e-4


def test_spectrum_nonfinite_time_step_or_factors_return_none() -> None:
    """Invalid time_step or NaN factors yield None."""
    edge_index = _path_edge_index(3)
    k_self = torch.eye(2)
    k_nbr = torch.zeros(2, 2)
    assert (
        spectrum_k_eff_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=3,
            adjacency="symmetric",
            time_step=0.0,
        )
        is None
    )
    k_self = k_self.clone()
    k_self[0, 0] = torch.nan
    assert (
        spectrum_k_eff_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=3,
            adjacency="symmetric",
            time_step=1.0,
        )
        is None
    )


def test_spectrum_random_walk_residual_failure_returns_none() -> None:
    """Spectrum helper returns None when RW adjacency residual fails."""
    edge_index = _path_edge_index(4)
    with patch(
        "koopman_graph.operators.kronecker_spectrum._random_walk_eigendecomposition",
        return_value=None,
    ):
        result = spectrum_k_eff_kronecker_sum(
            k_self=torch.eye(2),
            k_nbr=torch.zeros(2, 2),
            edge_index=edge_index,
            num_nodes=4,
            adjacency="random_walk",
            time_step=1.0,
        )
    assert result is None


def test_mode_amplitudes_roundtrip_smoke() -> None:
    """Reconstructed eigenvectors support KoopmanSpectrum.mode_amplitudes."""
    torch.manual_seed(2)
    num_nodes = 4
    latent_dim = 2
    edge_index = _path_edge_index(num_nodes)
    op = GraphKoopmanOperator(latent_dim, init_mode="identity_noise", init_scale=0.05)
    spectrum = spectrum_k_eff_kronecker_sum(
        k_self=op.K_self,
        k_nbr=op.K_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency="symmetric",
        time_step=1.0,
    )
    assert spectrum is not None
    flat = torch.randn(num_nodes * latent_dim, dtype=torch.float32)
    amplitudes = spectrum.mode_amplitudes(flat)
    coeffs = amplitudes.to(spectrum.eigenvectors.dtype)
    reconstructed = (spectrum.eigenvectors @ coeffs).real
    assert torch.allclose(reconstructed, flat, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("topology", ["path", "ring"])
@pytest.mark.parametrize("adjacency", ["symmetric", "random_walk"])
@pytest.mark.parametrize(("num_nodes", "latent_dim"), [(4, 2), (6, 3)])
def test_spectrum_l_eff_matches_generator_spectrum_and_residuals(
    topology: str,
    adjacency: str,
    num_nodes: int,
    latent_dim: int,
) -> None:
    """Continuous Kronecker spectrum matches dense generator oracle + residuals.

    Tolerance rtol=atol=1e-5 (float32); residual ≤ 1e-4 on ambient columns.
    """
    torch.manual_seed(4)
    edge_index = (
        _path_edge_index(num_nodes)
        if topology == "path"
        else _ring_edge_index(num_nodes)
    )
    op = ContinuousGraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        adjacency=adjacency,  # type: ignore[arg-type]
    )
    if latent_dim == 2:
        l_self = torch.tensor([[-0.4, 0.1], [-0.05, -0.5]])
    else:
        l_self = -0.5 * torch.eye(latent_dim) + 0.05 * torch.randn(
            latent_dim, latent_dim
        )
    op.set_dense_matrices(l_self, 0.12 * torch.eye(latent_dim))
    generator = op.effective_generator(edge_index, num_nodes)
    dense = compute_generator_spectrum(generator)
    reduced = spectrum_l_eff_kronecker_sum(
        l_self=op.L_self,
        l_nbr=op.L_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
    )
    assert reduced is not None
    ambient = num_nodes * latent_dim
    assert reduced.eigenvalues.shape == (ambient,)
    assert reduced.eigenvectors.shape == (ambient, ambient)
    assert reduced.time_step == 1.0
    assert _eigvals_match(
        reduced.eigenvalues,
        dense.eigenvalues,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(reduced.growth_rates, reduced.eigenvalues.real, atol=0.0)
    assert torch.allclose(
        reduced.frequencies,
        reduced.eigenvalues.imag / (2 * torch.pi),
        atol=1e-6,
    )
    generator_c = generator.to(dtype=reduced.eigenvectors.dtype)
    for mode in range(ambient):
        vector = reduced.eigenvectors[:, mode]
        residual = torch.linalg.vector_norm(
            generator_c @ vector - reduced.eigenvalues[mode] * vector
        ) / torch.linalg.vector_norm(vector).clamp_min(1e-30)
        assert float(residual.detach()) <= 1e-4
