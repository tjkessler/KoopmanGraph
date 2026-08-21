"""Targeted coverage for operator and spectrum validation branches."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from koopman_graph.operators import (
    BoundaryNilpotencyReport,
    CochainKoopmanOperator,
    CochainState,
    GraphKoopmanOperator,
    MatrixFreeGraphLinearOperator,
    ParametricKoopmanOperator,
    PolynomialGraphLinearOperator,
    boundary_nilpotency,
    leave_one_regime_out,
)
from koopman_graph.operators.linear import (
    _SymmetricGraphLinearMap,
    _taylor_expm_action,
)
from koopman_graph.operators.parametric import LENGTHSCALE_FLOOR
from koopman_graph.spectrum_types import (
    DefectiveSpectrumError,
    KoopmanSpectrum,
    SpectralDiagnostics,
    compute_generator_spectrum,
    compute_spectral_diagnostics,
    compute_spectrum,
    discrete_spectrum_at_delta_t,
)

_N_NODES = 3
_LATENT_DIM = 2
_ATOL = 1e-6


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Undirected path used as a small teaching graph."""
    tails = torch.arange(num_nodes - 1, dtype=torch.long)
    forward = torch.stack((tails, tails + 1), dim=0)
    return torch.cat((forward, forward.flip(0)), dim=1)


def _identity_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Well-conditioned self / neighbor factors."""
    identity = torch.eye(_LATENT_DIM, dtype=torch.float64)
    return 0.6 * identity, 0.05 * identity


def _one_tap_operator() -> MatrixFreeGraphLinearOperator:
    """Tiny one-tap wrapper for public validation paths."""
    k_self, k_nbr = _identity_pair()
    return MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=_path_edges(_N_NODES),
        num_nodes=_N_NODES,
    )


def _poly_operator() -> PolynomialGraphLinearOperator:
    """Degree-2 wrapper on the same path graph."""
    k_self, k_nbr = _identity_pair()
    k_two = 0.01 * torch.eye(_LATENT_DIM, dtype=torch.float64)
    return PolynomialGraphLinearOperator(
        (k_self, k_nbr, k_two),
        edge_index=_path_edges(_N_NODES),
        num_nodes=_N_NODES,
    )


def _valid_diagnostics(*, n: int = 2) -> SpectralDiagnostics:
    """Minimal finite diagnostics aligned with ``n`` modes."""
    return SpectralDiagnostics(
        eigenvector_condition=1.0,
        eigenvalue_sensitivities=torch.ones(n, dtype=torch.float64),
        departure_from_normality=0.0,
        departure_from_normality_relative=0.0,
        nyquist_frequency=0.5,
        aliasing_warning=torch.zeros(n, dtype=torch.bool),
        sign_of_eigenvalue=torch.ones(n, dtype=torch.float64),
    )


def _spectrum_from_vectors(
    vectors: torch.Tensor,
    *,
    diagnostics: SpectralDiagnostics | None = None,
) -> KoopmanSpectrum:
    """Build a spectrum with prescribed eigenvectors."""
    n = int(vectors.shape[0])
    eigenvalues = torch.ones(n, dtype=torch.complex128)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=vectors.to(dtype=torch.complex128),
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.zeros(n, dtype=torch.float64),
        frequencies=torch.zeros(n, dtype=torch.float64),
        time_step=1.0,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# cochain.py
# ---------------------------------------------------------------------------


def test_cochain_state_rejects_non_tensor_and_bad_dtypes() -> None:
    """Feature tables must be real finite floating-point matrices."""
    edge = torch.ones(2, 2)
    with pytest.raises(TypeError, match="must be a Tensor"):
        CochainState(node="latents", edge=edge)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be real"):
        CochainState(node=torch.ones(2, 2, dtype=torch.complex64), edge=edge)
    with pytest.raises(ValueError, match="floating-point"):
        CochainState(node=torch.ones(2, 2, dtype=torch.int64), edge=edge)
    with pytest.raises(ValueError, match="n_cells, d"):
        CochainState(node=torch.ones(2), edge=torch.ones(2))
    with pytest.raises(ValueError, match="latent width must be >= 1"):
        CochainState(node=torch.ones(2, 0), edge=torch.ones(2, 0))


def test_boundary_nilpotency_report_and_helper_guards() -> None:
    """Nilpotency report scalars and incidence tables are validated."""
    product = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="product must have shape"):
        BoundaryNilpotencyReport(
            product=torch.zeros(3),
            max_abs=0.0,
            atol=1e-6,
            nilpotent=True,
        )
    with pytest.raises(ValueError, match="max_abs"):
        BoundaryNilpotencyReport(
            product=product,
            max_abs=float("nan"),
            atol=1e-6,
            nilpotent=False,
        )
    with pytest.raises(ValueError, match="max_abs"):
        BoundaryNilpotencyReport(
            product=product,
            max_abs=-0.1,
            atol=1e-6,
            nilpotent=False,
        )
    with pytest.raises(ValueError, match="atol"):
        BoundaryNilpotencyReport(
            product=product,
            max_abs=0.0,
            atol=-1.0,
            nilpotent=True,
        )
    with pytest.raises(TypeError, match="must be tensors"):
        boundary_nilpotency(torch.ones(2, 2), "b2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 2-D"):
        boundary_nilpotency(torch.ones(3), torch.ones(3, 1))
    with pytest.raises(ValueError, match="must be finite"):
        boundary_nilpotency(
            torch.tensor([[float("nan"), 0.0]]),
            torch.ones(2, 1),
        )
    with pytest.raises(ValueError, match="atol must be a finite"):
        boundary_nilpotency(torch.ones(2, 2), torch.ones(2, 1), atol=float("inf"))


def test_cochain_operator_constructor_and_advance_guards() -> None:
    """Operator construction and incidence alignment name the broken field."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="latent_dim"):
        CochainKoopmanOperator(True, edge_index, num_nodes=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latent_dim"):
        CochainKoopmanOperator(0, edge_index, num_nodes=3)
    with pytest.raises(ValueError, match="num_nodes"):
        CochainKoopmanOperator(2, edge_index, num_nodes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="use_cross_degree"):
        CochainKoopmanOperator(2, edge_index, num_nodes=3, use_cross_degree=1)  # type: ignore[arg-type]
    operator = CochainKoopmanOperator(2, edge_index, num_nodes=3)
    with pytest.raises(ValueError, match="num_edges"):
        operator.advance(
            CochainState(node=torch.ones(3, 2), edge=torch.ones(1, 2)),
        )
    wide = CochainState(node=torch.ones(3, 3), edge=torch.ones(2, 3))
    with pytest.raises(ValueError, match="latent width must equal latent_dim"):
        operator.advance(wide)
    state = CochainState(node=torch.ones(3, 2), edge=torch.zeros(2, 2))
    got = operator.forward(state)
    assert got.node.shape == (3, 2)
    assert got.edge.shape == (2, 2)


# ---------------------------------------------------------------------------
# linear.py
# ---------------------------------------------------------------------------


def test_linear_operator_flat_vector_guards() -> None:
    """Public matvec / residual paths refuse non-flat real vectors."""
    operator = _one_tap_operator()
    dim = operator.dim
    with pytest.raises(TypeError, match="must be a Tensor"):
        operator.matvec([0.0] * dim)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be real"):
        operator.matvec(torch.ones(dim, dtype=torch.complex128))
    with pytest.raises(ValueError, match="floating-point"):
        operator.matvec(torch.ones(dim, dtype=torch.int64))
    with pytest.raises(ValueError, match="must have shape"):
        operator.matvec(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="must be finite"):
        operator.residual_norm(
            torch.ones(dim, dtype=torch.float64),
            torch.tensor([float("nan")] * dim, dtype=torch.float64),
        )


def test_linear_operator_expm_action_guards_and_zero_time() -> None:
    """Taylor action rejects bad time / term counts and is identity at t=0."""
    operator = _one_tap_operator()
    rhs = torch.arange(operator.dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="t must be a finite float"):
        operator.expm_action(True, rhs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="t must be finite"):
        operator.expm_action(float("inf"), rhs)
    got = operator.expm_action(0.0, rhs)
    assert torch.equal(got, rhs)
    assert got is not rhs
    with pytest.raises(ValueError, match="n_terms"):
        _taylor_expm_action(operator.matvec, 0.1, rhs, n_terms=0)


def test_linear_operator_hop_and_topology_guards() -> None:
    """Hop banks and topology are checked before any matvec."""
    edges = _path_edges(_N_NODES)
    identity = torch.eye(_LATENT_DIM, dtype=torch.float64)
    empty = torch.zeros(0, 0, dtype=torch.float64)
    with pytest.raises(ValueError, match="hop_matrices must be non-empty"):
        _SymmetricGraphLinearMap((), edges, _N_NODES, None)
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        PolynomialGraphLinearOperator(
            (empty, empty, empty),
            edge_index=edges,
            num_nodes=_N_NODES,
        )
    with pytest.raises(TypeError, match="hop_matrices\\[1\\] must be a Tensor"):
        PolynomialGraphLinearOperator(
            (identity, "nbr", identity),  # type: ignore[list-item]
            edge_index=edges,
            num_nodes=_N_NODES,
        )
    with pytest.raises(ValueError, match="must have shape"):
        PolynomialGraphLinearOperator(
            (identity, torch.ones(2, 3, dtype=torch.float64), identity),
            edge_index=edges,
            num_nodes=_N_NODES,
        )
    with pytest.raises(ValueError, match="real floating-point"):
        PolynomialGraphLinearOperator(
            (identity, identity.to(dtype=torch.complex128), identity),
            edge_index=edges,
            num_nodes=_N_NODES,
        )
    nan_hop = identity.clone()
    nan_hop[0, 0] = float("nan")
    with pytest.raises(ValueError, match="hop_matrices must be finite"):
        PolynomialGraphLinearOperator(
            (identity, nan_hop, identity),
            edge_index=edges,
            num_nodes=_N_NODES,
        )
    with pytest.raises(ValueError, match="num_nodes"):
        PolynomialGraphLinearOperator(
            (identity, identity, identity),
            edge_index=edges,
            num_nodes=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="num_nodes"):
        MatrixFreeGraphLinearOperator(
            identity,
            identity,
            edge_index=edges,
            num_nodes=0,
        )
    with pytest.raises(ValueError, match="edge_index must have shape"):
        MatrixFreeGraphLinearOperator(
            identity,
            identity,
            edge_index=torch.ones(3),
            num_nodes=_N_NODES,
        )
    with pytest.raises(ValueError, match="edge_index must have shape"):
        MatrixFreeGraphLinearOperator(
            identity,
            identity,
            edge_index=torch.ones(3, 2, dtype=torch.long),
            num_nodes=_N_NODES,
        )


def test_linear_operator_leading_eigpairs_guards_and_polynomial_path() -> None:
    """Arnoldi wrappers validate ``k``; the polynomial path returns Ritz pairs."""
    one_tap = _one_tap_operator()
    poly = _poly_operator()
    with pytest.raises(ValueError, match="k must be a positive int"):
        one_tap.leading_eigpairs(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be a positive int"):
        poly.leading_eigpairs(0)
    with pytest.raises(ValueError, match="k must be <="):
        poly.leading_eigpairs(poly.dim + 1)
    result = poly.leading_eigpairs(1, tol=1e-3)
    assert result.eigenvalues.shape == (1,)
    assert result.residual_norms.shape == (1,)


# ---------------------------------------------------------------------------
# graph.py
# ---------------------------------------------------------------------------


def test_graph_higher_hop_reset_bound_and_set_dense_guards() -> None:
    """``P>=2`` allocates hop banks; dense writes check factor lengths."""
    identity = torch.eye(2)
    dual = GraphKoopmanOperator(
        2,
        init_mode="identity",
        filter_degree=2,
        adjacency="dual_random_walk",
    )
    dual.reset_parameters()
    metric = dual.bound_metric()
    assert metric.ndim == 0
    assert torch.isfinite(metric)
    unit = GraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="hop_matrices requires filter_degree"):
        unit.set_dense_matrices(identity, identity, hop_matrices=(identity,))
    poly = GraphKoopmanOperator(2, init_mode="identity", filter_degree=2)
    with pytest.raises(ValueError, match="hop_matrices must have length"):
        poly.set_dense_matrices(identity, identity, hop_matrices=(identity, identity))
    with pytest.raises(ValueError, match="bwd_hop_matrices is only valid"):
        poly.set_dense_matrices(identity, identity, bwd_hop_matrices=(identity,))
    with pytest.raises(ValueError, match="bwd_hop_matrices must have length"):
        dual.set_dense_matrices(
            identity,
            identity,
            bwd_hop_matrices=(identity, identity),
        )
    assert unit._extra_hop_matrices() == ()
    assert unit._extra_bwd_hop_matrices() == ()


def test_graph_higher_hop_symmetric_random_walk_and_dual() -> None:
    """Dense and sparse ``P=2`` paths run for every adjacency kind."""
    edges = _path_edges(3)
    identity = 0.5 * torch.eye(2)
    hop = 0.04 * torch.eye(2)
    z = torch.tensor([[0.2, -0.1], [0.0, 0.3], [0.4, 0.1]])
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        operator = GraphKoopmanOperator(
            2,
            init_mode="identity",
            filter_degree=2,
            adjacency=adjacency,  # type: ignore[arg-type]
        )
        kwargs: dict[str, Any] = {"hop_matrices": (hop,)}
        if adjacency == "dual_random_walk":
            kwargs["k_bwd"] = 0.08 * torch.eye(2)
            kwargs["bwd_hop_matrices"] = (0.02 * torch.eye(2),)
        operator.set_dense_matrices(identity, 0.1 * torch.eye(2), **kwargs)
        advanced = operator(z, edges)
        assembled = operator.effective_matrix(edges, 3)
        dense_next = (assembled @ z.reshape(-1)).view_as(z)
        assert torch.allclose(advanced, dense_next, atol=1e-5), adjacency


def test_graph_effective_matrix_self_block_overrides() -> None:
    """``P=0`` and ``P>=1`` reject malformed per-node self blocks."""
    edges = _path_edges(3)
    identity = torch.eye(2)
    self_only = GraphKoopmanOperator(2, init_mode="identity", filter_degree=0)
    kron = self_only.effective_matrix(edges, 3, k_self=0.4 * identity)
    assert kron.shape == (6, 6)
    assert torch.allclose(kron, torch.kron(torch.eye(3), 0.4 * identity), atol=_ATOL)
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        self_only.effective_matrix(edges, 3, k_self_blocks=torch.ones(2, 2))
    blocks = torch.stack([0.4 * identity] * 3)
    blocked = self_only.effective_matrix(edges, 3, k_self_blocks=blocks)
    assert blocked.shape == (6, 6)
    assert torch.allclose(blocked, torch.block_diag(*blocks.unbind(0)), atol=_ATOL)
    one_hop = GraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        one_hop.effective_matrix(edges, 3, k_self_blocks=torch.ones(3, 2))


def test_graph_bilinear_inverse_requires_control_after_dim_cleared() -> None:
    """Inner bilinear inverse still demands a control once ``control_dim`` is 0."""
    edges = _path_edges(3)
    operator = GraphKoopmanOperator(
        2,
        init_mode="identity",
        control_dim=1,
        control_mode="bilinear",
    )
    operator.control_dim = 0
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="control input is required"):
        operator.inverse_advance(z, edge_index=edges)
    with pytest.raises(ValueError, match="Per-node control has"):
        operator.inverse_advance(z, control=torch.ones(2, 1), edge_index=edges)
    with pytest.raises(ValueError, match="control input must have shape"):
        operator.inverse_advance(z, control=torch.ones(1, 1, 1), edge_index=edges)


# ---------------------------------------------------------------------------
# parametric.py
# ---------------------------------------------------------------------------


def test_parametric_ridge_edmd_and_lengthscale_guards() -> None:
    """Leave-one-out helpers and length-scale writes reject invalid inputs."""
    z = torch.eye(2)
    with pytest.raises(ValueError, match="2-D latents"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), torch.ones(3), torch.ones(3)),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=1,
        )
    with pytest.raises(ValueError, match="must share shape"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), torch.ones(3, 2), torch.ones(2, 2)),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=1,
        )
    with pytest.raises(ValueError, match="at least one pair"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), torch.zeros(0, 2), torch.zeros(0, 2)),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=1,
        )
    with pytest.raises(ValueError, match="held-out mu"):
        leave_one_regime_out(
            [
                (torch.tensor(0.0), z, z),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=0,
        )
    operator = ParametricKoopmanOperator(2)
    with pytest.raises(ValueError, match="positivity floor"):
        operator.set_lengthscale(LENGTHSCALE_FLOOR)
    with pytest.raises(ValueError, match="anchors must be finite"):
        operator.set_anchors(torch.tensor([[0.0], [float("nan")]]))
    with pytest.raises(ValueError, match="parameters must be finite"):
        operator.advance(torch.randn(3, 2), parameters=torch.tensor([float("nan")]))


def test_leave_one_regime_out_requires_a_train_regime() -> None:
    """A length-2 sequence that only yields the hold-out has no train map."""

    class _HoldOutOnly:
        """Sequence whose iteration stops after the held-out regime."""

        def __len__(self) -> int:
            return 2

        def __getitem__(
            self, index: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if index != 0:
                raise IndexError
            pairs = torch.eye(2)
            return torch.tensor([0.0]), pairs, pairs

    with pytest.raises(ValueError, match="at least one train regime"):
        leave_one_regime_out(_HoldOutOnly(), hold_out=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# spectrum_types.py
# ---------------------------------------------------------------------------


def test_spectral_diagnostics_field_guards() -> None:
    """Diagnostic tensors and scalars reject empty, NaN, and out-of-range values."""
    good = _valid_diagnostics()
    with pytest.raises(ValueError, match="nonempty 1-D"):
        replace(good, eigenvalue_sensitivities=torch.ones(2, 1))
    with pytest.raises(ValueError, match="must not contain NaN"):
        replace(
            good,
            eigenvalue_sensitivities=torch.tensor([1.0, float("nan")]),
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(good, eigenvalue_sensitivities=torch.tensor([1.0, -0.1]))
    with pytest.raises(ValueError, match="aliasing_warning must have shape"):
        replace(good, aliasing_warning=torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="aliasing_warning must be a bool"):
        replace(good, aliasing_warning=torch.zeros(2))
    with pytest.raises(ValueError, match="sign_of_eigenvalue must have shape"):
        replace(good, sign_of_eigenvalue=torch.ones(1))
    with pytest.raises(ValueError, match="sign_of_eigenvalue must be real"):
        replace(
            good,
            sign_of_eigenvalue=torch.ones(2, dtype=torch.complex128),
        )
    with pytest.raises(ValueError, match="sign_of_eigenvalue must not contain NaN"):
        replace(good, sign_of_eigenvalue=torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match=r"must be in \{-1, 0, 1\}"):
        replace(good, sign_of_eigenvalue=torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="nyquist_frequency"):
        replace(good, nyquist_frequency=0.0)
    with pytest.raises(ValueError, match="must not be NaN"):
        replace(good, eigenvector_condition=float("nan"))
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(good, departure_from_normality=-1.0)


def test_koopman_spectrum_rejects_misaligned_diagnostics() -> None:
    """Attached diagnostics must match the stored eigenpair count."""
    diag = _valid_diagnostics(n=2)
    with pytest.raises(ValueError, match="diagnostics.eigenvalue_sensitivities"):
        KoopmanSpectrum(
            eigenvalues=torch.ones(3, dtype=torch.complex128),
            eigenvectors=torch.eye(3, dtype=torch.complex128),
            magnitudes=torch.ones(3),
            growth_rates=torch.zeros(3),
            frequencies=torch.zeros(3),
            time_step=1.0,
            diagnostics=diag,
        )


def test_compute_spectral_diagnostics_shape_guards() -> None:
    """The diagnostic helper names square-size and emptiness failures."""
    matrix = torch.eye(2, dtype=torch.float64)
    vectors = torch.eye(2, dtype=torch.complex128)
    values = torch.ones(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="matrix must be square"):
        compute_spectral_diagnostics(
            torch.ones(2, 3),
            vectors,
            eigenvalues=values,
        )
    with pytest.raises(ValueError, match="eigenvectors must be square"):
        compute_spectral_diagnostics(
            matrix,
            torch.ones(2, 3, dtype=torch.complex128),
            eigenvalues=values,
        )
    with pytest.raises(ValueError, match="must share dimension"):
        compute_spectral_diagnostics(
            matrix,
            torch.eye(3, dtype=torch.complex128),
            eigenvalues=torch.ones(3, dtype=torch.complex128),
        )
    with pytest.raises(ValueError, match="matrix must be non-empty"):
        compute_spectral_diagnostics(
            torch.zeros(0, 0),
            torch.zeros(0, 0),
            eigenvalues=torch.zeros(0),
        )


def test_mode_amplitudes_wraps_solve_failure_as_defective() -> None:
    """A finite stored κ(V) still raises when the solve is singular."""
    vectors = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    spectrum = _spectrum_from_vectors(vectors, diagnostics=_valid_diagnostics())
    with pytest.raises(DefectiveSpectrumError, match="Schur"):
        spectrum.mode_amplitudes(torch.ones(2, dtype=torch.float64))


def test_koopman_spectrum_residual_and_amplitude_guards() -> None:
    """Residuals must align with eigenpairs; amplitudes need a trailing dim."""
    eigenvalues = torch.ones(2, dtype=torch.complex128)
    vectors = torch.eye(2, dtype=torch.complex128)
    kwargs = {
        "eigenvalues": eigenvalues,
        "eigenvectors": vectors,
        "magnitudes": eigenvalues.abs(),
        "growth_rates": torch.zeros(2),
        "frequencies": torch.zeros(2),
        "time_step": 1.0,
    }
    with pytest.raises(ValueError, match="residuals must have shape"):
        KoopmanSpectrum(**kwargs, residuals=torch.ones(3))
    with pytest.raises(ValueError, match="residuals must be finite"):
        KoopmanSpectrum(**kwargs, residuals=torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="residuals must be non-negative"):
        KoopmanSpectrum(**kwargs, residuals=torch.tensor([1.0, -0.1]))
    spectrum = KoopmanSpectrum(**kwargs, residuals=torch.zeros(2))
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        spectrum.mode_amplitudes(torch.tensor(1.0))
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        spectrum.mode_amplitudes(torch.ones(2, 3))


def test_spectrum_assembly_helpers_reject_invalid_operators() -> None:
    """Discrete / generator assembly names shape, emptiness, and Δt failures."""
    with pytest.raises(ValueError, match="operator must be a square"):
        compute_spectrum(torch.ones(2, 3), time_step=1.0)
    with pytest.raises(ValueError, match="operator must be non-empty"):
        compute_spectrum(torch.zeros(0, 0), time_step=1.0)
    with pytest.raises(ValueError, match="time_step must be positive"):
        compute_spectrum(torch.eye(2), time_step=0.0)
    with pytest.raises(TypeError, match="floating-point or complex"):
        compute_spectrum(torch.eye(2, dtype=torch.int64), time_step=1.0)
    with pytest.raises(ValueError, match="generator must be a square"):
        compute_generator_spectrum(torch.ones(3))
    with pytest.raises(ValueError, match="generator must be non-empty"):
        compute_generator_spectrum(torch.zeros(0, 0))
    with pytest.raises(TypeError, match="floating-point or complex"):
        compute_generator_spectrum(torch.eye(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="delta_t must be positive"):
        discrete_spectrum_at_delta_t(torch.eye(2), delta_t=-0.1)
    spectrum = discrete_spectrum_at_delta_t(
        torch.diag(torch.tensor([-0.5, -1.0])),
        delta_t=0.1,
    )
    assert spectrum.eigenvalues.numel() == 2


def test_condition_number_failures_are_reported_as_infinite() -> None:
    """``cond`` errors or non-finite values become ``+inf`` diagnostics."""
    matrix = torch.eye(2, dtype=torch.float64)
    vectors = torch.eye(2, dtype=torch.complex128)
    values = torch.ones(2, dtype=torch.complex128)
    spectrum = _spectrum_from_vectors(vectors)
    with patch(
        "koopman_graph.spectrum_types.torch.linalg.cond",
        side_effect=RuntimeError("cond failed"),
    ):
        with pytest.raises(DefectiveSpectrumError, match="Schur"):
            spectrum.mode_amplitudes(torch.ones(2, dtype=torch.float64))
        diag = compute_spectral_diagnostics(
            matrix,
            vectors,
            eigenvalues=values,
            time_step=1.0,
        )
    assert math.isinf(diag.eigenvector_condition)
    with patch(
        "koopman_graph.spectrum_types.torch.linalg.cond",
        return_value=torch.tensor(float("inf")),
    ):
        with pytest.raises(DefectiveSpectrumError, match="Schur"):
            spectrum.mode_amplitudes(torch.ones(2, dtype=torch.float64))
        inf_diag = compute_spectral_diagnostics(
            matrix,
            vectors,
            eigenvalues=values,
            time_step=1.0,
        )
    assert math.isinf(inf_diag.eigenvector_condition)
