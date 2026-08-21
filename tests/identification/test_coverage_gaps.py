"""Coverage for identification validation and unused error branches.

These tests target public guards and dataclass ``__post_init__`` checks
that the scientific suites leave unhit. They do not change identification
formulas or invent forecast numbers.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from koopman_graph.identification import (
    ClosedFormBackend,
    IdentificationConfig,
    LatentPairs,
    LatentRankReport,
    OperatorSnapshot,
    ResDMDGateCandidate,
    ResDMDGateResult,
    SparseFactorReport,
    SubspaceInvarianceReport,
    apply_operator_snapshot,
    build_identification_report,
    identify_operator,
    identify_sparse_graph_factors,
    select_latent_rank,
    select_resdmd_gated,
    subspace_invariance_report,
)
from koopman_graph.identification.solvers import (
    _fit_ridge_row_operator,
    _fit_tls_row_operator,
    _scale_spectral_radius,
)
from koopman_graph.identification.sparse_factors import (
    _group_masks,
    _group_soft_threshold,
)
from koopman_graph.operators import KoopmanOperator


def _line_encodings(*, n_times: int = 8, n_nodes: int = 4) -> torch.Tensor:
    """Time-major encodings on :math:`\\mathrm{span}\\{e_1\\}`.

    Parameters
    ----------
    n_times, n_nodes : int
        Layout.

    Returns
    -------
    Tensor
        ``(T, N, 2)`` float64 encodings.
    """
    encodings = torch.zeros(n_times, n_nodes, 2, dtype=torch.float64)
    encodings[:, :, 0] = torch.linspace(
        0.2, 1.6, n_times, dtype=torch.float64
    ).unsqueeze(1)
    return encodings


def _linear_pairs(
    true_k: torch.Tensor,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> LatentPairs:
    """Build consecutive encodings from a known linear map.

    Parameters
    ----------
    true_k : Tensor
        Ground-truth ``K``.
    n_samples, seed : int
        Pair count and generator seed.

    Returns
    -------
    LatentPairs
        ``z_t`` / ``z_next`` matching ``true_k`` dtype.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    dim = true_k.shape[0]
    z_t = torch.randn(n_samples, dim, dtype=true_k.dtype, generator=generator)
    return LatentPairs(z_t=z_t, z_next=z_t @ true_k.T)


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Undirected path ``edge_index``.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    Tensor
        COO edges.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _rank_report_kwargs(**overrides: object) -> dict[str, object]:
    """Valid :class:`LatentRankReport` constructor kwargs.

    Parameters
    ----------
    **overrides
        Field replacements.

    Returns
    -------
    dict
        Constructor mapping.
    """
    payload: dict[str, object] = {
        "selected_rank": 1,
        "criterion": "vamp2",
        "candidates": (1,),
        "scores": (0.5,),
        "rejected_alternatives": (),
        "numerical_rank": 1,
        "n_samples": 4,
    }
    payload.update(overrides)
    return payload


def _sparse_report_kwargs(**overrides: object) -> dict[str, object]:
    """Valid :class:`SparseFactorReport` constructor kwargs.

    Parameters
    ----------
    **overrides
        Field replacements.

    Returns
    -------
    dict
        Constructor mapping.
    """
    eye = torch.eye(2, dtype=torch.float64)
    mask = torch.ones(2, 2, dtype=torch.bool)
    payload: dict[str, object] = {
        "K_self": eye,
        "K_nbr": eye.clone(),
        "active_mask_self": mask,
        "active_mask_nbr": mask.clone(),
        "residual": 0.0,
        "nnz": 4,
        "n_samples": 8,
        "method": "stlsq",
        "group": "none",
        "threshold": 0.1,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------


def test_invariance_report_rejects_malformed_fields() -> None:
    """``SubspaceInvarianceReport`` validates leakage, counts, rank, and flag."""
    with pytest.raises(ValueError, match="leakage must be a finite float"):
        SubspaceInvarianceReport(
            leakage=True,  # type: ignore[arg-type]
            n_samples=4,
            rank=1,
            held_out=True,
        )
    with pytest.raises(ValueError, match="leakage must be a finite float"):
        SubspaceInvarianceReport(
            leakage="0.1",  # type: ignore[arg-type]
            n_samples=4,
            rank=1,
            held_out=True,
        )
    with pytest.raises(ValueError, match="finite non-negative float"):
        SubspaceInvarianceReport(leakage=-0.1, n_samples=4, rank=1, held_out=True)
    with pytest.raises(ValueError, match="finite non-negative float"):
        SubspaceInvarianceReport(
            leakage=float("nan"), n_samples=4, rank=1, held_out=True
        )
    with pytest.raises(ValueError, match="n_samples must be a positive int"):
        SubspaceInvarianceReport(
            leakage=0.0,
            n_samples=True,  # type: ignore[arg-type]
            rank=1,
            held_out=True,
        )
    with pytest.raises(ValueError, match="n_samples must be a positive int"):
        SubspaceInvarianceReport(
            leakage=0.0,
            n_samples=4.0,  # type: ignore[arg-type]
            rank=1,
            held_out=True,
        )
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        SubspaceInvarianceReport(leakage=0.0, n_samples=0, rank=1, held_out=True)
    with pytest.raises(ValueError, match="rank must be a positive int"):
        SubspaceInvarianceReport(
            leakage=0.0,
            n_samples=4,
            rank=True,  # type: ignore[arg-type]
            held_out=True,
        )
    with pytest.raises(ValueError, match="rank must be >= 1"):
        SubspaceInvarianceReport(leakage=0.0, n_samples=4, rank=0, held_out=True)
    with pytest.raises(ValueError, match="held_out must be a bool"):
        SubspaceInvarianceReport(
            leakage=0.0,
            n_samples=4,
            rank=1,
            held_out=1,  # type: ignore[arg-type]
        )


def test_invariance_report_rejects_layout_and_matrix_mismatches() -> None:
    """``subspace_invariance_report`` refuses bad layouts and ``K`` shapes."""
    encodings = _line_encodings()
    matrix = torch.eye(2, dtype=torch.float64)
    with pytest.raises(TypeError, match="held_out must be a bool"):
        subspace_invariance_report(encodings, matrix, held_out=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="\\(T, d\\) or \\(T, N, d\\)"):
        subspace_invariance_report(torch.zeros(2, 2, 2, 2), matrix, held_out=False)
    with pytest.raises(ValueError, match="matrix must be square 2-D"):
        subspace_invariance_report(encodings, torch.ones(2, 3), held_out=False)
    with pytest.raises(ValueError, match="matrix must be square 2-D"):
        subspace_invariance_report(encodings, torch.ones(2), held_out=False)
    with pytest.raises(ValueError, match="trailing width must match"):
        subspace_invariance_report(encodings, torch.eye(3), held_out=False)
    with pytest.raises(ValueError, match="at least one snapshot"):
        subspace_invariance_report(torch.empty(0, 2), matrix, held_out=False)
    with pytest.raises(ValueError, match="at least one encoding row"):
        subspace_invariance_report(
            torch.zeros(4, 0, 2, dtype=torch.float64),
            matrix,
            held_out=True,
        )


def test_invariance_report_rejects_degenerate_encoding_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero or over-truncated encoding cloud has no numerical basis."""
    matrix = torch.eye(2, dtype=torch.float64)
    zeros = torch.zeros(6, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="encoding basis is degenerate"):
        subspace_invariance_report(zeros, matrix, held_out=False)
    monkeypatch.setattr(
        "koopman_graph.identification.invariance.SINGULAR_VALUE_REL_CUTOFF",
        2.0,
    )
    encodings = _line_encodings(n_times=6, n_nodes=1)
    with pytest.raises(ValueError, match="encoding basis rank is 0"):
        subspace_invariance_report(encodings, matrix, held_out=False)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_resdmd_candidate_rejects_non_finite_scores() -> None:
    """Candidate MSE and residual must be finite non-negative floats."""
    with pytest.raises(ValueError, match="mse must be a finite float"):
        ResDMDGateCandidate(name="ok", mse=True, residual_max=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residual_max must be a finite float"):
        ResDMDGateCandidate(name="ok", mse=0.1, residual_max="high")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite non-negative float"):
        ResDMDGateCandidate(name="ok", mse=-0.1, residual_max=0.0)
    with pytest.raises(ValueError, match="finite non-negative float"):
        ResDMDGateCandidate(name="ok", mse=0.1, residual_max=float("inf"))


def test_resdmd_result_rejects_malformed_fields() -> None:
    """``ResDMDGateResult`` validates names, rejected list, tolerance, and flag."""
    with pytest.raises(ValueError, match="selected must be a non-empty str"):
        ResDMDGateResult(
            selected="",
            rejected_alternatives=(),
            residual_tolerance=0.01,
            gated=True,
        )
    with pytest.raises(ValueError, match="rejected_alternatives must be a tuple"):
        ResDMDGateResult(
            selected="ok",
            rejected_alternatives=["bad"],  # type: ignore[arg-type]
            residual_tolerance=0.01,
            gated=True,
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        ResDMDGateResult(
            selected="ok",
            rejected_alternatives=("",),
            residual_tolerance=0.01,
            gated=True,
        )
    with pytest.raises(ValueError, match="residual_tolerance must be a finite float"):
        ResDMDGateResult(
            selected="ok",
            rejected_alternatives=(),
            residual_tolerance=True,  # type: ignore[arg-type]
            gated=True,
        )
    with pytest.raises(ValueError, match="finite non-negative float"):
        ResDMDGateResult(
            selected="ok",
            rejected_alternatives=(),
            residual_tolerance=-0.1,
            gated=True,
        )
    with pytest.raises(ValueError, match="gated must be a bool"):
        ResDMDGateResult(
            selected="ok",
            rejected_alternatives=(),
            residual_tolerance=0.01,
            gated=1,  # type: ignore[arg-type]
        )


def test_select_resdmd_gated_rejects_tolerance_and_pool_types() -> None:
    """``select_resdmd_gated`` validates tolerance and candidate sequence types."""
    ok = ResDMDGateCandidate(name="ok", mse=0.1, residual_max=0.0)
    with pytest.raises(ValueError, match="residual_tolerance must be a finite float"):
        select_resdmd_gated((ok,), residual_tolerance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite non-negative float"):
        select_resdmd_gated((ok,), residual_tolerance=float("nan"))
    with pytest.raises(ValueError, match="sequence of ResDMDGateCandidate"):
        select_resdmd_gated("ok")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entries must be ResDMDGateCandidate"):
        select_resdmd_gated((ok, "nope"))  # type: ignore[arg-type]
    custom = select_resdmd_gated((ok,), residual_tolerance=0.5, gate_resdmd=False)
    assert custom.selected == "ok"
    assert custom.residual_tolerance == pytest.approx(0.5)
    assert custom.gated is False


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------


def test_identify_operator_rejects_flat_empty_and_underdetermined() -> None:
    """Closed-form solvers refuse 1-D, empty, and rank-deficient ridge=0 fits."""
    vector = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 2-D"):
        identify_operator(
            LatentPairs(z_t=vector, z_next=vector),
            IdentificationConfig(solver="ridge", ridge=0.0),
        )
    empty = torch.empty(0, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one latent pair"):
        identify_operator(
            LatentPairs(z_t=empty, z_next=empty),
            IdentificationConfig(solver="ridge", ridge=1e-4),
        )
    thin = torch.ones(1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least latent_dim"):
        identify_operator(
            LatentPairs(z_t=thin, z_next=thin),
            IdentificationConfig(solver="ridge", ridge=0.0),
        )


def test_identify_operator_tls_and_ridge_internal_shape_guards() -> None:
    """TLS empty snapshots and mismatched private helper inputs raise."""
    empty_pairs = torch.zeros(0, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="TLS truncation rank must be >= 1"):
        _fit_tls_row_operator(empty_pairs, empty_pairs)
    left = torch.ones(4, 2)
    with pytest.raises(ValueError, match="ridge left/right must share shape"):
        _fit_ridge_row_operator(left, torch.ones(3, 2), ridge=0.0)
    with pytest.raises(ValueError, match="TLS left/right must share shape"):
        _fit_tls_row_operator(left, torch.ones(3, 2))
    with pytest.raises(ValueError, match="TLS left/right must be 2-D"):
        _fit_tls_row_operator(torch.ones(2, 2, 2), torch.ones(2, 2, 2))


def test_identify_operator_tls_wraps_rank_deficient_pinv() -> None:
    """A rank-deficient truncated ``U_x`` becomes a ``ValueError``."""
    true_k = torch.diag(torch.tensor([0.8, 0.4], dtype=torch.float64))
    pairs = _linear_pairs(true_k)

    def _boom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("singular")

    with (
        patch("torch.linalg.pinv", side_effect=_boom),
        pytest.raises(ValueError, match="rank-deficient"),
    ):
        identify_operator(pairs, IdentificationConfig(solver="tls"))


def test_constrained_ls_leaves_contractive_maps_and_rejects_nonfinite() -> None:
    """Unit-disk projection is a no-op when ``ρ≤1`` and refuses NaN radii."""
    true_k = torch.diag(torch.tensor([0.5, 0.2], dtype=torch.float64))
    pairs = _linear_pairs(true_k)
    snapshot = identify_operator(
        pairs, IdentificationConfig(solver="constrained_ls", ridge=0.0)
    )
    assert snapshot.matrix is not None
    rho = float(torch.linalg.eigvals(snapshot.matrix).abs().max().real)
    assert rho == pytest.approx(0.5, rel=1e-6, abs=1e-8)
    with pytest.raises(ValueError, match="spectral radius is non-finite"):
        _scale_spectral_radius(
            torch.tensor([[float("nan")]], dtype=torch.float64),
            1.0,
        )


def test_identify_operator_rejects_unknown_solver_via_mutated_config() -> None:
    """A solver name that bypasses config validation is still refused."""
    pairs = _linear_pairs(torch.eye(2, dtype=torch.float64), n_samples=8)
    config = IdentificationConfig(solver="ridge")
    object.__setattr__(config, "solver", "not-a-solver")
    with pytest.raises(ValueError, match="unsupported identification solver"):
        identify_operator(pairs, config)


def test_apply_operator_snapshot_rejects_non_dense_and_missing_matrix() -> None:
    """Writes require a dense uncontrolled operator and a filled matrix slot."""
    odo = KoopmanOperator(latent_dim=2, parameterization="odo")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        apply_operator_snapshot(odo, OperatorSnapshot(matrix=torch.eye(2)))
    dense = KoopmanOperator(latent_dim=2, parameterization="dense")
    with pytest.raises(ValueError, match="OperatorSnapshot.matrix is required"):
        apply_operator_snapshot(dense, OperatorSnapshot(k_self=torch.eye(2)))
    with pytest.raises(ValueError, match="KoopmanOperator only"):
        apply_operator_snapshot(nn.Linear(2, 2), OperatorSnapshot(matrix=torch.eye(2)))


def test_closed_form_backend_fit_operator_matches_identify() -> None:
    """``ClosedFormBackend.fit_operator`` dispatches to ``identify_operator``."""
    true_k = torch.diag(torch.tensor([0.9, 0.3], dtype=torch.float64))
    pairs = _linear_pairs(true_k)
    config = IdentificationConfig(solver="ridge", ridge=1e-4)
    direct = identify_operator(pairs, config)
    via_backend = ClosedFormBackend().fit_operator(pairs, config)
    assert direct.matrix is not None and via_backend.matrix is not None
    torch.testing.assert_close(
        via_backend.matrix,
        direct.matrix,
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_build_identification_report_validates_kwargs_and_empty_pools() -> None:
    """Report builder checks flags, horizon, tolerance, matrix, and pair lists."""
    true_k = torch.diag(torch.tensor([0.6, 0.4], dtype=torch.float64))
    pairs = _linear_pairs(true_k, n_samples=8)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    with pytest.raises(TypeError, match="gate_resdmd must be a bool"):
        build_identification_report(pairs, snapshot, gate_resdmd=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rollout_horizon must be >= 1"):
        build_identification_report(pairs, snapshot, rollout_horizon=0)
    with pytest.raises(ValueError, match="residual_tolerance must be a finite float"):
        build_identification_report(
            pairs,
            snapshot,
            residual_tolerance=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite non-negative float"):
        build_identification_report(pairs, snapshot, residual_tolerance=-0.1)
    with pytest.raises(ValueError, match="requires snapshot.matrix"):
        build_identification_report(pairs, OperatorSnapshot(k_self=true_k))
    with pytest.raises(ValueError, match="spectral radius must be a finite"):
        build_identification_report(
            pairs,
            OperatorSnapshot(
                matrix=torch.tensor([[float("nan")]], dtype=torch.float64)
            ),
        )
    with pytest.raises(ValueError, match="at least one LatentPairs"):
        build_identification_report((), snapshot)


def test_build_identification_report_skips_empty_rollouts_and_pools_blocks() -> None:
    """Zero-length pairs leave rollout empty; mixed pools skip empty MSE blocks."""
    true_k = torch.diag(torch.tensor([0.7, 0.2], dtype=torch.float64))
    filled = _linear_pairs(true_k, n_samples=6)
    empty = LatentPairs(
        z_t=torch.empty(0, 2, dtype=torch.float64),
        z_next=torch.empty(0, 2, dtype=torch.float64),
    )
    snapshot = identify_operator(
        filled,
        IdentificationConfig(solver="ridge", ridge=0.0),
    )
    vacant = build_identification_report(empty, snapshot, rollout_horizon=3)
    assert vacant.one_step.mse is None
    assert vacant.one_step.n_samples is None
    assert vacant.rollout.mse is None
    pooled = build_identification_report((empty, filled), snapshot, rollout_horizon=2)
    assert pooled.one_step.n_samples == 6 * 2
    assert pooled.one_step.mse == pytest.approx(0.0, abs=1e-12)
    assert pooled.rollout.mse is not None


def test_build_identification_report_rejects_nonfinite_resdmd_residual() -> None:
    """A non-finite max residual from ResDMD is refused when gating."""
    true_k = torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64))
    pairs = _linear_pairs(true_k, n_samples=10)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    fake = SimpleNamespace(residuals=torch.tensor([float("nan")]))
    with (
        patch("koopman_graph.analysis.resdmd.resdmd", return_value=fake),
        pytest.raises(ValueError, match="residual_max must be a finite"),
    ):
        build_identification_report(pairs, snapshot, gate_resdmd=True)
    fake_neg = SimpleNamespace(residuals=torch.tensor([-0.1]))
    with (
        patch("koopman_graph.analysis.resdmd.resdmd", return_value=fake_neg),
        pytest.raises(ValueError, match="residual_max must be a finite"),
    ):
        build_identification_report(pairs, snapshot, gate_resdmd=True)


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


def test_latent_rank_report_rejects_malformed_fields() -> None:
    """``LatentRankReport`` validates criterion, ranks, scores, and rejected names."""
    with pytest.raises(ValueError, match="criterion must be one of"):
        LatentRankReport(**_rank_report_kwargs(criterion="aic"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selected_rank must be an int >= 1"):
        LatentRankReport(**_rank_report_kwargs(selected_rank=0))
    with pytest.raises(ValueError, match="numerical_rank must be an int >= 1"):
        LatentRankReport(**_rank_report_kwargs(numerical_rank=0))
    with pytest.raises(ValueError, match="n_samples must be an int >= 1"):
        LatentRankReport(**_rank_report_kwargs(n_samples=0))
    with pytest.raises(ValueError, match="same length"):
        LatentRankReport(**_rank_report_kwargs(candidates=(1, 2), scores=(0.1,)))
    with pytest.raises(ValueError, match="not among scored"):
        LatentRankReport(**_rank_report_kwargs(selected_rank=2))
    with pytest.raises(ValueError, match="candidates must be positive ints"):
        LatentRankReport(
            **_rank_report_kwargs(selected_rank=1, candidates=(1.0,), scores=(0.1,))
        )
    with pytest.raises(ValueError, match="scores must be finite floats"):
        LatentRankReport(**_rank_report_kwargs(scores=(True,)))
    with pytest.raises(ValueError, match="scores must be finite"):
        LatentRankReport(**_rank_report_kwargs(scores=(float("inf"),)))
    with pytest.raises(ValueError, match="tuple of non-empty strings"):
        LatentRankReport(**_rank_report_kwargs(rejected_alternatives=("4", "")))
    with pytest.raises(ValueError, match="tuple of non-empty strings"):
        LatentRankReport(**_rank_report_kwargs(rejected_alternatives=["4"]))


def test_select_latent_rank_rejects_layout_candidates_and_penalties() -> None:
    """``select_latent_rank`` checks encodings, candidate type, and scalars."""
    encodings = torch.randn(8, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="\\(T, d\\) or \\(T, N, d\\)"):
        select_latent_rank(torch.randn(2, 2, 2, 2), (1,), criterion="vamp2")
    with pytest.raises(ValueError, match="must be floating-point"):
        select_latent_rank(
            torch.zeros(5, 2, dtype=torch.int64),
            (1,),
            criterion="vamp2",
        )
    with pytest.raises(ValueError, match="trailing width d >= 1"):
        select_latent_rank(torch.zeros(5, 0), (1,), criterion="vamp2")
    with pytest.raises(ValueError, match="sequence of positive ints"):
        select_latent_rank(encodings, "12", criterion="vamp2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ridge must be a finite"):
        select_latent_rank(encodings, (1, 2), criterion="vamp2", ridge=float("inf"))
    with pytest.raises(ValueError, match="stability_penalty must be a finite"):
        select_latent_rank(
            encodings,
            (1, 2),
            criterion="stability_penalized",
            stability_penalty=-1.0,
        )


def test_select_latent_rank_rejects_degenerate_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero cloud or over-truncated SVD cutoff has no numerical rank."""
    zeros = torch.zeros(8, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="encoding basis is degenerate"):
        select_latent_rank(zeros, (1, 2), criterion="vamp2")
    monkeypatch.setattr(
        "koopman_graph.identification.rank.SINGULAR_VALUE_REL_CUTOFF",
        2.0,
    )
    encodings = torch.randn(8, 3, dtype=torch.float64)
    encodings[:, 1:] = 0.0
    encodings[:, 0] = torch.linspace(0.2, 1.4, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="encoding basis rank is 0"):
        select_latent_rank(encodings, (1,), criterion="vamp2")


def test_stability_penalized_rank_rejects_missing_or_nonfinite_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stability scoring requires a dense finite identified matrix."""
    encodings = torch.randn(10, 3, dtype=torch.float64)

    def _factors_only(
        pairs: LatentPairs,
        _config: IdentificationConfig,
    ) -> OperatorSnapshot:
        width = int(pairs.z_t.shape[-1])
        return OperatorSnapshot(k_self=torch.eye(width, dtype=pairs.z_t.dtype))

    monkeypatch.setattr(
        "koopman_graph.identification.rank.identify_operator",
        _factors_only,
    )
    with pytest.raises(ValueError, match="must return a dense matrix"):
        select_latent_rank(encodings, (1,), criterion="stability_penalized")

    def _nan_matrix(
        pairs: LatentPairs,
        _config: IdentificationConfig,
    ) -> OperatorSnapshot:
        width = int(pairs.z_t.shape[-1])
        return OperatorSnapshot(
            matrix=torch.full((width, width), float("nan"), dtype=pairs.z_t.dtype)
        )

    monkeypatch.setattr(
        "koopman_graph.identification.rank.identify_operator",
        _nan_matrix,
    )
    with pytest.raises(ValueError, match="spectral radius is non-finite"):
        select_latent_rank(encodings, (1,), criterion="stability_penalized")


# ---------------------------------------------------------------------------
# sparse factors
# ---------------------------------------------------------------------------


def test_sparse_factor_report_rejects_malformed_fields() -> None:
    """``SparseFactorReport`` validates shapes, residual, counts, and threshold."""
    with pytest.raises(ValueError, match="K_self and K_nbr must share shape"):
        SparseFactorReport(
            **_sparse_report_kwargs(K_nbr=torch.eye(3, dtype=torch.float64))
        )
    with pytest.raises(ValueError, match="K_self must be square 2-D"):
        rect = torch.ones(2, 3, dtype=torch.float64)
        SparseFactorReport(**_sparse_report_kwargs(K_self=rect, K_nbr=rect.clone()))
    with pytest.raises(ValueError, match="active_mask_self must match"):
        SparseFactorReport(
            **_sparse_report_kwargs(active_mask_self=torch.ones(3, 3, dtype=torch.bool))
        )
    with pytest.raises(ValueError, match="active_mask_nbr must match"):
        SparseFactorReport(
            **_sparse_report_kwargs(active_mask_nbr=torch.ones(3, 3, dtype=torch.bool))
        )
    with pytest.raises(ValueError, match="residual must be a finite"):
        SparseFactorReport(**_sparse_report_kwargs(residual=float("nan")))
    with pytest.raises(ValueError, match="nnz must be non-negative"):
        SparseFactorReport(**_sparse_report_kwargs(nnz=-1))
    with pytest.raises(ValueError, match="n_samples >= 1"):
        SparseFactorReport(**_sparse_report_kwargs(n_samples=0))
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        SparseFactorReport(**_sparse_report_kwargs(threshold=-0.1))


def test_identify_sparse_graph_factors_rejects_edges_layout_and_iter() -> None:
    """Sparse-factor ID checks COO bounds, empty axes, dtype, and ``max_iter``."""
    z_pairs = torch.randn(4, 3, 2, dtype=torch.float64)
    edges = _path_edges(3)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        identify_sparse_graph_factors(
            z_pairs,
            torch.zeros(3, 4, dtype=torch.long),
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="edge_index entries must lie"):
        identify_sparse_graph_factors(
            z_pairs,
            torch.tensor([[0, 9], [1, 0]], dtype=torch.long),
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="N >= 1 and d >= 1"):
        identify_sparse_graph_factors(
            torch.randn(4, 0, 2, dtype=torch.float64),
            edges,
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="N >= 1 and d >= 1"):
        identify_sparse_graph_factors(
            torch.randn(4, 3, 0, dtype=torch.float64),
            edges,
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="z_pairs must be floating-point"):
        identify_sparse_graph_factors(
            torch.zeros(4, 3, 2, dtype=torch.int64),
            edges,
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        identify_sparse_graph_factors(z_pairs, edges, threshold=0.1, max_iter=0)


def test_identify_sparse_graph_factors_empty_support_and_soft_threshold() -> None:
    """A huge cutoff zeros both factors; a zero group is skipped in ISTA prox."""
    z_pairs = torch.randn(5, 3, 2, dtype=torch.float64)
    edges = _path_edges(3)
    weights = torch.ones(edges.shape[1], dtype=torch.float64)
    wiped = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="self_nbr",
        method="stlsq",
        threshold=1e6,
        edge_weight=weights,
        max_iter=2,
    )
    assert wiped.nnz == 0
    assert torch.equal(wiped.K_self, torch.zeros(2, 2, dtype=torch.float64))
    lasso = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="self_nbr",
        method="group_lasso",
        threshold=0.05,
        adjacency="symmetric",
    )
    assert lasso.method == "group_lasso"
    assert torch.isfinite(lasso.K_self).all()
    masks = _group_masks("self_nbr", 2, device=torch.device("cpu"))
    zeroed = _group_soft_threshold(torch.zeros(4, 2), masks, tau=0.1)
    assert torch.equal(zeroed, torch.zeros(4, 2))
