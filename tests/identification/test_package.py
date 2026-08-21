"""Tests for the identification package surface (types, not solvers)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

import koopman_graph
import koopman_graph.identification as identification
from koopman_graph.identification import (
    IDENTIFICATION_SOLVERS,
    IdentificationBackend,
    IdentificationConfig,
    IdentificationReport,
    InvarianceBlock,
    LatentPairs,
    LatentRankReport,
    MetricBlock,
    OperatorSnapshot,
    SparseFactorReport,
    SpectralReliabilityBlock,
    StabilityBlock,
    SubspaceInvarianceReport,
    identify_sparse_graph_factors,
    select_latent_rank,
)


def test_identification_not_on_root_facade() -> None:
    """Config/report types stay off the thin root façade."""
    for name in (
        "IdentificationConfig",
        "IdentificationReport",
        "IdentificationBackend",
        "LatentPairs",
        "OperatorSnapshot",
        "MetricBlock",
        "identify_operator",
        "ClosedFormBackend",
        "SubspaceInvarianceReport",
        "subspace_invariance_report",
        "select_resdmd_gated",
        "ResDMDGateCandidate",
        "ResDMDGateResult",
        "DEFAULT_RESDMD_GATE_TOLERANCE",
        "SparseFactorReport",
        "identify_sparse_graph_factors",
        "LatentRankReport",
        "select_latent_rank",
    ):
        assert name not in koopman_graph.__all__
        assert not hasattr(koopman_graph, name)
        assert name in identification.__all__
    with pytest.raises(ImportError):
        exec("from koopman_graph import IdentificationReport")


def test_package_reexports_match_submodules() -> None:
    """Package ``__all__`` names resolve to the submodule objects."""
    assert identification.IdentificationConfig is IdentificationConfig
    assert identification.IdentificationReport is IdentificationReport
    assert identification.IDENTIFICATION_SOLVERS is IDENTIFICATION_SOLVERS
    assert identification.LatentPairs is LatentPairs
    assert identification.OperatorSnapshot is OperatorSnapshot
    assert identification.identify_operator is not None
    assert identification.subspace_invariance_report is not None
    assert identification.SubspaceInvarianceReport is SubspaceInvarianceReport
    assert identification.select_resdmd_gated is not None
    assert identification.ResDMDGateCandidate is not None
    assert identification.ResDMDGateResult is not None
    assert identification.DEFAULT_RESDMD_GATE_TOLERANCE == 1e-2
    assert identification.identify_sparse_graph_factors is identify_sparse_graph_factors
    assert identification.SparseFactorReport is SparseFactorReport
    assert identification.select_latent_rank is select_latent_rank
    assert identification.LatentRankReport is LatentRankReport


def test_default_report_has_design_field_groups() -> None:
    """Empty report exposes reconstruction through stability plus rank slots."""
    report = IdentificationReport()
    assert report.reconstruction == MetricBlock()
    assert report.one_step == MetricBlock()
    assert report.rollout == MetricBlock()
    assert report.closure == MetricBlock()
    assert report.invariance == InvarianceBlock()
    assert report.spectral == SpectralReliabilityBlock()
    assert report.stability == StabilityBlock()
    assert report.selected_rank is None
    assert report.rejected_alternatives == ()


def test_report_and_config_are_frozen() -> None:
    """Public identification records reject in-place mutation."""
    config = IdentificationConfig()
    report = IdentificationReport()
    with pytest.raises(FrozenInstanceError):
        config.ridge = 0.1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.selected_rank = 2  # type: ignore[misc]


def test_config_defaults_and_solver_names() -> None:
    """Default config matches the documented opt-in ridge settings."""
    config = IdentificationConfig()
    assert config.solver == "ridge"
    assert config.ridge == pytest.approx(1e-4)
    assert config.select_on == ("rollout", "invariance", "resdmd")
    assert config.gate_resdmd is False
    assert {
        "ridge",
        "tls",
        "constrained_ls",
        "varpro",
        "alternating",
    } == IDENTIFICATION_SOLVERS
    for solver in sorted(IDENTIFICATION_SOLVERS):
        IdentificationConfig(solver=solver)  # type: ignore[arg-type]


def test_config_rejects_unknown_solver_and_negative_ridge() -> None:
    """Invalid solver names and non-finite ridge weights raise ValueError."""
    with pytest.raises(ValueError, match="solver must be one of"):
        IdentificationConfig(solver="adam")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ridge must be a finite"):
        IdentificationConfig(ridge=-1e-3)
    with pytest.raises(ValueError, match="ridge must be a finite"):
        IdentificationConfig(ridge=float("nan"))
    with pytest.raises(ValueError, match="gate_resdmd must be a bool"):
        IdentificationConfig(gate_resdmd=1)  # type: ignore[arg-type]


def test_metric_block_rejects_nonfinite_mse() -> None:
    """MSE slots are finite floats; sample counts are non-negative."""
    with pytest.raises(ValueError, match="mse must be a finite"):
        MetricBlock(mse=float("inf"))
    with pytest.raises(ValueError, match="n_samples must be a non-negative"):
        MetricBlock(n_samples=-1)
    block = MetricBlock(mse=0.25, n_samples=8)
    assert block.mse == pytest.approx(0.25)
    assert block.n_samples == 8


def test_invariance_and_stability_reject_negative_scalars() -> None:
    """Leakage and spectral radius are non-negative when set."""
    with pytest.raises(ValueError, match="leakage must be non-negative"):
        InvarianceBlock(leakage=-0.1)
    with pytest.raises(ValueError, match="spectral_radius must be non-negative"):
        StabilityBlock(spectral_radius=-0.01)
    assert InvarianceBlock(leakage=0.0).leakage == pytest.approx(0.0)


def test_latent_pairs_require_matching_tensors() -> None:
    """Consecutive encodings must share shape, dtype, and device."""
    z_t = torch.zeros(4, 2)
    z_next = torch.ones(4, 2)
    pairs = LatentPairs(z_t=z_t, z_next=z_next)
    assert pairs.z_t.shape == (4, 2)
    with pytest.raises(ValueError, match="must share shape"):
        LatentPairs(z_t=z_t, z_next=torch.zeros(3, 2))
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        LatentPairs(z_t=z_t, z_next=[0.0])  # type: ignore[arg-type]


def test_operator_snapshot_requires_a_factor() -> None:
    """Empty snapshots are rejected; a dense matrix is enough."""
    with pytest.raises(ValueError, match="requires matrix, k_self, or k_nbr"):
        OperatorSnapshot()
    matrix = torch.eye(4)
    snap = OperatorSnapshot(matrix=matrix)
    assert snap.matrix is matrix
    assert snap.k_self is None


def test_runtime_checkable_backend_protocol() -> None:
    """A duck-typed object with fit_operator satisfies the protocol."""

    class _RidgeStub:
        def fit_operator(
            self,
            encodings: LatentPairs,
            config: IdentificationConfig,
        ) -> OperatorSnapshot:
            del encodings, config
            return OperatorSnapshot(matrix=torch.eye(2))

    stub = _RidgeStub()
    assert isinstance(stub, IdentificationBackend)
    pairs = LatentPairs(z_t=torch.zeros(3, 2), z_next=torch.zeros(3, 2))
    out = stub.fit_operator(pairs, IdentificationConfig())
    assert out.matrix is not None
    assert tuple(out.matrix.shape) == (2, 2)
