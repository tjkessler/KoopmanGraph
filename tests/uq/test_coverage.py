"""Tests for named coverage specs and conformal target naming."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.uq import ConformalKoopmanUQ, JointCoverageSpec
from koopman_graph.uq.coverage import require_shipped_coverage


def test_joint_coverage_spec_defaults_to_per_node_marginal() -> None:
    """Default estimand is named per-node marginal coverage."""
    spec = JointCoverageSpec()
    assert spec.target == "per_node_marginal"
    assert spec.alpha == 0.1
    assert spec.block == "none"
    require_shipped_coverage(spec)


def test_joint_coverage_spec_refuses_unshipped_estimands() -> None:
    """Simultaneous / event / block targets are named but not implemented."""
    with pytest.raises(ValueError, match="alpha"):
        JointCoverageSpec(alpha=0.0)
    with pytest.raises(ValueError, match="per_node_marginal"):
        require_shipped_coverage(
            JointCoverageSpec(target="simultaneous_node_feature_horizon")
        )
    with pytest.raises(ValueError, match="per_node_marginal"):
        require_shipped_coverage(JointCoverageSpec(target="event"))
    with pytest.raises(ValueError, match="none"):
        require_shipped_coverage(JointCoverageSpec(block="temporal"))


def test_conformal_coverage_is_named_not_implied() -> None:
    """Conformal wrappers always expose a named per-node-marginal spec."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
    )
    wrapper = ConformalKoopmanUQ(model)
    named = wrapper.coverage
    assert named.target == "per_node_marginal"
    assert named.block == "none"
    assert named.alpha == 0.1
    explicit = JointCoverageSpec(target="per_node_marginal", alpha=0.2)
    named_wrapper = ConformalKoopmanUQ(model, coverage=explicit)
    assert named_wrapper.coverage.alpha == 0.2
    with pytest.raises(ValueError, match="not implemented"):
        ConformalKoopmanUQ(
            model,
            coverage=JointCoverageSpec(target="simultaneous_node_feature_horizon"),
        )


def test_conformal_calibrate_alpha_must_match_named_spec() -> None:
    """calibrate alpha cannot silently disagree with the named spec."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
    )
    wrapper = ConformalKoopmanUQ(
        model,
        coverage=JointCoverageSpec(target="per_node_marginal", alpha=0.2),
    )
    with pytest.raises(ValueError, match="coverage.alpha"):
        wrapper.calibrate([], steps=1, alpha=0.1)


def test_coverage_module_does_not_import_model_or_operators() -> None:
    """Coverage records stay below model / operator layers."""
    source = Path(__file__).resolve().parents[2] / "src/koopman_graph/uq/coverage.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = ("koopman_graph.model", "koopman_graph.operators")
    offenders = [
        name
        for name in imported
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not offenders
