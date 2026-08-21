"""Labeled synthetic SCM interventions (TASK-2348)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
from koopman_graph.analysis import (
    SyntheticInterventionReport,
    SyntheticSCM,
    granger_latent_influence,
    recover_synthetic_interventional_edges,
    teaching_three_node_scm,
)
from koopman_graph.analysis.causal_intervention import (
    DEFAULT_ATE_THRESHOLD,
    sample_synthetic_intervention,
    sample_synthetic_observational,
)


def test_intervention_types_are_off_root_and_on_analysis_all() -> None:
    """Synthetic SCM types stay off the root façade."""
    exported = set(koopman_graph.__all__)
    assert "SyntheticSCM" not in exported
    assert "SyntheticInterventionReport" not in exported
    assert "teaching_three_node_scm" not in exported
    assert "recover_synthetic_interventional_edges" not in exported
    assert "SyntheticSCM" in koopman_graph.analysis.__all__
    assert "recover_synthetic_interventional_edges" in koopman_graph.analysis.__all__
    assert "teaching_three_node_scm" in koopman_graph.analysis.__all__


def test_granger_docstring_stays_non_interventional() -> None:
    """Default Granger helper remains labeled non-interventional."""
    doc = granger_latent_influence.__doc__
    assert doc is not None
    assert "non-interventional" in doc
    assert "do-operator" in doc


def test_causal_intervention_does_not_import_model() -> None:
    """The intervention leaf must not import ``model``."""
    import koopman_graph.analysis.causal_intervention as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all(not name.startswith("koopman_graph.model") for name in imported)


def test_teaching_scm_recovers_known_interventional_edge() -> None:
    """Seeded three-node SCM recovers the known do-edge ``0 → 1``."""
    scm = teaching_three_node_scm(seed=0)
    assert scm.labeled_synthetic is True
    assert scm.true_edges == ((0, 1),)
    report = recover_synthetic_interventional_edges(scm, seed=0)
    assert isinstance(report, SyntheticInterventionReport)
    assert report.labeled_synthetic is True
    assert report.true_edges == ((0, 1),)
    assert (0, 1) in report.recovered_edges
    assert (1, 0) not in report.recovered_edges
    assert (0, 2) not in report.recovered_edges
    assert (2, 0) not in report.recovered_edges
    assert float(report.scores[0, 1]) > DEFAULT_ATE_THRESHOLD
    assert float(report.scores[1, 0]) < DEFAULT_ATE_THRESHOLD
    assert float(report.scores[0, 2]) < DEFAULT_ATE_THRESHOLD


def test_do_intervention_shifts_child_not_parent() -> None:
    """``do(X_0 = 1)`` moves node 1 and leaves independent node 2 near 0."""
    scm = teaching_three_node_scm(seed=1, noise_scale=0.01)
    intervened = sample_synthetic_intervention(
        scm,
        source=0,
        value=1.0,
        n_samples=400,
        generator=torch.Generator().manual_seed(7),
    )
    observational = sample_synthetic_observational(
        scm,
        400,
        generator=torch.Generator().manual_seed(8),
    )
    assert intervened.shape == (400, 3)
    assert torch.allclose(intervened[:, 0], torch.ones(400, dtype=intervened.dtype))
    assert abs(float(intervened[:, 1].mean()) - 0.8) < 0.05
    assert abs(float(intervened[:, 2].mean())) < 0.05
    assert abs(float(observational[:, 0].mean())) < 0.05


def test_cyclic_weights_are_refused() -> None:
    """Cyclic contemporaneous weights raise."""
    weights = torch.zeros(2, 2)
    weights[0, 1] = 0.4
    weights[1, 0] = 0.4
    with pytest.raises(ValueError, match="acyclic"):
        SyntheticSCM(weights=weights, noise_scale=0.1, seed=0)


def test_recover_requires_distinct_do_values() -> None:
    """Paired interventions must use two distinct do-values."""
    scm = teaching_three_node_scm()
    with pytest.raises(ValueError, match="must differ"):
        recover_synthetic_interventional_edges(
            scm,
            intervention_low=0.5,
            intervention_high=0.5,
        )
