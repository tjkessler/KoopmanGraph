"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

from koopman_graph.operators.continuous import ContinuousKoopmanOperator
from koopman_graph.operators.discrete import KoopmanOperator


def test_structural_assembly_factors_dense_returns_empty() -> None:
    """Dense discrete/continuous parameterization hits the empty-factor branch."""
    discrete = KoopmanOperator(latent_dim=2, parameterization="dense")
    continuous = ContinuousKoopmanOperator(latent_dim=2, parameterization="dense")
    assert discrete._structural_assembly_factors() == ()
    assert continuous._structural_assembly_factors() == ()
