"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import KoopmanOperator
from koopman_graph.baselines.base import (
    fit_fb_row_operator,
    fit_tls_row_operator,
)
from koopman_graph.operators.discrete_parameterizations import (
    assemble_doubly_stochastic_matrix,
    assemble_symplectic_matrix,
)


def test_fb_and_tls_row_operator_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward-backward invertibility and TLS shape/rank guards."""
    left = torch.randn(6, 3)
    right = torch.randn(6, 3)
    with pytest.raises(ValueError, match="share shape"):
        fit_tls_row_operator(left, right[:, :2], rank=None)
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_tls_row_operator(left[0], right[0], rank=None)

    def boom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("singular")

    monkeypatch.setattr(torch.linalg, "inv", boom)
    with pytest.raises(ValueError, match="invertible backward operator"):
        fit_fb_row_operator(left, right, rank=None)

    import koopman_graph.baselines.base as base_mod

    monkeypatch.setattr(base_mod, "resolve_fit_rank", lambda *_a, **_k: 0)
    with pytest.raises(ValueError, match="truncation rank must be >= 1"):
        fit_tls_row_operator(left, right, rank=1)


def test_row_stochastic_conserves_latent_mass() -> None:
    """Row-stochastic ``K`` conserves the sum of latent coordinates."""
    op = KoopmanOperator(4, parameterization="row_stochastic", init_mode="identity")
    matrix = op.matrix
    assert torch.allclose(matrix.sum(dim=-1), torch.ones(4), atol=1e-5)
    z = torch.randn(3, 4)
    advanced = op.advance(z)
    assert torch.allclose(advanced.sum(dim=-1), z.sum(dim=-1), atol=1e-5)


def test_doubly_stochastic_row_and_column_sums() -> None:
    """Sinkhorn assembly is approximately doubly stochastic."""
    raw = torch.randn(5, 5)
    matrix = assemble_doubly_stochastic_matrix(raw)
    ones = torch.ones(5)
    assert torch.allclose(matrix.sum(dim=-1), ones, atol=1e-4)
    assert torch.allclose(matrix.sum(dim=-2), ones, atol=1e-4)


def test_symplectic_even_dim_and_odd_reject() -> None:
    """Symplectic maps require even width and satisfy the Cayley construction."""
    with pytest.raises(ValueError, match="even"):
        KoopmanOperator(3, parameterization="symplectic")
    op = KoopmanOperator(4, parameterization="symplectic", init_mode="identity")
    assembled = assemble_symplectic_matrix(op.symplectic_raw)
    assert assembled.shape == (4, 4)
    z = torch.randn(2, 4)
    out = op.advance(z)
    assert out.shape == z.shape


def test_symplectic_rejects_odd_latent_dim() -> None:
    """Symplectic ``K`` requires even latent width."""
    with pytest.raises(ValueError, match="even latent_dim"):
        KoopmanOperator(3, parameterization="symplectic")
