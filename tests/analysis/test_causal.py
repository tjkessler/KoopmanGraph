"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis import (
    granger_latent_influence,
)


def test_causal_rejects_bad_rank_and_short_series() -> None:
    """Granger helper validates ``(T, N, d)`` with ``T >= 3``."""
    with pytest.raises(ValueError, match="shape \\(T, N, d\\)"):
        granger_latent_influence(torch.randn(4, 3))
    with pytest.raises(ValueError, match="T >= 3"):
        granger_latent_influence(torch.randn(2, 3, 2))


def test_granger_docstring_is_non_interventional() -> None:
    """Acceptance: Granger remains labeled non-interventional."""
    doc = granger_latent_influence.__doc__
    assert doc is not None
    assert "non-interventional" in doc
