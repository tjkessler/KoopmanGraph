"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.baselines import DMDBaseline
from koopman_graph.baselines.base import (
    gavish_donoho_omega,
    optimal_hard_threshold_rank,
    resolve_fit_rank,
)


def test_rank_auto_validation_branches() -> None:
    """Gavish–Donoho / resolve_fit_rank / DMDBaseline constructor guards."""
    with pytest.raises(ValueError, match="beta must lie"):
        gavish_donoho_omega(0.0)
    with pytest.raises(ValueError, match="beta must lie"):
        gavish_donoho_omega(1.5)
    with pytest.raises(ValueError, match="1-D"):
        optimal_hard_threshold_rank(torch.ones(2, 2), num_rows=2, num_cols=2)
    with pytest.raises(ValueError, match="num_rows and num_cols"):
        optimal_hard_threshold_rank(torch.ones(3), num_rows=0, num_cols=2)
    assert optimal_hard_threshold_rank(torch.tensor([]), num_rows=2, num_cols=2) == 0
    with pytest.raises(ValueError, match="finite"):
        optimal_hard_threshold_rank(
            torch.tensor([1.0, float("nan")]), num_rows=2, num_cols=2
        )
    assert optimal_hard_threshold_rank(torch.zeros(4), num_rows=4, num_cols=4) == 0

    assert resolve_fit_rank(torch.randn(8, 4), None) is None
    with pytest.raises(ValueError, match="2-D"):
        resolve_fit_rank(torch.randn(4), "auto")
    with pytest.raises(ValueError, match="int, None, or 'auto'"):
        resolve_fit_rank(torch.randn(8, 4), "full")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must be >= 1"):
        resolve_fit_rank(torch.randn(8, 4), 0)
    with pytest.raises(ValueError, match="rank must be <="):
        resolve_fit_rank(torch.randn(8, 4), 20)

    with pytest.raises(ValueError, match="int, None, or 'auto'"):
        DMDBaseline(rank="full")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must be >= 1"):
        DMDBaseline(rank=-1)
