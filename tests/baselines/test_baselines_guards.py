"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.baselines.base import (
    fit_opt_row_operator,
    streaming_gram_init,
    streaming_gram_update,
)


def test_baseline_validation_and_dead_eigenvalue_branch() -> None:
    """Optimized and streaming DMD validate iterations and vector ranks."""
    left = torch.randn(6, 2)
    right = torch.randn(6, 2)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        fit_opt_row_operator(left, right, None, max_iter=0)
    operator = fit_opt_row_operator(left, right, None, max_iter=1)
    assert operator.shape == (2, 2)

    gram, cross = streaming_gram_init(
        2, dtype=torch.float32, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="1-D state vectors"):
        streaming_gram_update(gram, cross, torch.zeros(1, 2), torch.zeros(2))
