"""Coverage and error-path tests for :mod:`koopman_graph.statistics`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.statistics import _hann_window, largest_lyapunov_exponent


def test_statistics_hann_and_constant_theiler_power() -> None:
    """Direct Hann guard + constant-series Theiler power fallback."""
    with pytest.raises(ValueError, match="segment_length must be >= 2"):
        _hann_window(1, dtype=torch.float32, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="finite divergence|Theiler|too short"):
        largest_lyapunov_exponent(
            torch.ones(64),
            embedding_dim=3,
            delay=1,
            theiler=None,
            trajectory_len=4,
            min_neighbors=5,
        )
