"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest

from koopman_graph.operators import (
    sample_graphon_adjacency,
)


def test_graphon_product_kernel_and_small_n() -> None:
    """Product graphon samples; ``N < 2`` is rejected."""
    with pytest.raises(ValueError, match=">= 2"):
        sample_graphon_adjacency(1)
    edges = sample_graphon_adjacency(5, kernel="product")
    assert edges.shape[0] == 2


def test_graphon_unknown_kernel() -> None:
    """Unknown graphon kernels raise an actionable error."""
    with pytest.raises(ValueError, match="unknown graphon"):
        sample_graphon_adjacency(4, kernel="bogus")
