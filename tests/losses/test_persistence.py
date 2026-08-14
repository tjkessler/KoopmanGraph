"""Coverage and error-path tests for :mod:`koopman_graph.losses`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.losses import PersistenceRegularizer


def test_persistence_regularizer_weight_and_empty_pairs() -> None:
    """Persistence regularizer rejects negative weights and empty diagrams."""
    with pytest.raises(ValueError, match="non-negative"):
        PersistenceRegularizer(weight=-1.0)
    loss = PersistenceRegularizer()(torch.zeros(1, 2))
    assert float(loss) == 0.0
