"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.operators import (
    MixtureKoopmanOperator,
    SwitchedKoopmanOperator,
)


def test_switched_and_mixture_operator_surface() -> None:
    """Switched/mixture banks cover mode errors, matrix views, inverse, forward."""
    with pytest.raises(ValueError, match="num_modes"):
        SwitchedKoopmanOperator(3, num_modes=0)
    with pytest.raises(ValueError, match="num_modes"):
        MixtureKoopmanOperator(3, num_modes=0)
    switched = SwitchedKoopmanOperator(3, num_modes=2)
    with pytest.raises(ValueError, match="mode_index"):
        switched.set_mode(9)
    z = torch.randn(4, 3)
    assert 0 <= switched.infer_mode(z) < 2
    assert switched.matrix.shape == (3, 3)
    assert switched.bound_metric().ndim == 0
    _ = switched.stability_certificate()
    assert switched.inverse_advance(z).shape == z.shape
    assert switched(z).shape == z.shape
    mixture = MixtureKoopmanOperator(3, num_modes=2)
    assert mixture.matrix.shape == (3, 3)
    assert mixture.bound_metric().ndim == 0
    assert mixture.inverse_advance(z).shape == z.shape
    assert mixture(z).shape == z.shape
