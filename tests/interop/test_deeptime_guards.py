"""Coverage and error-path tests for :mod:`koopman_graph.interop`."""

from __future__ import annotations

import pytest


def test_deeptime_feature_edge_cases() -> None:
    """Integer coercion, empty matrices, and unrecognized payloads."""
    pytest.importorskip("deeptime")
    import numpy as np

    from koopman_graph.interop.deeptime import (
        trajectory_features_from_deeptime,
        trajectory_features_to_deeptime,
    )

    features = np.arange(12, dtype=np.int64).reshape(6, 2)
    dataset = trajectory_features_to_deeptime(features, lag=1)
    recovered = trajectory_features_from_deeptime(dataset)
    assert recovered.dtype == np.float64
    assert np.allclose(recovered, features.astype(np.float64))

    with pytest.raises(ValueError, match="non-empty in both dimensions"):
        trajectory_features_to_deeptime(np.zeros((0, 3)), lag=1)
    with pytest.raises(ValueError, match="non-empty in both dimensions"):
        trajectory_features_to_deeptime(np.zeros((5, 0)), lag=1)
    with pytest.raises(ValueError, match="expected a deeptime"):
        trajectory_features_from_deeptime(object())
