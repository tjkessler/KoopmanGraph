"""Coverage and error-path tests for :mod:`koopman_graph.interop`."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from koopman_graph.interop.deeptime import (
    require_deeptime,
    trajectory_features_from_deeptime,
    trajectory_features_to_deeptime,
)


def test_deeptime_bridge_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deeptime conversion helpers run against a TrajectoryDataset stand-in."""
    fake_pkg = ModuleType("deeptime")
    fake_util = ModuleType("deeptime.util")
    fake_data = ModuleType("deeptime.util.data")

    class _Dataset:
        def __init__(self, lag: int, array: np.ndarray) -> None:
            self.lagtime = lag
            self.trajectory = array

    fake_data.TrajectoryDataset = _Dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deeptime", fake_pkg)
    monkeypatch.setitem(sys.modules, "deeptime.util", fake_util)
    monkeypatch.setitem(sys.modules, "deeptime.util.data", fake_data)
    features = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    dataset = trajectory_features_to_deeptime(features, lag=2)
    recovered = trajectory_features_from_deeptime(dataset)
    assert recovered.shape == (10, 2)
    multi = SimpleNamespace(trajectories=[np.ones((6, 2))])
    assert trajectory_features_from_deeptime(multi).shape == (6, 2)
    with pytest.raises(ValueError, match="lag must be"):
        trajectory_features_to_deeptime(features, lag=0)
    with pytest.raises(ValueError, match="num_timesteps"):
        trajectory_features_to_deeptime(torch.ones(1, 2), lag=1)
    with pytest.raises(ValueError, match="single trajectory"):
        trajectory_features_from_deeptime(SimpleNamespace(trajectories=[1, 2]))
    with pytest.raises(ValueError, match="shape"):
        trajectory_features_from_deeptime(SimpleNamespace(trajectory=np.ones(3)))
    with pytest.raises(ValueError, match="non-empty"):
        trajectory_features_from_deeptime(SimpleNamespace(trajectory=np.zeros((0, 2))))
    with pytest.raises(ValueError, match="expected a deeptime"):
        trajectory_features_from_deeptime(object())
    monkeypatch.delitem(sys.modules, "deeptime", raising=False)
    real_import = builtins.__import__

    def _block(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "deeptime" or name.startswith("deeptime."):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="koopman-graph\\[msm\\]"):
        require_deeptime()
