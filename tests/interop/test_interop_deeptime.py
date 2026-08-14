"""Tests for optional deeptime trajectory-feature interop helpers."""

from __future__ import annotations

import builtins

import pytest
import torch
from tests.helpers import REPO_ROOT

pytest.importorskip("deeptime")

from koopman_graph.interop import (  # noqa: E402
    require_deeptime,
    trajectory_features_from_deeptime,
    trajectory_features_to_deeptime,
)


def test_round_trip_numpy_features() -> None:
    """NumPy ``(T, F)`` features round-trip through ``TrajectoryDataset``."""
    import numpy as np

    features = np.linspace(0.0, 1.0, num=24, dtype=np.float64).reshape(12, 2)
    dataset = trajectory_features_to_deeptime(features, lag=2)
    recovered = trajectory_features_from_deeptime(dataset)
    assert recovered.shape == (12, 2)
    assert np.allclose(recovered, features)
    assert int(dataset.lagtime) == 2


def test_round_trip_torch_features() -> None:
    """Torch tensors detach to CPU NumPy before wrapping."""
    import numpy as np

    features = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    dataset = trajectory_features_to_deeptime(features, lag=1)
    recovered = trajectory_features_from_deeptime(dataset)
    assert recovered.shape == (10, 3)
    assert np.allclose(recovered, features.numpy())


def test_from_single_trajectories_dataset() -> None:
    """Single-member ``TrajectoriesDataset`` unwraps to one feature matrix."""
    import numpy as np
    from deeptime.util.data import TrajectoriesDataset

    features = np.random.default_rng(0).normal(size=(16, 4))
    multi = TrajectoriesDataset.from_numpy(1, [features])
    recovered = trajectory_features_from_deeptime(multi)
    assert np.allclose(recovered, features)


def test_multi_trajectory_rejected() -> None:
    """Multi-trajectory deeptime payloads raise a clear ``ValueError``."""
    import numpy as np
    from deeptime.util.data import TrajectoriesDataset

    multi = TrajectoriesDataset.from_numpy(
        1,
        [
            np.zeros((10, 2), dtype=np.float64),
            np.ones((12, 2), dtype=np.float64),
        ],
    )
    with pytest.raises(ValueError, match="single trajectory"):
        trajectory_features_from_deeptime(multi)


def test_lag_and_shape_validation() -> None:
    """Invalid lag / shapes raise ``ValueError`` before deeptime asserts."""
    import numpy as np

    features = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="lag must be >= 1"):
        trajectory_features_to_deeptime(features, lag=0)
    with pytest.raises(ValueError, match="num_timesteps > lag"):
        trajectory_features_to_deeptime(features, lag=4)
    with pytest.raises(ValueError, match="\\(num_timesteps, num_features\\)"):
        trajectory_features_to_deeptime(np.zeros(5), lag=1)


def test_require_deeptime_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing deeptime raises an install-hinted ``ImportError``."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "deeptime" or name.startswith("deeptime."):
            raise ImportError("no deeptime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"koopman-graph\[msm\]"):
        require_deeptime()
    with pytest.raises(ImportError, match=r"koopman-graph\[msm\]"):
        trajectory_features_to_deeptime([[0.0, 1.0], [1.0, 0.0]])


def test_no_non_interop_package_module_imports_interop() -> None:
    """Layer rule: only ``interop/`` may import ``koopman_graph.interop``."""
    package_root = REPO_ROOT / "src" / "koopman_graph"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if "interop" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        mentions_interop = (
            "koopman_graph.interop" in text
            or "from koopman_graph import interop" in text
        )
        if mentions_interop:
            offenders.append(str(path.relative_to(package_root)))
    assert offenders == []
