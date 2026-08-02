"""Lazy ``[msm]`` deeptime trajectory-feature conversion helpers.

Importing this module does **not** import deeptime. Call
:func:`require_deeptime` (or the conversion helpers below) at use sites so
core ``import koopman_graph`` stays free of the optional extra.

These helpers are a **diagnostic / teaching** bridge for MSM-adjacent
oracles (implied timescales, rank-order checks). They are not a PyEMMA
replacement and do not fit MSMs.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from torch import Tensor

_DEEPTIME_INSTALL_HINT = (
    "deeptime is required for the MSM / GraphVAMP toolchain. "
    "Install with: pip install 'koopman-graph[msm]'"
)


def require_deeptime() -> ModuleType:
    """Import ``deeptime`` or raise a guided ``ImportError``.

    Returns
    -------
    module
        The ``deeptime`` package.

    Raises
    ------
    ImportError
        If deeptime is not installed (``pip install 'koopman-graph[msm]'``).
    """
    try:
        import deeptime
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise ImportError(_DEEPTIME_INSTALL_HINT) from exc
    return deeptime


def trajectory_features_to_deeptime(
    features: Tensor | Any,
    *,
    lag: int = 1,
) -> Any:
    """Wrap ``(T, F)`` trajectory features as a deeptime ``TrajectoryDataset``.

    Parameters
    ----------
    features : Tensor or array-like
        Feature matrix with shape ``(num_timesteps, num_features)``. Torch
        tensors are detached and moved to CPU before conversion.
    lag : int, optional
        Positive lag in **frames** (integer steps) forwarded to
        :class:`deeptime.util.data.TrajectoryDataset`. Default is ``1``.

    Returns
    -------
    deeptime.util.data.TrajectoryDataset
        Time-lagged dataset whose ``.trajectory`` recovers the feature
        matrix (see :func:`trajectory_features_from_deeptime`).

    Raises
    ------
    ImportError
        If the ``[msm]`` extra is not installed.
    ValueError
        If ``lag`` / shapes are invalid.
    """
    require_deeptime()
    from deeptime.util.data import TrajectoryDataset

    array = _as_feature_ndarray(features)
    if lag < 1:
        msg = f"lag must be >= 1 frame steps, got {lag}"
        raise ValueError(msg)
    if array.shape[0] <= lag:
        msg = f"need num_timesteps > lag ({lag}), got {array.shape[0]}"
        raise ValueError(msg)
    return TrajectoryDataset(int(lag), array)


def trajectory_features_from_deeptime(dataset: Any) -> Any:
    """Extract a ``(T, F)`` NumPy feature matrix from a deeptime trajectory.

    Parameters
    ----------
    dataset : TrajectoryDataset or TrajectoriesDataset or array-like
        A :class:`deeptime.util.data.TrajectoryDataset` (preferred), a
        single-trajectory :class:`~deeptime.util.data.TrajectoriesDataset`,
        or an array-like already shaped ``(T, F)``.

    Returns
    -------
    numpy.ndarray
        Feature matrix with shape ``(num_timesteps, num_features)`` and
        floating dtype.

    Raises
    ------
    ImportError
        If the ``[msm]`` extra is not installed.
    ValueError
        If the object is not a recognized single-trajectory payload.
    """
    require_deeptime()

    if hasattr(dataset, "trajectory"):
        return _as_feature_ndarray(dataset.trajectory)
    if hasattr(dataset, "trajectories"):
        trajectories = list(dataset.trajectories)
        if len(trajectories) != 1:
            msg = (
                "trajectory_features_from_deeptime accepts a single "
                f"trajectory; got {len(trajectories)} trajectories"
            )
            raise ValueError(msg)
        return _as_feature_ndarray(trajectories[0])
    try:
        return _as_feature_ndarray(dataset)
    except (TypeError, ValueError) as exc:
        msg = (
            "expected a deeptime TrajectoryDataset, a single-trajectory "
            "TrajectoriesDataset, or a (T, F) array-like"
        )
        raise ValueError(msg) from exc


def _as_feature_ndarray(features: Tensor | Any) -> Any:
    """Normalize features to a contiguous 2-D floating NumPy array.

    Parameters
    ----------
    features
        See signature.

    Returns
    -------
        See signature.
    """
    import numpy as np

    if isinstance(features, Tensor):
        array = features.detach().cpu().numpy()
    else:
        array = np.asarray(features)
    if array.ndim != 2:
        msg = (
            "trajectory features must have shape (num_timesteps, "
            f"num_features), got shape {tuple(array.shape)}"
        )
        raise ValueError(msg)
    if array.shape[0] < 1 or array.shape[1] < 1:
        msg = (
            "trajectory features must be non-empty in both dimensions, "
            f"got shape {tuple(array.shape)}"
        )
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64, copy=False)
    return np.ascontiguousarray(array)
