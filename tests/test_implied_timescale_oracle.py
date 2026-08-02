"""Rank-order implied-timescale oracle: in-repo vs deeptime (G7).

Contract
--------
On the seeded synthetic two-state fixture we assert **rank order** of the
leading implied timescales and a **loose** relative bound on the slowest
mode — not exact numeric agreement.

Why rank order rather than exact values: the in-repo path uses an empirical
least-squares Koopman matrix on mean-centered features, while deeptime's
VAMP estimator uses a covariance / SVD pipeline. Different dictionaries and
estimators yield different finite-sample spectra; tight equality would be an
overclaim. Rank order of the slow process (one dominant slow timescale ≫
faster residuals) is the verifiable G7 outcome.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("deeptime")

from deeptime.decomposition import VAMP  # noqa: E402

from koopman_graph.analysis import implied_timescales  # noqa: E402
from koopman_graph.datasets.molecular import (  # noqa: E402
    generate_synthetic_two_state,
)
from koopman_graph.interop import trajectory_features_to_deeptime  # noqa: E402

# Loose relative bound on the slowest ranked timescale only.
_SLOW_RELATIVE_TOL = 0.5
# Slow mode must clearly outrank the next finite timescale.
_SLOW_TO_NEXT_RATIO = 5.0
_LAG_STEPS = 1
_NUM_TIMESTEPS = 4000
_SEED = 1
_TOP_K = 2


def _flattened_node_features(traj) -> np.ndarray:
    """Stack per-frame node features to ``(T, N·F)`` float64."""
    frames = [
        traj.sequence[t].x.detach().cpu().numpy().reshape(-1)
        for t in range(traj.sequence.num_timesteps)
    ]
    return np.asarray(frames, dtype=np.float64)


def _inrepo_timescales_desc(features: np.ndarray, *, lag_steps: int) -> np.ndarray:
    """Empirical Koopman → :func:`implied_timescales`, sorted descending."""
    x = torch.as_tensor(features[:-lag_steps], dtype=torch.float64)
    y = torch.as_tensor(features[lag_steps:], dtype=torch.float64)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    koopman = torch.linalg.lstsq(x, y).solution
    eigenvalues = torch.linalg.eigvals(koopman)
    report = implied_timescales(eigenvalues, lag_steps=lag_steps)
    values = report.timescales[report.valid].detach().cpu().numpy()
    values = values[np.isfinite(values) & (values > 0.0)]
    return np.sort(values)[::-1]


def _deeptime_timescales_desc(features: np.ndarray, *, lag_steps: int) -> np.ndarray:
    """deeptime VAMP implied timescales, sorted descending.

    Features are wrapped with :func:`trajectory_features_to_deeptime` so the
    interop boundary is exercised; VAMP still fits the raw ``(T, F)`` array.
    """
    dataset = trajectory_features_to_deeptime(features, lag=lag_steps)
    assert dataset.trajectory.shape == features.shape
    model = VAMP(lagtime=lag_steps).fit(features).fetch_model()
    values = np.asarray(model.timescales(), dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    return np.sort(values)[::-1]


def test_rank_order_top_two_timescales_vs_deeptime() -> None:
    """Leading two timescales share rank structure and a loose slow bound."""
    traj = generate_synthetic_two_state(
        num_timesteps=_NUM_TIMESTEPS,
        seed=_SEED,
    )
    features = _flattened_node_features(traj)
    inrepo = _inrepo_timescales_desc(features, lag_steps=_LAG_STEPS)
    deeptime_ts = _deeptime_timescales_desc(features, lag_steps=_LAG_STEPS)

    assert inrepo.shape[0] >= _TOP_K
    assert deeptime_ts.shape[0] >= _TOP_K

    inrepo_top = inrepo[:_TOP_K]
    deeptime_top = deeptime_ts[:_TOP_K]

    # Rank order: both estimators report a slow mode that clearly outranks
    # the next finite timescale (same descending ranking of the top two).
    assert inrepo_top[0] > _SLOW_TO_NEXT_RATIO * inrepo_top[1]
    assert deeptime_top[0] > _SLOW_TO_NEXT_RATIO * deeptime_top[1]
    assert np.argsort(-inrepo_top).tolist() == np.argsort(-deeptime_top).tolist()

    slow_rel = abs(inrepo_top[0] - deeptime_top[0]) / max(
        inrepo_top[0],
        deeptime_top[0],
    )
    assert slow_rel < _SLOW_RELATIVE_TOL

    # Optional honesty check vs the closed-form oracle (loose factor-of-two).
    oracle = traj.oracle_slow_timescale_steps
    assert inrepo_top[0] == pytest.approx(oracle, rel=1.0)
    assert deeptime_top[0] == pytest.approx(oracle, rel=1.0)


def test_state_indicator_smoke_agrees_on_single_slow_mode() -> None:
    """One-hot state indicators: both recover one comparable slow timescale."""
    traj = generate_synthetic_two_state(
        num_timesteps=_NUM_TIMESTEPS,
        seed=_SEED,
    )
    labels = traj.state_labels.detach().cpu().numpy()
    features = np.eye(2, dtype=np.float64)[labels]

    inrepo = _inrepo_timescales_desc(features, lag_steps=_LAG_STEPS)
    deeptime_ts = _deeptime_timescales_desc(features, lag_steps=_LAG_STEPS)

    assert inrepo.shape[0] >= 1
    assert deeptime_ts.shape[0] >= 1
    slow_rel = abs(inrepo[0] - deeptime_ts[0]) / max(inrepo[0], deeptime_ts[0])
    assert slow_rel < _SLOW_RELATIVE_TOL


def test_graph_vamp_smoke_on_synthetic_fixture() -> None:
    """GraphVAMP fit/score smoke on the synthetic contact trajectory.

    This does **not** assert timescale rank order against deeptime — GraphVAMP
    embeddings are a different dictionary than state indicators / raw features.
    """
    from koopman_graph.baselines import GraphVAMPBaseline

    traj = generate_synthetic_two_state(num_timesteps=64, seed=0)
    model = GraphVAMPBaseline(
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        num_layers=1,
    )
    model.fit(
        traj.sequence,
        lag=_LAG_STEPS,
        epochs=5,
        lr=1e-2,
        edge_index=traj.edge_index,
    )
    score = model.score(traj.sequence, lag=_LAG_STEPS)
    assert score == score  # not NaN
