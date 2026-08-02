"""Tests for topology-blind VAMP-2 precursor (TASK-1850)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from koopman_graph.baselines import vamp2_loss, vamp2_score
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.training import LossWeights, compute_training_loss

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "koopman_graph"
    / "baselines"
    / "vamp2.py"
)


def _seeded_lag_features(
    *,
    n_samples: int = 64,
    n_features: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build correlated lag features with a controllable linear map."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, n_features, generator=generator)
    weight = torch.randn(n_features, n_features, generator=generator) * 0.5
    y = x @ weight.T + 0.05 * torch.randn(n_samples, n_features, generator=generator)
    return x, y


def test_vamp2_score_rejects_invalid_inputs() -> None:
    """Validation guards reject bad epsilon, shape, and sample counts."""
    x, y = _seeded_lag_features()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        vamp2_score(x, y, epsilon=0.0)
    with pytest.raises(ValueError, match="2D"):
        vamp2_score(x[:, 0], y[:, 0])
    with pytest.raises(ValueError, match="share shape"):
        vamp2_score(x, y[:, :3])
    with pytest.raises(ValueError, match="at least 2 lag samples"):
        vamp2_score(x[:1], y[:1])
    with pytest.raises(ValueError, match="n_features must be positive"):
        vamp2_score(torch.zeros(4, 0), torch.zeros(4, 0))


def test_vamp2_score_finite_and_loss_negation() -> None:
    """Seeded lag pairs yield a finite positive score; loss is its negation."""
    x, y = _seeded_lag_features()
    score = vamp2_score(x, y)
    assert torch.isfinite(score)
    assert float(score) > 0.0
    assert torch.allclose(vamp2_loss(x, y), -score)


def test_vamp2_score_gradient_flow() -> None:
    """VAMP-2 score admits gradients w.r.t. feature matrices."""
    x, y = _seeded_lag_features(n_samples=32, n_features=3)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    vamp2_score(x, y).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert y.grad is not None and y.grad.abs().sum() > 0


def test_compute_training_loss_and_fit_record_vamp2() -> None:
    """``LossWeights.vamp2`` enters the training loss path and FitHistory."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    sequence = GraphSnapshotSequence.from_arrays(
        torch.randn(6, 3, 2),
        edge_index,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
    )
    weights = LossWeights(reconstruction=1.0, vamp2=0.1)
    breakdown = compute_training_loss(model, sequence, weights)
    assert torch.isfinite(breakdown.vamp2)
    assert torch.isfinite(breakdown.total)
    expected_total = (
        weights.reconstruction * breakdown.reconstruction
        + weights.vamp2 * breakdown.vamp2
    )
    assert torch.allclose(breakdown.total, expected_total)

    history = model.fit(sequence, epochs=1, loss_weights=weights)
    assert len(history.vamp2_loss) == 1
    assert math.isfinite(history.vamp2_loss[0])


def test_module_doc_honesty_keywords() -> None:
    """Module docstring excludes GraphVAMPnets / MD; points to 0.11 roadmap."""
    text = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "vamp-2" in text or "vamp2" in text
    assert "graphvampnets" in text
    assert "0.11" in text
    assert "molecular" in text or "md" in text
    assert "not" in text


def test_deeptime_oracle_agrees_when_installed() -> None:
    """Optional deeptime VAMP-2 oracle on a shared AR(1) trajectory.

    Compares :func:`vamp2_score` on lag-1 pairs to deeptime ``VAMP.score(2)``.
    deeptime's score includes the constant singular value (+1); our mean-free
    Frobenius score omits it, so the oracle is ``dt_score - 1``.
    """
    deeptime = pytest.importorskip("deeptime")
    from deeptime.decomposition import VAMP

    generator = torch.Generator().manual_seed(7)
    state = torch.randn(3, generator=generator)
    frames = [state]
    for _ in range(200):
        state = 0.7 * state + 0.3 * torch.randn(3, generator=generator)
        frames.append(state)
    trajectory = torch.stack(frames)
    x = trajectory[:-1]
    y = trajectory[1:]
    our_score = float(vamp2_score(x, y))
    model = VAMP(lagtime=1, dim=None).fit(trajectory.numpy())
    dt_score = float(model.fetch_model().score(2))
    assert our_score == pytest.approx(dt_score - 1.0, rel=0.05, abs=0.05)
    del deeptime
