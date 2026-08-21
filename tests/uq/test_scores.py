"""Tests for Gaussian and ensemble proper scores."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.uq import energy_score, gaussian_crps, gaussian_nll


def test_gaussian_nll_and_crps_prefer_the_true_oracle() -> None:
    """The matching Gaussian scores lower than a shifted mean."""
    observation = torch.zeros(8)
    mean = torch.zeros(8)
    variance = torch.ones(8)
    true_nll = gaussian_nll(observation, mean, variance)
    wrong_nll = gaussian_nll(observation, mean + 1.5, variance)
    assert float(true_nll) < float(wrong_nll)
    true_crps = gaussian_crps(observation, mean, variance)
    wrong_crps = gaussian_crps(observation, mean + 1.5, variance)
    assert float(true_crps) < float(wrong_crps)
    expected_nll = 0.5 * math.log(2.0 * math.pi)
    assert float(true_nll) == pytest.approx(expected_nll, rel=0.0, abs=1e-6)


def test_energy_score_prefers_centered_samples() -> None:
    """An ensemble around the observation scores better than a shift."""
    generator = torch.Generator().manual_seed(0)
    observation = torch.zeros(2)
    samples = torch.randn(16, 2, generator=generator)
    shifted = samples + 2.0
    true_score = energy_score(observation, samples)
    wrong_score = energy_score(observation, shifted)
    assert float(true_score) < float(wrong_score)


def test_score_guards() -> None:
    """Variance, reduction, and ensemble shape errors raise."""
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(y, y, torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        gaussian_crps(y, y, torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="reduction"):
        gaussian_nll(y, y, torch.ones(2), reduction="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2"):
        energy_score(y, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="samples\\[1:\\]"):
        energy_score(torch.zeros(3), torch.zeros(4, 2))
