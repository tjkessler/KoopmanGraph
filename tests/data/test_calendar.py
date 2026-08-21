"""Tests for time-of-day control and phase-index recipes."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from koopman_graph.data import diurnal_control_features, diurnal_phase_index
from koopman_graph.data.calendar import MAX_HARMONICS


def test_diurnal_control_features_midpoint_and_period_wrap() -> None:
    """Quarter-period is (1, 0); times 0 and T match."""
    period = 24.0
    features = diurnal_control_features(
        torch.tensor([0.0, period / 4.0, period]),
        period=period,
    )
    assert features.shape == (3, 2)
    assert features.dtype == torch.float32
    torch.testing.assert_close(features[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        features[1],
        torch.tensor([1.0, 0.0]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(features[2], features[0])


def test_diurnal_control_features_second_harmonic_shape() -> None:
    """Two harmonics yield four columns: sin/cos for k=1 and k=2."""
    period = 8.0
    times = torch.tensor([0.0, period / 4.0])
    features = diurnal_control_features(times, period=period, harmonics=2)
    assert features.shape == (2, 4)
    quarter = features[1]
    expected = torch.tensor(
        [
            math.sin(math.pi / 2.0),
            math.cos(math.pi / 2.0),
            math.sin(math.pi),
            math.cos(math.pi),
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(quarter, expected, atol=1e-6, rtol=0.0)


def test_diurnal_phase_index_bins_and_wrap() -> None:
    """Phase bins use floor(n * frac(t/T)) and wrap at the period."""
    period = 24.0
    times = torch.tensor([0.0, 6.0, 12.0, 24.0])
    bins = diurnal_phase_index(times, period=period, num_phases=4)
    assert bins.dtype == torch.int64
    assert bins.tolist() == [0, 1, 2, 0]


def test_diurnal_helpers_reject_invalid_inputs() -> None:
    """Empty, non-1-D, non-finite, and bound errors raise."""
    with pytest.raises(ValueError, match="at least one"):
        diurnal_control_features(torch.tensor([]), period=1.0)
    with pytest.raises(ValueError, match="1-D"):
        diurnal_control_features(torch.zeros(2, 2), period=1.0)
    with pytest.raises(ValueError, match="finite"):
        diurnal_control_features(torch.tensor([float("nan")]), period=1.0)
    with pytest.raises(ValueError, match="positive finite"):
        diurnal_control_features(torch.tensor([0.0]), period=0.0)
    with pytest.raises(ValueError, match="harmonics"):
        diurnal_control_features(torch.tensor([0.0]), period=1.0, harmonics=0)
    with pytest.raises(ValueError, match="harmonics"):
        diurnal_control_features(
            torch.tensor([0.0]),
            period=1.0,
            harmonics=MAX_HARMONICS + 1,
        )
    with pytest.raises(ValueError, match="num_phases"):
        diurnal_phase_index(torch.tensor([0.0]), period=1.0, num_phases=0)


def test_calendar_module_does_not_import_nn_or_model() -> None:
    """Calendar recipes must not import L4 model or L3 nn."""
    source = Path(__file__).resolve().parents[2] / "src/koopman_graph/data/calendar.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = ("koopman_graph.nn", "koopman_graph.model")
    offenders = [
        name
        for name in imported
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not offenders
