"""Tests for discrete Koopman spectrum plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import koopman_graph
from koopman_graph.analysis import plot_spectrum
from koopman_graph.analysis.plotting import _data_limits, _unit_disk_limits
from koopman_graph.spectrum_types import KoopmanSpectrum


def _spectrum_from_eigenvalues(eigenvalues: torch.Tensor) -> KoopmanSpectrum:
    """Build a minimal :class:`KoopmanSpectrum` for plotting tests."""
    magnitudes = eigenvalues.abs()
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=torch.eye(eigenvalues.numel(), dtype=torch.complex128),
        magnitudes=magnitudes,
        growth_rates=torch.log(magnitudes.clamp_min(1e-12)),
        frequencies=torch.angle(eigenvalues) / (2 * torch.pi),
        time_step=1.0,
    )


def test_plot_spectrum_not_in_root_all() -> None:
    """Capability-only helper stays off the thin root façade."""
    assert "plot_spectrum" not in koopman_graph.__all__
    assert callable(plot_spectrum)


def test_unit_disk_limits_match_teaching_view() -> None:
    """Default unit-disk pad reproduces the classic ``[-1.15, 1.15]²`` frame."""
    assert _unit_disk_limits(pad=0.15) == (-1.15, 1.15, -1.15, 1.15)


def test_data_limits_tighter_than_unit_disk_for_clustered_modes() -> None:
    """Near-circle clusters zoom in under ``limits='data'``."""
    eigenvalues = torch.tensor(
        [0.98 + 0.02j, 0.99 + 0.01j, 0.985 - 0.015j, 0.97 + 0.0j],
        dtype=torch.complex128,
    )
    spectrum = _spectrum_from_eigenvalues(eigenvalues)

    fig_disk, ax_disk = plt.subplots()
    fig_data, ax_data = plt.subplots()
    try:
        plot_spectrum(spectrum, ax=ax_disk, limits="unit_disk")
        plot_spectrum(spectrum, ax=ax_data, limits="data")

        disk_xlim = ax_disk.get_xlim()
        data_xlim = ax_data.get_xlim()
        data_ylim = ax_data.get_ylim()
        disk_span = disk_xlim[1] - disk_xlim[0]
        data_span = data_xlim[1] - data_xlim[0]
        assert data_span < disk_span
        assert ax_data.get_aspect() in {1.0, "equal"}

        # Frame stays close to the data bounding box (modest pad only).
        real = eigenvalues.real.numpy()
        imag = eigenvalues.imag.numpy()
        max_span = max(float(real.max() - real.min()), float(imag.max() - imag.min()))
        assert data_span == pytest.approx(max_span * 1.05, rel=1e-6)
        assert data_xlim[0] <= float(real.min())
        assert data_xlim[1] >= float(real.max())
        assert data_ylim[0] <= float(imag.min())
        assert data_ylim[1] >= float(imag.max())
    finally:
        plt.close(fig_disk)
        plt.close(fig_data)


def test_plot_spectrum_empty_and_identical_do_not_crash() -> None:
    """Empty, single-point, and identical eigenvalues remain plottable."""
    cases = [
        np.asarray([], dtype=np.complex128),
        np.asarray([0.9 + 0.1j], dtype=np.complex128),
        np.asarray([0.95 + 0.0j, 0.95 + 0.0j, 0.95 + 0.0j], dtype=np.complex128),
    ]
    for values in cases:
        fig, ax = plt.subplots()
        try:
            scatter = plot_spectrum(values, ax=ax, limits="data")
            assert scatter is not None
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            assert xlim[1] > xlim[0]
            assert ylim[1] > ylim[0]
        finally:
            plt.close(fig)


def test_plot_spectrum_rejects_bad_limits_and_pad() -> None:
    """Invalid ``limits`` / negative ``pad`` raise ``ValueError``."""
    values = np.asarray([0.5 + 0.0j], dtype=np.complex128)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="limits must be"):
            plot_spectrum(values, ax=ax, limits="half_plane")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="pad must be non-negative"):
            plot_spectrum(values, ax=ax, pad=-0.1)
    finally:
        plt.close(fig)


def test_data_limits_helper_uses_absolute_floor() -> None:
    """Identical points still get a positive equal-aspect window."""
    xmin, xmax, ymin, ymax = _data_limits(
        np.asarray([1.0]),
        np.asarray([0.0]),
        pad=0.05,
    )
    assert xmax - xmin == pytest.approx(ymax - ymin)
    assert xmax - xmin == pytest.approx(2.0 * 1e-3 * 1.05)
    assert xmax - xmin > 0.0


def test_data_limits_match_epidemic_style_cluster_tightly() -> None:
    """Schur-like near-circle clusters are not inflated by a large absolute floor."""
    real = np.asarray([0.968, 0.985, 0.995, 0.990, 0.978])
    imag = np.asarray([-0.012, 0.0, 0.008, -0.005, 0.014])
    xmin, xmax, ymin, ymax = _data_limits(real, imag, pad=0.05)
    span = xmax - xmin
    data_span = max(float(real.max() - real.min()), float(imag.max() - imag.min()))
    assert span == pytest.approx(data_span * 1.05, rel=1e-6)
    # Previous floor (half-span 0.05) produced ~0.1+ windows; stay well below that.
    assert span < 0.06
    assert xmin <= float(real.min()) <= float(real.max()) <= xmax
    assert ymin <= float(imag.min()) <= float(imag.max()) <= ymax
