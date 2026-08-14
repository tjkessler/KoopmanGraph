"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

from dataclasses import replace

import torch

from koopman_graph.analysis.plotting import plot_spectrum
from koopman_graph.spectrum_types import KoopmanSpectrum


def _spectrum_from_eigenvalues(eigenvalues: torch.Tensor) -> KoopmanSpectrum:
    magnitudes = eigenvalues.abs()
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=torch.eye(eigenvalues.numel(), dtype=torch.complex128),
        magnitudes=magnitudes,
        growth_rates=torch.log(magnitudes.clamp_min(1e-12)),
        frequencies=torch.angle(eigenvalues) / (2 * torch.pi),
        time_step=1.0,
    )


def test_plot_annotate_all_trustworthy_is_noop_overlay() -> None:
    """annotate_untrustworthy with all residuals below tolerance skips overlay."""
    import matplotlib.pyplot as plt

    annotated = replace(
        _spectrum_from_eigenvalues(
            torch.tensor([0.7 + 0.0j, 0.5 + 0.1j], dtype=torch.complex128)
        ),
        residuals=torch.tensor([1e-6, 1e-6], dtype=torch.float64),
    )
    fig, ax = plt.subplots()
    try:
        plot_spectrum(
            annotated,
            ax=ax,
            annotate_untrustworthy=True,
            residual_tolerance=1e-2,
        )
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)
