"""Re-export spectral conditioning diagnostics from the neutral leaf.

Compute lives in :mod:`koopman_graph.spectrum_types` so operators and the
model façade never import this package for eigendecomposition. Resolvent
grids remain opt-in in :mod:`koopman_graph.analysis.pseudospectra` and are
not populated by :func:`~koopman_graph.spectrum_types.compute_spectrum`.
Discrete Nyquist frequency is :math:`1/(2\\Delta t)` in cycles per unit
time; generator spectra leave it unset. ``mode_amplitudes`` warns when
:math:`\\kappa(V)` exceeds ``CONDITION_WARN`` and raises
:class:`~koopman_graph.spectrum_types.DefectiveSpectrumError` when
:math:`V` is singular.
"""

from __future__ import annotations

from koopman_graph.spectrum_types import (
    CONDITION_WARN,
    DefectiveSpectrumError,
    SpectralDiagnostics,
    compute_spectral_diagnostics,
)

__all__ = [
    "CONDITION_WARN",
    "DefectiveSpectrumError",
    "SpectralDiagnostics",
    "compute_spectral_diagnostics",
]
