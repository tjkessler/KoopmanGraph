"""Discrete Koopman spectrum plotting helpers.

Matplotlib is an optional dependency (available via the ``[dev]`` extra).
Importing this module succeeds without it; :func:`plot_spectrum` raises a
helpful ``ImportError`` at call time when Matplotlib is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from torch import Tensor

from koopman_graph.spectrum_types import KoopmanSpectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection

SpectrumLimits = Literal["unit_disk", "data"]

_MATPLOTLIB_IMPORT_ERROR = (
    "Matplotlib is required for plot_spectrum. "
    "Install with: pip install matplotlib  (or pip install 'koopman-graph[dev]')"
)

_MIN_DATA_HALF_SPAN = 0.05
_DEFAULT_UNIT_DISK_PAD = 0.15
_DEFAULT_DATA_PAD = 0.15


def _require_pyplot():  # noqa: ANN202 — returns pyplot module
    """Import Matplotlib pyplot or raise install guidance.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised when mpl missing
        raise ImportError(_MATPLOTLIB_IMPORT_ERROR) from exc
    return plt


def _as_complex_eigenvalues(
    spectrum: KoopmanSpectrum | Tensor | np.ndarray | Sequence[complex],
) -> np.ndarray:
    """Normalize spectrum inputs to a 1-D complex NumPy array.

    Parameters
    ----------
    spectrum : KoopmanSpectrum or complex array-like
        Discrete-time eigenvalues or a packaged spectrum.

    Returns
    -------
    ndarray
        Complex eigenvalues with shape ``(n,)``.
    """
    if isinstance(spectrum, KoopmanSpectrum):
        values = spectrum.eigenvalues.detach().cpu().numpy()
    elif isinstance(spectrum, Tensor):
        values = spectrum.detach().cpu().numpy()
    else:
        values = np.asarray(spectrum)

    flat = np.asarray(values, dtype=np.complex128).reshape(-1)
    return flat


def _unit_disk_limits(*, pad: float) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` for the padded unit disk.

    Parameters
    ----------
    pad : float
        Padding beyond the unit circle as a fraction of radius 1.

    Returns
    -------
    tuple of float
        Axis limits ``(xmin, xmax, ymin, ymax)``.
    """
    extent = 1.0 + pad
    return (-extent, extent, -extent, extent)


def _data_limits(
    real: np.ndarray,
    imag: np.ndarray,
    *,
    pad: float,
) -> tuple[float, float, float, float]:
    """Return padded equal-aspect limits framing the eigenvalue cloud.

    Parameters
    ----------
    real, imag : ndarray
        Real and imaginary parts of the eigenvalues.
    pad : float
        Padding as a fraction of the equal-aspect half-span.

    Returns
    -------
    tuple of float
        Axis limits ``(xmin, xmax, ymin, ymax)``. Empty inputs fall back to
        the default unit-disk frame.
    """
    if real.size == 0:
        return _unit_disk_limits(pad=_DEFAULT_UNIT_DISK_PAD)

    x_min = float(np.min(real))
    x_max = float(np.max(real))
    y_min = float(np.min(imag))
    y_max = float(np.max(imag))

    x_span = x_max - x_min
    y_span = y_max - y_min
    half_span = 0.5 * max(x_span, y_span, 0.0)
    # Identical / near-identical eigenvalues need a visible absolute floor.
    half_span = max(half_span, _MIN_DATA_HALF_SPAN)
    pad_span = max(pad, 0.0) * half_span * 2.0
    half_span = half_span + 0.5 * pad_span

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    return (
        x_center - half_span,
        x_center + half_span,
        y_center - half_span,
        y_center + half_span,
    )


def plot_spectrum(
    spectrum: KoopmanSpectrum | Tensor | np.ndarray | Sequence[complex],
    *,
    ax: Axes | None = None,
    limits: SpectrumLimits = "unit_disk",
    pad: float | None = None,
    show_unit_circle: bool = True,
    cmap: str = "viridis",
    point_size: float = 45.0,
) -> PathCollection:
    """Plot discrete Koopman eigenvalues in the complex plane.

    Parameters
    ----------
    spectrum : KoopmanSpectrum or complex array-like
        Discrete-time eigenvalues. When a :class:`KoopmanSpectrum` is passed,
        points are colored by ``magnitudes``; otherwise by ``|λ|`` of the
        supplied values.
    ax : matplotlib.axes.Axes, optional
        Target axes. When omitted, a new square figure is created.
    limits : {"unit_disk", "data"}, optional
        Axis framing mode. ``"unit_disk"`` (default) shows the full unit circle
        with fixed padded disk limits (stability-context / teaching view).
        ``"data"`` zooms to the eigenvalue bounding box with padding while
        keeping equal aspect ratio.
    pad : float, optional
        Padding as a fraction of the half-span. Defaults to ``0.15`` for both
        modes (``unit_disk`` → limits ``[-1.15, 1.15]²``). For ``"data"``, a
        small absolute floor keeps single-point / identical eigenvalues visible.
    show_unit_circle : bool, optional
        Draw the unit circle (clipped to the axes). Default is ``True``.
    cmap : str, optional
        Matplotlib colormap for ``|λ|``. Default is ``"viridis"``.
    point_size : float, optional
        Scatter marker size. Default is ``45``.

    Returns
    -------
    matplotlib.collections.PathCollection
        Scatter collection (suitable for ``fig.colorbar``).

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If ``limits`` is not ``"unit_disk"`` or ``"data"``, or ``pad`` is
        negative.
    """
    plt = _require_pyplot()

    if limits not in ("unit_disk", "data"):
        msg = f'limits must be "unit_disk" or "data", got {limits!r}'
        raise ValueError(msg)

    if pad is None:
        pad = _DEFAULT_UNIT_DISK_PAD if limits == "unit_disk" else _DEFAULT_DATA_PAD
    if pad < 0:
        msg = f"pad must be non-negative, got {pad}"
        raise ValueError(msg)

    eigenvalues = _as_complex_eigenvalues(spectrum)
    real = eigenvalues.real
    imag = eigenvalues.imag
    if isinstance(spectrum, KoopmanSpectrum):
        magnitudes = spectrum.magnitudes.detach().cpu().numpy().reshape(-1)
        if magnitudes.shape != eigenvalues.shape:
            magnitudes = np.abs(eigenvalues)
    else:
        magnitudes = np.abs(eigenvalues)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    if show_unit_circle:
        theta = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            color="0.6",
            linewidth=1.5,
            zorder=1,
            label="unit circle",
        )

    scatter = ax.scatter(
        real,
        imag,
        c=magnitudes if magnitudes.size else None,
        cmap=cmap if magnitudes.size else None,
        s=point_size,
        zorder=3,
    )
    ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_aspect("equal", adjustable="box")

    if limits == "unit_disk":
        xmin, xmax, ymin, ymax = _unit_disk_limits(pad=pad)
    else:
        xmin, xmax, ymin, ymax = _data_limits(real, imag, pad=pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return scatter
