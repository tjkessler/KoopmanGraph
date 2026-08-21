"""Combinatorial Hodge split of Koopman mode shapes.

:func:`hodge_decompose_modes` projects eigenvector columns onto the
gradient / curl / harmonic subspaces of a static signed incidence
:math:`B_1` (``Lim2020Hodge``):

.. math::

    z_k = d_{k-1}\\alpha + \\delta_{k+1}\\beta + \\gamma.

For :math:`k=0`, :math:`d_{-1}=0` so the gradient block is zero and
``curl`` is the image of :math:`\\delta_1=B_1`. For :math:`k=1`
without faces, :math:`\\delta_2=0` so the curl block is zero and
``gradient`` is the image of :math:`d_0=B_1^{\\top}`. A consistently
oriented cycle has a one-dimensional harmonic 1-cochain (circulation).
That algebraic kernel is **not** a physical current or validated
circulation.

This helper is analysis-only. It does not change the factory default,
is **not** :class:`~koopman_graph.operators.HodgeKoopmanOperator`, and
is **not** TopologicX / sheaf parity (``TopoX2024``).

This module must not import :mod:`koopman_graph.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.observables import boundary_incidence_b1
from koopman_graph.spectrum_types import KoopmanSpectrum

DEFAULT_HODGE_ATOL = 1e-6

__all__ = [
    "DEFAULT_HODGE_ATOL",
    "HodgeModeComponents",
    "hodge_decompose_modes",
]


@dataclass(frozen=True)
class HodgeModeComponents:
    """Gradient / curl / harmonic parts of stored mode-shape columns.

    Attributes
    ----------
    gradient : Tensor
        Image of :math:`d_{k-1}`. Shape ``(n_cells, n_modes)``. Same
        units as the eigenvector entries (dimensionless mode
        coordinates). Zero for :math:`k=0`.
    curl : Tensor
        Image of :math:`\\delta_{k+1}`. Shape ``(n_cells, n_modes)``.
        Zero for :math:`k=1` when no faces are supplied.
    harmonic : Tensor
        Component in :math:`\\ker L_k`. Shape ``(n_cells, n_modes)``.
        Not a physical circulation certificate.
    """

    gradient: Tensor
    curl: Tensor
    harmonic: Tensor

    def __post_init__(self) -> None:
        """Validate matching column layouts.

        Raises
        ------
        ValueError
            If the three blocks do not share a 2-D shape.
        """
        shape = tuple(self.gradient.shape)
        if self.gradient.ndim != 2 or int(self.gradient.shape[1]) < 1:
            msg = f"gradient must have shape (n_cells, n_modes), got {shape}"
            raise ValueError(msg)
        if tuple(self.curl.shape) != shape or tuple(self.harmonic.shape) != shape:
            msg = (
                "gradient, curl, and harmonic must share shape "
                f"{shape}; got {tuple(self.curl.shape)} and "
                f"{tuple(self.harmonic.shape)}"
            )
            raise ValueError(msg)


def _project_image(basis: Tensor, vectors: Tensor) -> Tensor:
    """Project columns of ``vectors`` onto :math:`\\operatorname{im}(basis)`.

    Parameters
    ----------
    basis : Tensor
        Matrix whose columns span the target subspace, shape
        ``(n_cells, n_gen)``.
    vectors : Tensor
        Columns to project, shape ``(n_cells, n_modes)``.

    Returns
    -------
    Tensor
        Projected columns, same shape as ``vectors``.
    """
    if int(basis.shape[1]) == 0:
        return torch.zeros_like(vectors)
    coefficients = torch.linalg.lstsq(basis, vectors).solution
    return basis @ coefficients


def _resolve_degree(
    n_rows: int,
    *,
    num_nodes: int,
    num_edges: int,
    degree: int | None,
) -> int:
    """Infer or validate the cochain degree from eigenvector length.

    Parameters
    ----------
    n_rows : int
        Eigenvector row count.
    num_nodes : int
        Number of 0-cells.
    num_edges : int
        Number of 1-cells.
    degree : int or None
        Caller degree, or ``None`` to infer.

    Returns
    -------
    int
        ``0`` or ``1``.

    Raises
    ------
    ValueError
        If the degree is unsupported or the row count does not match.
    """
    if degree is None:
        if n_rows == int(num_nodes):
            return 0
        if n_rows == int(num_edges):
            return 1
        msg = (
            "eigenvector rows must equal num_nodes or num_edges to infer "
            f"degree, got rows={n_rows}, N={num_nodes}, E={num_edges}"
        )
        raise ValueError(msg)
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise ValueError(f"degree must be 0 or 1, got {degree!r}")
    if int(degree) not in {0, 1}:
        msg = f"degree must be 0 or 1, got {degree}; k=2 curl is not shipped"
        raise ValueError(msg)
    expected = int(num_nodes) if int(degree) == 0 else int(num_edges)
    if n_rows != expected:
        kind = "num_nodes" if int(degree) == 0 else "num_edges"
        msg = (
            f"degree={degree} requires eigenvector rows equal to {kind} "
            f"{expected}, got {n_rows}"
        )
        raise ValueError(msg)
    return int(degree)


def hodge_decompose_modes(
    spectrum: KoopmanSpectrum,
    edge_index: Tensor,
    *,
    num_nodes: int,
    degree: int | None = None,
) -> HodgeModeComponents:
    """Split stored eigenvectors into gradient / curl / harmonic parts.

    Eigenvector columns are treated as real or complex cochains on the
    oriented 1-skeleton. Complex columns are split into real and
    imaginary parts, projected, and reassembled. The split is the
    linear algebra of :math:`B_1`, not a physical circulation and not
    a sheaf Laplacian (``Lim2020Hodge``, ``TopoX2024``).

    Parameters
    ----------
    spectrum : KoopmanSpectrum
        Stored eigenpairs. Eigenvector rows must equal ``num_nodes``
        (:math:`k=0`) or the number of ``edge_index`` columns
        (:math:`k=1`).
    edge_index : Tensor
        Oriented 1-cells ``(2, E)``. One column per undirected edge.
    num_nodes : int
        Number of 0-cells :math:`N`.
    degree : int or None, optional
        Cochain degree ``0`` or ``1``. Default infers from the
        eigenvector row count. When ``num_nodes == num_edges``,
        that inference is ``0``; pass ``degree=1`` for a
        1-cochain on a cycle.

    Returns
    -------
    HodgeModeComponents
        Three blocks with shape ``(n_cells, n_modes)``. Reconstruction
        is ``gradient + curl + harmonic``.

    Raises
    ------
    TypeError
        If ``spectrum`` is not a :class:`KoopmanSpectrum`.
    ValueError
        If ``num_nodes``, ``edge_index``, ``degree``, or eigenvector
        shape is invalid.

    Notes
    -----
    Face-based :math:`k=1` curl (:math:`\\operatorname{im} B_2`) is
    not implemented. Isolated-node graphs have :math:`L_0=0`, so every
    0-cochain is harmonic. On a consistently oriented cycle the
    constant 1-cochain spans :math:`\\ker L_1`. When
    ``num_nodes == num_edges`` and ``degree`` is omitted, inference
    returns ``0``.

    References
    ----------
    Lim, L.-H. (2020). Hodge Laplacians on graphs. *SIAM Review*
    62:685–715. doi:10.1137/18M1223101 (``Lim2020Hodge``).
    Hajij, M. et al. (2024). TopoX: a suite of Python packages for
    machine learning on topological domains. arXiv:2402.02441
    (``TopoX2024``).
    """
    if not isinstance(spectrum, KoopmanSpectrum):
        msg = f"spectrum must be a KoopmanSpectrum, got {type(spectrum).__name__}"
        raise TypeError(msg)
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise ValueError(f"num_nodes must be a positive int, got {num_nodes!r}")
    if int(num_nodes) < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    vectors = spectrum.eigenvectors
    if vectors.ndim != 2 or int(vectors.shape[0]) < 1 or int(vectors.shape[1]) < 1:
        msg = (
            "spectrum.eigenvectors must have shape (n_cells, n_modes) "
            f"with n_cells >= 1, got {tuple(vectors.shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(vectors).all().item()):
        raise ValueError("spectrum.eigenvectors must be finite")
    incidence = boundary_incidence_b1(edge_index, num_nodes=int(num_nodes))
    resolved = _resolve_degree(
        int(vectors.shape[0]),
        num_nodes=int(num_nodes),
        num_edges=int(incidence.shape[1]),
        degree=degree,
    )
    working = vectors.detach().to(dtype=torch.complex128)
    real = working.real.to(dtype=torch.float64)
    imag = working.imag.to(dtype=torch.float64)
    incidence64 = incidence.to(dtype=torch.float64, device=working.device)
    if resolved == 0:
        gradient_real = torch.zeros_like(real)
        curl_real = _project_image(incidence64, real)
        harmonic_real = real - curl_real
        gradient_imag = torch.zeros_like(imag)
        curl_imag = _project_image(incidence64, imag)
        harmonic_imag = imag - curl_imag
    else:
        coboundary = incidence64.transpose(0, 1)
        gradient_real = _project_image(coboundary, real)
        curl_real = torch.zeros_like(real)
        harmonic_real = real - gradient_real
        gradient_imag = _project_image(coboundary, imag)
        curl_imag = torch.zeros_like(imag)
        harmonic_imag = imag - gradient_imag
    if not vectors.is_complex():
        return HodgeModeComponents(
            gradient=gradient_real.to(dtype=vectors.dtype),
            curl=curl_real.to(dtype=vectors.dtype),
            harmonic=harmonic_real.to(dtype=vectors.dtype),
        )
    return HodgeModeComponents(
        gradient=torch.complex(gradient_real, gradient_imag).to(dtype=vectors.dtype),
        curl=torch.complex(curl_real, curl_imag).to(dtype=vectors.dtype),
        harmonic=torch.complex(harmonic_real, harmonic_imag).to(dtype=vectors.dtype),
    )
