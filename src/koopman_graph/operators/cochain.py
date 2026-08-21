"""Degree-specific cochain Koopman maps for k<=1.

:class:`CochainState` holds node (0-cochain) and edge (1-cochain) latents.
:class:`CochainKoopmanOperator` advances each degree with a feature-axis
map :math:`K_k` and optional incidence cross-degree terms. Incidence is
the static signed :math:`B_1` from
:func:`~koopman_graph.observables.boundary_incidence_b1`.

This is **not** :class:`~koopman_graph.operators.HodgeKoopmanOperator`
(node Laplacian neighbor term; ``koopman="hodge"``), **not** a factory
kind, and **not** TopologicX / sheaf parity (``TopoX2024``,
``Lim2020Hodge``). Face latents may be stored; :math:`k=2` is not
evolved.

Feature-axis intertwining :math:`d_0(z_0 K_0^{\\top})\\approx
(d_0 z_0)K_1^{\\top}` with :math:`d_0=B_1^{\\top}` reduces to
:math:`K_0\\approx K_1` on the image of :math:`d_0`. That residual is
exposed as a loss. Boundary nilpotency :math:`B_1 B_2\\approx 0` is
checked on caller-supplied incidences (``Lim2020Hodge``).

This module must not import :mod:`koopman_graph.model`,
:mod:`koopman_graph.data`, or :mod:`koopman_graph.nn`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from koopman_graph.observables import boundary_incidence_b1

DEFAULT_NILPOTENCY_ATOL = 1e-6

__all__ = [
    "DEFAULT_NILPOTENCY_ATOL",
    "BoundaryNilpotencyReport",
    "CochainKoopmanOperator",
    "CochainState",
    "boundary_nilpotency",
]


def _require_feature_matrix(name: str, value: Tensor) -> None:
    """Refuse empty, non-real, or non-finite feature tables.

    Parameters
    ----------
    name : str
        Field name for the error message.
    value : Tensor
        Candidate ``(n_cells, d)`` table.

    Raises
    ------
    TypeError
        If ``value`` is not a tensor.
    ValueError
        If rank, dtype, or finiteness is invalid.
    """
    if not isinstance(value, Tensor):
        msg = f"{name} must be a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    if value.is_complex():
        raise ValueError(f"{name} must be real")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if int(value.ndim) != 2 or int(value.shape[0]) < 0:
        raise ValueError(
            f"{name} must have shape (n_cells, d), got {tuple(value.shape)}"
        )
    if int(value.shape[1]) < 1:
        raise ValueError(f"{name} latent width must be >= 1, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class CochainState:
    """Typed :math:`k`-cochain latents for :math:`k\\in\\{0,1\\}`.

    Attributes
    ----------
    node : Tensor
        0-cochain latents with shape ``(N, d)``.
    edge : Tensor
        1-cochain latents with shape ``(E, d)``. Same trailing width as
        ``node``.
    face : Tensor or None
        Optional 2-cochain table ``(F, d2)``. Stored only; the operator
        does not advance :math:`k=2`.
    """

    node: Tensor
    edge: Tensor
    face: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate node / edge tables and an optional face table.

        Raises
        ------
        TypeError
            If a field is not a tensor.
        ValueError
            If shapes, dtypes, or finiteness are invalid.
        """
        _require_feature_matrix("node", self.node)
        _require_feature_matrix("edge", self.edge)
        if int(self.node.shape[1]) != int(self.edge.shape[1]):
            msg = (
                "node and edge must share latent width d, "
                f"got {tuple(self.node.shape)} and {tuple(self.edge.shape)}"
            )
            raise ValueError(msg)
        if self.face is not None:
            _require_feature_matrix("face", self.face)


@dataclass(frozen=True)
class BoundaryNilpotencyReport:
    """Boundary product :math:`B_1 B_2` and a documented tolerance flag.

    Attributes
    ----------
    product : Tensor
        :math:`B_1 B_2` with shape ``(N, F)``. Dimensionless.
    max_abs : float
        :math:`\\|B_1 B_2\\|_{\\infty}` over entries.
    atol : float
        Absolute tolerance used for ``nilpotent``. Default matches the
        float32 teaching triangle (``1e-6``).
    nilpotent : bool
        ``True`` when ``max_abs <= atol``.
    """

    product: Tensor
    max_abs: float
    atol: float
    nilpotent: bool

    def __post_init__(self) -> None:
        """Validate the product table and tolerance.

        Raises
        ------
        ValueError
            If the product is empty or the scalars are invalid.
        """
        if self.product.ndim != 2:
            msg = f"product must have shape (N, F), got {tuple(self.product.shape)}"
            raise ValueError(msg)
        if not math.isfinite(self.max_abs) or self.max_abs < 0.0:
            raise ValueError(
                f"max_abs must be a finite non-negative float, got {self.max_abs!r}"
            )
        if not math.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError(
                f"atol must be a finite non-negative float, got {self.atol!r}"
            )


def boundary_nilpotency(
    b1: Tensor,
    b2: Tensor,
    *,
    atol: float = DEFAULT_NILPOTENCY_ATOL,
) -> BoundaryNilpotencyReport:
    """Score :math:`B_1 B_2 \\approx 0` on caller-supplied incidences.

    :math:`B_1` has shape ``(N, E)`` and :math:`B_2` has shape
    ``(E, F)``. The product is the chain-complex form of
    :math:`\\partial\\partial=0` (``Lim2020Hodge``). This helper does
    not build :math:`B_2`; use
    :func:`~koopman_graph.nn.cell_complex.boundary_incidence_b2` in
    tests or callers.

    Parameters
    ----------
    b1 : Tensor
        Signed node–edge incidence ``(N, E)``.
    b2 : Tensor
        Signed edge–face incidence ``(E, F)``.
    atol : float, optional
        Absolute entrywise tolerance. Default is ``1e-6``.

    Returns
    -------
    BoundaryNilpotencyReport
        Product, max absolute entry, and flag.

    Raises
    ------
    TypeError
        If an argument is not a tensor.
    ValueError
        If shapes, dtypes, or ``atol`` are invalid.
    """
    if not isinstance(b1, Tensor) or not isinstance(b2, Tensor):
        raise TypeError("b1 and b2 must be tensors")
    if b1.ndim != 2 or b2.ndim != 2:
        msg = f"b1 and b2 must be 2-D, got {tuple(b1.shape)} and {tuple(b2.shape)}"
        raise ValueError(msg)
    if int(b1.shape[1]) != int(b2.shape[0]):
        msg = (
            "b1 columns must match b2 rows (shared E), "
            f"got {tuple(b1.shape)} and {tuple(b2.shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(b1).all().item()) or not bool(
        torch.isfinite(b2).all().item()
    ):
        raise ValueError("b1 and b2 must be finite")
    if not math.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError(f"atol must be a finite non-negative float, got {atol!r}")
    product = b1 @ b2
    max_abs = float(product.abs().max().item()) if int(product.numel()) else 0.0
    return BoundaryNilpotencyReport(
        product=product,
        max_abs=max_abs,
        atol=float(atol),
        nilpotent=max_abs <= float(atol),
    )


class CochainKoopmanOperator(nn.Module):
    """Advance 0- and 1-cochain latents on a static signed incidence.

    Row-vector convention matches discrete operators
    (:math:`z\\mapsto z K^{\\top}`). Default maps are
    :math:`\\tfrac12 I`. Optional cross-degree terms use the bound
    :math:`B_1`:

    .. math::

        z_0^{+} = z_0 K_0^{\\top} + (B_1 z_1) C_{0\\leftarrow 1}^{\\top}

        z_1^{+} = z_1 K_1^{\\top} + (B_1^{\\top} z_0) C_{1\\leftarrow 0}^{\\top}

    Parameters
    ----------
    latent_dim : int
        Shared feature width :math:`d` for node and edge latents.
    edge_index : Tensor
        Oriented 1-cells ``(2, E)``. One column per undirected edge.
    num_nodes : int
        Number of 0-cells :math:`N`.
    use_cross_degree : bool, optional
        If ``True``, apply the incidence cross terms. Default ``False``.

    Notes
    -----
    Distinct from ``koopman="hodge"``. Not a factory kind. No checkpoint
    key. Combinatorial incidence only — not a sheaf Laplacian
    (``Lim2020Hodge``, ``TopoX2024``).

    References
    ----------
    Lim, L.-H. (2020). Hodge Laplacians on graphs. *SIAM Review*
    62:685–715. doi:10.1137/18M1223101 (``Lim2020Hodge``).
    Hajij, M. et al. (2024). TopoX: a suite of Python packages for
    machine learning on topological domains. arXiv:2402.02441
    (``TopoX2024``).
    """

    def __init__(
        self,
        latent_dim: int,
        edge_index: Tensor,
        *,
        num_nodes: int,
        use_cross_degree: bool = False,
    ) -> None:
        """Allocate per-degree maps and bind a static :math:`B_1`.

        Parameters
        ----------
        latent_dim : int
            Shared feature width.
        edge_index : Tensor
            Oriented edges ``(2, E)``.
        num_nodes : int
            Node count.
        use_cross_degree : bool, optional
            Enable incidence cross terms. Default ``False``.

        Raises
        ------
        ValueError
            If dimensions or ``edge_index`` are invalid.
        """
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise ValueError(f"num_nodes must be a positive int, got {num_nodes!r}")
        if int(num_nodes) < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if not isinstance(use_cross_degree, bool):
            msg = f"use_cross_degree must be a bool, got {use_cross_degree!r}"
            raise ValueError(msg)
        width = int(latent_dim)
        nodes = int(num_nodes)
        incidence = boundary_incidence_b1(edge_index, num_nodes=nodes)
        self.latent_dim = width
        self.num_nodes = nodes
        self.num_edges = int(incidence.shape[1])
        self.use_cross_degree = bool(use_cross_degree)
        self.register_buffer("incidence", incidence)
        self.k_node = nn.Parameter(0.5 * torch.eye(width))
        self.k_edge = nn.Parameter(0.5 * torch.eye(width))
        self.c_node_from_edge = nn.Parameter(torch.zeros(width, width))
        self.c_edge_from_node = nn.Parameter(torch.zeros(width, width))

    def _incidence_for(self, state: CochainState) -> Tensor:
        """Return :math:`B_1` on the state's device and dtype.

        Parameters
        ----------
        state : CochainState
            Current cochains.

        Returns
        -------
        Tensor
            Incidence ``(N, E)``.

        Raises
        ------
        ValueError
            If cell counts do not match the bound incidence.
        """
        if int(state.node.shape[0]) != self.num_nodes:
            msg = (
                "state.node rows must equal num_nodes "
                f"{self.num_nodes}, got {tuple(state.node.shape)}"
            )
            raise ValueError(msg)
        if int(state.edge.shape[0]) != self.num_edges:
            msg = (
                "state.edge rows must equal num_edges "
                f"{self.num_edges}, got {tuple(state.edge.shape)}"
            )
            raise ValueError(msg)
        if int(state.node.shape[1]) != self.latent_dim:
            msg = (
                "state latent width must equal latent_dim "
                f"{self.latent_dim}, got {tuple(state.node.shape)}"
            )
            raise ValueError(msg)
        return self.incidence.to(dtype=state.node.dtype, device=state.node.device)

    def commuting_residual(self, state: CochainState) -> Tensor:
        """Return :math:`d_0(z_0 K_0^{\\top})-(d_0 z_0)K_1^{\\top}`.

        :math:`d_0=B_1^{\\top}` acts on cells; :math:`K_k` acts on
        features. The residual vanishes when :math:`K_0=K_1`.

        Parameters
        ----------
        state : CochainState
            Current 0-cochain used to form :math:`d_0 z_0`.

        Returns
        -------
        Tensor
            Residual with shape ``(E, d)`` in latent units.
        """
        incidence = self._incidence_for(state)
        coboundary = incidence.transpose(0, 1) @ state.node
        left = incidence.transpose(0, 1) @ (state.node @ self.k_node.transpose(-1, -2))
        right = coboundary @ self.k_edge.transpose(-1, -2)
        return left - right

    def commuting_loss(self, state: CochainState) -> Tensor:
        """Mean squared intertwining residual.

        Parameters
        ----------
        state : CochainState
            Current 0-cochain.

        Returns
        -------
        Tensor
            Scalar mean square of :meth:`commuting_residual`.
        """
        residual = self.commuting_residual(state)
        return residual.square().mean()

    def advance(self, state: CochainState) -> CochainState:
        """Apply one per-degree step on the bound incidence.

        Parameters
        ----------
        state : CochainState
            Current node / edge latents. ``face`` is copied through.

        Returns
        -------
        CochainState
            Next 0- and 1-cochains. Face latents are unchanged.
        """
        incidence = self._incidence_for(state)
        node_next = state.node @ self.k_node.transpose(-1, -2)
        edge_next = state.edge @ self.k_edge.transpose(-1, -2)
        if self.use_cross_degree:
            node_next = node_next + (
                incidence @ state.edge
            ) @ self.c_node_from_edge.transpose(-1, -2)
            edge_next = edge_next + (
                incidence.transpose(0, 1) @ state.node
            ) @ self.c_edge_from_node.transpose(-1, -2)
        return CochainState(node=node_next, edge=edge_next, face=state.face)

    def forward(self, state: CochainState) -> CochainState:
        """Alias of :meth:`advance`.

        Parameters
        ----------
        state : CochainState
            Current cochains.

        Returns
        -------
        CochainState
            Next cochains.
        """
        return self.advance(state)
