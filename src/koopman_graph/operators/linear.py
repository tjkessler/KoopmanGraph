"""Matrix-free linear-operator protocol for networked Koopman maps.

:class:`LinearOperatorProtocol` is the common surface (``matvec``,
``rmatvec``, ``solve``, ``expm_action``, leading eigpairs, residual
norms). :class:`PolynomialGraphLinearOperator` wraps a monomial
degree-:math:`P\\ge 2` graph polynomial.
:class:`MatrixFreeGraphLinearOperator` wraps the existing one-tap
:func:`~koopman_graph.operators.matrix_free.apply_k_eff_graph` /
:func:`~koopman_graph.operators.matrix_free.invert_k_eff_graph` /
:func:`~koopman_graph.operators.matrix_free.spectrum_k_eff_graph`
path.

Dense assembly of :math:`K_{\\mathrm{eff}}` is refused when
:math:`N\\cdot d` exceeds :data:`MAX_DENSE_LINEAR_OPERATOR_SIZE`
(same teaching ceiling as assembled eig-reg / joint Schur). Trainer
DDP / ``[distributed]`` extras do **not** shrink that representation.
Exact Kronecker spectrum remains a special case — leading eigpairs
here are Arnoldi Ritz values, not ``eig(B(λ))``.
:math:`10^{5}`-node scaling is not a release gate.

This module must not import :mod:`koopman_graph.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from koopman_graph.graph_utils import (
    dense_symmetric_normalized_adjacency,
    symmetric_normalized_adjacency_matvec,
)
from koopman_graph.operators.matrix_free import (
    DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    DEFAULT_MATRIX_FREE_INVERSE_TOL,
    DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    _apply_self_preconditioner,
    _arnoldi_eigs,
    _invert_square_factor,
    _richardson_invert,
    apply_k_eff_graph,
    flatten_node_latents,
    invert_k_eff_graph,
    spectrum_k_eff_graph,
    unflatten_node_latents,
)
from koopman_graph.operators.polynomial_graph import (
    apply_monomial_powers,
    dense_polynomial_kronecker,
)

# Match assembled eig-reg / joint-Schur teaching ceiling (``N·d``).
MAX_DENSE_LINEAR_OPERATOR_SIZE = 4096
DEFAULT_EXPM_TAYLOR_TERMS = 16

__all__ = [
    "DEFAULT_EXPM_TAYLOR_TERMS",
    "EigResult",
    "LinearOperatorProtocol",
    "MAX_DENSE_LINEAR_OPERATOR_SIZE",
    "MatrixFreeGraphLinearOperator",
    "MemoryEstimate",
    "PolynomialGraphLinearOperator",
]


@dataclass(frozen=True)
class MemoryEstimate:
    """Declared cost of a linear operator without assembling it.

    Attributes
    ----------
    dim : int
        Flat width ``N·d``.
    dense_entries : int
        Entries in a dense ``(N·d, N·d)`` layout.
    dense_allowed : bool
        ``True`` when dense assembly is under the teaching ceiling.
    notes : str
        Honesty string. Trainer DDP does not change this estimate.
    """

    dim: int
    dense_entries: int
    dense_allowed: bool
    notes: str


@dataclass(frozen=True)
class EigResult:
    """Leading-modulus Ritz pairs from a matrix-free Arnoldi pass.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex values ``(k,)``, largest-modulus first.
    residual_norms : Tensor
        Per-mode ``||A v - λ v|| / ||v||`` with shape ``(k,)``.
    converged : bool
        ``True`` when every returned mode met the requested ``tol``.
    """

    eigenvalues: Tensor
    residual_norms: Tensor
    converged: bool


@runtime_checkable
class LinearOperatorProtocol(Protocol):
    """Common matrix-free algebra surface for networked maps.

    Implementers apply a linear map on flat ``(N·d,)`` vectors without
    assembling :math:`K_{\\mathrm{eff}}`. This is operator math — **not**
    trainer DDP.

    Notes
    -----
    ``@runtime_checkable`` verifies method presence only. Exact
    Kronecker spectrum remains a special case.
    """

    def matvec(self, x: Tensor) -> Tensor:
        """Apply ``A x``.

        Parameters
        ----------
        x : Tensor
            Flat vector ``(N·d,)``.

        Returns
        -------
        Tensor
            Flat result ``(N·d,)``.
        """
        ...

    def rmatvec(self, x: Tensor) -> Tensor:
        """Apply ``A^T x``.

        Parameters
        ----------
        x : Tensor
            Flat vector ``(N·d,)``.

        Returns
        -------
        Tensor
            Flat result ``(N·d,)``.
        """
        ...

    def solve(self, b: Tensor, *, tol: float, max_iters: int) -> Tensor:
        """Solve ``A x = b`` without a dense factorization.

        Parameters
        ----------
        b : Tensor
            Flat right-hand side ``(N·d,)``.
        tol : float
            Relative residual tolerance.
        max_iters : int
            Iteration budget.

        Returns
        -------
        Tensor
            Flat solution ``(N·d,)``.
        """
        ...

    def expm_action(self, t: float, b: Tensor) -> Tensor:
        """Apply ``exp(t A) b`` from matvecs (Taylor teaching path).

        Parameters
        ----------
        t : float
            Time scale.
        b : Tensor
            Flat right-hand side ``(N·d,)``.

        Returns
        -------
        Tensor
            Flat result ``(N·d,)``.
        """
        ...

    def leading_eigpairs(self, k: int, *, tol: float) -> EigResult:
        """Return ``k`` largest-modulus Ritz pairs and residuals.

        Parameters
        ----------
        k : int
            Number of modes.
        tol : float
            Ritz residual tolerance.

        Returns
        -------
        EigResult
            Accepted pairs. Not a Kronecker exact spectrum.
        """
        ...

    def residual_norm(self, x: Tensor, b: Tensor) -> Tensor:
        """Return ``||A x - b||_2``.

        Parameters
        ----------
        x, b : Tensor
            Flat vectors ``(N·d,)``.

        Returns
        -------
        Tensor
            Scalar residual norm.
        """
        ...

    @property
    def memory_estimate(self) -> MemoryEstimate:
        """Declared dense-vs-matrix-free cost.

        Returns
        -------
        MemoryEstimate
            ``N·d`` width and whether dense assembly is allowed.
        """
        ...


def _require_flat(name: str, value: Tensor, *, dim: int) -> Tensor:
    """Refuse empty, non-real, or wrong-width flat vectors.

    Parameters
    ----------
    name : str
        Field name for the error message.
    value : Tensor
        Candidate ``(dim,)`` vector.
    dim : int
        Expected length.

    Returns
    -------
    Tensor
        The validated vector.

    Raises
    ------
    TypeError
        If ``value`` is not a tensor.
    ValueError
        If rank, dtype, finiteness, or length is invalid.
    """
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor, got {type(value).__name__}")
    if value.is_complex():
        raise ValueError(f"{name} must be real")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 1 or int(value.shape[0]) != int(dim):
        raise ValueError(f"{name} must have shape ({dim},), got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _taylor_expm_action(
    matvec: object,
    t: float,
    b: Tensor,
    *,
    n_terms: int = DEFAULT_EXPM_TAYLOR_TERMS,
) -> Tensor:
    """Evaluate ``exp(t A) b`` from repeated ``matvec`` (Taylor).

    Parameters
    ----------
    matvec : callable
        ``A @ v`` on flat vectors.
    t : float
        Time scale. Must be finite.
    b : Tensor
        Flat right-hand side.
    n_terms : int, optional
        Number of Taylor terms after the identity. Default
        :data:`DEFAULT_EXPM_TAYLOR_TERMS`.

    Returns
    -------
    Tensor
        Flat ``exp(t A) b`` in the dtype of ``b``.

    Raises
    ------
    ValueError
        If ``t`` or ``n_terms`` is invalid.
    """
    if not isinstance(t, int | float) or isinstance(t, bool):
        raise ValueError(f"t must be a finite float, got {t!r}")
    if not torch.isfinite(torch.tensor(float(t))):
        raise ValueError(f"t must be finite, got {t!r}")
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError(f"n_terms must be a positive int, got {n_terms!r}")
    if float(t) == 0.0:
        return b.clone()
    working = b.detach().to(dtype=torch.float64)
    term = working.clone()
    out = working.clone()
    scale = float(t)
    apply = matvec
    for order in range(1, n_terms + 1):
        term = apply(term) * (scale / float(order))
        out = out + term
    return out.to(dtype=b.dtype)


class _SymmetricGraphLinearMap:
    """Shared symmetric-adjacency bookkeeping for graph wrappers.

    Parameters
    ----------
    hop_matrices : tuple of Tensor
        ``K_0, …, K_P`` each ``(d, d)``.
    edge_index : Tensor
        COO edges ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    edge_weight : Tensor or None
        Optional scalar edge weights.
    """

    def __init__(
        self,
        hop_matrices: tuple[Tensor, ...],
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None,
    ) -> None:
        """Validate hops, topology, and the symmetric teaching path.

        Parameters
        ----------
        hop_matrices : tuple of Tensor
            Square latent factors.
        edge_index : Tensor
            COO edges.
        num_nodes : int
            Node count.
        edge_weight : Tensor or None
            Optional weights.

        Raises
        ------
        ValueError
            If hops, ``num_nodes``, or ``edge_index`` are invalid.
        """
        if len(hop_matrices) < 1:
            raise ValueError("hop_matrices must be non-empty")
        latent_dim = int(hop_matrices[0].shape[0])
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        expected = (latent_dim, latent_dim)
        for index, matrix in enumerate(hop_matrices):
            if not isinstance(matrix, Tensor):
                raise TypeError(
                    f"hop_matrices[{index}] must be a Tensor, "
                    f"got {type(matrix).__name__}"
                )
            if matrix.shape != expected:
                msg = (
                    f"hop_matrices[{index}] must have shape {expected}, "
                    f"got {tuple(matrix.shape)}"
                )
                raise ValueError(msg)
            if matrix.is_complex() or not matrix.is_floating_point():
                raise ValueError("hop_matrices must be real floating-point")
            if not bool(torch.isfinite(matrix).all().item()):
                raise ValueError("hop_matrices must be finite")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise ValueError(f"num_nodes must be a positive int, got {num_nodes!r}")
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if not isinstance(edge_index, Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if int(edge_index.shape[0]) != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
            )
        self.hop_matrices = hop_matrices
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = int(num_nodes)
        self.latent_dim = latent_dim
        self.dim = self.num_nodes * self.latent_dim

    def _adj_matvec(self, states: Tensor) -> Tensor:
        """Apply symmetric-normalized ``Â`` to node latents.

        Parameters
        ----------
        states : Tensor
            Node table ``(N, d)``.

        Returns
        -------
        Tensor
            ``Â Z``.
        """
        return symmetric_normalized_adjacency_matvec(
            self.edge_index,
            states,
            edge_weight=self.edge_weight,
            num_nodes=self.num_nodes,
        )

    def _apply_hops(self, flat: Tensor, hops: tuple[Tensor, ...]) -> Tensor:
        """Apply a Kronecker polynomial via sparse adjacency matvecs.

        Parameters
        ----------
        flat : Tensor
            Flat state ``(N·d,)``.
        hops : tuple of Tensor
            Factors ``K_0, …, K_P``.

        Returns
        -------
        Tensor
            Flat result.
        """
        states = unflatten_node_latents(
            flat,
            num_nodes=self.num_nodes,
            latent_dim=self.latent_dim,
        )
        out = states @ hops[0].transpose(-2, -1)
        if len(hops) > 1:
            out = out + apply_monomial_powers(
                states,
                hops[1:],
                self._adj_matvec,
                min_power=1,
            )
        return flatten_node_latents(out)

    def matvec(self, x: Tensor) -> Tensor:
        """Apply the stored polynomial to a flat vector.

        Parameters
        ----------
        x : Tensor
            Flat state ``(N·d,)``.

        Returns
        -------
        Tensor
            ``K_eff x``.
        """
        return self._apply_hops(_require_flat("x", x, dim=self.dim), self.hop_matrices)

    def rmatvec(self, x: Tensor) -> Tensor:
        """Apply the adjoint (symmetric ``Â`` ⇒ transposed factors).

        Parameters
        ----------
        x : Tensor
            Flat state ``(N·d,)``.

        Returns
        -------
        Tensor
            ``K_eff^T x``.
        """
        transposed = tuple(matrix.transpose(-2, -1) for matrix in self.hop_matrices)
        return self._apply_hops(_require_flat("x", x, dim=self.dim), transposed)

    def residual_norm(self, x: Tensor, b: Tensor) -> Tensor:
        """Return ``||K_eff x - b||_2``.

        Parameters
        ----------
        x, b : Tensor
            Flat vectors ``(N·d,)``.

        Returns
        -------
        Tensor
            Scalar residual norm.
        """
        left = self.matvec(_require_flat("x", x, dim=self.dim))
        rhs = _require_flat("b", b, dim=self.dim)
        return torch.linalg.vector_norm(left - rhs)

    def expm_action(self, t: float, b: Tensor) -> Tensor:
        """Taylor action ``exp(t K_eff) b`` (no dense exponential).

        Parameters
        ----------
        t : float
            Time scale.
        b : Tensor
            Flat right-hand side.

        Returns
        -------
        Tensor
            Flat result.
        """
        rhs = _require_flat("b", b, dim=self.dim)
        return _taylor_expm_action(self.matvec, t, rhs)

    def _solve_richardson(
        self,
        b: Tensor,
        *,
        tol: float,
        max_iters: int,
    ) -> Tensor:
        """Self-preconditioned Richardson solve.

        Parameters
        ----------
        b : Tensor
            Flat right-hand side.
        tol : float
            Relative residual tolerance.
        max_iters : int
            Iteration budget.

        Returns
        -------
        Tensor
            Flat solution.
        """
        rhs = _require_flat("b", b, dim=self.dim)
        k_self_inv = _invert_square_factor(self.hop_matrices[0])

        def _apply_m_inv(flat: Tensor) -> Tensor:
            """Apply ``I ⊗ K_0^{-1}``.

            Parameters
            ----------
            flat : Tensor
                Flat residual ``(N·d,)``.

            Returns
            -------
            Tensor
                Preconditioned residual.
            """
            return _apply_self_preconditioner(
                flat,
                k_self_inv,
                num_nodes=self.num_nodes,
                latent_dim=self.latent_dim,
            )

        result = _richardson_invert(
            self.matvec,
            _apply_m_inv,
            rhs,
            max_iters=max_iters,
            tol=tol,
            x0=None,
        )
        return result.solution

    def _leading_from_matvec(self, k: int, *, tol: float) -> EigResult:
        """Arnoldi on the stored matvec.

        Parameters
        ----------
        k : int
            Number of modes.
        tol : float
            Ritz residual tolerance.

        Returns
        -------
        EigResult
            Accepted pairs.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if k > self.dim:
            raise ValueError(f"k must be <= dim={self.dim}, got {k}")
        ncv = min(self.dim, max(2 * k + 2, 20))
        raw = _arnoldi_eigs(
            self.matvec,
            self.dim,
            num_modes=k,
            tol=tol,
            ncv=ncv,
            seed=0,
            device=self.hop_matrices[0].device,
            dtype=self.hop_matrices[0].dtype,
        )
        return EigResult(
            eigenvalues=raw.eigenvalues,
            residual_norms=raw.residual_norms,
            converged=raw.converged,
        )

    def dense_matrix(self) -> Tensor:
        """Assemble ``K_eff`` or refuse above the teaching ceiling.

        Returns
        -------
        Tensor
            Dense ``(N·d, N·d)`` map.

        Raises
        ------
        ValueError
            If ``N·d`` exceeds :data:`MAX_DENSE_LINEAR_OPERATOR_SIZE`.
        """
        if self.dim > MAX_DENSE_LINEAR_OPERATOR_SIZE:
            msg = (
                "dense assembly refused: N·d = "
                f"{self.dim} exceeds MAX_DENSE_LINEAR_OPERATOR_SIZE="
                f"{MAX_DENSE_LINEAR_OPERATOR_SIZE}. Use matvec / solve / "
                "leading_eigpairs. Trainer DDP does not raise this ceiling."
            )
            raise ValueError(msg)
        adjacency = dense_symmetric_normalized_adjacency(
            self.edge_index,
            self.num_nodes,
            edge_weight=self.edge_weight,
            dtype=self.hop_matrices[0].dtype,
        )
        return dense_polynomial_kronecker(adjacency, self.hop_matrices)

    @property
    def memory_estimate(self) -> MemoryEstimate:
        """Report ``N·d`` and whether dense assembly is allowed.

        Returns
        -------
        MemoryEstimate
            Declared cost. Trainer DDP does not change it.
        """
        allowed = self.dim <= MAX_DENSE_LINEAR_OPERATOR_SIZE
        return MemoryEstimate(
            dim=self.dim,
            dense_entries=self.dim * self.dim,
            dense_allowed=allowed,
            notes=(
                "matrix-free operator math; trainer DDP / [distributed] "
                "does not shrink the dense N·d representation"
            ),
        )


class PolynomialGraphLinearOperator(_SymmetricGraphLinearMap):
    """Monomial graph polynomial :math:`P\\ge 2` as a linear operator.

    ``hop_matrices`` is ``(K_0, K_1, …, K_P)`` with ``P = len-1 >= 2``.
    Not a factory kind. Exact Kronecker spectrum is out of scope here.

    Parameters
    ----------
    hop_matrices : sequence of Tensor
        Square factors, length at least 3.
    edge_index : Tensor
        COO edges.
    num_nodes : int
        Node count.
    edge_weight : Tensor or None, optional
        Optional edge weights.
    """

    def __init__(
        self,
        hop_matrices: tuple[Tensor, ...] | list[Tensor],
        *,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
    ) -> None:
        """Bind a degree-``P>=2`` monomial and a symmetric graph.

        Parameters
        ----------
        hop_matrices : sequence of Tensor
            ``K_0, …, K_P``.
        edge_index : Tensor
            COO edges.
        num_nodes : int
            Node count.
        edge_weight : Tensor or None, optional
            Optional weights.

        Raises
        ------
        ValueError
            If fewer than three factors are supplied.
        """
        hops = tuple(hop_matrices)
        if len(hops) < 3:
            raise ValueError(
                "PolynomialGraphLinearOperator requires P>=2 "
                f"(at least three hop matrices), got {len(hops)}"
            )
        super().__init__(hops, edge_index, num_nodes, edge_weight)

    def solve(
        self,
        b: Tensor,
        *,
        tol: float = DEFAULT_MATRIX_FREE_INVERSE_TOL,
        max_iters: int = DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    ) -> Tensor:
        """Richardson solve on the polynomial map.

        Parameters
        ----------
        b : Tensor
            Flat right-hand side.
        tol : float, optional
            Relative residual tolerance.
        max_iters : int, optional
            Iteration budget.

        Returns
        -------
        Tensor
            Flat solution.
        """
        return self._solve_richardson(b, tol=tol, max_iters=max_iters)

    def leading_eigpairs(
        self,
        k: int,
        *,
        tol: float = DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    ) -> EigResult:
        """Arnoldi Ritz pairs (not Kronecker ``eig(B(λ))``).

        Parameters
        ----------
        k : int
            Number of modes.
        tol : float, optional
            Ritz residual tolerance.

        Returns
        -------
        EigResult
            Accepted pairs.
        """
        return self._leading_from_matvec(k, tol=tol)


class MatrixFreeGraphLinearOperator(_SymmetricGraphLinearMap):
    """One-tap ``I⊗K_self + Â⊗K_nbr`` wrapper around ``matrix_free``.

    Parameters
    ----------
    k_self, k_nbr : Tensor
        Square ``(d, d)`` factors.
    edge_index : Tensor
        COO edges.
    num_nodes : int
        Node count.
    edge_weight : Tensor or None, optional
        Optional edge weights.
    """

    def __init__(
        self,
        k_self: Tensor,
        k_nbr: Tensor,
        *,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
    ) -> None:
        """Bind the existing one-tap matrix-free path.

        Parameters
        ----------
        k_self, k_nbr : Tensor
            Self / neighbor factors.
        edge_index : Tensor
            COO edges.
        num_nodes : int
            Node count.
        edge_weight : Tensor or None, optional
            Optional weights.
        """
        super().__init__((k_self, k_nbr), edge_index, num_nodes, edge_weight)

    def matvec(self, x: Tensor) -> Tensor:
        """Delegate to :func:`apply_k_eff_graph`.

        Parameters
        ----------
        x : Tensor
            Flat state.

        Returns
        -------
        Tensor
            ``K_eff x``.
        """
        flat = _require_flat("x", x, dim=self.dim)
        return apply_k_eff_graph(
            flat,
            k_self=self.hop_matrices[0],
            k_nbr=self.hop_matrices[1],
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            adjacency="symmetric",
            edge_weight=self.edge_weight,
        )

    def solve(
        self,
        b: Tensor,
        *,
        tol: float = DEFAULT_MATRIX_FREE_INVERSE_TOL,
        max_iters: int = DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    ) -> Tensor:
        """Delegate to :func:`invert_k_eff_graph`.

        Parameters
        ----------
        b : Tensor
            Flat right-hand side.
        tol : float, optional
            Relative residual tolerance.
        max_iters : int, optional
            Iteration budget.

        Returns
        -------
        Tensor
            Flat solution.
        """
        rhs = _require_flat("b", b, dim=self.dim)
        result = invert_k_eff_graph(
            rhs,
            k_self=self.hop_matrices[0],
            k_nbr=self.hop_matrices[1],
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            adjacency="symmetric",
            edge_weight=self.edge_weight,
            max_iters=max_iters,
            tol=tol,
        )
        return result.solution

    def leading_eigpairs(
        self,
        k: int,
        *,
        tol: float = DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    ) -> EigResult:
        """Delegate to :func:`spectrum_k_eff_graph` (Arnoldi, not Kronecker).

        Parameters
        ----------
        k : int
            Number of modes.
        tol : float, optional
            Ritz residual tolerance.

        Returns
        -------
        EigResult
            Accepted pairs.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        raw = spectrum_k_eff_graph(
            k_self=self.hop_matrices[0],
            k_nbr=self.hop_matrices[1],
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_modes=k,
            adjacency="symmetric",
            edge_weight=self.edge_weight,
            tol=tol,
        )
        return EigResult(
            eigenvalues=raw.eigenvalues,
            residual_norms=raw.residual_norms,
            converged=raw.converged,
        )
