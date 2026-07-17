"""Shared Koopman operator contract, types, and structural helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

InitMode = Literal["identity", "identity_noise", "xavier"]
Parameterization = Literal["dense", "odo", "schur", "dissipative", "lyapunov"]
#: Canonical discrete vs continuous dynamics vocabulary. Re-exported from
#: :mod:`koopman_graph.protocols` as :data:`~koopman_graph.protocols.DynamicsMode`.
DynamicsMode = Literal["discrete", "continuous"]
#: Built-in discrete factory kind for :class:`~koopman_graph.model.GraphKoopmanModel`.
KoopmanKind = Literal["pernode", "graph"]

STABILITY_EPS_MARGIN = 1e-4
DISSIPATIVE_MIN_EIGENVALUE = 1e-3


@dataclass(frozen=True)
class StabilityCertificate:
    """Lyapunov or spectral stability certificate for constrained operators.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles).

    Attributes
    ----------
    margin : Tensor
        Positive stability margin from the active certificate. Discrete
        structural modes (``"schur"``, ``"dissipative"``, ``"lyapunov"``) all
        report the unit-disk gap ``1 - bound_metric`` (equivalently
        ``1 - max |d_i|`` for Lyapunov). Continuous structural modes use the
        gap of ``bound_metric`` below the Hurwitz boundary (``-bound_metric``).
    lyapunov_matrix : Tensor or None
        Positive-definite Lyapunov matrix when available (``"lyapunov"`` mode);
        ``None`` for spectral-margin-only certificates.
    """

    margin: Tensor
    lyapunov_matrix: Tensor | None = None


@runtime_checkable
class KoopmanOperatorContract(Protocol):
    """Shared contract for discrete and continuous Koopman operators.

    Both :class:`KoopmanOperator`,
    :class:`~koopman_graph.operators.ContinuousKoopmanOperator`, and
    :class:`~koopman_graph.operators.GraphKoopmanOperator` implement this
    surface. Domain-specific names (``K`` / ``L``, ``spectral_radius`` /
    ``max_real_part``, ``forward`` / ``inverse_step``) remain as thin aliases
    for notebooks and existing call sites. ``spectral_radius`` /
    ``max_real_part`` always report the true spectrum via ``eigvals``;
    :meth:`bound_metric` is the cheap soft/structural monitoring bound.
    Networked operators additionally accept optional ``edge_index`` /
    ``edge_weight`` on :meth:`advance` / :meth:`inverse_advance`; per-node
    operators ignore those kwargs.

    Attributes
    ----------
    latent_dim : int
        Latent state dimension.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Soft or structural parameterization mode.
    """

    latent_dim: int
    control_dim: int
    parameterization: Parameterization

    @property
    def matrix(self) -> Tensor:
        """Assembled operator matrix (``K`` or generator ``L``).

        Returns
        -------
        Tensor
            Square assembled matrix with shape ``(latent_dim, latent_dim)``.
        """
        ...

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states one step (discrete) or over ``Δt`` (continuous).

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Continuous integration interval. Ignored for discrete operators;
            required for continuous operators.
        control : Tensor or None, optional
            Exogenous control input when ``control_dim > 0``.
        edge_index : Tensor or None, optional
            Graph topology for networked operators. Ignored by per-node
            operators; required by
            :class:`~koopman_graph.operators.GraphKoopmanOperator`.
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.
        """
        ...

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Recover the previous latent state (discrete or continuous).

        Parameters
        ----------
        z : Tensor
            Latent states after a forward step.
        delta_t : float, Tensor, or None, optional
            Continuous integration interval. Ignored for discrete operators;
            required for continuous operators.
        control : Tensor or None, optional
            Control that drove the forward transition.
        inverse_matrix : Tensor or None, optional
            Optional precomputed discrete inverse; ignored for continuous
            operators.
        edge_index : Tensor or None, optional
            Graph topology for networked operators. Ignored by per-node
            operators.
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Recovered previous latent states.
        """
        ...

    def bound_metric(self) -> Tensor:
        """Return the cheap soft/structural monitoring bound.

        This is the non-eigendecomposition bound used for certificates and
        training monitors. For ``"odo"``, it is the **diagonal-factor** bound,
        not the true spectrum of assembled ``K`` / ``L``. Prefer
        :meth:`~koopman_graph.operators.KoopmanOperator.spectral_radius` or
        :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.max_real_part`
        for the true spectrum via ``eigvals``.

        Returns
        -------
        Tensor
            Scalar bound metric for the active parameterization.
        """
        ...


def resolve_factory_stability_bound(
    operator: object,
    *,
    dynamics_mode: DynamicsMode,
) -> float:
    """Map built-in operator bound fields to ``koopman_max_spectral_radius``.

    Factory, checkpoint, and model construction share one neutral knob name.
    Domain-specific operator attributes stay as-is; call sites should use this
    helper (or the mapping below) instead of inventing a third field name.

    * Factory / checkpoint key: ``koopman_max_spectral_radius`` (both modes)
    * Discrete operator attribute: ``max_spectral_radius``
    * Continuous operator attribute: ``max_real_eigenvalue``

    Parameters
    ----------
    operator : object
        Built-in discrete or continuous Koopman operator.
    dynamics_mode : {"discrete", "continuous"}
        Model dynamics mode selecting which attribute to read.

    Returns
    -------
    float
        Stability bound suitable for ``koopman_max_spectral_radius``.

    Raises
    ------
    TypeError
        If ``dynamics_mode`` is invalid or the expected attribute is missing.
    """
    if dynamics_mode == "continuous":
        bound = getattr(operator, "max_real_eigenvalue", None)
        attr_name = "max_real_eigenvalue"
    elif dynamics_mode == "discrete":
        bound = getattr(operator, "max_spectral_radius", None)
        attr_name = "max_spectral_radius"
    else:
        msg = f"dynamics_mode must be 'discrete' or 'continuous', got {dynamics_mode!r}"
        raise TypeError(msg)
    if bound is None:
        msg = (
            f"{type(operator).__name__} missing {attr_name} for "
            f"dynamics_mode={dynamics_mode!r}"
        )
        raise TypeError(msg)
    return float(bound)


def strict_spectral_bound(max_spectral_radius: float) -> float:
    """Return the strict interior bound ``max_spectral_radius - epsilon``.

    Parameters
    ----------
    max_spectral_radius : float
        Configured spectral-radius upper limit.

    Returns
    -------
    float
        Strict interior bound used by structurally stable modes.
    """
    return max(max_spectral_radius - STABILITY_EPS_MARGIN, STABILITY_EPS_MARGIN)


def cayley_orthogonal(skew_params: Tensor) -> Tensor:
    """Build an orthogonal matrix via the Cayley transform.

    Parameters
    ----------
    skew_params : Tensor
        Square parameter matrix; only its skew-symmetric part is used.

    Returns
    -------
    Tensor
        Orthogonal matrix with the same shape as ``skew_params``.
    """
    skew = 0.5 * (skew_params - skew_params.T)
    identity = torch.eye(skew.shape[0], device=skew.device, dtype=skew.dtype)
    return torch.linalg.solve(identity - skew, identity + skew)


def _bounded_diagonal(raw: Tensor, max_radius: float) -> Tensor:
    """Map unconstrained parameters to a bounded diagonal matrix.

    Parameters
    ----------
    raw : Tensor
        Unconstrained diagonal parameters with shape ``(latent_dim,)``.
    max_radius : float
        Maximum absolute value on the diagonal.

    Returns
    -------
    Tensor
        Diagonal matrix with spectral values in ``[-max_radius, max_radius]``.
    """
    values = torch.tanh(raw) * max_radius
    return torch.diag(values)


def _strict_diagonal_values(raw: Tensor, max_spectral_radius: float) -> Tensor:
    """Map raw parameters to strictly bounded diagonal eigenvalues.

    Parameters
    ----------
    raw : Tensor
        Unconstrained diagonal parameters with shape ``(latent_dim,)``.
    max_spectral_radius : float
        Configured spectral-radius upper limit.

    Returns
    -------
    Tensor
        Diagonal eigenvalues strictly inside ``[-bound, bound]``.
    """
    bound = strict_spectral_bound(max_spectral_radius)
    return torch.tanh(raw) * bound


def _safe_diagonal_inverse(diag_values: Tensor) -> Tensor:
    """Return ``diag(1 / d_i)`` with magnitude flooring that preserves sign.

    Parameters
    ----------
    diag_values : Tensor
        Diagonal entries (may be negative).

    Returns
    -------
    Tensor
        Diagonal inverse matrix. Entries with ``|d_i| < eps`` are replaced by
        ``sign(d_i) * eps`` before inversion so negative eigenvalues are not
        flipped by a one-sided ``clamp_min``.
    """
    eps = torch.finfo(diag_values.dtype).eps
    safe = torch.copysign(diag_values.abs().clamp_min(eps), diag_values)
    return torch.diag(1.0 / safe)
