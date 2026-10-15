"""Log-concentration transform and element-balance violation.

Concentrations in kinetics span many decades, so models often train on
``log c``. :class:`LogConcentrationTransform` clamps at a recorded positive
``floor`` before the logarithm (so zero or slightly negative integrator
output stays finite) and serializes that floor for dataset cards. Values
below the floor do not round-trip: ``inverse(forward(c)) == max(c, floor)``.

:func:`element_balance_violation` measures how far predicted
concentrations drift from the element totals of a reference state, the
conservation check a learned operator should pass on a closed system.

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.system import ReactionSystem

DEFAULT_LOG_FLOOR = 1e-12
"""Default concentration floor applied before ``log`` (same units as ``c``)."""

_TRANSFORM_KIND = "log_concentration"


@dataclass(frozen=True)
class LogConcentrationTransform:
    """Elementwise ``z = log(max(c, floor))`` with exact inverse ``c = exp(z)``.

    Attributes
    ----------
    floor : float
        Positive, finite clamp applied before the logarithm, in the same
        units as the concentrations. Default is :data:`DEFAULT_LOG_FLOOR`.
    """

    floor: float = DEFAULT_LOG_FLOOR

    def __post_init__(self) -> None:
        """Validate that ``floor`` is finite and positive.

        Raises
        ------
        ValueError
            If ``floor`` is not a finite number greater than zero.
        """
        try:
            floor = float(self.floor)
        except (TypeError, ValueError) as exc:
            msg = f"floor must be a real number, got {self.floor!r}"
            raise ValueError(msg) from exc
        if not math.isfinite(floor) or floor <= 0.0:
            msg = f"floor must be finite and > 0, got {self.floor!r}"
            raise ValueError(msg)
        object.__setattr__(self, "floor", floor)

    def forward(self, concentrations: Tensor) -> Tensor:
        """Map concentrations to floored log-concentrations.

        Parameters
        ----------
        concentrations : Tensor
            Floating tensor of any shape. NaN entries stay NaN.

        Returns
        -------
        Tensor
            ``log(clamp(c, min=floor))`` with the input's shape and dtype.
        """
        c = _floating_tensor(concentrations, "concentrations")
        return torch.log(torch.clamp(c, min=self.floor))

    def inverse(self, log_concentrations: Tensor) -> Tensor:
        """Map log-concentrations back to concentrations.

        Parameters
        ----------
        log_concentrations : Tensor
            Floating tensor of any shape.

        Returns
        -------
        Tensor
            ``exp(z)`` with the input's shape and dtype (always positive).
        """
        z = _floating_tensor(log_concentrations, "log_concentrations")
        return torch.exp(z)

    def floored_mask(self, concentrations: Tensor) -> Tensor:
        """Return which entries fall below the floor and lose information.

        Parameters
        ----------
        concentrations : Tensor
            Floating tensor of any shape.

        Returns
        -------
        Tensor
            Boolean tensor, ``True`` where ``c < floor``.
        """
        c = _floating_tensor(concentrations, "concentrations")
        return c < self.floor

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transform for a dataset card.

        Returns
        -------
        dict
            ``{"kind": "log_concentration", "floor": floor}``.
        """
        return {"kind": _TRANSFORM_KIND, "floor": self.floor}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogConcentrationTransform:
        """Rebuild a transform from :meth:`to_dict` output.

        Parameters
        ----------
        payload : dict
            Mapping with ``kind == "log_concentration"`` and ``floor``.

        Returns
        -------
        LogConcentrationTransform
            Transform with the recorded floor.

        Raises
        ------
        ValueError
            If ``kind`` is not ``"log_concentration"``.
        KeyError
            If ``floor`` is missing.
        """
        kind = payload.get("kind")
        if kind != _TRANSFORM_KIND:
            msg = f"expected kind {_TRANSFORM_KIND!r}, got {kind!r}"
            raise ValueError(msg)
        return cls(floor=payload["floor"])


def element_balance_violation(
    concentrations: Tensor,
    reference: Tensor,
    composition: ReactionSystem | Tensor,
) -> Tensor:
    """Return the signed element-total drift ``E (c - c_ref)``.

    Parameters
    ----------
    concentrations : Tensor
        Floating tensor with shape ``(..., num_species)``, e.g. a predicted
        trajectory ``(T, num_species)``.
    reference : Tensor
        Reference state broadcastable to ``concentrations``, typically the
        initial condition with shape ``(num_species,)``.
    composition : ReactionSystem or Tensor
        A system with an ``element_matrix``, or the element matrix itself
        with shape ``(num_elements, num_species)``.

    Returns
    -------
    Tensor
        Violation with shape ``(..., num_elements)``. Zero for an exact
        closed-system trajectory; take ``.abs().amax(-1)`` for a scalar per
        state.

    Raises
    ------
    TypeError
        If inputs are not floating Tensors (or ``composition`` is neither a
        system nor a Tensor).
    ValueError
        If the system has no element matrix or species dimensions disagree.
    """
    c = _floating_tensor(concentrations, "concentrations")
    ref = _floating_tensor(reference, "reference")
    if isinstance(composition, ReactionSystem):
        if composition.element_matrix is None:
            msg = f"{composition.name!r} has no element_matrix"
            raise ValueError(msg)
        element_matrix = composition.element_matrix
    elif isinstance(composition, Tensor):
        element_matrix = composition
    else:
        msg = (
            "composition must be a ReactionSystem or Tensor, got "
            f"{type(composition).__name__}"
        )
        raise TypeError(msg)
    if element_matrix.ndim != 2:
        msg = (
            "element matrix must have shape (num_elements, num_species), got "
            f"{tuple(element_matrix.shape)}"
        )
        raise ValueError(msg)
    num_species = int(element_matrix.shape[1])
    for name, tensor in (("concentrations", c), ("reference", ref)):
        if tensor.ndim == 0 or int(tensor.shape[-1]) != num_species:
            msg = (
                f"{name} must have trailing dimension {num_species}, got "
                f"{tuple(tensor.shape)}"
            )
            raise ValueError(msg)
    delta = c - ref.to(dtype=c.dtype, device=c.device)
    return delta @ element_matrix.to(dtype=c.dtype, device=c.device).T


def _floating_tensor(value: Any, name: str) -> Tensor:
    """Require a floating-point Tensor.

    Parameters
    ----------
    value
        See signature.
    name
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(value, Tensor):
        msg = f"{name} must be a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    if not value.is_floating_point():
        msg = f"{name} must be floating point, got {value.dtype}"
        raise TypeError(msg)
    return value
