"""Identification backend protocol and encoding / operator snapshots.

:class:`IdentificationBackend.fit_operator` is the plug-in for closed-form
solvers. The default Adam ``fit`` path does not call this protocol.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from koopman_graph.identification.config import IdentificationConfig

__all__ = [
    "IdentificationBackend",
    "LatentPairs",
    "OperatorSnapshot",
]


@dataclass(frozen=True)
class LatentPairs:
    """Consecutive latent encodings for a linear operator fit.

    ``z_t`` and ``z_next`` share ``shape``, dtype, and device. Trailing
    width is the latent dimension :math:`d` (or :math:`N\\cdot d` when
    flattened).

    Attributes
    ----------
    z_t : Tensor
        Encodings at time :math:`t`.
    z_next : Tensor
        Encodings at time :math:`t+1`, same shape as ``z_t``.
    """

    z_t: Tensor
    z_next: Tensor

    def __post_init__(self) -> None:
        """Validate matching shape, dtype, and device.

        Raises
        ------
        TypeError
            If either field is not a ``Tensor``.
        ValueError
            If shape, dtype, or device differ.
        """
        z_t = self.z_t
        z_next = self.z_next
        if not isinstance(z_t, Tensor) or not isinstance(z_next, Tensor):
            msg = "z_t and z_next must be torch.Tensor"
            raise TypeError(msg)
        if z_t.shape != z_next.shape:
            msg = (
                "z_t and z_next must share shape, "
                f"got {tuple(z_t.shape)} vs {tuple(z_next.shape)}"
            )
            raise ValueError(msg)
        if z_t.dtype != z_next.dtype:
            msg = f"z_t and z_next must share dtype, got {z_t.dtype} vs {z_next.dtype}"
            raise ValueError(msg)
        if z_t.device != z_next.device:
            msg = (
                f"z_t and z_next must share device, got {z_t.device} vs {z_next.device}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class OperatorSnapshot:
    """Identified operator factors or a dense matrix.

    At least one of ``matrix``, ``k_self``, or ``k_nbr`` must be set.
    Tensors are borrowed, not cloned.

    Attributes
    ----------
    matrix : Tensor or None
        Dense :math:`(n, n)` map on flattened latents.
    k_self : Tensor or None
        Shared self factor :math:`(d, d)`.
    k_nbr : Tensor or None
        Shared neighbor factor :math:`(d, d)`.
    """

    matrix: Tensor | None = None
    k_self: Tensor | None = None
    k_nbr: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate presence and types of factor slots.

        Raises
        ------
        TypeError
            If a provided slot is not a ``Tensor``.
        ValueError
            If every slot is ``None``.
        """
        for name, value in (
            ("matrix", self.matrix),
            ("k_self", self.k_self),
            ("k_nbr", self.k_nbr),
        ):
            if value is not None and not isinstance(value, Tensor):
                msg = f"{name} must be a Tensor or None"
                raise TypeError(msg)
        if self.matrix is None and self.k_self is None and self.k_nbr is None:
            msg = "OperatorSnapshot requires matrix, k_self, or k_nbr"
            raise ValueError(msg)


@runtime_checkable
class IdentificationBackend(Protocol):
    """Closed-form operator estimator on frozen encodings.

    Notes
    -----
    Implementers return an :class:`OperatorSnapshot`. Opt-in
    ``fit(..., identification=...)`` calls
    :func:`~koopman_graph.identification.identify_operator` by default.
    """

    def fit_operator(
        self,
        encodings: LatentPairs,
        config: IdentificationConfig,
    ) -> OperatorSnapshot:
        """Fit an operator snapshot from consecutive latent encodings.

        Parameters
        ----------
        encodings : LatentPairs
            Frozen ``z_t`` / ``z_next`` pairs.
        config : IdentificationConfig
            Solver name and ridge / selection options.

        Returns
        -------
        OperatorSnapshot
            Dense map and/or factorized :math:`d\\times d` blocks.
        """
        ...
