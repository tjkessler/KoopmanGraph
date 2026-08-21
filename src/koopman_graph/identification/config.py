"""Frozen configuration for opt-in operator identification.

Pass an instance to :meth:`~koopman_graph.model.GraphKoopmanModel.fit` as
``identification=...``. ``identification=None`` (the default) keeps the
Adam path. This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

IdentificationSolver = Literal[
    "ridge",
    "tls",
    "constrained_ls",
    "varpro",
    "alternating",
]

IDENTIFICATION_SOLVERS: frozenset[str] = frozenset(
    {"ridge", "tls", "constrained_ls", "varpro", "alternating"}
)

__all__ = [
    "IDENTIFICATION_SOLVERS",
    "IdentificationConfig",
    "IdentificationSolver",
]


@dataclass(frozen=True)
class IdentificationConfig:
    """Options for closed-form operator identification at fit time.

    ``ridge`` is a dimensionless Tikhonov weight on the least-squares
    Gram, not a physical regularization in feature units.
    ``solver="alternating"`` uses the ridge formula; the fit loop
    supplies encoder/operator alternation. ``solver="varpro"`` raises
    :class:`NotImplementedError`. ``gate_resdmd=True`` fills the final
    identification report's finite-dictionary ``spectral`` block; it
    does not abort ``fit``. Residual-aware reject uses
    :func:`~koopman_graph.identification.select_resdmd_gated` or
    :class:`~koopman_graph.training.ResDMDFitCallback` ``mode="gate"``.

    Attributes
    ----------
    solver : {"ridge", "tls", "constrained_ls", "varpro", "alternating"}
        Closed-form ``K`` method. Default ``"ridge"``.
    ridge : float
        Non-negative finite Tikhonov weight. Default ``1e-4``. Ignored
        by ``tls``.
    select_on : tuple of str
        Model-selection keys a later gate may consult. Default
        ``("rollout", "invariance", "resdmd")``. Empty is allowed.
    gate_resdmd : bool
        When ``True``, fill ``spectral`` on the final identification
        report. Default ``False`` (unset spectral block).
    """

    solver: IdentificationSolver = "ridge"
    ridge: float = 1e-4
    select_on: tuple[str, ...] = ("rollout", "invariance", "resdmd")
    gate_resdmd: bool = False

    def __post_init__(self) -> None:
        """Validate solver name, ridge weight, and selection keys.

        Raises
        ------
        ValueError
            If ``solver`` is unknown, ``ridge`` is negative or non-finite,
            ``select_on`` is not a string sequence, or ``gate_resdmd`` is
            not a ``bool``.
        """
        if self.solver not in IDENTIFICATION_SOLVERS:
            allowed = ", ".join(sorted(IDENTIFICATION_SOLVERS))
            msg = f"solver must be one of {{{allowed}}}; got {self.solver!r}"
            raise ValueError(msg)
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            msg = f"ridge must be a finite non-negative float, got {self.ridge!r}"
            raise ValueError(msg)
        if not isinstance(self.select_on, Sequence) or isinstance(
            self.select_on, (str, bytes)
        ):
            msg = "select_on must be a sequence of strings"
            raise ValueError(msg)
        keys = tuple(self.select_on)
        if any(not isinstance(key, str) or key == "" for key in keys):
            msg = "select_on entries must be non-empty strings"
            raise ValueError(msg)
        if keys != self.select_on:
            object.__setattr__(self, "select_on", keys)
        if type(self.gate_resdmd) is not bool:
            msg = f"gate_resdmd must be a bool, got {type(self.gate_resdmd).__name__}"
            raise ValueError(msg)
