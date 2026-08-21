"""Opt-in decoded-space constraint heads.

:class:`MassConservingDecoder` projects named channels onto
:math:`\\{x:\\mathbf{1}^{\\top}x=M\\}` by softmax (simplex) or affine
correction. :class:`PositivityDecoder` applies ``softplus`` or ``exp``
to named channels. :class:`LinearConservingDecoder` is the min-norm
affine projector onto :math:`\\{x:Cx=c_0\\}` for a fixed node count.

These heads act **after** an inner decoder. Latent symplectic
:math:`K` alone does not conserve decoded mass
(``Greydanus2019HNN``). IEEE-118 trajectories remain Laplacian
diffusion — constraint heads do not create AC power-flow consistency.

This module must not import :mod:`koopman_graph.model`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

DEFAULT_CONSERVATION_ATOL = 1e-6
MassMethod = Literal["softmax", "affine"]
PositivityMethod = Literal["softplus", "exp"]

__all__ = [
    "DEFAULT_CONSERVATION_ATOL",
    "LinearConservingDecoder",
    "MassConservingDecoder",
    "MassMethod",
    "PositivityDecoder",
    "PositivityMethod",
    "project_linear_conservation",
]


def _as_channel_index(channels: Sequence[int], *, n_features: int) -> tuple[int, ...]:
    """Validate named feature-column indices.

    Parameters
    ----------
    channels : sequence of int
        Feature columns to constrain.
    n_features : int
        Trailing decoded width.

    Returns
    -------
    tuple[int, ...]
        Coerced channel indices.

    Raises
    ------
    ValueError
        If a channel is repeated or out of range.
    """
    if len(channels) < 1:
        raise ValueError("channels must be a non-empty sequence of ints")
    resolved: list[int] = []
    seen: set[int] = set()
    for raw in channels:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"channel indices must be ints, got {raw!r}")
        if int(raw) < 0 or int(raw) >= int(n_features):
            msg = (
                f"channel {raw} is outside [0, {n_features}), "
                f"got channels={tuple(channels)}"
            )
            raise ValueError(msg)
        if int(raw) in seen:
            raise ValueError(f"channels must be unique, got {tuple(channels)}")
        seen.add(int(raw))
        resolved.append(int(raw))
    return tuple(resolved)


def _require_decoded_table(name: str, value: Tensor) -> None:
    """Refuse empty, non-real, or non-finite decoded tables.

    Parameters
    ----------
    name : str
        Field name for the error message.
    value : Tensor
        Candidate ``(N, d)`` table.

    Raises
    ------
    TypeError
        If ``value`` is not a tensor.
    ValueError
        If rank, dtype, or finiteness is invalid.
    """
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor, got {type(value).__name__}")
    if value.is_complex():
        raise ValueError(f"{name} must be real")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim != 2 or int(value.shape[0]) < 1 or int(value.shape[1]) < 1:
        raise ValueError(
            f"{name} must have shape (n_nodes, n_features) with both "
            f"axes >= 1, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _call_inner(
    decoder: nn.Module,
    z: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
) -> Tensor:
    """Decode latents with the inner module.

    Parameters
    ----------
    decoder : nn.Module
        Inner decoder ``(z, edge_index, edge_weight=None) -> (N, d)``.
    z : Tensor
        Latents ``(N, latent_dim)``.
    edge_index : Tensor
        COO edges ``(2, E)``.
    edge_weight : Tensor or None
        Optional scalar edge weights.

    Returns
    -------
    Tensor
        Decoded table ``(N, d)``.
    """
    decoded = decoder(z, edge_index, edge_weight)
    _require_decoded_table("decoded", decoded)
    return decoded


def project_linear_conservation(
    values: Tensor,
    constraint: Tensor,
    target: Tensor,
) -> Tensor:
    """Min-norm affine correction onto :math:`\\{x:Cx=c_0\\}`.

    Parameters
    ----------
    values : Tensor
        Unconstrained 1-D field ``(N,)``.
    constraint : Tensor
        Constraint matrix ``C`` with shape ``(n_eq, N)``.
    target : Tensor
        Target ``c_0`` with shape ``(n_eq,)``.

    Returns
    -------
    Tensor
        Projected field ``(N,)`` in the dtype of ``values``.

    Raises
    ------
    ValueError
        If shapes are incompatible, values are non-finite, or
        ``constraint`` is row-rank deficient.
    """
    _require_decoded_table("values", values.unsqueeze(-1))
    if constraint.ndim != 2 or int(constraint.shape[1]) != int(values.shape[0]):
        msg = (
            "constraint must have shape (n_eq, n_nodes), "
            f"got {tuple(constraint.shape)} for n_nodes={int(values.shape[0])}"
        )
        raise ValueError(msg)
    if target.ndim != 1 or int(target.shape[0]) != int(constraint.shape[0]):
        msg = (
            "target must have shape (n_eq,), "
            f"got {tuple(target.shape)} for n_eq={int(constraint.shape[0])}"
        )
        raise ValueError(msg)
    if int(constraint.shape[0]) < 1:
        raise ValueError("constraint must have at least one equation")
    if not bool(torch.isfinite(constraint).all().item()) or not bool(
        torch.isfinite(target).all().item()
    ):
        raise ValueError("constraint and target must be finite")
    working = values.detach().to(dtype=torch.float64)
    matrix = constraint.to(dtype=torch.float64, device=working.device)
    rhs = target.to(dtype=torch.float64, device=working.device)
    residual = matrix @ working - rhs
    # Min-norm correction C^+ r via the Gram system (C C^T) λ = r,
    # x ← x − C^T λ. ``lstsq`` on the fat (n_eq < N) factor is
    # driver-dependent: CPU GELS on Linux can leave a large residual
    # on later rows while the first equation looks exact.
    gram = matrix @ matrix.T
    try:
        multiplier = torch.linalg.solve(gram, residual)
    except torch.linalg.LinAlgError as exc:
        msg = (
            "constraint must have full row rank so Cx = c0 is a "
            "well-posed affine section, "
            f"got C with shape {tuple(matrix.shape)}"
        )
        raise ValueError(msg) from exc
    delta = matrix.T @ multiplier
    return (working - delta).to(dtype=values.dtype)


class _ConstraintHead(nn.Module):
    """Shared inner-decoder wrapper.

    Parameters
    ----------
    decoder : nn.Module
        Inner decode map.
    channels : sequence of int
        Feature columns to constrain. Validated on the first forward.
    """

    def __init__(self, decoder: nn.Module, channels: Sequence[int]) -> None:
        """Store the inner decoder and channel request.

        Parameters
        ----------
        decoder : nn.Module
            Inner decoder.
        channels : sequence of int
            Named feature columns.

        Raises
        ------
        TypeError
            If ``decoder`` is not a module.
        ValueError
            If ``channels`` is empty.
        """
        super().__init__()
        if not isinstance(decoder, nn.Module):
            raise TypeError(
                f"decoder must be an nn.Module, got {type(decoder).__name__}"
            )
        if len(channels) < 1:
            raise ValueError("channels must be a non-empty sequence of ints")
        self.decoder = decoder
        self._requested_channels = tuple(channels)

    def _resolved_channels(self, decoded: Tensor) -> tuple[int, ...]:
        """Validate channels against a decoded table.

        Parameters
        ----------
        decoded : Tensor
            Inner decoder output ``(N, d)``.

        Returns
        -------
        tuple[int, ...]
            Channel indices.
        """
        return _as_channel_index(
            self._requested_channels,
            n_features=int(decoded.shape[1]),
        )


class MassConservingDecoder(_ConstraintHead):
    """Project named decoded channels onto a constant node-sum.

    Softmax maps each named column to the simplex scaled by ``mass``.
    Affine subtracts the min-norm correction so
    :math:`\\mathbf{1}^{\\top}x=M`. Other columns are copied.

    Parameters
    ----------
    decoder : nn.Module
        Inner decoder ``(z, edge_index, edge_weight=None) -> (N, d)``.
    channels : sequence of int
        Feature columns to conserve.
    mass : float
        Target sum :math:`M` (same units as the named channel).
    method : {"softmax", "affine"}, optional
        Simplex or affine projector. Default ``"affine"``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        channels: Sequence[int],
        mass: float,
        method: MassMethod = "affine",
    ) -> None:
        """Bind an inner decoder and a scalar mass target.

        Parameters
        ----------
        decoder : nn.Module
            Inner decoder.
        channels : sequence of int
            Named feature columns.
        mass : float
            Target node-sum.
        method : {"softmax", "affine"}, optional
            Projection kind.

        Raises
        ------
        ValueError
            If ``mass`` or ``method`` is invalid.
        """
        super().__init__(decoder, channels)
        if not isinstance(mass, int | float) or isinstance(mass, bool):
            raise ValueError(f"mass must be a finite float, got {mass!r}")
        if not torch.isfinite(torch.tensor(float(mass))):
            raise ValueError(f"mass must be finite, got {mass!r}")
        if method not in {"softmax", "affine"}:
            raise ValueError(f"method must be 'softmax' or 'affine', got {method!r}")
        if method == "softmax" and float(mass) <= 0.0:
            raise ValueError(f"softmax mass must be > 0, got {mass}")
        self.mass = float(mass)
        self.method: MassMethod = method

    def project(self, decoded: Tensor) -> Tensor:
        """Apply the mass map to an already-decoded table.

        Parameters
        ----------
        decoded : Tensor
            Unconstrained reconstruction ``(N, d)``.

        Returns
        -------
        Tensor
            Projected table, same shape.
        """
        _require_decoded_table("decoded", decoded)
        channels = self._resolved_channels(decoded)
        out = decoded.clone()
        ones = torch.ones(
            1,
            int(decoded.shape[0]),
            dtype=torch.float64,
            device=decoded.device,
        )
        target = torch.tensor([self.mass], dtype=torch.float64, device=decoded.device)
        for channel in channels:
            column = out[:, channel]
            if self.method == "softmax":
                weights = torch.softmax(column.to(dtype=torch.float64), dim=0)
                out[:, channel] = (weights * self.mass).to(dtype=decoded.dtype)
            else:
                out[:, channel] = project_linear_conservation(column, ones, target)
        return out

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode then project named channels.

        Parameters
        ----------
        z : Tensor
            Latents ``(N, latent_dim)``.
        edge_index : Tensor
            COO edges.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Constrained reconstruction ``(N, d)``.
        """
        return self.project(_call_inner(self.decoder, z, edge_index, edge_weight))


class PositivityDecoder(_ConstraintHead):
    """Map named decoded channels through ``softplus`` or ``exp``.

    Other columns are copied. This does not conserve mass.

    Parameters
    ----------
    decoder : nn.Module
        Inner decoder.
    channels : sequence of int
        Feature columns forced non-negative (``softplus``) or strictly
        positive (``exp``).
    method : {"softplus", "exp"}, optional
        Pointwise map. Default ``"softplus"``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        channels: Sequence[int],
        method: PositivityMethod = "softplus",
    ) -> None:
        """Bind an inner decoder and a positivity map.

        Parameters
        ----------
        decoder : nn.Module
            Inner decoder.
        channels : sequence of int
            Named feature columns.
        method : {"softplus", "exp"}, optional
            Pointwise map.

        Raises
        ------
        ValueError
            If ``method`` is invalid.
        """
        super().__init__(decoder, channels)
        if method not in {"softplus", "exp"}:
            raise ValueError(f"method must be 'softplus' or 'exp', got {method!r}")
        self.method: PositivityMethod = method

    def project(self, decoded: Tensor) -> Tensor:
        """Apply the positivity map to an already-decoded table.

        Parameters
        ----------
        decoded : Tensor
            Unconstrained reconstruction ``(N, d)``.

        Returns
        -------
        Tensor
            Projected table, same shape.
        """
        _require_decoded_table("decoded", decoded)
        channels = self._resolved_channels(decoded)
        out = decoded.clone()
        for channel in channels:
            column = out[:, channel]
            if self.method == "exp":
                out[:, channel] = torch.exp(column)
            else:
                out[:, channel] = torch.nn.functional.softplus(column)
        return out

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode then apply the positivity map.

        Parameters
        ----------
        z : Tensor
            Latents ``(N, latent_dim)``.
        edge_index : Tensor
            COO edges.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Constrained reconstruction ``(N, d)``.
        """
        return self.project(_call_inner(self.decoder, z, edge_index, edge_weight))


class LinearConservingDecoder(_ConstraintHead):
    """Min-norm affine projector onto :math:`\\{x:Cx=c_0\\}` per named channel.

    ``constraint`` has shape ``(n_eq, N)`` and is fixed at construction.
    Changing node count without rebuilding the head raises.

    Parameters
    ----------
    decoder : nn.Module
        Inner decoder.
    channels : sequence of int
        Feature columns that each receive the same ``C``.
    constraint : Tensor
        ``C`` with shape ``(n_eq, n_nodes)``.
    target : Tensor
        ``c_0`` with shape ``(n_eq,)``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        channels: Sequence[int],
        constraint: Tensor,
        target: Tensor,
    ) -> None:
        """Bind an inner decoder and a linear conservation law.

        Parameters
        ----------
        decoder : nn.Module
            Inner decoder.
        channels : sequence of int
            Named feature columns.
        constraint : Tensor
            ``C`` with shape ``(n_eq, n_nodes)``.
        target : Tensor
            ``c_0`` with shape ``(n_eq,)``.

        Raises
        ------
        ValueError
            If ``constraint`` / ``target`` shapes are invalid.
        """
        super().__init__(decoder, channels)
        if not isinstance(constraint, Tensor) or not isinstance(target, Tensor):
            raise TypeError("constraint and target must be tensors")
        if constraint.ndim != 2 or int(constraint.shape[0]) < 1:
            raise ValueError(
                "constraint must have shape (n_eq, n_nodes) with n_eq >= 1, "
                f"got {tuple(constraint.shape)}"
            )
        if target.ndim != 1 or int(target.shape[0]) != int(constraint.shape[0]):
            raise ValueError(
                "target must have shape (n_eq,), "
                f"got {tuple(target.shape)} for n_eq={int(constraint.shape[0])}"
            )
        if not bool(torch.isfinite(constraint).all().item()) or not bool(
            torch.isfinite(target).all().item()
        ):
            raise ValueError("constraint and target must be finite")
        self.num_nodes = int(constraint.shape[1])
        self.register_buffer("constraint", constraint.detach().to(dtype=torch.float64))
        self.register_buffer("target", target.detach().to(dtype=torch.float64))

    def project(self, decoded: Tensor) -> Tensor:
        """Apply :math:`Cx=c_0` to named columns.

        Parameters
        ----------
        decoded : Tensor
            Unconstrained reconstruction ``(N, d)``.

        Returns
        -------
        Tensor
            Projected table, same shape.

        Raises
        ------
        ValueError
            If the node count does not match ``constraint``.
        """
        _require_decoded_table("decoded", decoded)
        if int(decoded.shape[0]) != self.num_nodes:
            msg = (
                "decoded node count must equal constraint columns "
                f"{self.num_nodes}, got {int(decoded.shape[0])}"
            )
            raise ValueError(msg)
        channels = self._resolved_channels(decoded)
        out = decoded.clone()
        for channel in channels:
            out[:, channel] = project_linear_conservation(
                out[:, channel],
                self.constraint,
                self.target,
            )
        return out

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode then apply the linear conservation law.

        Parameters
        ----------
        z : Tensor
            Latents ``(N, latent_dim)``.
        edge_index : Tensor
            COO edges.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Constrained reconstruction ``(N, d)``.
        """
        return self.project(_call_inner(self.decoder, z, edge_index, edge_weight))
