"""Encoder vs discrete-graph-operator receptive-field check.

Compares spatial hop radii. Encoder neighborhood mixing does **not**
compensate for a one-hop (or shorter) Koopman factor. The helper warns
and never fails the call. Operators without
``receptive_field_hops`` (per-node, continuous graph, hetero,
hypergraph) are skipped. Decoder hops are ignored.

``nn`` duck-types the operator and must **not** import
:mod:`koopman_graph.operators`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from torch import nn

__all__ = [
    "ReceptiveFieldMismatchWarning",
    "ReceptiveFieldReport",
    "check_encoder_operator_receptive_field",
]


class ReceptiveFieldMismatchWarning(UserWarning):
    """Encoder hops exceed the discrete graph operator hop degree.

    Emitted by :func:`check_encoder_operator_receptive_field` and
    :meth:`~koopman_graph.model.GraphKoopmanModel.fit`. Training still
    proceeds. Encoder neighborhood mixing does not compensate for a
    one-hop Koopman factor.

    Notes
    -----
    Filter with ``warnings.filterwarnings`` on this class. Operators
    without ``receptive_field_hops`` do not emit the warning.
    """


@dataclass(frozen=True)
class ReceptiveFieldReport:
    """Hop counts from :func:`check_encoder_operator_receptive_field`.

    Attributes
    ----------
    encoder_hops : int or None
        Spatial encoder radius, or ``None`` when unknown.
    operator_hops : int or None
        Discrete graph operator radius ``P``, or ``None`` when the
        operator does not expose ``receptive_field_hops``.
    warned : bool
        ``True`` when a :class:`ReceptiveFieldMismatchWarning` was emitted.
    """

    encoder_hops: int | None
    operator_hops: int | None
    warned: bool


def _as_hop_count(value: object) -> int | None:
    """Return a non-negative hop integer, or ``None`` if unusable.

    Parameters
    ----------
    value : object
        Candidate hop count.

    Returns
    -------
    int or None
        ``value`` when it is a non-negative ``int`` (not ``bool``).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _encoder_hops(encoder: nn.Module) -> int | None:
    """Infer spatial hops from an encoder (unwrap delay wrappers).

    Parameters
    ----------
    encoder : nn.Module
        Topology-aware encoder or delay wrapper.

    Returns
    -------
    int or None
        Hop radius, or ``None`` when the encoder has no graph radius.
    """
    current: nn.Module = encoder
    for _ in range(8):
        getter = getattr(current, "receptive_field_hops", None)
        if callable(getter):
            try:
                hops = _as_hop_count(getter())
            except (TypeError, AttributeError, ValueError):
                hops = None
            if hops is not None:
                return hops
        diffusion = _as_hop_count(getattr(current, "diffusion_steps", None))
        layers = _as_hop_count(getattr(current, "num_layers", None))
        if diffusion is not None and layers is not None:
            return layers * diffusion
        if layers is not None:
            return layers
        base = getattr(current, "base_encoder", None)
        if isinstance(base, nn.Module):
            current = base
            continue
        return None
    return None


def _operator_hops(operator: nn.Module) -> int | None:
    """Return ``operator.receptive_field_hops()`` when that method exists.

    Parameters
    ----------
    operator : nn.Module
        Koopman operator. Duck-typed; no operator-package import.

    Returns
    -------
    int or None
        Operator hop degree, or ``None`` to skip the check.
    """
    getter = getattr(operator, "receptive_field_hops", None)
    if not callable(getter):
        return None
    try:
        return _as_hop_count(getter())
    except (TypeError, AttributeError, ValueError):
        return None


def check_encoder_operator_receptive_field(
    encoder: nn.Module,
    operator: nn.Module,
    *,
    stacklevel: int = 2,
) -> ReceptiveFieldReport:
    """Warn when encoder hops exceed discrete graph operator hops.

    GCN/GAT/SAGE/Transformer stacks count one hop per layer. DiffConv
    counts ``num_layers * diffusion_steps``. Delay wrappers contribute
    no spatial hops. Decoder depth is ignored. Missing hop metadata
    skips the warning (per-node and non-graph operators).

    Parameters
    ----------
    encoder : nn.Module
        Encoder (or delay wrapper around one).
    operator : nn.Module
        Koopman operator. Checked only when it exposes
        ``receptive_field_hops``.
    stacklevel : int, optional
        ``warnings.warn`` stacklevel. Default ``2`` points at the
        caller. ``GraphKoopmanModel.fit`` passes ``3``.

    Returns
    -------
    ReceptiveFieldReport
        Hop counts and whether a warning was emitted.

    Warnings
    --------
    ReceptiveFieldMismatchWarning
        Encoder hops exceed operator hops. Fit still proceeds. Encoder
        mixing does not compensate for a one-hop Koopman factor.
    """
    encoder_hops = _encoder_hops(encoder)
    operator_hops = _operator_hops(operator)
    if encoder_hops is None or operator_hops is None:
        return ReceptiveFieldReport(
            encoder_hops=encoder_hops,
            operator_hops=operator_hops,
            warned=False,
        )
    if encoder_hops <= operator_hops:
        return ReceptiveFieldReport(
            encoder_hops=encoder_hops,
            operator_hops=operator_hops,
            warned=False,
        )
    level = 2 if stacklevel < 1 else int(stacklevel)
    msg = (
        f"Encoder receptive field is {encoder_hops} hop(s) "
        f"({type(encoder).__name__}) but the graph operator mixes "
        f"{operator_hops} hop(s) ({type(operator).__name__}). "
        "Encoder neighborhood mixing does not compensate for a "
        "one-hop Koopman factor; raise koopman_filter_degree or "
        "reduce encoder depth."
    )
    warnings.warn(msg, ReceptiveFieldMismatchWarning, stacklevel=level)
    return ReceptiveFieldReport(
        encoder_hops=encoder_hops,
        operator_hops=operator_hops,
        warned=True,
    )
