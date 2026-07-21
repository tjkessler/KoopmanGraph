"""Online-adaptation façade bridge helpers for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. These helpers only wire the
sklearn-like adaptation methods to :mod:`koopman_graph.adaptation`; they do
**not** relocate :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` or
:class:`~koopman_graph.adaptation.KoopmanObserver`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.adaptation import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.operators.contract import KoopmanOperatorContract
from koopman_graph.protocols import DynamicsMode

EncodeSnapshotFn = Callable[[Data], Tensor]


def enable_online_adaptation(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    koopman: KoopmanOperatorContract,
    is_continuous: bool,
    forgetting_factor: float = 0.99,
    regularization: float = 1e3,
) -> RecursiveKoopmanAdapter:
    """Freeze encoder/decoder and construct an RLS adapter for dense operators.

    Parameters
    ----------
    encoder, decoder
        Modules whose parameters are frozen for online updates.
    koopman
        Dense-parameterized operator seed for the adapter.
    is_continuous : bool
        Selects continuous vs discrete adaptation mode.
    forgetting_factor : float, optional
        RLS forgetting factor in ``(0, 1]``. Default is ``0.99``.
    regularization : float, optional
        Initial covariance scale for the RLS regressor. Default is ``1e3``.

    Returns
    -------
    RecursiveKoopmanAdapter
        Adapter instance ready to store on the model façade.

    Raises
    ------
    ValueError
        If the Koopman operator is not densely parameterized.
    """
    if koopman.parameterization != "dense":
        msg = (
            "Online adaptation requires dense Koopman parameterization; "
            f"got {koopman.parameterization!r}."
        )
        raise ValueError(msg)

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    mode: DynamicsMode = "continuous" if is_continuous else "discrete"
    return RecursiveKoopmanAdapter.from_operator(
        koopman,
        mode=mode,
        forgetting_factor=forgetting_factor,
        regularization=regularization,
    )


def run_adapt_step(
    adapter: RecursiveKoopmanAdapter,
    *,
    encode: EncodeSnapshotFn,
    koopman: KoopmanOperatorContract,
    is_continuous: bool,
    time_step: float,
    snapshot_t: Data,
    snapshot_tp1: Data,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
) -> AdaptationStepResult:
    """Apply one online RLS update from a pair of graph snapshots.

    Parameters
    ----------
    adapter
        Active :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter`.
    encode
        Frozen encode callable ``Data -> latent``.
    koopman
        Operator written by ``adapter.apply_to``.
    is_continuous : bool
        Whether continuous ``delta_t`` soft-default applies.
    time_step : float
        Soft-default continuous interval when ``delta_t`` is omitted.
    snapshot_t, snapshot_tp1 : Data
        Source and target graph snapshots.
    control : Tensor or None, optional
        Control input applied during the transition.
    delta_t : float, Tensor, or None, optional
        Integration interval for continuous models.

    Returns
    -------
    AdaptationStepResult
        Diagnostics for the adaptation step.
    """
    resolved_delta = delta_t
    if is_continuous and resolved_delta is None:
        resolved_delta = time_step

    with torch.no_grad():
        z_t = encode(snapshot_t)
        z_tp1 = encode(snapshot_tp1)
        result = adapter.update(
            z_t,
            z_tp1,
            control=control,
            delta_t=resolved_delta,
        )
        adapter.apply_to(koopman)
    return result


def disable_online_adaptation(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    unfreeze: bool = True,
) -> None:
    """Optionally restore encoder/decoder trainability after adaptation.

    Parameters
    ----------
    encoder, decoder
        Modules previously frozen by :func:`enable_online_adaptation`.
    unfreeze : bool, optional
        When ``True``, restore ``requires_grad`` on encoder and decoder
        parameters. Default is ``True``.
    """
    if not unfreeze:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)


def freeze_modules(modules: Iterable[nn.Module]) -> None:
    """Freeze parameters on each module (shared by RL env construction).

    Parameters
    ----------
    modules : Iterable[nn.Module]
        Modules whose parameters should be set ``requires_grad=False``.
    """
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
