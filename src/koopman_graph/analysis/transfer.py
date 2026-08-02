"""Measured cross-topology transfer evaluation.

Honesty contract
----------------
This module **measures** zero-shot and fine-tune transfer of factorized
graph Koopman operators across a change in node count. It does **not**
claim that transfer succeeds. Naming is deliberately
``evaluate_topology_transfer`` rather than ``transfer()`` so callers cannot
read the API as a success path.

The report always includes a mandatory ``pernode`` control. Transfer
advantage follows the Appendix B contract: the subject configuration beats
``pernode`` only when its transfer MSE is strictly less than the control by
a documented tolerance :data:`TRANSFER_ADVANTAGE_EPSILON`. Negative advantage
(``False``) is an expected and reportable outcome — as on the seeded
path-diffusion fixture used by example 37.

Incompatible configurations
---------------------------
Some operator settings bind node cardinality and cannot transfer across a
change in :math:`N`:

* ``self_adaptive`` — ``learn_topology="self_adaptive"``
* ``orbit_partition`` — explicit orbit ties or ``koopman_auto_orbits``
* ``isotypic`` — ``koopman_symmetry="isotypic"`` (Aut orbit / isotypic ties)

**Implicit presence** on the template model is listed in
:attr:`TopologyTransferReport.excluded_configs` and evaluation continues
without propagating those settings into rebuilt control models.

**Explicit request** via ``request_excluded`` raises: callers cannot force
an incompatible path into the transfer protocol.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.online_adaptation import freeze_modules

# Absolute tolerance in the Appendix B transfer-advantage inequality.
# Single source of truth — tests import this constant rather than duplicating.
TRANSFER_ADVANTAGE_EPSILON: float = 1e-6

_SUPPORTED_CONTROLS: frozenset[str] = frozenset({"graph", "pernode"})
# Stable report / raise labels for N-binding configurations.
_KNOWN_INCOMPATIBLE: tuple[str, ...] = (
    "self_adaptive",
    "orbit_partition",
    "isotypic",
)
_KNOWN_INCOMPATIBLE_SET: frozenset[str] = frozenset(_KNOWN_INCOMPATIBLE)
_TransferMode = Literal["zero_shot", "finetune_koopman"]

__all__ = [
    "TRANSFER_ADVANTAGE_EPSILON",
    "TopologyTransferReport",
    "evaluate_topology_transfer",
]


@dataclass(frozen=True)
class TopologyTransferReport:
    """Structured result of a measured topology-transfer evaluation.

    Attributes
    ----------
    in_dist_mse : dict[str, float]
        Mean multi-step MSE on the in-distribution holdout, keyed by control
        kind (e.g. ``"graph"``, ``"pernode"``).
    transfer_mse : dict[str, float]
        Mean multi-step MSE on the transfer holdout (or finetune scored
        suffix), keyed by control kind.
    transfer_advantage : bool
        ``True`` only when the subject configuration's transfer MSE beats the
        ``pernode`` control by :data:`TRANSFER_ADVANTAGE_EPSILON` (Appendix B).
        ``False`` is a valid measured outcome, not a failure of this API.
    excluded_configs : tuple[str, ...]
        Incompatible configurations detected on the architecture template
        (e.g. ``"self_adaptive"``, ``"orbit_partition"``, ``"isotypic"``).
        Implicit presence is listed here and evaluation continues without
        those settings; use ``request_excluded`` to raise instead of listing.
    transfer_mode : {"zero_shot", "finetune_koopman"}
        Evaluation mode that produced the report.
    seed : int
        RNG seed used for model construction and training.
    steps : int
        Autoregressive rollout horizon used for MSE.
    """

    in_dist_mse: Mapping[str, float]
    transfer_mse: Mapping[str, float]
    transfer_advantage: bool
    excluded_configs: tuple[str, ...]
    transfer_mode: _TransferMode
    seed: int
    steps: int


def evaluate_topology_transfer(
    model: GraphKoopmanModel,
    train_sequence: GraphSnapshotSequence | Sequence[Any],
    holdout_in_distribution: GraphSnapshotSequence | Sequence[Any],
    holdout_transfer: GraphSnapshotSequence | Sequence[Any],
    *,
    steps: int = 8,
    controls: tuple[str, ...] = ("graph", "pernode"),
    transfer_mode: _TransferMode = "zero_shot",
    seed: int = 0,
    epochs: int = 100,
    lr: float = 5e-3,
    device: str | torch.device = "cpu",
    request_excluded: tuple[str, ...] = (),
    burn_in_timesteps: int = 8,
    finetune_epochs: int = 20,
    finetune_lr: float | None = None,
    **fit_kwargs: Any,
) -> TopologyTransferReport:
    """Measure topology transfer against a mandatory per-node control.

    Honesty contract
    ----------------
    This function evaluates transfer; it does **not** promise a transfer
    advantage. The returned :attr:`TopologyTransferReport.transfer_advantage`
    may be ``False``. Callers must not wrap this entry point as a success API.

    The ``model`` argument is an **architecture template**. For each kind in
    ``controls``, a fresh :class:`~koopman_graph.model.GraphKoopmanModel` is
    built (reinitialized encoder/decoder copies) and trained on
    ``train_sequence``. Pre-trained weights on the template are not reused.

    Transfer modes
    --------------
    ``zero_shot``
        Train on :math:`N_1`, score in-distribution and transfer holdouts with
        no further adaptation.
    ``finetune_koopman``
        Train on :math:`N_1`, freeze the encoder and decoder, fine-tune only
        operator factors on a short :math:`N_2` burn-in prefix of
        ``holdout_transfer``, then score the disjoint suffix. Every control
        (including ``pernode``) receives the same burn-in budget.

    Leakage rules (``finetune_koopman``)
    ------------------------------------
    The burn-in prefix and scored suffix are a contiguous split of
    ``holdout_transfer`` and are therefore disjoint by construction. Fine-tune
    uses only the burn-in prefix; transfer MSE uses only the scored suffix.
    In-distribution MSE is measured after :math:`N_1` training and **before**
    the :math:`N_2` fine-tune. Overlapping or empty slices raise.

    Incompatible configurations (raise vs list)
    -------------------------------------------
    Settings that bind node cardinality
    (``self_adaptive``, ``orbit_partition``, ``isotypic``) cannot transfer
    across a change in :math:`N`. Detection reads the template model — not
    caller-supplied labels on ``controls``.

    * **Implicit presence** on the template is recorded in
      :attr:`TopologyTransferReport.excluded_configs` and evaluation continues
      without propagating those settings into rebuilt control models.
    * **Explicit request** via ``request_excluded`` raises
      :class:`ValueError` naming the incompatible configuration.

    Parameters
    ----------
    model : GraphKoopmanModel
        Architecture template providing encoder/decoder structure and
        ``latent_dim`` / ``time_step`` (and related construction attributes).
    train_sequence : GraphSnapshotSequence or sequence
        Training snapshots at the source node count :math:`N_1`.
    holdout_in_distribution : GraphSnapshotSequence or sequence
        In-distribution holdout at :math:`N_1`.
    holdout_transfer : GraphSnapshotSequence or sequence
        Transfer holdout at a different node count :math:`N_2`. For
        ``finetune_koopman``, the first ``burn_in_timesteps`` snapshots are the
        burn-in prefix and the remainder is the scored suffix.
    steps : int, optional
        Autoregressive rollout horizon for MSE. Default ``8``.
    controls : tuple of str, optional
        Operator kinds to train and compare. Must include ``"pernode"`` and at
        least one subject kind (default ``("graph", "pernode")``). Supported
        kinds for this MVP are ``"graph"`` and ``"pernode"``.
    transfer_mode : {"zero_shot", "finetune_koopman"}, optional
        Evaluation mode. Default ``"zero_shot"``.
    seed : int, optional
        RNG seed for model construction and training. Default ``0``.
    epochs : int, optional
        :math:`N_1` training epochs forwarded to :meth:`GraphKoopmanModel.fit`.
        Default ``100`` (example-37 protocol).
    lr : float, optional
        Learning rate for :math:`N_1` training. Default ``5e-3``.
    device : str or torch.device, optional
        Training device. Default ``"cpu"``.
    request_excluded : tuple of str, optional
        Explicit request to evaluate under named incompatible configurations.
        Any known incompatible name raises. Default ``()``.
    burn_in_timesteps : int, optional
        Length of the :math:`N_2` burn-in prefix for ``finetune_koopman``.
        Ignored for ``zero_shot``. Default ``8``.
    finetune_epochs : int, optional
        Fine-tune epochs on the burn-in prefix. Default ``20``.
    finetune_lr : float or None, optional
        Learning rate for the operator-only fine-tune. When ``None``, uses
        ``lr``. Default ``None``.
    **fit_kwargs
        Additional keyword arguments forwarded to ``fit`` (both phases).

    Returns
    -------
    TopologyTransferReport
        Frozen report with per-control MSEs, the Appendix B advantage flag,
        and any implicitly excluded configs detected on the template.

    Raises
    ------
    ValueError
        If ``steps < 1``, ``controls`` omits ``"pernode"``, ``controls`` has no
        subject kind, a control kind is unsupported, a holdout is too short,
        burn-in / score slices would overlap or be empty, ``transfer_mode`` is
        unknown, or ``request_excluded`` names a known incompatible
        configuration.
    TypeError
        If ``model`` is not a :class:`~koopman_graph.model.GraphKoopmanModel`.
    """
    if not isinstance(model, GraphKoopmanModel):
        msg = (
            "model must be a GraphKoopmanModel architecture template, "
            f"got {type(model).__name__}"
        )
        raise TypeError(msg)
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)
    if transfer_mode not in ("zero_shot", "finetune_koopman"):
        msg = (
            "transfer_mode must be 'zero_shot' or 'finetune_koopman', "
            f"got {transfer_mode!r}"
        )
        raise ValueError(msg)

    _reject_requested_excluded(request_excluded)
    excluded_configs = _detect_excluded_configs(model)
    control_kinds = _validate_controls(controls)
    subject = next(kind for kind in control_kinds if kind != "pernode")

    train = _as_homogeneous_sequence(train_sequence, name="train_sequence")
    hold_in = _as_homogeneous_sequence(
        holdout_in_distribution, name="holdout_in_distribution"
    )
    hold_xfer = _as_homogeneous_sequence(holdout_transfer, name="holdout_transfer")
    _require_rollout_length(hold_in, steps=steps, name="holdout_in_distribution")

    if transfer_mode == "zero_shot":
        _require_rollout_length(hold_xfer, steps=steps, name="holdout_transfer")
        in_dist_mse, transfer_mse = _run_zero_shot(
            model,
            train,
            hold_in,
            hold_xfer,
            control_kinds=control_kinds,
            steps=steps,
            seed=seed,
            epochs=epochs,
            lr=lr,
            device=device,
            fit_kwargs=fit_kwargs,
        )
    else:
        burn_in, scored = _split_burn_in_and_score(
            hold_xfer,
            burn_in_timesteps=burn_in_timesteps,
            steps=steps,
        )
        resolved_finetune_lr = lr if finetune_lr is None else finetune_lr
        in_dist_mse, transfer_mse = _run_finetune_koopman(
            model,
            train,
            hold_in,
            burn_in,
            scored,
            control_kinds=control_kinds,
            steps=steps,
            seed=seed,
            epochs=epochs,
            lr=lr,
            device=device,
            finetune_epochs=finetune_epochs,
            finetune_lr=resolved_finetune_lr,
            fit_kwargs=fit_kwargs,
        )

    advantage = bool(
        transfer_mse[subject] < transfer_mse["pernode"] - TRANSFER_ADVANTAGE_EPSILON
    )
    return TopologyTransferReport(
        in_dist_mse=in_dist_mse,
        transfer_mse=transfer_mse,
        transfer_advantage=advantage,
        excluded_configs=excluded_configs,
        transfer_mode=transfer_mode,
        seed=seed,
        steps=steps,
    )


def _run_zero_shot(
    template: GraphKoopmanModel,
    train: GraphSnapshotSequence,
    hold_in: GraphSnapshotSequence,
    hold_xfer: GraphSnapshotSequence,
    *,
    control_kinds: tuple[str, ...],
    steps: int,
    seed: int,
    epochs: int,
    lr: float,
    device: str | torch.device,
    fit_kwargs: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Train on N1 and score both holdouts with no further adaptation.

    Parameters
    ----------
    template : GraphKoopmanModel
        Architecture template for rebuilt control models.
    train : GraphSnapshotSequence
        Source-topology training snapshots.
    hold_in : GraphSnapshotSequence
        In-distribution holdout at the source node count.
    hold_xfer : GraphSnapshotSequence
        Transfer holdout at the target node count.
    control_kinds : tuple of str
        Operator kinds to train and score.
    steps : int
        Autoregressive rollout horizon for MSE.
    seed : int
        RNG seed for model construction and training.
    epochs : int
        Training epochs forwarded to :meth:`GraphKoopmanModel.fit`.
    lr : float
        Learning rate for training.
    device : str or torch.device
        Training device.
    fit_kwargs : mapping
        Additional keyword arguments forwarded to ``fit``.

    Returns
    -------
    tuple of dict[str, float]
        In-distribution and transfer MSE maps keyed by control kind.
    """
    in_dist_mse: dict[str, float] = {}
    transfer_mse: dict[str, float] = {}
    for kind in control_kinds:
        torch.manual_seed(seed)
        fitted = _build_control_model(template, koopman=kind)
        fitted.fit(train, epochs=epochs, lr=lr, device=device, **fit_kwargs)
        in_dist_mse[kind] = _rollout_mse(fitted, hold_in, steps=steps)
        transfer_mse[kind] = _rollout_mse(fitted, hold_xfer, steps=steps)
    return in_dist_mse, transfer_mse


def _run_finetune_koopman(
    template: GraphKoopmanModel,
    train: GraphSnapshotSequence,
    hold_in: GraphSnapshotSequence,
    burn_in: GraphSnapshotSequence,
    scored: GraphSnapshotSequence,
    *,
    control_kinds: tuple[str, ...],
    steps: int,
    seed: int,
    epochs: int,
    lr: float,
    device: str | torch.device,
    finetune_epochs: int,
    finetune_lr: float,
    fit_kwargs: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Train on N1, freeze encode/decode, fine-tune K on burn-in, score suffix.

    Parameters
    ----------
    template : GraphKoopmanModel
        Architecture template for rebuilt control models.
    train : GraphSnapshotSequence
        Source-topology training snapshots.
    hold_in : GraphSnapshotSequence
        In-distribution holdout scored before fine-tuning.
    burn_in : GraphSnapshotSequence
        Target-topology burn-in prefix for operator-only fine-tuning.
    scored : GraphSnapshotSequence
        Disjoint target-topology suffix used for transfer MSE.
    control_kinds : tuple of str
        Operator kinds to train and score.
    steps : int
        Autoregressive rollout horizon for MSE.
    seed : int
        RNG seed for model construction and training.
    epochs : int
        Initial training epochs on the source topology.
    lr : float
        Learning rate for initial training.
    device : str or torch.device
        Training device.
    finetune_epochs : int
        Operator-only fine-tune epochs on ``burn_in``.
    finetune_lr : float
        Learning rate for the fine-tune phase.
    fit_kwargs : mapping
        Additional keyword arguments forwarded to ``fit``.

    Returns
    -------
    tuple of dict[str, float]
        In-distribution and transfer MSE maps keyed by control kind.
    """
    if finetune_epochs < 1:
        msg = f"finetune_epochs must be >= 1, got {finetune_epochs}"
        raise ValueError(msg)

    in_dist_mse: dict[str, float] = {}
    transfer_mse: dict[str, float] = {}
    for kind in control_kinds:
        torch.manual_seed(seed)
        fitted = _build_control_model(template, koopman=kind)
        fitted.fit(train, epochs=epochs, lr=lr, device=device, **fit_kwargs)
        in_dist_mse[kind] = _rollout_mse(fitted, hold_in, steps=steps)

        freeze_modules((fitted.encoder, fitted.decoder))
        fitted.fit(
            burn_in,
            epochs=finetune_epochs,
            lr=finetune_lr,
            device=device,
            **fit_kwargs,
        )
        transfer_mse[kind] = _rollout_mse(fitted, scored, steps=steps)
    return in_dist_mse, transfer_mse


def _split_burn_in_and_score(
    holdout_transfer: GraphSnapshotSequence,
    *,
    burn_in_timesteps: int,
    steps: int,
) -> tuple[GraphSnapshotSequence, GraphSnapshotSequence]:
    """Split transfer holdout into disjoint burn-in prefix and scored suffix.

    Parameters
    ----------
    holdout_transfer : GraphSnapshotSequence
        Full transfer holdout at the target node count.
    burn_in_timesteps : int
        Length of the burn-in prefix (must be >= 2).
    steps : int
        Rollout horizon required on the scored suffix.

    Returns
    -------
    tuple[GraphSnapshotSequence, GraphSnapshotSequence]
        ``(burn_in, scored)`` contiguous slices.
    """
    if burn_in_timesteps < 2:
        msg = (
            f"burn_in_timesteps must be >= 2 for fine-tune pairs, "
            f"got {burn_in_timesteps}"
        )
        raise ValueError(msg)

    n = len(holdout_transfer)
    need = burn_in_timesteps + steps + 1
    if n < need:
        msg = (
            "holdout_transfer must contain at least "
            "burn_in_timesteps + steps + 1 snapshots for a disjoint scored "
            f"suffix (need {need}, got {n})"
        )
        raise ValueError(msg)

    # Contiguous split [0, burn_in_timesteps) | [burn_in_timesteps, n) is
    # disjoint by construction; the length check above is the leakage guard.
    burn_in = holdout_transfer.slice(0, burn_in_timesteps)
    scored = holdout_transfer.slice(burn_in_timesteps, n)
    _require_rollout_length(scored, steps=steps, name="scored transfer holdout")
    return burn_in, scored


def _validate_controls(controls: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and return the control kinds tuple.

    Parameters
    ----------
    controls : tuple of str
        Requested operator kinds; must include ``\"pernode\"`` and a subject.

    Returns
    -------
    tuple of str
        Validated control kinds unchanged.
    """
    if not controls:
        msg = "controls must be a non-empty tuple of operator kind strings"
        raise ValueError(msg)
    if "pernode" not in controls:
        msg = (
            "controls must include the mandatory 'pernode' control; "
            f"got {controls!r}. Omitting pernode is rejected rather than "
            "silently added so callers cannot hide the control comparison."
        )
        raise ValueError(msg)
    if all(kind == "pernode" for kind in controls):
        msg = (
            "controls must include at least one subject kind besides "
            f"'pernode'; got {controls!r}"
        )
        raise ValueError(msg)
    unknown = sorted({kind for kind in controls if kind not in _SUPPORTED_CONTROLS})
    if unknown:
        supported = ", ".join(sorted(_SUPPORTED_CONTROLS))
        msg = (
            f"unsupported control kind(s) {unknown!r}; "
            f"supported for this MVP: {supported}"
        )
        raise ValueError(msg)
    return controls


def _reject_requested_excluded(request_excluded: tuple[str, ...]) -> None:
    """Raise when the caller explicitly requests an incompatible configuration.

    Parameters
    ----------
    request_excluded : tuple of str
        Incompatible configuration names explicitly requested by the caller.

    Raises
    ------
    ValueError
        If any requested name is unknown or known to be incompatible.
    """
    if not request_excluded:
        return
    unknown = sorted(
        {name for name in request_excluded if name not in _KNOWN_INCOMPATIBLE_SET}
    )
    if unknown:
        known = ", ".join(_KNOWN_INCOMPATIBLE)
        msg = (
            f"request_excluded contains unknown name(s) {unknown!r}; "
            f"known incompatible configurations: {known}"
        )
        raise ValueError(msg)
    for name in request_excluded:
        if name in _KNOWN_INCOMPATIBLE_SET:
            msg = (
                f"incompatible configuration {name!r} cannot transfer across "
                f"a change in node count N; omit request_excluded={name!r} "
                "and rely on excluded_configs listing when the setting is "
                "present on the template"
            )
            raise ValueError(msg)


def _detect_excluded_configs(model: GraphKoopmanModel) -> tuple[str, ...]:
    """Detect N-binding configs from the template model (not caller labels).

    Parameters
    ----------
    model : GraphKoopmanModel
        Architecture template inspected for incompatible settings.

    Returns
    -------
    tuple of str
        Stable labels for detected incompatible configurations.
    """
    found: list[str] = []
    if getattr(model, "learn_topology", None) == "self_adaptive":
        found.append("self_adaptive")

    koopman = getattr(model, "koopman", None)
    isotypic = bool(getattr(koopman, "isotypic_symmetry", False))
    if isotypic:
        # Prefer the isotypic label; internal auto_orbits=True must not also
        # report as a plain orbit_partition exclusion.
        found.append("isotypic")
    else:
        has_orbit = getattr(koopman, "orbit_partition", None) is not None
        has_auto_orbits = bool(getattr(koopman, "auto_orbits", False))
        if has_orbit or has_auto_orbits:
            found.append("orbit_partition")

    return tuple(name for name in _KNOWN_INCOMPATIBLE if name in found)


def _as_homogeneous_sequence(
    sequence: GraphSnapshotSequence | Sequence[Any],
    *,
    name: str,
) -> GraphSnapshotSequence:
    """Resolve a sequence and require a homogeneous GraphSnapshotSequence.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or sequence
        Input snapshots or an already-resolved container.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    GraphSnapshotSequence
        Resolved homogeneous snapshot sequence.

    Raises
    ------
    TypeError
        If resolution does not yield a :class:`GraphSnapshotSequence`.
    """
    resolved = resolve_sequence(sequence)
    if not isinstance(resolved, GraphSnapshotSequence):
        msg = (
            f"{name} must resolve to GraphSnapshotSequence for topology "
            f"transfer evaluation; got {type(resolved).__name__}"
        )
        raise TypeError(msg)
    return resolved


def _require_rollout_length(
    sequence: GraphSnapshotSequence,
    *,
    steps: int,
    name: str,
) -> None:
    """Require ``len(sequence) >= steps + 1`` for origin-0 rollout scoring.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Holdout sequence to validate.
    steps : int
        Required rollout horizon.
    name : str
        Parameter name used in error messages.

    Raises
    ------
    ValueError
        If the sequence is too short for an origin-0 rollout.
    """
    if len(sequence) < steps + 1:
        msg = (
            f"{name} must contain at least steps + 1 snapshots "
            f"(need {steps + 1}, got {len(sequence)})"
        )
        raise ValueError(msg)


def _build_control_model(
    template: GraphKoopmanModel,
    *,
    koopman: str,
) -> GraphKoopmanModel:
    """Build a fresh model for ``koopman`` from an architecture template.

    Parameters
    ----------
    template : GraphKoopmanModel
        Architecture template providing encoder/decoder structure.
    koopman : str
        Built-in operator kind for the rebuilt model.

    Returns
    -------
    GraphKoopmanModel
        Fresh model with reinitialized encoder and decoder weights.
    """
    encoder = copy.deepcopy(template.encoder)
    decoder = copy.deepcopy(template.decoder)
    _reinitialize_parameters(encoder)
    _reinitialize_parameters(decoder)
    return GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=template.latent_dim,
        time_step=template.time_step,
        dynamics_mode=template.dynamics_mode,
        koopman=koopman,  # type: ignore[arg-type]
        control_dim=template.control_dim,
        control_mode=template.control_mode,
        bilinear_rank=template.bilinear_rank,
        n_delays=template.n_delays,
        physics_dim=template.physics_dim,
        physics_preset=template.physics_preset,
        physics_lifting_fn=template.physics_lifting_fn,
        physics_position=template.physics_position,
    )


def _reinitialize_parameters(module: nn.Module) -> None:
    """Reset parameters on every submodule that exposes ``reset_parameters``.

    Parameters
    ----------
    module : nn.Module
        Root module whose subtree is reinitialized.
    """

    def _reset(sub: nn.Module) -> None:
        """Call ``reset_parameters`` when present on a submodule.

        Parameters
        ----------
        sub : nn.Module
            Submodule visited during ``module.apply``.
        """
        reset = getattr(sub, "reset_parameters", None)
        if callable(reset):
            reset()

    module.apply(_reset)


@torch.no_grad()
def _rollout_mse(
    model: GraphKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    steps: int,
) -> float:
    """Mean per-step MSE over an origin-0 autoregressive rollout of ``steps``.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted model used for prediction.
    sequence : GraphSnapshotSequence
        Holdout snapshots with at least ``steps + 1`` frames.
    steps : int
        Rollout horizon.

    Returns
    -------
    float
        Mean squared error averaged over rollout steps.
    """
    model.eval()
    preds = model.predict(sequence[0], steps)
    errors: list[torch.Tensor] = []
    for horizon in range(steps):
        target = sequence[horizon + 1].x
        pred = preds[horizon].x
        if pred is None or target is None:
            msg = (
                "rollout MSE requires non-None node features on predictions "
                f"and targets (horizon={horizon + 1})"
            )
            raise ValueError(msg)
        errors.append(torch.mean((pred - target) ** 2))
    return float(torch.stack(errors).mean().item())
