"""Multi-epoch fit-loop orchestration and early-stop / scheduler helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import HeteroData

from koopman_graph.data import (
    HeteroGraphSnapshotSequence,
    NeighborWindowSampler,
    RolloutStartIndices,
    SnapshotSequence,
    WindowLikeSampler,
    resolve_rollout_start_indices,
    resolve_window_sampler,
)
from koopman_graph.data.hetero_layout import (
    global_relation_edge_indices,
    snapshot_num_nodes_dict,
)
from koopman_graph.graph_utils import snapshot_hyperedge_index
from koopman_graph.nn.heterogeneous import resolve_multiplex_relation_inputs
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.batched_objectives import validate_graph_batching_request
from koopman_graph.training.callbacks import FitCallback
from koopman_graph.training.device import resolve_device, sequence_to_device
from koopman_graph.training.epochs import (
    eval_one_epoch,
    prepare_training_amp,
    train_one_epoch,
    train_windowed_epoch,
)
from koopman_graph.training.history import (
    EarlyStoppingMonitor,
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
)
from koopman_graph.training.schedules import resolve_loss_weights_for_epoch


def _build_fit_history(
    *,
    losses: list[float],
    reconstruction_losses: list[float],
    forward_losses: list[float],
    backward_losses: list[float],
    rollout_losses: list[float],
    eigenvalue_losses: list[float],
    lie_losses: list[float],
    pde_losses: list[float],
    sparsity_losses: list[float],
    worst_case_losses: list[float],
    vamp2_losses: list[float],
    topology_losses: list[float],
    presence_losses: list[float],
    val_losses: list[float] | None,
    val_reconstruction_losses: list[float] | None,
    val_forward_losses: list[float] | None,
    val_backward_losses: list[float] | None,
    val_rollout_losses: list[float] | None,
    val_eigenvalue_losses: list[float] | None,
    val_lie_losses: list[float] | None,
    val_pde_losses: list[float] | None,
    val_sparsity_losses: list[float] | None,
    val_worst_case_losses: list[float] | None,
    val_vamp2_losses: list[float] | None,
    val_topology_losses: list[float] | None,
    val_presence_losses: list[float] | None,
    stopped_early: bool,
    best_epoch: int | None,
    best_loss: float | None,
) -> FitHistory:
    """Build an immutable :class:`FitHistory` from in-progress epoch lists.

    Parameters
    ----------
    losses, reconstruction_losses, forward_losses, backward_losses,
    rollout_losses, eigenvalue_losses, lie_losses, pde_losses,
    sparsity_losses, worst_case_losses, vamp2_losses, topology_losses,
    presence_losses : list of float
        Per-epoch training loss series (same length as ``losses``).
    val_losses, val_reconstruction_losses, val_forward_losses,
    val_backward_losses, val_rollout_losses, val_eigenvalue_losses,
    val_lie_losses, val_pde_losses, val_sparsity_losses,
    val_worst_case_losses, val_vamp2_losses, val_topology_losses,
    val_presence_losses : list of float or None
        Optional per-epoch validation series; ``None`` when validation
        was not run.
    stopped_early : bool
        Whether early stopping terminated training.
    best_epoch : int or None
        Best epoch index when tracked.
    best_loss : float or None
        Best monitored loss when tracked.

    Returns
    -------
    FitHistory
        Frozen history with tuple series.
    """
    return FitHistory(
        loss=tuple(losses),
        epochs=len(losses),
        reconstruction_loss=tuple(reconstruction_losses),
        forward_loss=tuple(forward_losses),
        backward_loss=tuple(backward_losses),
        rollout_loss=tuple(rollout_losses),
        eigenvalue_loss=tuple(eigenvalue_losses),
        lie_loss=tuple(lie_losses),
        pde_loss=tuple(pde_losses),
        sparsity_loss=tuple(sparsity_losses),
        worst_case_loss=tuple(worst_case_losses),
        vamp2_loss=tuple(vamp2_losses),
        topology_loss=tuple(topology_losses),
        presence_loss=tuple(presence_losses),
        val_loss=None if val_losses is None else tuple(val_losses),
        val_reconstruction_loss=(
            None
            if val_reconstruction_losses is None
            else tuple(val_reconstruction_losses)
        ),
        val_forward_loss=(
            None if val_forward_losses is None else tuple(val_forward_losses)
        ),
        val_backward_loss=(
            None if val_backward_losses is None else tuple(val_backward_losses)
        ),
        val_rollout_loss=(
            None if val_rollout_losses is None else tuple(val_rollout_losses)
        ),
        val_eigenvalue_loss=(
            None if val_eigenvalue_losses is None else tuple(val_eigenvalue_losses)
        ),
        val_lie_loss=None if val_lie_losses is None else tuple(val_lie_losses),
        val_pde_loss=None if val_pde_losses is None else tuple(val_pde_losses),
        val_sparsity_loss=(
            None if val_sparsity_losses is None else tuple(val_sparsity_losses)
        ),
        val_worst_case_loss=(
            None if val_worst_case_losses is None else tuple(val_worst_case_losses)
        ),
        val_vamp2_loss=(None if val_vamp2_losses is None else tuple(val_vamp2_losses)),
        val_topology_loss=(
            None if val_topology_losses is None else tuple(val_topology_losses)
        ),
        val_presence_loss=(
            None if val_presence_losses is None else tuple(val_presence_losses)
        ),
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )


def _validate_identification_fit_request(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
    identification: object,
    *,
    window_sampler: object | None,
    batch_graphs: bool,
) -> None:
    """Reject identification layouts that this increment does not support.

    Lazy-imports identification types so ``import koopman_graph.training``
    does not load that package.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Fit target.
    train_sequences : sequence of SnapshotSequence
        Training trajectories.
    identification : object
        Expected :class:`~koopman_graph.identification.IdentificationConfig`.
    window_sampler : object or None
        Resolved window sampler; must be ``None``.
    batch_graphs : bool
        Multi-graph collate flag; must be ``False``.

    Raises
    ------
    TypeError
        If ``identification`` is not an ``IdentificationConfig``.
    ValueError
        If the operator, delays, controls, topology, or sampling path is
        outside the per-node dense discrete MVP.
    """
    from koopman_graph.identification import IdentificationConfig
    from koopman_graph.operators import KoopmanOperator

    if not isinstance(identification, IdentificationConfig):
        msg = (
            "identification must be IdentificationConfig or None, got "
            f"{type(identification).__name__}"
        )
        raise TypeError(msg)
    if window_sampler is not None:
        msg = (
            "fit(..., identification=...) does not support windowed sampling; "
            "use full-sequence fit (window_length=None, sampler=None)"
        )
        raise ValueError(msg)
    if batch_graphs:
        msg = (
            "fit(..., identification=...) is mutually exclusive with batch_graphs=True"
        )
        raise ValueError(msg)
    if any(
        isinstance(sequence, HeteroGraphSnapshotSequence)
        for sequence in train_sequences
    ):
        msg = "identification does not support HeteroGraphSnapshotSequence"
        raise ValueError(msg)
    if any(
        getattr(sequence, "allow_node_churn", False) for sequence in train_sequences
    ):
        msg = "identification does not support sequences with allow_node_churn=True"
        raise ValueError(msg)
    kind = getattr(model, "koopman_kind", "pernode")
    if kind != "pernode":
        msg = (
            "identification currently supports discrete per-node "
            f"KoopmanOperator only, got koopman_kind={kind!r}"
        )
        raise ValueError(msg)
    if getattr(model, "dynamics_mode", "discrete") != "discrete":
        msg = "identification requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if getattr(model, "n_delays", 1) != 1:
        msg = (
            "identification does not support delay embeddings "
            f"(n_delays={getattr(model, 'n_delays', None)})"
        )
        raise ValueError(msg)
    if int(getattr(model, "control_dim", 0)) != 0:
        msg = (
            "identification does not support controlled models "
            f"(control_dim={getattr(model, 'control_dim', None)})"
        )
        raise ValueError(msg)
    if getattr(model, "adaptive_topology", None) is not None:
        msg = "identification does not support learned pairwise topology"
        raise ValueError(msg)
    koopman = model.koopman
    if not isinstance(koopman, KoopmanOperator):
        msg = (
            "identification currently supports discrete per-node "
            f"KoopmanOperator only, got {type(koopman).__name__}"
        )
        raise ValueError(msg)


def _encode_sequence_latents(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
) -> Tensor:
    """Stack per-timestep encodings for one trajectory.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``encode_at`` or ``encode``.
    sequence : SnapshotSequence
        Training trajectory.

    Returns
    -------
    Tensor
        Latents with shape ``(T, N, d)``.
    """
    encode_at = getattr(model, "encode_at", None)
    stacked = []
    for index in range(sequence.num_timesteps):
        if encode_at is not None:
            stacked.append(encode_at(sequence, index))
        else:
            stacked.append(model.encode(sequence[index]))
    return torch.stack(stacked, dim=0)


def _observe_callback_encodings(
    callbacks: Sequence[object],
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
) -> None:
    """Push a frozen latent stack to callbacks that observe encodings.

    Encodes the first training sequence under ``eval`` / ``no_grad``.
    Layout matches ``evaluate(..., include_resdmd=True)``: time-major
    ``(T, N, d)``. No-ops when no callback exposes
    ``observe_encodings``.

    Parameters
    ----------
    callbacks : sequence of object
        Active fit callbacks.
    model : TrainableKoopmanModel
        Encoder source.
    sequence : SnapshotSequence
        Trajectory to encode (typically ``train_sequences[0]``).
    """
    observers = [
        callback
        for callback in callbacks
        if callable(getattr(callback, "observe_encodings", None))
    ]
    if not observers:
        return
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encodings = _encode_sequence_latents(model, sequence)
    finally:
        model.train(was_training)
    for callback in observers:
        callback.observe_encodings(encodings)


def _collect_identification_pairs(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
):
    """Encode frozen latent pairs for closed-form identification.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Encoder source.
    train_sequences : sequence of SnapshotSequence
        Trajectories to encode.

    Returns
    -------
    tuple
        ``(joint_pairs, per_sequence_pairs)`` as
        :class:`~koopman_graph.identification.LatentPairs` objects.

    Raises
    ------
    ValueError
        If a trajectory has fewer than two snapshots.
    """
    from koopman_graph.identification import LatentPairs

    was_training = model.training
    model.eval()
    per_sequence: list[LatentPairs] = []
    try:
        with torch.no_grad():
            for sequence in train_sequences:
                if sequence.num_timesteps < 2:
                    msg = "identification requires at least two snapshots"
                    raise ValueError(msg)
                latents = _encode_sequence_latents(model, sequence)
                per_sequence.append(LatentPairs(z_t=latents[:-1], z_next=latents[1:]))
    finally:
        model.train(was_training)
    width = per_sequence[0].z_t.shape[-1]
    z_t = torch.cat([pairs.z_t.reshape(-1, width) for pairs in per_sequence], dim=0)
    z_next = torch.cat(
        [pairs.z_next.reshape(-1, width) for pairs in per_sequence],
        dim=0,
    )
    return LatentPairs(z_t=z_t, z_next=z_next), tuple(per_sequence)


def _run_identification_step(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
    identification: object,
    *,
    attach_report: bool,
) -> None:
    """Closed-form ``K`` update from the current encoder.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Fit target; ``koopman`` is overwritten in place.
    train_sequences : sequence of SnapshotSequence
        Trajectories encoded under ``eval`` / ``no_grad``.
    identification : object
        :class:`~koopman_graph.identification.IdentificationConfig`.
    attach_report : bool
        When ``True``, store :class:`~koopman_graph.identification.IdentificationReport`
        on ``model.identification_report``.
    """
    from koopman_graph.identification import (
        apply_operator_snapshot,
        build_identification_report,
        identify_operator,
    )

    joint_pairs, per_sequence = _collect_identification_pairs(model, train_sequences)
    snapshot = identify_operator(joint_pairs, identification)
    apply_operator_snapshot(model.koopman, snapshot)
    if attach_report:
        model.identification_report = build_identification_report(  # type: ignore[attr-defined]
            per_sequence,
            snapshot,
            gate_resdmd=bool(getattr(identification, "gate_resdmd", False)),
        )


__all__ = [
    "bind_pending_orbit_ties",
    "eval_one_epoch",
    "prepare_training_amp",
    "resolve_early_stopping_monitor",
    "resolve_lr_scheduler",
    "run_fit_loop",
    "should_stop_early",
    "train_one_epoch",
    "train_windowed_epoch",
]


def resolve_early_stopping_monitor(
    monitor: EarlyStoppingMonitor,
    *,
    has_validation: bool,
) -> Literal["train", "val"]:
    """Resolve which loss early stopping should monitor.

    Parameters
    ----------
    monitor : {"auto", "train", "val"}
        Requested monitor mode.
    has_validation : bool
        Whether a validation sequence was provided to :meth:`fit`.

    Returns
    -------
    {"train", "val"}
        Resolved monitor target.

    Raises
    ------
    ValueError
        If ``monitor="val"`` but no validation sequence was provided.
    """
    if monitor == "auto":
        return "val" if has_validation else "train"
    if monitor == "val" and not has_validation:
        msg = "early_stopping_monitor='val' requires validation_sequence"
        raise ValueError(msg)
    return monitor


def bind_pending_orbit_ties(
    model: nn.Module,
    train_sequences: Sequence[SnapshotSequence],
) -> None:
    """Bind ``auto_orbits`` from the first train snapshot before the optimizer.

    Late binding during the first ``advance`` would allocate orbit ``K_self``
    factors after :func:`torch.optim.Optimizer` construction, leaving those
    parameters untrained. Call this before creating the optimizer whenever the
    Koopman module supports orbit ties.

    Parameters
    ----------
    model : nn.Module
        Trainable model that may expose a ``koopman`` submodule with
        :meth:`~koopman_graph.operators.orbit_ties.OrbitTiedSelfMixin.ensure_orbit_binding`
        (when present).
    train_sequences : sequence of SnapshotSequence
        Training trajectories already placed on the training device. The first
        snapshot of the first sequence supplies topology. Multiplex hetero
        sequences bind orbits from the **union** of relation banks;
        typed hetero binds per-type orbits from intra-type banks.

    Raises
    ------
    ValueError
        If a homogeneous bind path receives ``HeteroData``, or typed /
        multiplex hetero binding fails validation.
    """
    if not train_sequences:
        return
    koopman = getattr(model, "koopman", None)
    ensure = getattr(koopman, "ensure_orbit_binding", None)
    if koopman is None or ensure is None:
        return
    pending_symmetry = bool(getattr(koopman, "auto_orbits", False)) or bool(
        getattr(koopman, "isotypic_symmetry", False)
    )
    if not pending_symmetry:
        return
    if getattr(koopman, "orbit_partition", None) is not None:
        return
    if isinstance(train_sequences[0], HeteroGraphSnapshotSequence):
        snapshot = train_sequences[0][0]
        if not isinstance(snapshot, HeteroData):
            msg = (
                "hetero auto_orbits binding requires HeteroData snapshots; "
                f"got {type(snapshot).__name__}"
            )
            raise ValueError(msg)
        if bool(getattr(koopman, "is_typed", False)):
            typed_ensure = getattr(koopman, "ensure_typed_orbit_binding", None)
            if typed_ensure is None:
                msg = (
                    "typed hetero auto_orbits requires "
                    "ensure_typed_orbit_binding on the Koopman operator"
                )
                raise ValueError(msg)
            node_types = tuple(koopman.node_types)
            edge_types = tuple(koopman.edge_types)
            counts = snapshot_num_nodes_dict(snapshot, node_types)
            banks = global_relation_edge_indices(
                snapshot,
                edge_types,
                node_types,
                num_nodes_dict=counts,
            )
            typed_ensure(banks, counts)
            return
        # Match multiplex RelGraph bank order (sorted edge-type keys), not
        # the operator's default synthetic ``r0``/``r1`` labels.
        _x, banks, _weights = resolve_multiplex_relation_inputs(
            snapshot,
            None,
            None,
            num_relations=int(koopman.num_relations),
        )
        nonempty = [bank for bank in banks if bank.numel() > 0]
        if nonempty:
            union = torch.cat(nonempty, dim=1)
        else:
            device = banks[0].device if banks else _x.device
            union = torch.zeros(2, 0, dtype=torch.long, device=device)
        ensure(int(_x.shape[0]), edge_index=union)
        return
    snapshot = train_sequences[0][0]
    if isinstance(snapshot, HeteroData):
        msg = (
            "auto_orbits / orbit binding requires homogeneous Data snapshots; "
            "got HeteroData"
        )
        raise ValueError(msg)
    ensure(
        int(snapshot.num_nodes),
        edge_index=snapshot.edge_index,
        hyperedge_index=snapshot_hyperedge_index(snapshot),
    )


def resolve_lr_scheduler(
    lr_scheduler: LRScheduler | LRSchedulerFactory | None,
    optimizer: Optimizer,
) -> LRScheduler | None:
    """Instantiate an optional learning-rate scheduler.

    Parameters
    ----------
    lr_scheduler : LRScheduler, callable, or None
        Scheduler instance or factory ``optimizer -> scheduler``.
    optimizer : Optimizer
        Optimizer passed to a scheduler factory.

    Returns
    -------
    LRScheduler or None
        Resolved scheduler, if any.
    """
    if lr_scheduler is None:
        return None
    if isinstance(lr_scheduler, LRScheduler):
        return lr_scheduler
    return lr_scheduler(optimizer)


def should_stop_early(
    *,
    epoch_loss: float,
    best_loss: float,
    epochs_without_improvement: int,
    patience: int,
    min_delta: float,
) -> tuple[bool, float, int]:
    """Update early-stopping state after an epoch.

    Parameters
    ----------
    epoch_loss : float
        Training loss for the completed epoch.
    best_loss : float
        Best loss seen so far.
    epochs_without_improvement : int
        Consecutive epochs without sufficient improvement.
    patience : int
        Stop after this many non-improving epochs.
    min_delta : float
        Minimum decrease in loss to count as improvement.

    Returns
    -------
    tuple of (bool, float, int)
        Whether to stop, updated best loss, and updated non-improvement count.
    """
    if epoch_loss < best_loss - min_delta:
        return False, epoch_loss, 0
    updated_count = epochs_without_improvement + 1
    return updated_count >= patience, best_loss, updated_count


def run_fit_loop(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    optimizer: Callable[..., Optimizer] = torch.optim.Adam,
    device: str | torch.device | None = None,
    loss_weights: LossWeights | None = None,
    loss_weight_schedule: LossWeightSchedule | None = None,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
    lr_scheduler: LRScheduler | LRSchedulerFactory | None = None,
    window_length: int | None = None,
    batch_size: int = 8,
    windows_per_epoch: int | None = None,
    window_seed: int | None = None,
    sampler: WindowLikeSampler | None = None,
    max_grad_norm: float | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: Literal["train", "val"] = "train",
    val_sequences: Sequence[SnapshotSequence] | None = None,
    restore_best_weights: bool = False,
    checkpoint_path: str | Path | None = None,
    callbacks: Sequence[FitCallback] | None = None,
    batch_graphs: bool = False,
    identification: object | None = None,
    **optimizer_kwargs: Any,
) -> FitHistory:
    """Run the multi-epoch training loop for a trainable Koopman model.

    Canonical orchestration for :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
    Callers are responsible for resolving and validating sequences (controls,
    length, multi-trajectory layout) and for resolving
    ``early_stopping_monitor`` via :func:`resolve_early_stopping_monitor`
    before calling this helper (``"auto"`` is not accepted here).

    Callback hooks
    --------------
    When ``callbacks`` is set, each :class:`~koopman_graph.training.FitCallback`
    is invoked in list order as:

    * ``on_fit_start`` once after device / optimizer setup and before epoch 0
    * ``on_epoch_end`` after each epoch's train (and optional val) losses are
      recorded and best-epoch tracking is updated, before early-stop ``break``.
      Callbacks that expose ``observe_encodings`` receive a frozen time-major
      latent stack from the first training sequence first (identity-dictionary
      ResDMD layout).
    * ``on_fit_end`` once with the final :class:`FitHistory`, after optional
      best-weight restore / checkpoint write. ``ResDMDFitCallback(mode="gate")``
      may raise here without mutating parameters.

    Callbacks must not mutate model parameters or optimizer state. Default
    ``None`` preserves prior behavior.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        (also an ``nn.Module`` for device / state-dict operations).
    train_sequences : sequence of GraphSnapshotSequence
        Already-validated training trajectories.
    epochs : int, optional
        Number of training epochs. Default is ``100``.
    lr : float, optional
        Learning rate passed to the optimizer. Default is ``1e-3``.
    optimizer : callable, optional
        Optimizer class. Default is :class:`torch.optim.Adam`.
    device : str, torch.device, or None, optional
        Training device. Defaults to the model's parameter device, or CPU.
    loss_weights : LossWeights or None, optional
        Static loss weights for all epochs.
    loss_weight_schedule : callable or None, optional
        Per-epoch weight schedule; overrides ``loss_weights`` when set.
    extra_losses : ExtraLosses or None, optional
        Fit-time callables for enabled Lie and PDE residual terms.
    rollout_horizon : int or None, optional
        Autoregressive rollout steps when rollout weight is non-zero.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Rollout-loss origin indices.
    rollout_starts_per_epoch : int or None, optional
        Random sample size for rollout origins each epoch.
    rollout_start_seed : int or None, optional
        Base seed for rollout-origin sampling.
    lr_scheduler : LRScheduler or callable, optional
        Scheduler instance or ``optimizer -> scheduler`` factory.
    window_length : int or None, optional
        Fixed window length for mini-batch training; ``None`` uses full
        sequences. Mutually exclusive with ``sampler``. Supported for
        homogeneous and hetero
        (:class:`~koopman_graph.data.HeteroGraphSnapshotSequence`)
        trajectories via :class:`~koopman_graph.data.WindowSampler`.
    batch_size : int, optional
        Windows per optimizer step when windowed. Default is ``8``.
    windows_per_epoch : int or None, optional
        Cap on sampled windows per epoch.
    window_seed : int or None, optional
        Base seed for window shuffling.
    sampler : WindowSampler, NeighborWindowSampler, or None, optional
        Pre-built window sampler (temporal or neighbor-subgraph). When set,
        ``window_length`` / ``batch_size`` / ``windows_per_epoch`` /
        ``window_seed`` are ignored. :class:`~koopman_graph.data.WindowSampler`
        is allowed for hetero sequences;
        :class:`~koopman_graph.data.NeighborWindowSampler` remains
        homogeneous-only (induced-subgraph approximation).
    max_grad_norm : float or None, optional
        Global gradient-norm clip before each optimizer step.
    use_amp : bool, optional
        Enable CUDA automatic mixed precision (autocast + GradScaler).
        On CPU/MPS, warns once and runs FP32. Default is ``False``.
    amp_dtype : torch.dtype or None, optional
        Autocast dtype when AMP is active (default ``torch.float16``).
    early_stopping_patience : int or None, optional
        Stop after this many non-improving epochs. Disabled when ``None``.
    early_stopping_min_delta : float, optional
        Minimum monitored-loss decrease to count as improvement.
    early_stopping_monitor : {"train", "val"}, optional
        Resolved monitor (not ``"auto"``). Default is ``"train"``.
    val_sequences : sequence of GraphSnapshotSequence or None, optional
        Optional held-out trajectories already validated by the caller.
    restore_best_weights : bool, optional
        Reload in-memory weights from the lowest-loss epoch when ``True``.
    checkpoint_path : str, Path, or None, optional
        Write a checkpoint at the lowest-loss epoch when set.
    callbacks : sequence of FitCallback or None, optional
        Observe-only fit hooks. Default ``None`` skips all hook calls.
    batch_graphs : bool, optional
        When ``True``, collate homogeneous trajectories into one PyG
        ``Batch`` for the full-sequence epoch path. Default is ``False``
        (existing per-sequence ``MultiTrajectory`` loop). Mutually
        exclusive with ``window_length``, ``sampler``, and DDP.
    identification : IdentificationConfig or None, optional
        Opt-in closed-form operator identification. ``None`` (default)
        keeps the Adam path and does not import
        :mod:`koopman_graph.identification`. When set, each epoch freezes
        the encoder, fits dense per-node ``K`` (ridge / TLS / constrained
        LS), then takes encoder/decoder Adam steps. Graph, hetero,
        continuous, controlled, delay, windowed, and ``batch_graphs``
        layouts raise. The report is stored on
        ``model.identification_report``.
    **optimizer_kwargs
        Extra keyword arguments for the optimizer constructor.

    Returns
    -------
    FitHistory
        Per-epoch losses and early-stop metadata.

    Raises
    ------
    ValueError
        If         ``early_stopping_monitor="val"`` without ``val_sequences``, if both
        ``sampler`` and ``window_length`` are provided, if ``batch_graphs``
        is combined with windowed sampling, if a
        :class:`~koopman_graph.data.NeighborWindowSampler` is used with
        hetero training sequences, or if ``identification`` is set on an
        unsupported operator / sampling layout.
    """
    if early_stopping_monitor == "val" and val_sequences is None:
        msg = 'early_stopping_monitor="val" requires val_sequences'
        raise ValueError(msg)
    if sampler is not None and window_length is not None:
        msg = "pass sampler or window_length, not both"
        raise ValueError(msg)
    if batch_graphs and (sampler is not None or window_length is not None):
        msg = "batch_graphs=True is mutually exclusive with window_length and sampler"
        raise ValueError(msg)
    has_hetero_train = any(
        isinstance(sequence, HeteroGraphSnapshotSequence)
        for sequence in train_sequences
    )
    if has_hetero_train and isinstance(sampler, NeighborWindowSampler):
        msg = (
            "NeighborWindowSampler is homogeneous-only; "
            "HeteroGraphSnapshotSequence windowed fit requires WindowSampler "
            "or window_length=... (not neighbor-subgraph sampling)"
        )
        raise ValueError(msg)

    # Lazy import: avoid training → serialization → model edges at module load.
    from koopman_graph.serialization import save_checkpoint, snapshot_state_dict

    module = cast(nn.Module, model)
    train_device = resolve_device(model, device)
    module.to(train_device)
    train_sequences = [
        sequence_to_device(sequence, train_device) for sequence in train_sequences
    ]
    if val_sequences is not None:
        val_sequences = [
            sequence_to_device(sequence, train_device) for sequence in val_sequences
        ]

    if batch_graphs:
        validate_graph_batching_request(model, train_sequences)
        if val_sequences is not None:
            validate_graph_batching_request(model, val_sequences)

    # Allocate orbit-tied K_self before Adam sees module.parameters().
    bind_pending_orbit_ties(module, train_sequences)

    if sampler is not None:
        sampler.sequences = [
            sequence_to_device(sequence, train_device) for sequence in sampler.sequences
        ]
    window_sampler = resolve_window_sampler(
        train_sequences,
        window_length=window_length,
        batch_size=batch_size,
        windows_per_epoch=windows_per_epoch,
        window_seed=window_seed,
        sampler=sampler,
        distributed=False,
    )
    if identification is not None:
        _validate_identification_fit_request(
            model,
            train_sequences,
            identification,
            window_sampler=window_sampler,
            batch_graphs=batch_graphs,
        )
        koopman_ids = {id(parameter) for parameter in model.koopman.parameters()}
        trainable = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in koopman_ids
        ]
        if trainable:
            optim: Optimizer | None = optimizer(trainable, lr=lr, **optimizer_kwargs)
        else:
            if lr_scheduler is not None:
                msg = (
                    "identification fit with no trainable encoder/decoder "
                    "parameters cannot use lr_scheduler"
                )
                raise ValueError(msg)
            optim = None
    else:
        model.identification_report = None  # type: ignore[attr-defined]
        optim = optimizer(module.parameters(), lr=lr, **optimizer_kwargs)
    scheduler = None if optim is None else resolve_lr_scheduler(lr_scheduler, optim)
    amp_enabled, resolved_amp_dtype, grad_scaler = prepare_training_amp(
        use_amp,
        train_device,
        amp_dtype,
    )
    # Epoch helpers also accept use_amp; pass the resolved scaler so state
    # persists across epochs when AMP is active.
    amp_kwargs = {
        "use_amp": amp_enabled,
        "amp_dtype": resolved_amp_dtype,
        "grad_scaler": grad_scaler,
    }
    losses: list[float] = []
    reconstruction_losses: list[float] = []
    forward_losses: list[float] = []
    backward_losses: list[float] = []
    rollout_losses: list[float] = []
    eigenvalue_losses: list[float] = []
    lie_losses: list[float] = []
    pde_losses: list[float] = []
    sparsity_losses: list[float] = []
    worst_case_losses: list[float] = []
    vamp2_losses: list[float] = []
    topology_losses: list[float] = []
    presence_losses: list[float] = []
    val_losses: list[float] | None = [] if val_sequences is not None else None
    val_reconstruction_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    val_forward_losses: list[float] | None = [] if val_sequences is not None else None
    val_backward_losses: list[float] | None = [] if val_sequences is not None else None
    val_rollout_losses: list[float] | None = [] if val_sequences is not None else None
    val_eigenvalue_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    val_lie_losses: list[float] | None = [] if val_sequences is not None else None
    val_pde_losses: list[float] | None = [] if val_sequences is not None else None
    val_sparsity_losses: list[float] | None = [] if val_sequences is not None else None
    val_worst_case_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    val_vamp2_losses: list[float] | None = [] if val_sequences is not None else None
    val_topology_losses: list[float] | None = [] if val_sequences is not None else None
    val_presence_losses: list[float] | None = [] if val_sequences is not None else None
    best_loss_for_stop = float("inf")
    best_loss: float | None = None
    best_epoch: int | None = None
    best_state_dict: dict[str, Tensor] | None = None
    track_best = restore_best_weights or checkpoint_path is not None
    epochs_without_improvement = 0
    stopped_early = False
    active_callbacks: Sequence[FitCallback] = () if callbacks is None else callbacks
    fit_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "lr": lr,
        "device": train_device,
        "loss_weights": loss_weights,
        "loss_weight_schedule": loss_weight_schedule,
        "extra_losses": extra_losses,
        "rollout_horizon": rollout_horizon,
        "window_length": window_length,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "use_amp": use_amp,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": early_stopping_monitor,
        "restore_best_weights": restore_best_weights,
        "checkpoint_path": checkpoint_path,
        "has_val_sequences": val_sequences is not None,
        "batch_graphs": batch_graphs,
        "identification": identification is not None,
    }
    for callback in active_callbacks:
        callback.on_fit_start(model=model, fit_kwargs=fit_kwargs)

    for epoch in range(epochs):
        if identification is not None:
            _run_identification_step(
                model,
                train_sequences,
                identification,
                attach_report=False,
            )
        epoch_weights = resolve_loss_weights_for_epoch(
            epoch,
            loss_weights=loss_weights,
            loss_weight_schedule=loss_weight_schedule,
        )
        epoch_rollout_starts: list[int] | None = None
        if window_sampler is None:
            rollout_horizon_for_epoch = (
                train_sequences[0].num_timesteps - 1
                if rollout_horizon is None
                else rollout_horizon
            )
            epoch_rollout_starts = resolve_rollout_start_indices(
                train_sequences[0],
                horizon=rollout_horizon_for_epoch,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
                epoch=epoch,
            )
            if optim is None:
                breakdown = eval_one_epoch(
                    model,
                    train_sequences,
                    epoch_weights,
                    extra_losses=extra_losses,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=epoch_rollout_starts,
                    batch_graphs=batch_graphs,
                )
            else:
                breakdown = train_one_epoch(
                    model,
                    train_sequences,
                    optim,
                    epoch_weights,
                    extra_losses=extra_losses,
                    max_grad_norm=max_grad_norm,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=epoch_rollout_starts,
                    batch_graphs=batch_graphs,
                    **amp_kwargs,
                )
        else:
            if optim is None:
                msg = "windowed fit requires trainable parameters"
                raise ValueError(msg)
            breakdown = train_windowed_epoch(
                model,
                window_sampler,
                optim,
                epoch_weights,
                extra_losses=extra_losses,
                epoch=epoch,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
                **amp_kwargs,
            )
        if scheduler is not None:
            scheduler.step()

        term_values = breakdown.to_floats()
        losses.append(term_values["total"])
        reconstruction_losses.append(term_values["reconstruction"])
        forward_losses.append(term_values["forward"])
        backward_losses.append(term_values["backward"])
        rollout_losses.append(term_values["rollout"])
        eigenvalue_losses.append(term_values["eigenvalue"])
        lie_losses.append(term_values["lie"])
        pde_losses.append(term_values["pde"])
        sparsity_losses.append(term_values["sparsity"])
        worst_case_losses.append(term_values["worst_case"])
        vamp2_losses.append(term_values["vamp2"])
        topology_losses.append(term_values["topology"])
        presence_losses.append(term_values["presence"])

        monitored_loss = term_values["total"]
        epoch_val_breakdown = None
        if val_sequences is not None:
            epoch_val_breakdown = eval_one_epoch(
                model,
                val_sequences,
                epoch_weights,
                extra_losses=extra_losses,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=epoch_rollout_starts,
                batch_graphs=batch_graphs,
            )
            val_terms = epoch_val_breakdown.to_floats()
            assert val_losses is not None
            assert val_reconstruction_losses is not None
            assert val_forward_losses is not None
            assert val_backward_losses is not None
            assert val_rollout_losses is not None
            assert val_eigenvalue_losses is not None
            assert val_lie_losses is not None
            assert val_pde_losses is not None
            assert val_sparsity_losses is not None
            assert val_worst_case_losses is not None
            assert val_vamp2_losses is not None
            assert val_topology_losses is not None
            assert val_presence_losses is not None
            val_losses.append(val_terms["total"])
            val_reconstruction_losses.append(val_terms["reconstruction"])
            val_forward_losses.append(val_terms["forward"])
            val_backward_losses.append(val_terms["backward"])
            val_rollout_losses.append(val_terms["rollout"])
            val_eigenvalue_losses.append(val_terms["eigenvalue"])
            val_lie_losses.append(val_terms["lie"])
            val_pde_losses.append(val_terms["pde"])
            val_sparsity_losses.append(val_terms["sparsity"])
            val_worst_case_losses.append(val_terms["worst_case"])
            val_vamp2_losses.append(val_terms["vamp2"])
            val_topology_losses.append(val_terms["topology"])
            val_presence_losses.append(val_terms["presence"])
            if early_stopping_monitor == "val":
                monitored_loss = val_terms["total"]

        if track_best and (best_loss is None or monitored_loss < best_loss):
            best_loss = monitored_loss
            best_epoch = epoch
            best_state_dict = snapshot_state_dict(module)

        if active_callbacks:
            history_so_far = _build_fit_history(
                losses=losses,
                reconstruction_losses=reconstruction_losses,
                forward_losses=forward_losses,
                backward_losses=backward_losses,
                rollout_losses=rollout_losses,
                eigenvalue_losses=eigenvalue_losses,
                lie_losses=lie_losses,
                pde_losses=pde_losses,
                sparsity_losses=sparsity_losses,
                worst_case_losses=worst_case_losses,
                vamp2_losses=vamp2_losses,
                topology_losses=topology_losses,
                presence_losses=presence_losses,
                val_losses=val_losses,
                val_reconstruction_losses=val_reconstruction_losses,
                val_forward_losses=val_forward_losses,
                val_backward_losses=val_backward_losses,
                val_rollout_losses=val_rollout_losses,
                val_eigenvalue_losses=val_eigenvalue_losses,
                val_lie_losses=val_lie_losses,
                val_pde_losses=val_pde_losses,
                val_sparsity_losses=val_sparsity_losses,
                val_worst_case_losses=val_worst_case_losses,
                val_vamp2_losses=val_vamp2_losses,
                val_topology_losses=val_topology_losses,
                val_presence_losses=val_presence_losses,
                stopped_early=False,
                best_epoch=best_epoch,
                best_loss=best_loss,
            )
            _observe_callback_encodings(active_callbacks, model, train_sequences[0])
            for callback in active_callbacks:
                callback.on_epoch_end(
                    epoch=epoch,
                    train_breakdown=breakdown,
                    val_breakdown=epoch_val_breakdown,
                    history_so_far=history_so_far,
                )

        if early_stopping_patience is not None:
            stop, best_loss_for_stop, epochs_without_improvement = should_stop_early(
                epoch_loss=monitored_loss,
                best_loss=best_loss_for_stop,
                epochs_without_improvement=epochs_without_improvement,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
            if stop:
                stopped_early = True
                break

    if track_best and best_state_dict is not None:
        last_state_dict: dict[str, Tensor] | None = None
        if not restore_best_weights:
            last_state_dict = snapshot_state_dict(module)
        module.load_state_dict(best_state_dict)
        if checkpoint_path is not None:
            # File-path best-epoch checkpoints stay legacy .pt (not directories).
            save_checkpoint(
                model,  # type: ignore[arg-type]
                checkpoint_path,
                format="legacy_pt",
            )
        if not restore_best_weights and last_state_dict is not None:
            module.load_state_dict(last_state_dict)

    if identification is not None:
        _run_identification_step(
            model,
            train_sequences,
            identification,
            attach_report=True,
        )

    history = _build_fit_history(
        losses=losses,
        reconstruction_losses=reconstruction_losses,
        forward_losses=forward_losses,
        backward_losses=backward_losses,
        rollout_losses=rollout_losses,
        eigenvalue_losses=eigenvalue_losses,
        lie_losses=lie_losses,
        pde_losses=pde_losses,
        sparsity_losses=sparsity_losses,
        worst_case_losses=worst_case_losses,
        vamp2_losses=vamp2_losses,
        topology_losses=topology_losses,
        presence_losses=presence_losses,
        val_losses=val_losses,
        val_reconstruction_losses=val_reconstruction_losses,
        val_forward_losses=val_forward_losses,
        val_backward_losses=val_backward_losses,
        val_rollout_losses=val_rollout_losses,
        val_eigenvalue_losses=val_eigenvalue_losses,
        val_lie_losses=val_lie_losses,
        val_pde_losses=val_pde_losses,
        val_sparsity_losses=val_sparsity_losses,
        val_worst_case_losses=val_worst_case_losses,
        val_vamp2_losses=val_vamp2_losses,
        val_topology_losses=val_topology_losses,
        val_presence_losses=val_presence_losses,
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )
    for callback in active_callbacks:
        callback.on_fit_end(history=history)
    return history
