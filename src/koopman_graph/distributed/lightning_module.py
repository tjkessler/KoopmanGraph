"""Optional Lightning ``LightningModule`` façade for KoopmanGraph.

Composes a :class:`~koopman_graph.model.GraphKoopmanModel` (does not subclass
it). Trainer usage is sugar for users who standardize on Lightning; prefer
:func:`~koopman_graph.distributed.run_ddp_fit_loop` or
:func:`~koopman_graph.distributed.fit_with_fabric` when you need full loss
schedules, rank-aware window sampling, or the shared epoch driver.

Lightning is imported lazily so :mod:`koopman_graph.distributed` remains
importable without the ``[lightning]`` extra until this submodule is loaded.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training import (
    ExtraLosses,
    LossWeights,
    LRSchedulerFactory,
    compute_training_loss,
    mean_training_loss_breakdown,
)

__all__ = ["KoopmanLightningModule"]  # noqa: F822 — provided via __getattr__

_LIGHTNING_INSTALL_HINT = 'pip install "koopman-graph[lightning]"'
_MODULE_CLS: type[Any] | None = None
_SNAPSHOT_SEQUENCE_TYPES = (GraphSnapshotSequence, HeteroGraphSnapshotSequence)


def _import_lightning() -> Any:
    """Import the ``lightning`` package with an actionable install hint.

    Returns
    -------
    module
        The ``lightning`` package.

    Raises
    ------
    ImportError
        If Lightning is not installed.
    """
    try:
        return importlib.import_module("lightning")
    except ImportError as exc:
        msg = (
            "Lightning is required for KoopmanLightningModule; "
            f"install with: {_LIGHTNING_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


def _as_sequence_batch(
    batch: SnapshotSequence | Sequence[SnapshotSequence],
) -> list[SnapshotSequence]:
    """Normalize a Trainer batch to a list of snapshot sequences.

    Parameters
    ----------
    batch : SnapshotSequence or sequence of SnapshotSequence
        Collated batch from a DataLoader (single window or list of windows).
        Accepts homogeneous or multiplex trajectories.

    Returns
    -------
    list of SnapshotSequence
        Batch as a concrete list.

    Raises
    ------
    TypeError
        If ``batch`` is not a sequence container of snapshot sequences.
    """
    if isinstance(batch, _SNAPSHOT_SEQUENCE_TYPES):
        return [batch]
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        sequences = list(batch)
        if not sequences:
            msg = "training batch must contain at least one SnapshotSequence"
            raise ValueError(msg)
        for item in sequences:
            if not isinstance(item, _SNAPSHOT_SEQUENCE_TYPES):
                msg = (
                    "KoopmanLightningModule batches must be SnapshotSequence "
                    "(GraphSnapshotSequence or HeteroGraphSnapshotSequence) "
                    f"or list thereof; got {type(item)!r}"
                )
                raise TypeError(msg)
        return sequences
    msg = (
        "KoopmanLightningModule batches must be SnapshotSequence "
        "(GraphSnapshotSequence or HeteroGraphSnapshotSequence) or "
        f"list thereof; got {type(batch)!r}"
    )
    raise TypeError(msg)


def _build_koopman_lightning_module() -> type[Any]:
    """Construct the LightningModule subclass once Lightning is available.

    Returns
    -------
    type
        ``KoopmanLightningModule`` class bound to the installed Lightning API.
    """
    lightning = _import_lightning()

    class KoopmanLightningModule(lightning.LightningModule):
        """Thin Lightning façade around a composed GraphKoopman model.

        Notes
        -----
        Constructor parameters are documented on :meth:`__init__`. Prefer
        Fabric / native DDP for full loss schedules and rank-aware sampling.
        """

        def __init__(
            self,
            model: TrainableKoopmanModel,
            *,
            loss_weights: LossWeights | None = None,
            extra_losses: ExtraLosses | None = None,
            lr: float = 1e-3,
            optimizer: Callable[..., Optimizer] = torch.optim.Adam,
            lr_scheduler: LRSchedulerFactory | None = None,
            rollout_horizon: int | None = None,
            rollout_start_indices: Sequence[int] | None = None,
            **optimizer_kwargs: Any,
        ) -> None:
            """Store a composed model and static training configuration.

            Parameters
            ----------
            model : TrainableKoopmanModel
                Trainable Koopman façade held by composition (not subclassed).
            loss_weights : LossWeights or None, optional
                Static loss weights. Default is reconstruction-only when
                ``None``.
            extra_losses : ExtraLosses or None, optional
                Fit-time Lie / PDE callables.
            lr : float, optional
                Learning rate. Default is ``1e-3``.
            optimizer : callable, optional
                Optimizer class. Default is :class:`torch.optim.Adam`.
            lr_scheduler : LRScheduler factory or None, optional
                Optional ``optimizer -> scheduler`` factory.
            rollout_horizon : int or None, optional
                Autoregressive rollout horizon for the training loss.
            rollout_start_indices : sequence of int or None, optional
                Rollout origins (default ``[0]`` inside
                ``compute_training_loss``).
            optimizer_kwargs
                Extra optimizer constructor kwargs.
            """
            super().__init__()
            self.model = model
            self.loss_weights = (
                loss_weights if loss_weights is not None else LossWeights()
            )
            self.extra_losses = extra_losses
            self.lr = float(lr)
            self._optimizer_cls = optimizer
            self._lr_scheduler_factory = lr_scheduler
            self.rollout_horizon = rollout_horizon
            self.rollout_start_indices = rollout_start_indices
            self._optimizer_kwargs = optimizer_kwargs

        def _step_loss(
            self,
            batch: SnapshotSequence | Sequence[SnapshotSequence],
        ) -> tuple[torch.Tensor, int]:
            """Compute mean training loss over a batch of windows.

            Parameters
            ----------
            batch : SnapshotSequence or sequence of SnapshotSequence
                Trainer batch to evaluate (homogeneous or multiplex).

            Returns
            -------
            loss : Tensor
                Mean ``compute_training_loss(...).total`` over the batch.
            batch_size : int
                Number of sequences in the batch (for Lightning logging;
                avoids ``HeteroData`` batch-size inference).
            """
            sequences = _as_sequence_batch(batch)
            breakdowns = [
                compute_training_loss(
                    self.model,
                    sequence,
                    self.loss_weights,
                    extra_losses=self.extra_losses,
                    rollout_horizon=self.rollout_horizon,
                    rollout_start_indices=self.rollout_start_indices,
                )
                for sequence in sequences
            ]
            return mean_training_loss_breakdown(breakdowns).total, len(sequences)

        def training_step(
            self,
            batch: SnapshotSequence | Sequence[SnapshotSequence],
            batch_idx: int,
        ) -> torch.Tensor:
            """Lightning training step on window / trajectory batches.

            Parameters
            ----------
            batch : SnapshotSequence or sequence of SnapshotSequence
                Collated training batch (homogeneous or multiplex).
            batch_idx : int
                Batch index (unused).

            Returns
            -------
            Tensor
                Scalar training loss for automatic optimization.
            """
            del batch_idx
            loss, batch_size = self._step_loss(batch)
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            return loss

        def validation_step(
            self,
            batch: SnapshotSequence | Sequence[SnapshotSequence],
            batch_idx: int,
        ) -> torch.Tensor:
            """Lightning validation step on window / trajectory batches.

            Parameters
            ----------
            batch : SnapshotSequence or sequence of SnapshotSequence
                Collated validation batch (homogeneous or multiplex).
            batch_idx : int
                Batch index (unused).

            Returns
            -------
            Tensor
                Scalar validation loss.
            """
            del batch_idx
            loss, batch_size = self._step_loss(batch)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            return loss

        def configure_optimizers(self) -> Optimizer | dict[str, Any]:
            """Build the optimizer (and optional epoch scheduler).

            Returns
            -------
            Optimizer or dict
                Optimizer alone, or Lightning scheduler mapping when a
                factory is configured.
            """
            optim = self._optimizer_cls(
                self.model.parameters(),
                lr=self.lr,
                **self._optimizer_kwargs,
            )
            if self._lr_scheduler_factory is None:
                return optim
            scheduler = self._lr_scheduler_factory(optim)
            if not isinstance(scheduler, LRScheduler):
                msg = "lr_scheduler factory must return a torch LRScheduler"
                raise TypeError(msg)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        def export_format1_checkpoint(self, path: str | Path) -> None:
            """Write a format-1 KoopmanGraph checkpoint from the composed model.

            Parameters
            ----------
            path : str or Path
                Destination ``.pt`` path. Uses the unwrapped composed
                :class:`~koopman_graph.model.GraphKoopmanModel` (no Lightning
                ``module.`` prefix).
            """
            from koopman_graph.serialization import save_checkpoint

            save_checkpoint(self.model, path)

    return KoopmanLightningModule


def __getattr__(name: str) -> Any:
    """Lazy-build ``KoopmanLightningModule`` when Lightning is first needed.

    Parameters
    ----------
    name : str
        Attribute name requested on this module.

    Returns
    -------
    object
        Lazily constructed ``KoopmanLightningModule`` class.

    Raises
    ------
    AttributeError
        If ``name`` is not ``KoopmanLightningModule``.
    """
    global _MODULE_CLS
    if name == "KoopmanLightningModule":
        if _MODULE_CLS is None:
            _MODULE_CLS = _build_koopman_lightning_module()
        return _MODULE_CLS
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return module attributes including the lazy Lightning class.

    Returns
    -------
    list of str
        Sorted attribute names, including lazy ``__all__`` exports.
    """
    return sorted({*globals().keys(), *__all__})
