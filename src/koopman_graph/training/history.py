"""Training history and loss-weight / fit input types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence, MultiTrajectory


@dataclass(frozen=True)
class LossWeights:
    """Weights for reconstruction and consistency loss terms.

    Attributes
    ----------
    reconstruction : float
        Weight on the one-step reconstruction (MSE) loss.
    forward : float
        Weight on the forward consistency loss.
    backward : float
        Weight on the backward consistency loss.
    rollout : float
        Weight on the autoregressive rollout reconstruction loss.
    eigenvalue : float
        Weight on the unit-circle eigenvalue hinge penalty.
    """

    reconstruction: float = 1.0
    forward: float = 0.0
    backward: float = 0.0
    rollout: float = 0.0
    eigenvalue: float = 0.0


LossWeightSchedule = Callable[[int], LossWeights]
EarlyStoppingMonitor = Literal["auto", "train", "val"]
LRSchedulerFactory = Callable[[Optimizer], LRScheduler]
TrainingInput = (
    GraphSnapshotSequence
    | MultiTrajectory
    | Sequence[Data]
    | Sequence[GraphSnapshotSequence]
)
ValidationInput = (
    GraphSnapshotSequence
    | MultiTrajectory
    | Sequence[Data]
    | Sequence[GraphSnapshotSequence]
    | None
)


@dataclass(frozen=True)
class TrainingLossBreakdown:
    """Immutable snapshot of unweighted and weighted loss terms.

    Value/result type (not a mutable accumulator): call sites rebuild via
    constructors, :meth:`zeros`, or :func:`mean_training_loss_breakdown`.
    Module-visible for training utilities and tests; not a package ``__all__``
    export.

    Attributes
    ----------
    reconstruction : Tensor
        Mean one-step reconstruction loss.
    forward : Tensor
        Mean forward consistency loss.
    backward : Tensor
        Mean backward consistency loss.
    rollout : Tensor
        Mean rollout reconstruction loss.
    eigenvalue : Tensor
        Eigenvalue hinge regularization loss.
    total : Tensor
        Weighted sum of all active loss terms.
    """

    reconstruction: Tensor
    forward: Tensor
    backward: Tensor
    rollout: Tensor
    eigenvalue: Tensor
    total: Tensor

    @classmethod
    def zeros(cls, device: torch.device) -> TrainingLossBreakdown:
        """Return a zero breakdown on ``device``.

        Parameters
        ----------
        device : torch.device
            Device for the zero tensors.

        Returns
        -------
        TrainingLossBreakdown
            Breakdown with all scalar terms set to zero.
        """
        zero = torch.zeros((), device=device)
        return cls(zero, zero, zero, zero, zero, zero)

    def to_floats(self) -> dict[str, float]:
        """Convert tensor terms to detached Python floats.

        Returns
        -------
        dict of str to float
            Mapping with keys ``reconstruction``, ``forward``, ``backward``,
            ``rollout``, ``eigenvalue``, and ``total``.
        """
        return {
            "reconstruction": float(self.reconstruction.detach().cpu()),
            "forward": float(self.forward.detach().cpu()),
            "backward": float(self.backward.detach().cpu()),
            "rollout": float(self.rollout.detach().cpu()),
            "eigenvalue": float(self.eigenvalue.detach().cpu()),
            "total": float(self.total.detach().cpu()),
        }


def mean_training_loss_breakdown(
    breakdowns: Sequence[TrainingLossBreakdown],
) -> TrainingLossBreakdown:
    """Average loss breakdowns across multiple trajectories.

    Parameters
    ----------
    breakdowns : sequence of TrainingLossBreakdown
        Per-trajectory breakdowns to average.

    Returns
    -------
    TrainingLossBreakdown
        Element-wise mean across ``breakdowns``.

    Raises
    ------
    ValueError
        If ``breakdowns`` is empty.
    """
    if not breakdowns:
        msg = "breakdowns must contain at least one entry"
        raise ValueError(msg)
    count = len(breakdowns)
    return TrainingLossBreakdown(
        reconstruction=sum(b.reconstruction for b in breakdowns) / count,
        forward=sum(b.forward for b in breakdowns) / count,
        backward=sum(b.backward for b in breakdowns) / count,
        rollout=sum(b.rollout for b in breakdowns) / count,
        eigenvalue=sum(b.eigenvalue for b in breakdowns) / count,
        total=sum(b.total for b in breakdowns) / count,
    )


@dataclass(frozen=True)
class FitHistory:
    """Training history returned by :func:`run_fit_loop` / model ``fit``.

    Public result types in this package are frozen dataclasses with attribute
    access. Per-epoch series are immutable tuples (not in-place-mutable lists).

    Attributes
    ----------
    loss : tuple of float
        Per-epoch average combined training loss.
    epochs : int
        Number of training epochs completed (may be less than requested when
        early stopping triggers).
    reconstruction_loss : tuple of float
        Per-epoch unweighted reconstruction loss.
    forward_loss : tuple of float
        Per-epoch unweighted forward consistency loss.
    backward_loss : tuple of float
        Per-epoch unweighted backward consistency loss.
    rollout_loss : tuple of float
        Per-epoch unweighted rollout reconstruction loss.
    eigenvalue_loss : tuple of float
        Per-epoch unweighted eigenvalue regularization loss.
    val_loss : tuple of float or None
        Per-epoch validation loss when a validation sequence is provided.
    val_reconstruction_loss : tuple of float or None
        Per-epoch unweighted validation reconstruction loss.
    val_forward_loss : tuple of float or None
        Per-epoch unweighted validation forward consistency loss.
    val_backward_loss : tuple of float or None
        Per-epoch unweighted validation backward consistency loss.
    val_rollout_loss : tuple of float or None
        Per-epoch unweighted validation rollout loss.
    val_eigenvalue_loss : tuple of float or None
        Per-epoch unweighted validation eigenvalue loss.
    stopped_early : bool
        Whether training stopped before the requested epoch count.
    best_epoch : int or None
        Zero-based index of the lowest-loss epoch when best-epoch tracking is
        enabled via ``restore_best_weights`` or ``checkpoint_path`` in
        :func:`run_fit_loop`.
    best_loss : float or None
        Lowest monitored loss observed when best-epoch tracking is enabled.
    """

    loss: tuple[float, ...]
    epochs: int
    reconstruction_loss: tuple[float, ...] = ()
    forward_loss: tuple[float, ...] = ()
    backward_loss: tuple[float, ...] = ()
    rollout_loss: tuple[float, ...] = ()
    eigenvalue_loss: tuple[float, ...] = ()
    val_loss: tuple[float, ...] | None = None
    val_reconstruction_loss: tuple[float, ...] | None = None
    val_forward_loss: tuple[float, ...] | None = None
    val_backward_loss: tuple[float, ...] | None = None
    val_rollout_loss: tuple[float, ...] | None = None
    val_eigenvalue_loss: tuple[float, ...] | None = None
    stopped_early: bool = False
    best_epoch: int | None = None
    best_loss: float | None = None
