"""Loss-weight schedules for training epochs."""

from __future__ import annotations

from koopman_graph.training.history import LossWeights, LossWeightSchedule


def constant_loss_weights(
    *,
    reconstruction: float = 1.0,
    forward: float = 0.0,
    backward: float = 0.0,
    rollout: float = 0.0,
    eigenvalue: float = 0.0,
    lie: float = 0.0,
    pde: float = 0.0,
    sparsity: float = 0.0,
    worst_case: float = 0.0,
) -> LossWeights:
    """Build static loss weights for all training epochs.

    Parameters
    ----------
    reconstruction : float, optional
        Weight on the reconstruction loss. Default is ``1.0``.
    forward : float, optional
        Weight on the forward consistency loss. Default is ``0.0``.
    backward : float, optional
        Weight on the backward consistency loss. Default is ``0.0``.
    rollout : float, optional
        Weight on the rollout reconstruction loss. Default is ``0.0``.
    eigenvalue : float, optional
        Weight on the eigenvalue hinge penalty. Default is ``0.0``.
    lie : float, optional
        Weight on Lie consistency. Default is ``0.0``.
    pde : float, optional
        Weight on the decoded-field PDE residual. Default is ``0.0``.
    sparsity : float, optional
        Weight on Koopman-matrix sparsity. Default is ``0.0``.
    worst_case : float, optional
        Weight on the worst-case reconstruction term. Default is ``0.0``.

    Returns
    -------
    :class:`~koopman_graph.training.LossWeights`
        Fixed weights applied uniformly across epochs.
    """
    return LossWeights(
        reconstruction=reconstruction,
        forward=forward,
        backward=backward,
        rollout=rollout,
        eigenvalue=eigenvalue,
        lie=lie,
        pde=pde,
        sparsity=sparsity,
        worst_case=worst_case,
    )


def linear_ramp_loss_weights(
    start: LossWeights,
    end: LossWeights,
    ramp_epochs: int,
) -> LossWeightSchedule:
    """Linearly interpolate loss weights over the first ``ramp_epochs``.

    Parameters
    ----------
    start : :class:`~koopman_graph.training.LossWeights`
        Weights at epoch ``0``.
    end : :class:`~koopman_graph.training.LossWeights`
        Weights reached at epoch ``ramp_epochs - 1`` and held thereafter.
    ramp_epochs : int
        Number of epochs over which to interpolate. Must be >= 1.

    Returns
    -------
    callable
        Schedule mapping epoch index to :class:`~koopman_graph.training.LossWeights`.

    Raises
    ------
    ValueError
        If ``ramp_epochs < 1``.
    """
    if ramp_epochs < 1:
        msg = f"ramp_epochs must be >= 1, got {ramp_epochs}"
        raise ValueError(msg)

    def schedule(epoch: int) -> LossWeights:
        """Return interpolated loss weights for a training epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.

        Returns
        -------
        :class:`~koopman_graph.training.LossWeights`
            Interpolated weights for the given epoch.
        """
        if ramp_epochs == 1 or epoch >= ramp_epochs - 1:
            return end
        t = epoch / (ramp_epochs - 1)
        return LossWeights(
            reconstruction=start.reconstruction
            + t * (end.reconstruction - start.reconstruction),
            forward=start.forward + t * (end.forward - start.forward),
            backward=start.backward + t * (end.backward - start.backward),
            rollout=start.rollout + t * (end.rollout - start.rollout),
            eigenvalue=start.eigenvalue + t * (end.eigenvalue - start.eigenvalue),
            lie=start.lie + t * (end.lie - start.lie),
            pde=start.pde + t * (end.pde - start.pde),
            sparsity=start.sparsity + t * (end.sparsity - start.sparsity),
            worst_case=start.worst_case + t * (end.worst_case - start.worst_case),
        )

    return schedule


def resolve_loss_weights_for_epoch(
    epoch: int,
    *,
    loss_weights: LossWeights | None,
    loss_weight_schedule: LossWeightSchedule | None,
) -> LossWeights:
    """Resolve per-epoch loss weights from static weights or a schedule.

    Parameters
    ----------
    epoch : int
        Zero-based epoch index.
    loss_weights : :class:`~koopman_graph.training.LossWeights` or None
        Explicit static weights. When ``None`` and no schedule is provided,
        defaults to reconstruction-only weights.
    loss_weight_schedule : callable or None
        Optional per-epoch schedule. Takes precedence over ``loss_weights``.

    Returns
    -------
    :class:`~koopman_graph.training.LossWeights`
        Weights to use for the given epoch.
    """
    if loss_weight_schedule is not None:
        return loss_weight_schedule(epoch)
    if loss_weights is not None:
        return loss_weights
    return constant_loss_weights()
