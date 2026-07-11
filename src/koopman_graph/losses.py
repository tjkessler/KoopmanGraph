"""Loss functions for Koopman graph dynamics training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.operator import KoopmanOperator


class ForwardConsistencyLoss(nn.Module):
    """Penalize deviation from linear latent evolution under the Koopman operator.

    For latent encodings ``z_t`` and ``z_{t+1}``, the loss is the mean squared error
    between ``K @ z_t`` (implemented as ``z_t @ K.T``) and ``z_{t+1}``:

    .. math::

        \\mathcal{L}_{\\mathrm{fc}} = \\| K z_t - z_{t+1} \\|^2

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operator.KoopmanOperator`.
    """

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
        koopman: KoopmanOperator,
    ) -> Tensor:
        """Compute forward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : :class:`~koopman_graph.operator.KoopmanOperator`
            Learnable linear propagator applied to ``z_t``.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``koopman(z_t)`` and ``z_t1``.
        """
        z_pred = koopman(z_t)
        return nn.functional.mse_loss(z_pred, z_t1)


class BackwardConsistencyLoss(nn.Module):
    """Penalize deviation from inverse linear latent evolution under **K**.

    For latent encodings ``z_t`` and ``z_{t+1}`` with forward dynamics
    ``z_{t+1} = z_t @ K.T``, the backward (adjoint) consistency term is the
    mean squared error between ``z_t`` and the pseudo-inverse propagation of
    ``z_{t+1}``:

    .. math::

        \\mathcal{L}_{\\mathrm{bc}} = \\| z_t - z_{t+1} K^{\\dagger} \\|^2

    where ``K^{\\dagger}`` is the Moore-Penrose pseudo-inverse of ``K``.

    Trade-offs
    ----------
    **Benefits:** Enforces bidirectional linear consistency, which can improve
    Koopman operator identifiability and training stability when paired with
    the forward consistency term.

    **Costs:** Each evaluation computes a matrix pseudo-inverse, adding
    overhead relative to the forward term alone. When ``K`` is near-singular,
    pseudo-inverse recovery can be numerically sensitive; tune the backward
    weight accordingly or rely on gradient clipping and loss weight schedules
    in :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operator.KoopmanOperator`.
    """

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
        koopman: KoopmanOperator,
    ) -> Tensor:
        """Compute backward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : :class:`~koopman_graph.operator.KoopmanOperator`
            Learnable linear propagator whose inverse step is applied to
            ``z_t1``.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``z_t`` and the inverse
            propagation of ``z_t1``.
        """
        z_recovered = _inverse_koopman_step(z_t1, koopman)
        return nn.functional.mse_loss(z_recovered, z_t)


def _inverse_koopman_step(z: Tensor, koopman: KoopmanOperator) -> Tensor:
    """Apply one inverse Koopman step via the pseudo-inverse of ``K``.

    Parameters
    ----------
    z : Tensor
        Latent states at time ``t+1``, shape ``(..., latent_dim)``.
    koopman : :class:`~koopman_graph.operator.KoopmanOperator`
        Operator whose matrix ``K`` satisfies ``z_next = z @ K.T``.

    Returns
    -------
    Tensor
        Recovered latent states at time ``t``, same shape as ``z``.
    """
    k_inv = torch.linalg.pinv(koopman.K)
    return z @ k_inv.T


def rollout_sequence_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start: int = 0,
) -> Tensor:
    """Compute autoregressive rollout reconstruction loss from one start snapshot.

    Encodes ``sequence[start]`` once, advances the latent state with the Koopman
    operator for ``horizon`` steps, and compares decoded predictions to the
    observed snapshots ``sequence[start + 1 : start + horizon + 1]``. This term
    aligns training with :meth:`~koopman_graph.model.GraphKoopmanModel.predict`.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder``, ``koopman``, and ``decoder`` attributes.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with shared topology.
    horizon : int
        Number of rollout steps (must be >= 1).
    start : int, optional
        Index of the initial snapshot. Default is ``0``.

    Returns
    -------
    Tensor
        Scalar mean rollout reconstruction loss over ``horizon`` steps.

    Raises
    ------
    ValueError
        If ``horizon < 1``, ``start < 0``, or the sequence is too short.
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if start < 0:
        msg = f"start must be >= 0, got {start}"
        raise ValueError(msg)
    if start + horizon >= sequence.num_timesteps:
        msg = (
            f"sequence too short for rollout from start={start} "
            f"with horizon={horizon} (num_timesteps={sequence.num_timesteps})"
        )
        raise ValueError(msg)

    initial = sequence[start]
    edge_index = initial.edge_index
    z = model.encoder(initial, edge_index)

    total_loss = torch.zeros((), device=z.device)
    for step in range(1, horizon + 1):
        z = model.koopman(z)
        prediction = model.decoder(z, edge_index)
        total_loss = total_loss + nn.functional.mse_loss(
            prediction,
            sequence[start + step].x,
        )
    return total_loss / horizon
