"""Loss functions for Koopman graph dynamics training."""

from __future__ import annotations

from collections.abc import Sequence

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
        *,
        control: Tensor | None = None,
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
        control : Tensor or None, optional
            Control input driving the transition from ``t`` to ``t+1``.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``koopman(z_t, control)`` and ``z_t1``.
        """
        z_pred = koopman(z_t, control=control)
        return nn.functional.mse_loss(z_pred, z_t1)


class BackwardConsistencyLoss(nn.Module):
    """Penalize deviation from inverse linear latent evolution under **K**.

    For latent encodings ``z_t`` and ``z_{t+1}`` with forward dynamics
    ``z_{t+1} = z_t @ K.T``, the backward (adjoint) consistency term is the
    mean squared error between ``z_t`` and the inverse propagation of
    ``z_{t+1}``:

    .. math::

        \\mathcal{L}_{\\mathrm{bc}} = \\| z_t - z_{t+1} K^{\\dagger} \\|^2

    where ``K^{\\dagger}`` denotes an inverse or pseudo-inverse of ``K``.

    Trade-offs
    ----------
    **Benefits:** Enforces bidirectional linear consistency, which can improve
    Koopman operator identifiability and training stability when paired with
    the forward consistency term (see Lusch et al., 2018; Mezić, 2021).

    **Costs:** Dense unconstrained ``K`` requires a matrix inverse or
    pseudo-inverse. The ODO parameterization
    (:attr:`~koopman_graph.operator.KoopmanOperator.parameterization`
    ``"odo"``) provides a cheap exact factorized inverse and bounds the
    spectral radius. For dense ``K``, sequence-level training precomputes the
    inverse once per step rather than per snapshot pair.

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
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
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
        control : Tensor or None, optional
            Control input that drove the forward transition from ``t`` to
            ``t+1``.
        inverse_matrix : Tensor or None, optional
            Precomputed dense inverse matrix reused across pair evaluations.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``z_t`` and the inverse
            propagation of ``z_t1``.
        """
        z_recovered = koopman.inverse_step(
            z_t1,
            control=control,
            inverse_matrix=inverse_matrix,
        )
        return nn.functional.mse_loss(z_recovered, z_t)


class EigenvalueRegularizationLoss(nn.Module):
    """Penalize Koopman eigenvalues outside the unit circle.

    Implements a hinge-style eigenloss that activates only when eigenvalue
    magnitudes exceed one:

    .. math::

        \\mathcal{L}_{\\mathrm{eig}} =
        \\mathrm{mean}\\big(\\max(|\\lambda_i| - 1, 0)^2\\big)

    For the ODO parameterization, eigenvalues are read directly from the
    bounded diagonal factor, avoiding an explicit eigendecomposition.

    Trade-offs
    ----------
    **Benefits:** Encourages discrete-time stability without hard-constraining
    the operator parameterization. Complements spectrally constrained ODO
    initialization (DeepKoopFormer-style factorization; eigeninit/eigenloss
    literature).

    **Costs:** Dense ``K`` requires ``torch.linalg.eigvals`` each evaluation.
    Use the ODO parameterization when a hard spectral-radius bound is preferred.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with a
    :class:`~koopman_graph.operator.KoopmanOperator`.
    """

    def forward(self, koopman: KoopmanOperator) -> Tensor:
        """Compute the unit-circle eigenvalue hinge penalty.

        Parameters
        ----------
        koopman : :class:`~koopman_graph.operator.KoopmanOperator`
            Operator whose eigenvalue magnitudes are penalized.

        Returns
        -------
        Tensor
            Scalar hinge penalty (zero when all magnitudes are <= 1).
        """
        if koopman.parameterization == "odo":
            magnitudes = (
                torch.tanh(koopman.diag_raw).abs() * koopman.max_spectral_radius
            )
        else:
            magnitudes = torch.linalg.eigvals(koopman.K).abs()
        violation = torch.relu(magnitudes - 1.0)
        return (violation**2).mean()


def _inverse_koopman_step(
    z: Tensor,
    koopman: KoopmanOperator,
    *,
    inverse_matrix: Tensor | None = None,
) -> Tensor:
    """Apply one inverse Koopman step.

    Parameters
    ----------
    z : Tensor
        Latent states at time ``t+1``, shape ``(..., latent_dim)``.
    koopman : :class:`~koopman_graph.operator.KoopmanOperator`
        Operator whose inverse step is applied.
    inverse_matrix : Tensor or None, optional
        Precomputed dense inverse matrix reused across pair evaluations.

    Returns
    -------
    Tensor
        Recovered latent states at time ``t``, same shape as ``z``.
    """
    return koopman.inverse_step(z, inverse_matrix=inverse_matrix)


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
        Time-ordered snapshots. For dynamic-topology sequences, each decode step
        uses the target snapshot's ``edge_index``.
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
    edge_weight = getattr(initial, "edge_weight", None)
    z = model.encoder(initial, edge_index, edge_weight)

    total_loss = torch.zeros((), device=z.device)
    for step in range(1, horizon + 1):
        control = None
        if sequence.has_controls:
            control = sequence.control_at(start + step - 1)
        z = model.koopman(z, control=control)
        target = sequence[start + step]
        decode_edge_index = target.edge_index
        decode_edge_weight = getattr(target, "edge_weight", None)
        prediction = model.decoder(z, decode_edge_index, decode_edge_weight)
        total_loss = total_loss + nn.functional.mse_loss(
            prediction,
            target.x,
        )
    return total_loss / horizon


def rollout_multi_start_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
) -> Tensor:
    """Average rollout reconstruction loss over multiple start snapshots.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder``, ``koopman``, and ``decoder`` attributes.
    sequence : GraphSnapshotSequence
        Time-ordered snapshots.
    horizon : int
        Number of rollout steps (must be >= 1).
    start_indices : sequence of int
        Zero-based origin indices for each rollout.

    Returns
    -------
    Tensor
        Scalar mean rollout loss across origins.

    Raises
    ------
    ValueError
        If ``start_indices`` is empty or any origin is invalid.
    """
    if not start_indices:
        msg = "start_indices must contain at least one origin"
        raise ValueError(msg)

    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    for start in start_indices:
        total_loss = total_loss + rollout_sequence_loss(
            model,
            sequence,
            horizon=horizon,
            start=start,
        )
    return total_loss / len(start_indices)
