"""Loss functions for Koopman graph dynamics training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.continuous import ContinuousKoopmanOperator
from koopman_graph.data import (
    GraphSnapshotSequence,
    _snapshot_edge_weight,
    resolve_pair_delta_t,
)
from koopman_graph.operator import KoopmanOperator

KoopmanPropagator = KoopmanOperator | ContinuousKoopmanOperator


def _propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
) -> Tensor:
    """Advance latent states with discrete or continuous Koopman operators.

    Returns
    -------
    Tensor
        Advanced latent states.
    """
    if isinstance(koopman, ContinuousKoopmanOperator):
        interval = 1.0 if delta_t is None else delta_t
        return koopman.advance(z, interval, control=control)
    return koopman(z, control=control)


def _inverse_propagate_latent(
    koopman: KoopmanPropagator,
    z: Tensor,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    inverse_matrix: Tensor | None = None,
) -> Tensor:
    """Apply one inverse Koopman propagation step.

    Returns
    -------
    Tensor
        Recovered latent states.
    """
    if isinstance(koopman, ContinuousKoopmanOperator):
        interval = 1.0 if delta_t is None else delta_t
        return koopman.inverse_advance(z, interval, control=control)
    return koopman.inverse_step(
        z,
        control=control,
        inverse_matrix=inverse_matrix,
    )


def masked_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute mean squared error over observed nodes only.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features with shape ``(num_nodes, feature_dim)``.
    target : Tensor
        Ground-truth node features with the same shape as ``prediction``.
    mask : Tensor
        Boolean node mask with shape ``(num_nodes,)``. ``True`` marks an
        observed node included in the average.

    Returns
    -------
    Tensor
        Scalar masked mean squared error.
    """
    node_mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    if node_mask.dtype != prediction.dtype:
        node_mask = node_mask.to(dtype=prediction.dtype)
    expanded = node_mask.unsqueeze(-1)
    diff_sq = (prediction - target) ** 2
    denom = expanded.sum() * prediction.shape[-1]
    if denom <= 0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return (diff_sq * expanded).sum() / denom


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
        koopman: KoopmanPropagator,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> Tensor:
        """Compute forward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : KoopmanOperator or ContinuousKoopmanOperator
            Learnable linear propagator applied to ``z_t``.
        control : Tensor or None, optional
            Control input driving the transition from ``t`` to ``t+1``.
        delta_t : float, Tensor, or None, optional
            Integration interval for continuous-time operators. Ignored for
            discrete operators.

        Returns
        -------
        Tensor
            Scalar mean-squared error between propagated ``z_t`` and ``z_t1``.
        """
        z_pred = _propagate_latent(koopman, z_t, control=control, delta_t=delta_t)
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
        koopman: KoopmanPropagator,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        delta_t: float | Tensor | None = None,
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
        z_recovered = _inverse_propagate_latent(
            koopman,
            z_t1,
            control=control,
            inverse_matrix=inverse_matrix,
            delta_t=delta_t,
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

    def forward(self, koopman: KoopmanPropagator) -> Tensor:
        """Compute the stability eigenvalue hinge penalty.

        Discrete operators penalize magnitudes outside the unit circle.
        Continuous generators penalize positive real parts.

        Parameters
        ----------
        koopman : KoopmanOperator or ContinuousKoopmanOperator
            Operator whose eigenvalues are penalized.

        Returns
        -------
        Tensor
            Scalar hinge penalty.
        """
        if isinstance(koopman, ContinuousKoopmanOperator):
            if koopman.parameterization in {"odo", "schur", "lyapunov"}:
                if koopman.parameterization == "odo":
                    raw = koopman.diag_raw
                elif koopman.parameterization == "schur":
                    raw = koopman.schur_diag_raw
                else:
                    raw = koopman.lyap_diag_raw
                bound = max(koopman.max_real_eigenvalue - 1e-4, 1e-4)
                real_parts = -torch.tanh(raw).abs() * bound
            elif koopman.parameterization == "dissipative":
                return torch.zeros((), device=koopman.dissipative_L.device)
            else:
                real_parts = torch.linalg.eigvals(koopman.L).real
            violation = torch.relu(real_parts)
            return (violation**2).mean()

        if koopman.parameterization == "odo":
            magnitudes = (
                torch.tanh(koopman.diag_raw).abs() * koopman.max_spectral_radius
            )
        elif koopman.parameterization in {"schur", "lyapunov"}:
            raw = (
                koopman.schur_diag_raw
                if koopman.parameterization == "schur"
                else koopman.lyap_diag_raw
            )
            bound = max(
                koopman.max_spectral_radius - 1e-4,
                1e-4,
            )
            magnitudes = torch.tanh(raw).abs() * bound
        elif koopman.parameterization == "dissipative":
            return torch.zeros((), device=koopman.dissipative_L.device)
        else:
            magnitudes = torch.linalg.eigvals(koopman.K).abs()
        violation = torch.relu(magnitudes - 1.0)
        return (violation**2).mean()


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
    z = model.encode(initial)

    total_loss = torch.zeros((), device=z.device)
    time_step = getattr(model, "time_step", 1.0)
    for step in range(1, horizon + 1):
        control = None
        if sequence.has_controls:
            control = sequence.control_at(start + step - 1)
        delta_t = resolve_pair_delta_t(
            sequence,
            start + step - 1,
            default_time_step=time_step,
        )
        z = _propagate_latent(
            model.koopman,
            z,
            control=control,
            delta_t=delta_t,
        )
        target = sequence[start + step]
        decode_edge_index = target.edge_index
        decode_edge_weight = _snapshot_edge_weight(target)
        prediction = model.decoder(z, decode_edge_index, decode_edge_weight)
        if sequence.has_observation_masks:
            node_mask = sequence.observation_mask_at(start + step)
            step_loss = masked_mse_loss(prediction, target.x, node_mask)
        else:
            step_loss = nn.functional.mse_loss(prediction, target.x)
        total_loss = total_loss + step_loss
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
