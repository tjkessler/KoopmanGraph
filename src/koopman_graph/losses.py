"""Loss functions for Koopman graph dynamics training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.graph_utils import (
    KoopmanPropagator,
    autoregressive_latent_rollout,
    inverse_propagate_latent,
    propagate_latent,
    snapshot_topology_at,
)
from koopman_graph.protocols import DynamicsMode, TrainableKoopmanModel


def masked_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute mean squared error over observed nodes only.

    Averages squared errors over all feature channels of masked nodes::

        sum_{n in O, f} (pred_{n,f} - target_{n,f})^2 / (|O| * feature_dim)

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
    expanded = node_mask.unsqueeze(-1)
    diff_sq = (prediction - target) ** 2
    denom = expanded.sum() * prediction.shape[-1]
    if denom <= 0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return (diff_sq * expanded).sum() / denom


class ForwardConsistencyLoss(nn.Module):
    """Penalize deviation from linear latent evolution under the Koopman operator.

    For latent row encodings ``z_t`` and ``z_{t+1}``, the loss is the element-wise
    mean squared error between the propagated state ``z_t @ K.T`` (plus optional
    control) and ``z_{t+1}``:

    .. math::

        \\mathcal{L}_{\\mathrm{fc}}
        = \\mathrm{mean}\\big((z_t K^{\\top} - z_{t+1})^2\\big)

    This matches ``torch.nn.functional.mse_loss`` (average over all entries),
    not an unnormalized Frobenius or Euclidean squared norm.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operators.KoopmanOperator`.
    """

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
        koopman: KoopmanPropagator,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
        default_delta_t: float | Tensor = 1.0,
        mask: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
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
        default_delta_t : float or Tensor, optional
            Fallback when ``delta_t is None`` in continuous mode. Pass the
            model ``time_step`` for model-backed training; bare calls default
            to ``1.0``.
        mask : Tensor or None, optional
            Optional boolean node mask with shape ``(num_nodes,)``. When set,
            MSE is averaged over observed nodes only.
        edge_index : Tensor or None, optional
            Topology for networked operators (ignored by per-node operators).
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Scalar mean-squared error between propagated ``z_t`` and ``z_t1``.
        """
        z_pred = propagate_latent(
            koopman,
            z_t,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if mask is None:
            return nn.functional.mse_loss(z_pred, z_t1)
        return masked_mse_loss(z_pred, z_t1, mask)


class BackwardConsistencyLoss(nn.Module):
    """Penalize deviation from inverse linear latent evolution under **K**.

    For latent row encodings ``z_t`` and ``z_{t+1}`` with forward dynamics
    ``z_{t+1} = z_t @ K.T``, the backward consistency term is the element-wise
    mean squared error between ``z_t`` and the inverse propagation of
    ``z_{t+1}`` (``z_{t+1} @ (K^{\\dagger}).T`` after removing control):

    .. math::

        \\mathcal{L}_{\\mathrm{bc}}
        = \\mathrm{mean}\\big((z_t - z_{t+1} (K^{\\dagger})^{\\top})^2\\big)

    where ``K^{\\dagger}`` denotes an inverse or pseudo-inverse of ``K``.
    As with the forward term, this is ``mse_loss`` (mean over entries), not
    ``\\|\\cdot\\|_F^2``.

    Trade-offs
    ----------
    **Benefits:** Enforces bidirectional linear consistency, which can improve
    Koopman operator identifiability and training stability when paired with
    the forward consistency term (see Lusch et al., 2018; Mezić, 2021).

    **Costs:** Dense unconstrained ``K`` requires a matrix inverse or
    pseudo-inverse. The ODO parameterization
    (:attr:`~koopman_graph.operators.KoopmanOperator.parameterization`
    ``"odo"``) provides a cheap exact factorized inverse of the diagonal
    factors (discrete ODO still lacks a structural ε-interior certificate).
    For dense ``K``, sequence-level training precomputes the inverse once per
    step rather than per snapshot pair.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operators.KoopmanOperator`.
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
        default_delta_t: float | Tensor = 1.0,
        mask: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute backward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : :class:`~koopman_graph.operators.KoopmanOperator`
            Learnable linear propagator whose inverse step is applied to
            ``z_t1``.
        control : Tensor or None, optional
            Control input that drove the forward transition from ``t`` to
            ``t+1``.
        inverse_matrix : Tensor or None, optional
            Precomputed dense inverse matrix reused across pair evaluations.
        delta_t : float, Tensor, or None, optional
            Integration interval for continuous-time operators. Ignored for
            discrete operators.
        default_delta_t : float or Tensor, optional
            Fallback when ``delta_t is None`` in continuous mode. Pass the
            model ``time_step`` for model-backed training; bare calls default
            to ``1.0``.
        mask : Tensor or None, optional
            Optional boolean node mask with shape ``(num_nodes,)``. When set,
            MSE is averaged over observed nodes only.
        edge_index : Tensor or None, optional
            Topology for networked operators (ignored by per-node operators).
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``z_t`` and the inverse
            propagation of ``z_t1``.
        """
        z_recovered = inverse_propagate_latent(
            koopman,
            z_t1,
            control=control,
            inverse_matrix=inverse_matrix,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if mask is None:
            return nn.functional.mse_loss(z_recovered, z_t)
        return masked_mse_loss(z_recovered, z_t, mask)


class EigenvalueRegularizationLoss(nn.Module):
    """Penalize Koopman eigenvalues outside the stable region.

    Implements a hinge-style eigenloss. Discrete operators penalize magnitudes
    outside the unit circle:

    .. math::

        \\mathcal{L}_{\\mathrm{eig}} =
        \\mathrm{mean}\\big(\\max(|\\lambda_i| - 1, 0)^2\\big)

    Continuous generators penalize positive real parts.

    Path selection:

    - ``"dense"`` and ``"odo"`` — ``eigvals`` on
      :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix` (true
      spectrum). Continuous ODO can be Hurwitz-unstable even when diagonal
      factors are negative, so the factor
      :meth:`~koopman_graph.operators.KoopmanOperatorContract.bound_metric`
      must not be used here.
    - ``"schur"`` / ``"lyapunov"`` — cheap
      :meth:`~koopman_graph.operators.KoopmanOperatorContract.bound_metric`
      (closed-form certified bound).
    - ``"dissipative"`` — always zero (structurally Hurwitz / contractive).

    Trade-offs
    ----------
    **Benefits:** Encourages stability without hard-constraining dense
    operators. For continuous ODO, penalizes the assembled generator's true
    spectrum (DeepKoopFormer-style eigenloss literature).

    **Costs:** ``"dense"`` / ``"odo"`` require ``torch.linalg.eigvals`` each
    evaluation. Prefer structural modes (``schur`` / ``lyapunov`` /
    ``dissipative``) when a cheap ``bound_metric`` path is enough.

    Notes
    -----
    Pass ``dynamics_mode`` matching the operator semantics (the training
    loop uses :attr:`~koopman_graph.model.GraphKoopmanModel.dynamics_mode`).
    Defaults to ``"discrete"`` for standalone call sites.
    """

    def forward(
        self,
        koopman: KoopmanPropagator,
        *,
        dynamics_mode: DynamicsMode = "discrete",
    ) -> Tensor:
        """Compute the stability eigenvalue hinge penalty.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Operator whose eigenvalues (or bound metric) are penalized.
        dynamics_mode : {"discrete", "continuous"}, optional
            Selects the discrete unit-circle hinge vs continuous Hurwitz hinge.
            Default is ``"discrete"``.

        Returns
        -------
        Tensor
            Scalar hinge penalty.
        """
        if dynamics_mode not in {"discrete", "continuous"}:
            msg = (
                "dynamics_mode must be 'discrete' or 'continuous', "
                f"got {dynamics_mode!r}"
            )
            raise ValueError(msg)

        if koopman.parameterization == "dissipative":
            return torch.zeros((), device=koopman.matrix.device)

        if koopman.parameterization in {"schur", "lyapunov"}:
            bound = koopman.bound_metric()
            if dynamics_mode == "continuous":
                violation = torch.relu(bound)
            else:
                violation = torch.relu(bound - 1.0)
            return violation**2

        eigenvalues = torch.linalg.eigvals(koopman.matrix)
        if dynamics_mode == "continuous":
            violation = torch.relu(eigenvalues.real)
        else:
            violation = torch.relu(eigenvalues.abs() - 1.0)
        return (violation**2).mean()


def rollout_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start: int = 0,
) -> Tensor:
    """Compute autoregressive rollout reconstruction loss from one start snapshot.

    Encodes ``sequence[start]`` once via
    :meth:`~koopman_graph.protocols.TrainableKoopmanModel.encode`, advances the
    latent state with the model's Koopman operator for ``horizon`` steps, and
    compares decoded predictions to the observed snapshots
    ``sequence[start + 1 : start + horizon + 1]``. This term aligns training
    with :meth:`~koopman_graph.model.GraphKoopmanModel.predict` via the shared
    :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` primitive.
    Decode topology uses **teacher target** edges (per-step snapshot topology),
    whereas ``predict`` uses hold-last unless ``future_topologies`` are supplied
    — see :mod:`koopman_graph.graph_utils`.

    Parameters
    ----------
    model : :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        Trainable model exposing ``encode``, ``resolve_delta_t``, ``koopman``,
        and ``decoder``. :class:`~koopman_graph.model.GraphKoopmanModel` is the
        intended implementer; no encoder-only fallback is used.
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
    encode_at = getattr(model, "encode_at", None)
    z = encode_at(sequence, start) if callable(encode_at) else model.encode(initial)

    time_step = float(model.resolve_delta_t(None))
    targets = [sequence[start + step] for step in range(1, horizon + 1)]

    rollout = autoregressive_latent_rollout(
        model.koopman,
        model.decoder,
        z,
        steps=horizon,
        topology_at=snapshot_topology_at(targets),
        control_at=(
            None
            if not sequence.has_controls
            else (lambda step: sequence.control_at(start + step))
        ),
        delta_t_at=lambda step: resolve_pair_delta_t(
            sequence,
            start + step,
            default_time_step=time_step,
        ),
        default_delta_t=time_step,
    )

    total_loss = torch.zeros((), device=z.device)
    for step, (prediction, _, _) in enumerate(rollout):
        target = targets[step]
        if sequence.has_observation_masks:
            node_mask = sequence.observation_mask_at(start + step + 1)
            step_loss = masked_mse_loss(prediction, target.x, node_mask)
        else:
            step_loss = nn.functional.mse_loss(prediction, target.x)
        total_loss = total_loss + step_loss
    return total_loss / horizon


def rollout_multi_start_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
) -> Tensor:
    """Average rollout reconstruction loss over multiple start snapshots.

    Parameters
    ----------
    model : :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        Trainable model accepted by :func:`rollout_sequence_loss` (also uses
        ``parameters`` for device placement).
        :class:`~koopman_graph.model.GraphKoopmanModel` is the intended
        implementer.
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
