"""Koopman latent-space Kalman observer façade for partial observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.adaptation.impute import graph_diffuse_impute
from koopman_graph.adaptation.kalman import FilterResult
from koopman_graph.data import GraphSnapshotSequence, resolve_pair_delta_t
from koopman_graph.data.delay_windows import apply_observation_mask_to_features
from koopman_graph.graph_utils import propagate_latent, snapshot_edge_weight
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)

if TYPE_CHECKING:
    from koopman_graph.model import GraphKoopmanModel

ObservationModel = Literal["latent_encode", "decoder_jacobian"]


def _as_bool_mask(mask: Tensor | None, num_nodes: int) -> Tensor:
    """Return a length-``num_nodes`` boolean mask (all True when ``None``).

    Parameters
    ----------
    mask : Tensor or None
        Optional boolean observation mask with shape ``(num_nodes,)``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Tensor
        Boolean mask with shape ``(num_nodes,)``.
    """
    if mask is None:
        return torch.ones(num_nodes, dtype=torch.bool)
    return mask.bool()


class KoopmanObserver:
    """Latent-space Kalman filter / smoother for masked graph sequences.

    Latent dynamics use the library **row** convention::

        z_{t+1} = z_t @ K.T   # z: (num_nodes, latent_dim)

    Equivalently, each node's latent column evolves as ``s⁺ = K s``. On the
    flattened state ``x = z.reshape(-1)`` (row-major), the discrete process
    matrix is ``A = I_N ⊗ K`` for a dense
    :class:`~koopman_graph.operators.KoopmanOperator` (independent nodes).
    That process model is exactly linear-Gaussian once ``Q`` is set — not an
    approximate linearization. Observation handling depends on
    :attr:`observation_model`:

    ``"latent_encode"`` (default)
        Warm-start masked features (optional graph diffusion), encode to a
        noisy latent measurement, and keep **selected** rows of ``H = I`` for
        observed node blocks (selection matrix ``H_t = S_t``, not full ``I``
        under masks). Fast and CI-friendly, but **heuristic**:
        ``encode(impute(x))`` is not a true inverse of the decoder.

    ``"decoder_jacobian"``
        Measure node features with an EKF-style local linearization
        ``H = ∂decode/∂z`` (autograd), **dropping** rows for unobserved
        nodes. Aligns with sparse-measurement Koopman observer synthesis, but
        is **costly** (``O((N d)(N F))`` Jacobian work per step), fragile for
        deep GNN decoders / hybrid physics latents, and only locally valid.

    Notes
    -----
    Do **not** claim an exact Kalman filter in observation space when using
    a nonlinear decoder — the process model is exact (for linear ``K``);
    the observation model is EKF-local or encode-heuristic.

    Bilinear ``control_mode`` is unsupported (process model would be
    state-dependent). Networked
    :class:`~koopman_graph.operators.GraphKoopmanOperator` uses
    :meth:`~koopman_graph.operators.GraphKoopmanOperator.effective_matrix`.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        process_noise: float = 1e-3,
        observation_noise: float = 1e-2,
        observation_model: ObservationModel = "latent_encode",
        graph_diffusion_warm_start: bool = False,
        diffusion_iterations: int = 8,
        initial_covariance: float = 1.0,
    ) -> None:
        """Initialize a Koopman latent observer.

        Parameters
        ----------
        model : GraphKoopmanModel
            Fitted (or seeded) model providing encode / decode / Koopman step.
        process_noise : float, optional
            Isotropic process-noise scale ``q`` with ``Q = q I``. Default
            ``1e-3``.
        observation_noise : float, optional
            Isotropic observation-noise scale ``r`` with ``R = r I`` on the
            active measurement rows. Default ``1e-2``.
        observation_model : {"latent_encode", "decoder_jacobian"}, optional
            Observation linearization strategy. Default ``"latent_encode"``.
        graph_diffusion_warm_start : bool, optional
            When ``True``, fill masked features by neighbor averaging before
            encoding (``latent_encode``) or as a decode prior. Default
            ``False``.
        diffusion_iterations : int, optional
            Neighbor-average sweeps when warm-start is enabled. Default ``8``.
        initial_covariance : float, optional
            Initial covariance scale ``P_0 = p_0 I``. Default ``1.0``.

        Raises
        ------
        ValueError
            If noise scales are non-positive or ``observation_model`` is
            invalid.
        """
        if process_noise <= 0.0:
            msg = f"process_noise must be positive, got {process_noise}"
            raise ValueError(msg)
        if observation_noise <= 0.0:
            msg = f"observation_noise must be positive, got {observation_noise}"
            raise ValueError(msg)
        if initial_covariance <= 0.0:
            msg = f"initial_covariance must be positive, got {initial_covariance}"
            raise ValueError(msg)
        if observation_model not in {"latent_encode", "decoder_jacobian"}:
            msg = (
                "observation_model must be 'latent_encode' or "
                f"'decoder_jacobian', got {observation_model!r}"
            )
            raise ValueError(msg)
        if getattr(model, "control_mode", "additive") == "bilinear":
            msg = (
                "KoopmanObserver does not support bilinear control_mode; "
                "use additive control or an uncontrolled model"
            )
            raise ValueError(msg)

        self.model = model
        self.process_noise = float(process_noise)
        self.observation_noise = float(observation_noise)
        self.observation_model: ObservationModel = observation_model
        self.graph_diffusion_warm_start = bool(graph_diffusion_warm_start)
        self.diffusion_iterations = int(diffusion_iterations)
        self.initial_covariance = float(initial_covariance)

    def filter(self, sequence: GraphSnapshotSequence) -> FilterResult:
        """Forward Kalman filter over ``sequence`` using observation masks.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Trajectory; optional ``observation_masks`` mark measured nodes.

        Returns
        -------
        FilterResult
            Filtered latent means and covariances.
        """
        means, covs, _, _ = self._forward_filter(sequence)
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        return FilterResult(
            latents=means.reshape(-1, n_nodes, d),
            covariances=covs,
        )

    def smooth(self, sequence: GraphSnapshotSequence) -> FilterResult:
        """Rauch–Tung–Striebel smoother over ``sequence``.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Trajectory; optional ``observation_masks`` mark measured nodes.

        Returns
        -------
        FilterResult
            Smoothed latent means and covariances.
        """
        means, covs, pred_means, pred_covs = self._forward_filter(sequence)
        # Use the first-step transition for RTS gains when A is constant;
        # rebuild per-step when continuous / graph topology changes.
        transitions = self._transition_matrices(sequence)
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        t_steps = means.shape[0]
        sm_means = means.clone()
        sm_covs = covs.clone()
        for t in range(t_steps - 2, -1, -1):
            a_mat = transitions[t]
            gain = torch.linalg.solve(pred_covs[t + 1], a_mat @ covs[t].T).T
            sm_means[t] = means[t] + gain @ (sm_means[t + 1] - pred_means[t + 1])
            sm_covs[t] = covs[t] + gain @ (sm_covs[t + 1] - pred_covs[t + 1]) @ gain.T
            sm_covs[t] = 0.5 * (sm_covs[t] + sm_covs[t].T)
        return FilterResult(
            latents=sm_means.reshape(-1, n_nodes, d),
            covariances=sm_covs,
        )

    def impute(
        self,
        sequence: GraphSnapshotSequence,
        *,
        use_smoother: bool = True,
    ) -> GraphSnapshotSequence:
        """Decode filtered/smoothed latents and fill masked node features.

        Observed node features are preserved; unobserved entries are replaced
        by the decoder reconstruction of the estimated latent state.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Partially observed trajectory.
        use_smoother : bool, optional
            When ``True`` (default), run :meth:`smooth`; otherwise
            :meth:`filter`.

        Returns
        -------
        GraphSnapshotSequence
            Clone with imputed ``x`` values. Original ``observation_masks``
            are retained for provenance.
        """
        result = self.smooth(sequence) if use_smoother else self.filter(sequence)
        snapshots: list[Data] = []
        for t, snap in enumerate(sequence):
            edge_index = snap.edge_index
            edge_weight = snapshot_edge_weight(snap)
            z = result.latents[t]
            with torch.no_grad():
                x_hat = self.model.decoder(z, edge_index, edge_weight)
            x_out = snap.x.clone()
            if sequence.has_observation_masks:
                mask = sequence.observation_mask_at(t)
                x_out = torch.where(mask.unsqueeze(1), x_out, x_hat)
            else:
                x_out = x_hat
            data = Data(x=x_out, edge_index=edge_index)
            if edge_weight is not None:
                data.edge_weight = edge_weight
            snapshots.append(data)
        return GraphSnapshotSequence(
            snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs.clone()
            ),
            timestamps=(
                None if sequence.timestamps is None else sequence.timestamps.clone()
            ),
            observation_masks=(
                None
                if sequence.observation_masks is None
                else sequence.observation_masks.clone()
            ),
        )

    def _forward_filter(
        self,
        sequence: GraphSnapshotSequence,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shared forward pass returning filter and one-step predict caches.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory.

        Returns
        -------
        tuple of Tensor
            Filtered means, filtered covariances, predicted means, and
            predicted covariances on the flattened latent state.
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        state_dim = n_nodes * d
        t_steps = len(sequence)

        transitions = self._transition_matrices(sequence)
        biases = self._control_biases(
            sequence, state_dim=state_dim, device=device, dtype=dtype
        )
        q_mat = self.process_noise * torch.eye(state_dim, dtype=dtype, device=device)

        means = torch.empty(t_steps, state_dim, dtype=dtype, device=device)
        covs = torch.empty(t_steps, state_dim, state_dim, dtype=dtype, device=device)
        pred_means = torch.empty(t_steps, state_dim, dtype=dtype, device=device)
        pred_covs = torch.empty(
            t_steps, state_dim, state_dim, dtype=dtype, device=device
        )

        x_prev = self._initial_state(sequence, device=device, dtype=dtype)
        p_prev = self.initial_covariance * torch.eye(
            state_dim, dtype=dtype, device=device
        )
        eye = torch.eye(state_dim, dtype=dtype, device=device)

        for t in range(t_steps):
            if t == 0:
                x_pred = x_prev
                p_pred = p_prev
            else:
                x_pred = transitions[t - 1] @ x_prev + biases[t - 1]
                p_pred = transitions[t - 1] @ p_prev @ transitions[t - 1].T + q_mat
            pred_means[t] = x_pred
            pred_covs[t] = p_pred

            y, h_mat, r_mat = self._measurement(
                sequence, t, x_pred, device=device, dtype=dtype
            )
            innov_cov = h_mat @ p_pred @ h_mat.T + r_mat
            # Solve H P for gain = P H^T S^{-1} via S^{-1} (H P).
            gain = torch.linalg.solve(innov_cov, h_mat @ p_pred).T
            innov = y - h_mat @ x_pred
            x_filt = x_pred + gain @ innov
            p_filt = (eye - gain @ h_mat) @ p_pred
            p_filt = 0.5 * (p_filt + p_filt.T)
            means[t] = x_filt
            covs[t] = p_filt
            x_prev = x_filt
            p_prev = p_filt

        return means, covs, pred_means, pred_covs

    def _transition_matrices(self, sequence: GraphSnapshotSequence) -> list[Tensor]:
        """Return per-interval discrete transition matrices on flattened states.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory (topology / timestamps may vary per step).

        Returns
        -------
        list of Tensor
            Transition matrices with length ``T - 1`` and shape
            ``(N d, N d)`` each.
        """
        koopman = self.model.koopman
        n_nodes = sequence.num_nodes
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        matrices: list[Tensor] = []
        for t in range(len(sequence) - 1):
            snap = sequence[t]
            edge_index = snap.edge_index
            edge_weight = snapshot_edge_weight(snap)
            if isinstance(koopman, GraphKoopmanOperator):
                a_mat = koopman.effective_matrix(
                    edge_index,
                    n_nodes,
                    edge_weight=edge_weight,
                )
            elif isinstance(koopman, ContinuousKoopmanOperator):
                delta = resolve_pair_delta_t(
                    sequence,
                    t,
                    default_time_step=float(self.model.time_step),
                )
                phi = torch.linalg.matrix_exp(koopman.L.detach() * float(delta))
                a_mat = torch.kron(
                    torch.eye(n_nodes, dtype=dtype, device=device),
                    phi.to(device=device, dtype=dtype),
                )
            elif isinstance(koopman, KoopmanOperator):
                k_mat = koopman.matrix.detach().to(device=device, dtype=dtype)
                identity = torch.eye(n_nodes, dtype=dtype, device=device)
                a_mat = torch.kron(identity, k_mat)
            else:
                # Protocol-only injection: finite-difference the advance map.
                a_mat = self._finite_diff_transition(
                    sequence,
                    t,
                    device=device,
                    dtype=dtype,
                )
            matrices.append(a_mat.to(device=device, dtype=dtype))
        return matrices

    def _finite_diff_transition(
        self,
        sequence: GraphSnapshotSequence,
        t: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Approximate ``A`` by finite differences of ``propagate_latent``.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory.
        t : int
            Source timestep index for the interval ``t -> t+1``.
        device : torch.device
            Computation device.
        dtype : torch.dtype
            Computation dtype.

        Returns
        -------
        Tensor
            Approximate transition matrix with shape ``(N d, N d)``.
        """
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        state_dim = n_nodes * d
        snap = sequence[t]
        edge_index = snap.edge_index
        edge_weight = snapshot_edge_weight(snap)
        control = None if sequence.control_inputs is None else sequence.control_at(t)
        delta = resolve_pair_delta_t(
            sequence,
            t,
            default_time_step=float(self.model.time_step),
        )
        eps = 1e-4
        base = torch.zeros(n_nodes, d, dtype=dtype, device=device)
        with torch.no_grad():
            f0 = propagate_latent(
                self.model.koopman,
                base,
                control=control,
                delta_t=self.model.resolve_delta_t(delta),
                default_delta_t=self.model.time_step,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ).reshape(-1)
            columns: list[Tensor] = []
            for j in range(state_dim):
                pert = base.reshape(-1).clone()
                pert[j] += eps
                f1 = propagate_latent(
                    self.model.koopman,
                    pert.reshape(n_nodes, d),
                    control=control,
                    delta_t=self.model.resolve_delta_t(delta),
                    default_delta_t=self.model.time_step,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                ).reshape(-1)
                columns.append((f1 - f0) / eps)
        return torch.stack(columns, dim=1)

    def _control_biases(
        self,
        sequence: GraphSnapshotSequence,
        *,
        state_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        """Additive control offsets on the flattened state for each interval.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory that may carry controls.
        state_dim : int
            Flattened latent dimension ``N * d``.
        device : torch.device
            Computation device.
        dtype : torch.dtype
            Computation dtype.

        Returns
        -------
        list of Tensor
            Bias vectors with length ``T - 1`` and shape ``(state_dim,)``.
        """
        biases: list[Tensor] = []
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        for t in range(len(sequence) - 1):
            if sequence.control_inputs is None or self.model.control_dim == 0:
                biases.append(torch.zeros(state_dim, dtype=dtype, device=device))
                continue
            control = sequence.control_at(t)
            snap = sequence[t]
            z0 = torch.zeros(n_nodes, d, dtype=dtype, device=device)
            with torch.no_grad():
                z1 = propagate_latent(
                    self.model.koopman,
                    z0,
                    control=control,
                    delta_t=self.model.resolve_delta_t(
                        resolve_pair_delta_t(
                            sequence,
                            t,
                            default_time_step=float(self.model.time_step),
                        )
                    ),
                    default_delta_t=self.model.time_step,
                    edge_index=snap.edge_index,
                    edge_weight=snapshot_edge_weight(snap),
                )
            biases.append(z1.reshape(-1))
        return biases

    def _initial_state(
        self,
        sequence: GraphSnapshotSequence,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Encode the first snapshot (with optional mask warm-start).

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory.
        device : torch.device
            Computation device.
        dtype : torch.dtype
            Computation dtype.

        Returns
        -------
        Tensor
            Flattened prior latent mean with shape ``(N d,)``.
        """
        snap = sequence[0]
        mask = (
            sequence.observation_mask_at(0)
            if sequence.has_observation_masks
            else torch.ones(sequence.num_nodes, dtype=torch.bool, device=device)
        )
        x = self._prepare_features(snap.x.to(device=device, dtype=dtype), mask, snap)
        data = Data(x=x, edge_index=snap.edge_index.to(device))
        ew = snapshot_edge_weight(snap)
        if ew is not None:
            data.edge_weight = ew.to(device=device, dtype=dtype)
        with torch.no_grad():
            z = self.model.encode(data)
        return z.reshape(-1).to(device=device, dtype=dtype)

    def _prepare_features(self, x: Tensor, mask: Tensor, snap: Data) -> Tensor:
        """Optionally diffuse-impute, then zero residual unobserved rows.

        Parameters
        ----------
        x : Tensor
            Node features with shape ``(num_nodes, feature_dim)``.
        mask : Tensor
            Boolean observation mask with shape ``(num_nodes,)``.
        snap : Data
            Snapshot providing ``edge_index`` for diffusion warm-start.

        Returns
        -------
        Tensor
            Prepared features for encoding or Jacobian evaluation.
        """
        prepared = x
        if self.graph_diffusion_warm_start and not bool(mask.all()):
            prepared = graph_diffuse_impute(
                prepared,
                mask,
                snap.edge_index,
                iterations=self.diffusion_iterations,
            )
        if self.observation_model == "latent_encode":
            # Encoder still sees zeros on remaining holes when diffusion is off.
            prepared = apply_observation_mask_to_features(prepared, mask)
        return prepared

    def _measurement(
        self,
        sequence: GraphSnapshotSequence,
        t: int,
        x_pred: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build measurement ``y``, Jacobian ``H``, and ``R`` at time ``t``.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Input trajectory.
        t : int
            Timestep index.
        x_pred : Tensor
            Predicted flattened latent state with shape ``(N d,)``.
        device : torch.device
            Computation device.
        dtype : torch.dtype
            Computation dtype.

        Returns
        -------
        tuple of Tensor
            Observed measurement vector, observation matrix, and observation
            noise covariance for the active rows.
        """
        snap = sequence[t]
        n_nodes = sequence.num_nodes
        d = self.model.latent_dim
        mask = _as_bool_mask(
            sequence.observation_mask_at(t) if sequence.has_observation_masks else None,
            n_nodes,
        ).to(device=device)

        if self.observation_model == "latent_encode":
            x = self._prepare_features(
                snap.x.to(device=device, dtype=dtype), mask, snap
            )
            data = Data(x=x, edge_index=snap.edge_index.to(device))
            ew = snapshot_edge_weight(snap)
            if ew is not None:
                data.edge_weight = ew.to(device=device, dtype=dtype)
            with torch.no_grad():
                z_meas = self.model.encode(data)
            y_full = z_meas.reshape(-1)
            h_full = torch.eye(n_nodes * d, dtype=dtype, device=device)
            node_expand = mask.repeat_interleave(d)
            return self._select_observed(y_full, h_full, node_expand)

        # decoder_jacobian
        feat_dim = int(snap.x.shape[1])
        z_mat = x_pred.detach().reshape(n_nodes, d).requires_grad_(True)
        edge_index = snap.edge_index.to(device)
        edge_weight = snapshot_edge_weight(snap)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=device, dtype=dtype)
        x_hat = self.model.decoder(z_mat, edge_index, edge_weight)
        y_full = snap.x.to(device=device, dtype=dtype).reshape(-1)
        # Autograd Jacobian of flatten(decode(z)) w.r.t. flatten(z).
        flat_pred = x_hat.reshape(-1)
        h_rows: list[Tensor] = []
        for i in range(flat_pred.numel()):
            grads = torch.autograd.grad(
                flat_pred[i],
                z_mat,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grads is None:
                h_rows.append(torch.zeros(n_nodes * d, dtype=dtype, device=device))
            else:
                h_rows.append(grads.reshape(-1))
        h_full = torch.stack(h_rows, dim=0)
        node_expand = mask.repeat_interleave(feat_dim)
        return self._select_observed(y_full, h_full, node_expand)

    def _select_observed(
        self,
        y_full: Tensor,
        h_full: Tensor,
        observed_rows: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Restrict measurements to observed rows; build matching ``R``.

        Parameters
        ----------
        y_full : Tensor
            Full measurement vector.
        h_full : Tensor
            Full observation matrix.
        observed_rows : Tensor
            Boolean mask over measurement rows.

        Returns
        -------
        tuple of Tensor
            Restricted ``y``, ``H``, and isotropic ``R``.

        Raises
        ------
        ValueError
            If no measurement rows are observed.
        """
        if not bool(observed_rows.any()):
            msg = "observation mask has no observed entries at this timestep"
            raise ValueError(msg)
        y = y_full[observed_rows]
        h_mat = h_full[observed_rows]
        r_mat = self.observation_noise * torch.eye(
            y.shape[0],
            dtype=y.dtype,
            device=y.device,
        )
        return y, h_mat, r_mat
