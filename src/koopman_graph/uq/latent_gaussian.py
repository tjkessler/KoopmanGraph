"""Linear-Gaussian latent forecast UQ for GraphKoopman models.

Propagates a Gaussian latent state under the linear (or locally linearized)
Koopman map and forms observation-space predictive intervals by Monte Carlo
decoding. Optional mid-horizon observations apply a Kalman update in the
same process-model convention as
:class:`~koopman_graph.adaptation.KoopmanObserver`.

This is **not** Deep Probabilistic Koopman (DPK), which typically predicts
time-varying distribution parameters, and **not** a full K²VAE
(VAE encoder + KalmanNet). It is a linear-Gaussian / Kalman-refined latent
forecast peer related to the Kalman half of K²VAE-style pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    hold_last_topology_at,
    propagate_latent,
    resolve_delta_t,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.uq.common import (
    PredictionInterval,
    quantile_levels,
    snapshot_with_features,
)


@dataclass(frozen=True)
class LatentGaussianForecast:
    """Closed-form latent means and covariances for a Gaussian forecast.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles).

    Attributes
    ----------
    means : Tensor
        Latent means with shape ``(steps, num_nodes, latent_dim)``.
    covariances : Tensor
        Flattened-state covariances with shape
        ``(steps, num_nodes * latent_dim, num_nodes * latent_dim)``.
        Node blocks are stacked in row-major order matching
        ``means.reshape(steps, -1)``.
    """

    means: Tensor
    covariances: Tensor


def propagate_gaussian_covariance(
    transition: Tensor,
    covariance: Tensor,
    process_noise: Tensor | float,
) -> Tensor:
    """One-step linear-Gaussian covariance update ``A P Aᵀ + Q``.

    Parameters
    ----------
    transition : Tensor
        Discrete transition matrix ``A`` with shape ``(D, D)``.
    covariance : Tensor
        Prior covariance ``P`` with shape ``(D, D)``.
    process_noise : Tensor or float
        Process-noise covariance ``Q``. A scalar ``q`` means ``Q = q I``.

    Returns
    -------
    Tensor
        Posterior predictive covariance, symmetrized.
    """
    if transition.shape != covariance.shape:
        msg = (
            "transition and covariance must share shape; "
            f"got {tuple(transition.shape)} vs {tuple(covariance.shape)}"
        )
        raise ValueError(msg)
    dim = transition.shape[0]
    if isinstance(process_noise, (float, int)):
        q_mat = float(process_noise) * torch.eye(
            dim, dtype=covariance.dtype, device=covariance.device
        )
    else:
        q_mat = process_noise
        if q_mat.shape != covariance.shape:
            msg = (
                "process_noise tensor must match covariance shape; "
                f"got {tuple(q_mat.shape)} vs {tuple(covariance.shape)}"
            )
            raise ValueError(msg)
    updated = transition @ covariance @ transition.T + q_mat
    return 0.5 * (updated + updated.T)


def dense_nodewise_transition(k_matrix: Tensor, num_nodes: int) -> Tensor:
    """Build ``A = I_N ⊗ K`` for independent per-node dense Koopman maps.

    Parameters
    ----------
    k_matrix : Tensor
        Dense Koopman matrix ``K`` with shape ``(d, d)``.
    num_nodes : int
        Number of graph nodes ``N``.

    Returns
    -------
    Tensor
        Flattened-state transition with shape ``(N d, N d)``.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    identity = torch.eye(num_nodes, dtype=k_matrix.dtype, device=k_matrix.device)
    return torch.kron(identity, k_matrix)


class LatentGaussianKoopmanUQ:
    """Linear-Gaussian latent forecast with optional Kalman refinement.

    Composes a fitted :class:`~koopman_graph.model.GraphKoopmanModel` without
    subclassing or forking latent rollout helpers. Open-loop forecasts
    propagate ``z ↦ Kz`` (or continuous ``exp(L Δt)``) with covariance
    ``P ← A P Aᵀ + Q``. Observation-space intervals are Monte Carlo quantiles
    of decoded latent samples — not a claim of exact KF coverage under a
    nonlinear GNN decoder.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted (or seeded) model providing encode / decode / Koopman step.
    process_noise : float, optional
        Isotropic process-noise scale ``q`` with ``Q = q I``. Default ``1e-3``.
    observation_noise : float, optional
        Isotropic observation-noise scale ``r`` with ``R = r I`` on encoded
        measurement rows when ``observations`` are supplied. Default ``1e-2``.
    initial_covariance : float, optional
        Initial covariance scale ``P_0 = p_0 I``. Default ``1.0``.
    n_samples : int, optional
        Monte Carlo draws for observation-space quantiles. Default ``64``.

    Notes
    -----
    Bilinear ``control_mode`` is unsupported (state-dependent process model).
    Networked :class:`~koopman_graph.operators.GraphKoopmanOperator` uses
    :meth:`~koopman_graph.operators.GraphKoopmanOperator.effective_matrix`.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        process_noise: float = 1e-3,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1.0,
        n_samples: int = 64,
    ) -> None:
        """Validate noise scales and store the composed model.

        Notes
        -----
        Constructor parameters are documented on the class.
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
        if n_samples < 1:
            msg = f"n_samples must be >= 1; got {n_samples}"
            raise ValueError(msg)
        if getattr(model, "control_mode", "additive") == "bilinear":
            msg = (
                "LatentGaussianKoopmanUQ does not support bilinear "
                "control_mode; use additive control or an uncontrolled model"
            )
            raise ValueError(msg)

        self.model = model
        self.process_noise = float(process_noise)
        self.observation_noise = float(observation_noise)
        self.initial_covariance = float(initial_covariance)
        self.n_samples = int(n_samples)

    def forecast_latents(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        observations: Sequence[Data] | None = None,
    ) -> LatentGaussianForecast:
        """Propagate latent means and covariances for ``steps`` forecasts.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot.
        steps : int
            Number of forecast steps (must be >= 1).
        edge_index, edge_weight, controls, future_topologies, history
            Forwarded to encoding / topology resolution (same semantics as
            :meth:`~koopman_graph.model.GraphKoopmanModel.predict`).
        observations : sequence of Data or None, optional
            Optional mid-horizon observations (length ``steps``). When
            provided, each step applies a Kalman update after the predict
            step using ``latent_encode`` measurements (heuristic under
            nonlinear encoders).

        Returns
        -------
        LatentGaussianForecast
            Closed-form latent means and covariances.
        """
        if steps < 1:
            msg = f"steps must be >= 1; got {steps}"
            raise ValueError(msg)
        if observations is not None and len(observations) != steps:
            msg = (
                "observations must have length equal to steps; "
                f"got len={len(observations)}, steps={steps}"
            )
            raise ValueError(msg)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self._forecast_latents_impl(
                    initial_graph,
                    steps,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    controls=controls,
                    future_topologies=future_topologies,
                    history=history,
                    observations=observations,
                )
        finally:
            self.model.train(was_training)

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        observations: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Decode the latent-mean forecast.

        Returns
        -------
        list of Data
            Mean forecast snapshots using the model's topology contract.
        """
        interval = self.predict_interval(
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
            observations=observations,
            level=0.9,
        )
        return interval.mean

    def predict_interval(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        *args: Any,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        observations: Sequence[Data] | None = None,
        level: float = 0.9,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> PredictionInterval:
        """Return mean decode plus Monte Carlo predictive quantiles.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot for the rollout.
        steps : int
            Number of forecast steps.
        *args, **kwargs
            Accepted for :class:`~koopman_graph.uq.IntervalForecastModel`
            compatibility; unexpected keywords raise ``TypeError``.
        edge_index, edge_weight, controls, future_topologies, history
            Same semantics as :meth:`forecast_latents`.
        observations : sequence of Data or None, optional
            Optional mid-horizon Kalman updates.
        level : float, optional
            Nominal central coverage in ``(0, 1)``. Default ``0.9``.
        generator : torch.Generator or None, optional
            RNG for latent Monte Carlo draws.

        Returns
        -------
        PredictionInterval
            Mean forecast plus lower/upper empirical quantiles. The
            ``n_members`` field reports :attr:`n_samples` (Monte Carlo draws).
        """
        if args:
            msg = (
                "LatentGaussianKoopmanUQ.predict_interval takes no "
                "positional args after steps"
            )
            raise TypeError(msg)
        if kwargs:
            msg = "unexpected keyword arguments for predict_interval: " + ", ".join(
                sorted(kwargs)
            )
            raise TypeError(msg)

        lower_q, upper_q = quantile_levels(level)
        forecast = self.forecast_latents(
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
            observations=observations,
        )
        _, init_edge, init_weight = self.model.encode_rollout_origin(
            initial_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history=history,
        )
        topology_at = hold_last_topology_at(
            init_edge,
            init_weight,
            future_topologies=future_topologies,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                mean_snaps: list[Data] = []
                lower_snaps: list[Data] = []
                upper_snaps: list[Data] = []
                for step in range(steps):
                    edge_t, weight_t = topology_at(step)
                    template = Data(x=forecast.means[step], edge_index=edge_t)
                    if weight_t is not None:
                        template.edge_weight = weight_t
                    z_mean = forecast.means[step]
                    mean_x = self.model.decoder(z_mean, edge_t, weight_t)
                    samples = self._sample_latents(
                        forecast.means[step],
                        forecast.covariances[step],
                        generator=generator,
                    )
                    decoded = [
                        self.model.decoder(sample, edge_t, weight_t)
                        for sample in samples
                    ]
                    stacked = torch.stack(decoded, dim=0)
                    if stacked.shape[0] == 1:
                        lower_x = mean_x.clone()
                        upper_x = mean_x.clone()
                    else:
                        q = torch.tensor(
                            [lower_q, upper_q],
                            device=stacked.device,
                            dtype=stacked.dtype,
                        )
                        bounds = torch.quantile(stacked.float(), q, dim=0).to(
                            dtype=stacked.dtype
                        )
                        lower_x = bounds[0]
                        upper_x = bounds[1]
                    mean_snaps.append(snapshot_with_features(template, mean_x))
                    lower_snaps.append(snapshot_with_features(template, lower_x))
                    upper_snaps.append(snapshot_with_features(template, upper_x))
        finally:
            self.model.train(was_training)

        return PredictionInterval(
            mean=mean_snaps,
            lower=lower_snaps,
            upper=upper_snaps,
            level=level,
            n_members=self.n_samples,
        )

    def _forecast_latents_impl(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        *,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
        controls: Sequence[Tensor] | None,
        future_topologies: Sequence[Data] | None,
        history: Sequence[Data] | None,
        observations: Sequence[Data] | None,
    ) -> LatentGaussianForecast:
        """Propagate latent moments while the caller manages evaluation mode.

        Returns
        -------
        LatentGaussianForecast
            Per-step latent means and covariances.
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        z0, init_edge, init_weight = self.model.encode_rollout_origin(
            initial_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history=history,
        )
        z0 = z0.to(device=device, dtype=dtype)
        n_nodes = z0.shape[0]
        d = self.model.latent_dim
        state_dim = n_nodes * d

        mean = z0.reshape(-1)
        cov = self.initial_covariance * torch.eye(state_dim, dtype=dtype, device=device)
        q_mat = self.process_noise * torch.eye(state_dim, dtype=dtype, device=device)

        means = torch.empty(steps, n_nodes, d, dtype=dtype, device=device)
        covs = torch.empty(steps, state_dim, state_dim, dtype=dtype, device=device)

        topology_at = hold_last_topology_at(
            init_edge,
            init_weight,
            future_topologies=future_topologies,
        )
        control_at = None if controls is None else (lambda step: controls[step])
        default_delta_t = self.model.time_step

        for t in range(steps):
            edge_t, weight_t = topology_at(t)
            control = None if control_at is None else control_at(t)
            transition, bias = self._transition_and_bias(
                n_nodes=n_nodes,
                edge_index=edge_t,
                edge_weight=weight_t,
                control=control,
                default_delta_t=default_delta_t,
                device=device,
                dtype=dtype,
            )
            mean = transition @ mean + bias
            cov = propagate_gaussian_covariance(transition, cov, q_mat)

            if observations is not None:
                mean, cov = self._kalman_update_latent_encode(
                    mean,
                    cov,
                    observations[t],
                    device=device,
                    dtype=dtype,
                )

            means[t] = mean.reshape(n_nodes, d)
            covs[t] = cov

        return LatentGaussianForecast(means=means, covariances=covs)

    def _transition_and_bias(
        self,
        *,
        n_nodes: int,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        control: Tensor | None,
        default_delta_t: float | Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Build transition and additive control bias on flattened state.

        Returns
        -------
        tuple of Tensor
            Dense state transition and flattened additive bias.
        """
        koopman = self.model.koopman
        d = self.model.latent_dim
        state_dim = n_nodes * d
        delta = resolve_delta_t(None, default_delta_t=default_delta_t)

        if isinstance(koopman, GraphKoopmanOperator):
            transition = koopman.effective_matrix(
                edge_index,
                n_nodes,
                edge_weight=edge_weight,
            ).to(device=device, dtype=dtype)
        elif isinstance(koopman, ContinuousKoopmanOperator):
            phi = torch.linalg.matrix_exp(koopman.L.detach() * float(delta))
            transition = dense_nodewise_transition(
                phi.to(device=device, dtype=dtype), n_nodes
            )
        elif isinstance(koopman, KoopmanOperator):
            k_mat = koopman.matrix.detach().to(device=device, dtype=dtype)
            transition = dense_nodewise_transition(k_mat, n_nodes)
        else:
            msg = (
                "LatentGaussianKoopmanUQ requires a built-in discrete, "
                "continuous, or graph Koopman operator"
            )
            raise TypeError(msg)

        bias = torch.zeros(state_dim, dtype=dtype, device=device)
        if control is not None and getattr(self.model, "control_dim", 0) > 0:
            z0 = torch.zeros(n_nodes, d, dtype=dtype, device=device)
            z1 = propagate_latent(
                self.model.koopman,
                z0,
                control=control,
                delta_t=self.model.resolve_delta_t(delta),
                default_delta_t=default_delta_t,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            bias = z1.reshape(-1)
        return transition, bias

    def _kalman_update_latent_encode(
        self,
        mean: Tensor,
        cov: Tensor,
        observation: Data,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Apply a Kalman update using an encoded latent measurement.

        Returns
        -------
        tuple of Tensor
            Updated latent mean and covariance.
        """
        with torch.no_grad():
            z_meas = self.model.encode(observation).to(device=device, dtype=dtype)
        y = z_meas.reshape(-1)
        state_dim = mean.shape[0]
        h_mat = torch.eye(state_dim, dtype=dtype, device=device)
        r_mat = self.observation_noise * torch.eye(
            state_dim, dtype=dtype, device=device
        )
        innov_cov = h_mat @ cov @ h_mat.T + r_mat
        gain = torch.linalg.solve(innov_cov, h_mat @ cov).T
        innov = y - h_mat @ mean
        mean_upd = mean + gain @ innov
        eye = torch.eye(state_dim, dtype=dtype, device=device)
        cov_upd = (eye - gain @ h_mat) @ cov
        cov_upd = 0.5 * (cov_upd + cov_upd.T)
        return mean_upd, cov_upd

    def _sample_latents(
        self,
        mean: Tensor,
        covariance: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Draw ``n_samples`` latent states from ``N(mean, covariance)``.

        Parameters
        ----------
        mean : Tensor
            Latent mean with shape ``(num_nodes, latent_dim)``.
        covariance : Tensor
            Flattened covariance with shape ``(N d, N d)``.
        generator : torch.Generator or None
            Optional RNG.

        Returns
        -------
        Tensor
            Samples with shape ``(n_samples, num_nodes, latent_dim)``.
        """
        n_nodes, d = mean.shape
        flat_mean = mean.reshape(-1)
        state_dim = flat_mean.shape[0]
        # Jitter for numerical Cholesky stability.
        jitter = 1e-6 * torch.eye(
            state_dim, dtype=covariance.dtype, device=covariance.device
        )
        scale = torch.linalg.cholesky(covariance + jitter)
        noise = torch.randn(
            self.n_samples,
            state_dim,
            dtype=covariance.dtype,
            device=covariance.device,
            generator=generator,
        )
        samples = flat_mean.unsqueeze(0) + noise @ scale.T
        return samples.reshape(self.n_samples, n_nodes, d)
