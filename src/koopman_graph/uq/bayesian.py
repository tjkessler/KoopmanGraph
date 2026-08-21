"""Diagonal Laplace Bayesian UQ over dense Koopman operator factors.

Fits an approximate Gaussian posterior over **linear factor** entries only
(``K`` for per-node dense operators; ``K_self`` / ``K_nbr`` and optional
``K_bwd`` for graph operators). Encoder and decoder weights remain
point-estimated. The curvature is a **diagonal** generalized Gauss–Newton /
empirical Fisher approximation of the one-step latent mean-squared error
(MSE) likelihood plus an isotropic Gaussian prior — not a full
:math:`P \\times P` Hessian. Monte Carlo bands are approximate **credible**
intervals under that diagonal Laplace posterior, **not** a finite-sample
coverage guarantee.

This peer is **not** a Bayesian neural network (BNN) over the encoder or
decoder, **not** Deep Probabilistic Koopman (DPK), and **not** a
:math:`K^2`VAE reimplementation. Heterogeneous operators are unsupported in
v1; use :class:`~koopman_graph.uq.ConformalKoopmanUQ` for hetero intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    call_topology_at,
    hold_last_topology_at,
    pack_rollout_snapshots,
    propagate_latent,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)
from koopman_graph.graph_utils.topology import (
    random_walk_normalized_adjacency_matvec,
    symmetric_normalized_adjacency_matvec,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.validation import validate_controls
from koopman_graph.nn.predicted_topology import resolve_rollout_topology_at
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator
from koopman_graph.uq.common import (
    PredictionInterval,
    quantile_levels,
    snapshot_with_features,
)

CalibrationSequence = Sequence[Data]


@dataclass(frozen=True)
class LaplaceFactorSpec:
    """Layout of one dense factor block inside a flattened parameter vector.

    Attributes
    ----------
    name : str
        Factor name (``"K"``, ``"K_self"``, ``"K_nbr"``, or ``"K_bwd"``).
    shape : tuple of int
        Dense matrix shape, typically ``(d, d)``.
    offset : int
        Starting index in the flattened parameter vector.
    numel : int
        Number of scalar entries (``prod(shape)``).
    """

    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int


@dataclass(frozen=True)
class LaplacePosterior:
    """Diagonal Laplace posterior over dense operator factors.

    Attributes
    ----------
    mean : Tensor
        MAP factor vector with shape ``(P,)`` (current fitted factors).
    diag_variance : Tensor
        Diagonal posterior variances with shape ``(P,)``.
    factors : tuple of LaplaceFactorSpec
        Block layout for unflattening ``mean`` / samples.
    n_data : int
        Number of one-step latent transition pairs used for the Fisher.
    prior_precision : float
        Isotropic Gaussian prior precision ``τ``.
    """

    mean: Tensor
    diag_variance: Tensor
    factors: tuple[LaplaceFactorSpec, ...]
    n_data: int
    prior_precision: float


def _as_data_sequence(sequence: CalibrationSequence | Any) -> list[Data]:
    """Normalize a calibration trajectory to a list of homogeneous ``Data``.

    Parameters
    ----------
    sequence : sequence of Data
        Trajectory snapshots (or a sequence-like container of ``Data``).

    Returns
    -------
    list of Data
        Materialized snapshot list.

    Raises
    ------
    TypeError
        If the trajectory is empty or contains non-``Data`` entries.
    ValueError
        If any snapshot is ``HeteroData``.
    """
    snaps = list(sequence)
    if not snaps:
        msg = "calibration sequences must be non-empty"
        raise ValueError(msg)
    for snap in snaps:
        if isinstance(snap, HeteroData):
            msg = (
                "BayesianKoopmanUQ does not support HeteroData trajectories; "
                "use ConformalKoopmanUQ for hetero intervals"
            )
            raise ValueError(msg)
        if not isinstance(snap, Data):
            msg = (
                "BayesianKoopmanUQ calibration sequences require homogeneous "
                f"Data snapshots, got {type(snap).__name__}"
            )
            raise TypeError(msg)
    return snaps


def _factor_specs(
    koopman: KoopmanOperator | GraphKoopmanOperator,
) -> tuple[LaplaceFactorSpec, ...]:
    """Build flattened factor layout for a dense discrete or graph operator.

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Dense-parameterized operator whose factors are flattened.

    Returns
    -------
    tuple of LaplaceFactorSpec
        Ordered factor blocks.
    """
    d = int(koopman.latent_dim)
    specs: list[LaplaceFactorSpec] = []
    offset = 0
    if isinstance(koopman, GraphKoopmanOperator):
        names = ["K_self", "K_nbr"]
        if getattr(koopman, "adjacency", None) == "dual_random_walk":
            names.append("K_bwd")
        for name in names:
            specs.append(
                LaplaceFactorSpec(name=name, shape=(d, d), offset=offset, numel=d * d)
            )
            offset += d * d
        return tuple(specs)
    specs.append(LaplaceFactorSpec(name="K", shape=(d, d), offset=0, numel=d * d))
    return tuple(specs)


def _read_factor_vector(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    factors: tuple[LaplaceFactorSpec, ...],
) -> Tensor:
    """Flatten current dense factors into a detached MAP vector.

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Operator supplying factor matrices.
    factors : tuple of LaplaceFactorSpec
        Flatten layout.

    Returns
    -------
    Tensor
        Concatenated factor entries with shape ``(P,)``.
    """
    pieces: list[Tensor] = []
    for spec in factors:
        if spec.name == "K":
            assert isinstance(koopman, KoopmanOperator)
            mat = koopman.K.detach()
        elif spec.name == "K_self":
            assert isinstance(koopman, GraphKoopmanOperator)
            mat = koopman.K_self.detach()
        elif spec.name == "K_nbr":
            assert isinstance(koopman, GraphKoopmanOperator)
            mat = koopman.K_nbr.detach()
        elif spec.name == "K_bwd":
            assert isinstance(koopman, GraphKoopmanOperator)
            mat = koopman.K_bwd.detach()
        else:
            msg = f"unknown factor name {spec.name!r}"
            raise ValueError(msg)
        pieces.append(mat.reshape(-1))
    return torch.cat(pieces, dim=0)


def _matrices_from_theta(
    theta: Tensor,
    factors: tuple[LaplaceFactorSpec, ...],
) -> dict[str, Tensor]:
    """Unflatten a parameter vector into named factor matrices.

    Parameters
    ----------
    theta : Tensor
        Flattened factors with shape ``(P,)``.
    factors : tuple of LaplaceFactorSpec
        Block layout.

    Returns
    -------
    dict of str to Tensor
        Mapping factor name → dense matrix.
    """
    out: dict[str, Tensor] = {}
    for spec in factors:
        block = theta[spec.offset : spec.offset + spec.numel]
        out[spec.name] = block.reshape(spec.shape)
    return out


def _apply_theta(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    theta: Tensor,
    factors: tuple[LaplaceFactorSpec, ...],
) -> None:
    """Write flattened factors onto a dense operator in place.

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Target dense operator.
    theta : Tensor
        Flattened factor vector.
    factors : tuple of LaplaceFactorSpec
        Block layout.
    """
    mats = _matrices_from_theta(theta, factors)
    if isinstance(koopman, GraphKoopmanOperator):
        koopman.set_dense_matrices(
            mats["K_self"],
            mats["K_nbr"],
            k_bwd=mats.get("K_bwd"),
        )
        return
    koopman.set_dense_matrix(mats["K"])


def _control_offset(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    z: Tensor,
    control: Tensor | None,
) -> Tensor:
    """Additive control offset with the operator's fixed ``B`` (not in ``θ``).

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Operator owning the control term.
    z : Tensor
        Latent states ``(N, d)`` used for broadcasting.
    control : Tensor or None
        Control for this step, or ``None`` when uncontrolled.

    Returns
    -------
    Tensor
        Offset with shape ``(N, d)``, or zeros when there is no control.
    """
    control_dim = int(getattr(koopman, "control_dim", 0) or 0)
    if control_dim == 0:
        if control is not None:
            msg = "control input provided to an uncontrolled operator"
            raise ValueError(msg)
        return torch.zeros_like(z)
    if control is None:
        msg = "control input is required when control_dim > 0"
        raise ValueError(msg)
    # Control lives on the self-factor module for graph operators.
    owner = (
        koopman._self  # noqa: SLF001
        if isinstance(koopman, GraphKoopmanOperator)
        else koopman
    )
    offset = owner.control_term(control, num_nodes=z.shape[0])
    if offset.ndim == 1:
        return offset.unsqueeze(0).expand_as(z)
    return offset


def _neighbor_term_from_factors(
    koopman: GraphKoopmanOperator,
    z: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    k_nbr: Tensor,
    k_bwd: Tensor | None,
) -> Tensor:
    """Sparse neighbor contribution using explicit factor matrices.

    Parameters
    ----------
    koopman : GraphKoopmanOperator
        Operator supplying adjacency mode.
    z : Tensor
        Latent node states ``(N, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    edge_weight : Tensor or None
        Optional edge weights.
    k_nbr : Tensor
        Dense neighbor / forward factor.
    k_bwd : Tensor or None
        Dense backward factor when ``adjacency='dual_random_walk'``.

    Returns
    -------
    Tensor
        Neighbor contribution ``(N, d)``.
    """
    if koopman.adjacency == "symmetric":
        neighbor = symmetric_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=edge_weight,
            num_nodes=z.shape[0],
        )
        return neighbor @ k_nbr.T

    neighbor_fwd = random_walk_normalized_adjacency_matvec(
        edge_index,
        z,
        edge_weight=edge_weight,
        num_nodes=z.shape[0],
        direction="forward",
    )
    term = neighbor_fwd @ k_nbr.T
    if koopman.adjacency == "random_walk":
        return term
    if k_bwd is None:
        msg = "K_bwd is required for dual_random_walk adjacency"
        raise ValueError(msg)
    neighbor_bwd = random_walk_normalized_adjacency_matvec(
        edge_index,
        z,
        edge_weight=edge_weight,
        num_nodes=z.shape[0],
        direction="backward",
    )
    return term + neighbor_bwd @ k_bwd.T


def _advance_from_theta(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    z: Tensor,
    theta: Tensor,
    factors: tuple[LaplaceFactorSpec, ...],
    *,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    control: Tensor | None,
) -> Tensor:
    """Differentiable one-step latent advance reconstructed from ``θ``.

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Operator providing control / adjacency metadata.
    z : Tensor
        Latent states ``(N, d)``.
    theta : Tensor
        Flattened factors (may require grad).
    factors : tuple of LaplaceFactorSpec
        Block layout.
    edge_index : Tensor or None
        Topology for graph operators.
    edge_weight : Tensor or None
        Optional edge weights.
    control : Tensor or None
        Optional additive control (``B`` held at MAP).

    Returns
    -------
    Tensor
        Advanced latents ``(N, d)``.
    """
    mats = _matrices_from_theta(theta, factors)
    if isinstance(koopman, GraphKoopmanOperator):
        if edge_index is None:
            msg = "graph Bayesian UQ requires edge_index for each transition"
            raise ValueError(msg)
        z_next = z @ mats["K_self"].T + _neighbor_term_from_factors(
            koopman,
            z,
            edge_index,
            edge_weight,
            mats["K_nbr"],
            mats.get("K_bwd"),
        )
        return z_next + _control_offset(koopman, z, control)
    z_next = z @ mats["K"].T
    return z_next + _control_offset(koopman, z, control)


class BayesianKoopmanUQ:
    """Diagonal Laplace UQ over dense Koopman factors with Monte Carlo forecasts.

    Composes a fitted :class:`~koopman_graph.model.GraphKoopmanModel` without
    subclassing. ``fit_posterior`` treats the current dense factors as the MAP
    and estimates a diagonal Laplace posterior from one-step latent MSE.
    ``sample_forecast`` draws factor samples, writes them via
    ``set_dense_matrix`` / ``set_dense_matrices``, rolls out, then restores the
    MAP factors.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted (or seeded) model with ``parameterization='dense'``.
    prior_precision : float, optional
        Isotropic Gaussian prior precision ``τ``. Default ``1.0``.
    observation_noise : float, optional
        Latent observation-noise scale ``σ`` in the one-step MSE likelihood
        and optional aleatoric composition. Default ``1.0``.
    n_samples : int, optional
        Default Monte Carlo factor draws for ``sample_forecast``. Default
        ``32``.

    Notes
    -----
    Requires dense discrete per-node or graph operators. Rejects hetero,
    bilinear control, and non-dense parameterizations. Posterior intervals
    have **no** finite-sample coverage guarantee; prefer conformal methods
    when calibration guarantees matter.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        prior_precision: float = 1.0,
        observation_noise: float = 1.0,
        n_samples: int = 32,
    ) -> None:
        """Validate the composed model and store UQ hyperparameters.

        Parameters
        ----------
        model : GraphKoopmanModel
            See class docstring.
        prior_precision : float, optional
            See class docstring.
        observation_noise : float, optional
            See class docstring.
        n_samples : int, optional
            See class docstring.
        """
        if prior_precision <= 0.0:
            msg = f"prior_precision must be positive, got {prior_precision}"
            raise ValueError(msg)
        if observation_noise <= 0.0:
            msg = f"observation_noise must be positive, got {observation_noise}"
            raise ValueError(msg)
        if n_samples < 1:
            msg = f"n_samples must be >= 1; got {n_samples}"
            raise ValueError(msg)

        self.model = model
        self.prior_precision = float(prior_precision)
        self.observation_noise = float(observation_noise)
        self.n_samples = int(n_samples)
        self._posterior: LaplacePosterior | None = None
        self._validate_model()

    def _validate_model(self) -> None:
        """Reject unsupported operator / control configurations.

        Raises
        ------
        ValueError
            If the model uses hetero, bilinear, continuous, or non-dense
            factorizations.
        TypeError
            If the Koopman module is not a dense discrete or graph operator.
        """
        if bool(getattr(self.model, "uses_hetero_koopman", False)):
            msg = (
                "BayesianKoopmanUQ does not support hetero Koopman models; "
                "use ConformalKoopmanUQ for hetero intervals"
            )
            raise ValueError(msg)
        koopman = self.model.koopman
        control_mode = getattr(self.model, "control_mode", None)
        if control_mode is None:
            control_mode = getattr(koopman, "control_mode", "additive")
        if control_mode == "bilinear":
            msg = (
                "BayesianKoopmanUQ does not support bilinear control_mode; "
                "use additive control or an uncontrolled model"
            )
            raise ValueError(msg)
        if not isinstance(koopman, (KoopmanOperator, GraphKoopmanOperator)):
            msg = (
                "BayesianKoopmanUQ requires a dense discrete KoopmanOperator "
                "or GraphKoopmanOperator (continuous / hypergraph / custom "
                "operators are unsupported); got "
                f"{type(koopman).__name__}"
            )
            raise TypeError(msg)
        parameterization = getattr(koopman, "parameterization", None)
        if parameterization != "dense":
            msg = (
                "BayesianKoopmanUQ requires parameterization='dense' over "
                "linear factors; got "
                f"{parameterization!r}. Non-dense modes (schur/odo/lyapunov/"
                "dissipative/auxiliary_spectral) are unsupported in v1"
            )
            raise ValueError(msg)

    @property
    def posterior(self) -> LaplacePosterior | None:
        """Most recent :meth:`fit_posterior` result, or ``None`` if unset.

        Returns
        -------
        object
            Function result.
        """
        return self._posterior

    def fit_posterior(
        self,
        calibration_sequences: Sequence[CalibrationSequence],
        *,
        controls: Sequence[Sequence[Tensor] | None] | None = None,
    ) -> LaplacePosterior:
        """Estimate a diagonal Laplace posterior at the current MAP factors.

        Collects consecutive encoded pairs ``(z_t, z_{t+1})`` from each
        trajectory, accumulates the diagonal empirical Fisher of the one-step
        latent MSE likelihood, and forms
        ``Var_i = 1 / (τ + Fisher_ii)``.

        Parameters
        ----------
        calibration_sequences : sequence of trajectories
            Homogeneous ``Data`` trajectories of length ≥ 2. Hetero sequences
            are rejected.
        controls : sequence of control sequences or None, optional
            Optional per-trajectory controls aligned with transitions
            (length ``len(sequence) - 1`` when provided).

        Returns
        -------
        LaplacePosterior
            Diagonal Laplace posterior (also stored on the wrapper).

        Raises
        ------
        ValueError
            If sequences are empty, too short, or hetero / unsupported.
        """
        self._validate_model()
        if not calibration_sequences:
            msg = "calibration_sequences must be non-empty"
            raise ValueError(msg)
        if controls is not None and len(controls) != len(calibration_sequences):
            msg = (
                "controls must align with calibration_sequences; "
                f"got {len(controls)} vs {len(calibration_sequences)}"
            )
            raise ValueError(msg)

        koopman = self.model.koopman
        assert isinstance(koopman, (KoopmanOperator, GraphKoopmanOperator))
        factors = _factor_specs(koopman)
        mean = _read_factor_vector(koopman, factors)
        device = mean.device
        dtype = mean.dtype
        fisher = torch.zeros_like(mean)
        n_data = 0

        was_training = self.model.training
        self.model.eval()
        try:
            for seq_idx, sequence in enumerate(calibration_sequences):
                snaps = _as_data_sequence(sequence)
                if len(snaps) < 2:
                    msg = (
                        "each calibration sequence must contain at least two "
                        f"snapshots; got length {len(snaps)}"
                    )
                    raise ValueError(msg)
                seq_controls = None if controls is None else controls[seq_idx]
                if seq_controls is not None and len(seq_controls) != len(snaps) - 1:
                    msg = (
                        "controls[i] must have length len(sequence) - 1; "
                        f"got {len(seq_controls)} vs {len(snaps) - 1}"
                    )
                    raise ValueError(msg)

                for t in range(len(snaps) - 1):
                    z0 = self.model.encode(snaps[t]).detach()
                    z1 = self.model.encode(snaps[t + 1]).detach()
                    edge_index = getattr(snaps[t], "edge_index", None)
                    edge_weight = getattr(snaps[t], "edge_weight", None)
                    control = None if seq_controls is None else seq_controls[t]
                    if control is not None:
                        control = control.to(device=device, dtype=dtype)

                    theta = mean.detach().clone().requires_grad_(True)
                    pred = _advance_from_theta(
                        koopman,
                        z0.to(device=device, dtype=dtype),
                        theta,
                        factors,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        control=control,
                    )
                    residual = pred - z1.to(device=device, dtype=dtype)
                    loss = 0.5 * (residual.square().sum()) / (self.observation_noise**2)
                    (grad,) = torch.autograd.grad(loss, theta)
                    fisher = fisher + grad.detach().square()
                    n_data += 1
        finally:
            self.model.train(was_training)

        if n_data < 1:
            msg = "fit_posterior collected no transition pairs"
            raise ValueError(msg)

        precision = self.prior_precision + fisher
        diag_variance = (1.0 / precision.clamp(min=1e-12)).detach()
        posterior = LaplacePosterior(
            mean=mean.detach().clone(),
            diag_variance=diag_variance,
            factors=factors,
            n_data=n_data,
            prior_precision=self.prior_precision,
        )
        self._posterior = posterior
        return posterior

    def sample_forecast(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        *args: Any,
        level: float = 0.9,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        compose_latent_gaussian: bool = False,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        topology_policy: str = "auto",
        **kwargs: Any,
    ) -> PredictionInterval:
        """Monte Carlo predictive interval from diagonal Laplace factor draws.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Rollout origin (homogeneous only).
        steps : int
            Forecast horizon (``≥ 1``).
        *args, **kwargs
            Rejected; reserved for :class:`~koopman_graph.uq.IntervalForecastModel`
            compatibility.
        level : float, optional
            Nominal central coverage in ``(0, 1)``. Default ``0.9``.
        n_samples : int or None, optional
            Factor draws. Defaults to :attr:`n_samples`.
        generator : torch.Generator or None, optional
            RNG for factor (and optional aleatoric) draws.
        compose_latent_gaussian : bool, optional
            When ``True``, add isotropic latent noise of scale
            :attr:`observation_noise` at decode time only (aleatoric
            composition). Default ``False``.
        edge_index, edge_weight, controls, future_topologies, history
            Same semantics as
            :meth:`~koopman_graph.model.GraphKoopmanModel.predict`.
        topology_policy : {"auto", "recursive", "hold_last"}, optional
            Same semantics as
            :meth:`~koopman_graph.model.GraphKoopmanModel.predict`.

        Returns
        -------
        PredictionInterval
            Mean / lower / upper length-``steps`` tuples of ``Data`` with
            ``x`` shape ``(N, F)``. ``n_members`` reports the number of
            factor samples.

        Raises
        ------
        ValueError
            If no posterior is fitted, shapes are invalid, or hetero inputs
            are supplied.
        TypeError
            If unexpected positional / keyword arguments are passed.
        """
        if args:
            msg = (
                "BayesianKoopmanUQ.sample_forecast takes no positional args after steps"
            )
            raise TypeError(msg)
        if kwargs:
            msg = "unexpected keyword arguments for sample_forecast: " + ", ".join(
                sorted(kwargs)
            )
            raise TypeError(msg)
        if isinstance(initial_graph, HeteroData):
            msg = (
                "BayesianKoopmanUQ does not support HeteroData origins; "
                "use ConformalKoopmanUQ for hetero intervals"
            )
            raise ValueError(msg)
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if self._posterior is None:
            msg = "call fit_posterior before sample_forecast"
            raise ValueError(msg)

        n_draw = self.n_samples if n_samples is None else int(n_samples)
        if n_draw < 1:
            msg = f"n_samples must be >= 1; got {n_draw}"
            raise ValueError(msg)

        lower_q, upper_q = quantile_levels(level)
        posterior = self._posterior
        koopman = self.model.koopman
        assert isinstance(koopman, (KoopmanOperator, GraphKoopmanOperator))

        validate_controls(
            control_dim=int(getattr(self.model, "control_dim", 0) or 0),
            controls=controls,
            steps=steps,
        )

        map_theta = _read_factor_vector(koopman, posterior.factors)
        device = map_theta.device
        dtype = map_theta.dtype
        std = posterior.diag_variance.to(device=device, dtype=dtype).sqrt()
        mean = posterior.mean.to(device=device, dtype=dtype)

        was_training = self.model.training
        self.model.eval()
        member_features: list[list[Tensor]] = []
        init_edge: Tensor
        init_weight: Tensor | None
        try:
            with torch.no_grad():
                z0, init_edge, init_weight = self.model.encode_rollout_origin(
                    initial_graph,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    history=history,
                )
                hyperedge_index = (
                    snapshot_hyperedge_index(initial_graph)
                    if isinstance(initial_graph, Data)
                    else None
                )
                hyperedge_weight = (
                    snapshot_hyperedge_weight(initial_graph)
                    if isinstance(initial_graph, Data)
                    else None
                )
                topology_at = resolve_rollout_topology_at(
                    self.model,
                    init_edge,
                    init_weight,
                    future_topologies,
                    topology_policy,
                )
                control_at = None if controls is None else (lambda step: controls[step])

                for _ in range(n_draw):
                    eps = torch.randn(
                        mean.shape,
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                    theta = mean + std * eps
                    _apply_theta(koopman, theta, posterior.factors)
                    if compose_latent_gaussian:
                        snaps = self._rollout_with_decode_noise(
                            z0,
                            steps=steps,
                            topology_at=topology_at,
                            control_at=control_at,
                            generator=generator,
                        )
                    else:
                        rollout = autoregressive_latent_rollout(
                            self.model.koopman,
                            self.model.decoder,
                            z0,
                            steps=steps,
                            topology_at=topology_at,
                            control_at=control_at,
                            default_delta_t=self.model.time_step,
                            hyperedge_index=hyperedge_index,
                            hyperedge_weight=hyperedge_weight,
                        )
                        snaps = pack_rollout_snapshots(rollout)
                    member_features.append([snap.x for snap in snaps])
        finally:
            _apply_theta(koopman, map_theta, posterior.factors)
            self.model.train(was_training)

        topology_at = hold_last_topology_at(
            init_edge,
            init_weight,
            future_topologies,
        )
        mean_snaps: list[Data] = []
        lower_snaps: list[Data] = []
        upper_snaps: list[Data] = []
        for step in range(steps):
            stacked = torch.stack(
                [features[step] for features in member_features],
                dim=0,
            )
            edge_t, weight_t = topology_at(step)
            template = Data(x=stacked.mean(dim=0), edge_index=edge_t)
            if weight_t is not None:
                template.edge_weight = weight_t
            mean_x = stacked.mean(dim=0)
            if stacked.shape[0] == 1:
                lower_x = mean_x.clone()
                upper_x = mean_x.clone()
            else:
                q = torch.tensor(
                    [lower_q, upper_q],
                    device=stacked.device,
                    dtype=torch.float32,
                )
                bounds = torch.quantile(stacked.float(), q, dim=0).to(
                    dtype=stacked.dtype
                )
                lower_x = bounds[0]
                upper_x = bounds[1]
            mean_snaps.append(snapshot_with_features(template, mean_x))
            lower_snaps.append(snapshot_with_features(template, lower_x))
            upper_snaps.append(snapshot_with_features(template, upper_x))

        return PredictionInterval(
            mean=mean_snaps,
            lower=lower_snaps,
            upper=upper_snaps,
            level=level,
            n_members=n_draw,
        )

    def _rollout_with_decode_noise(
        self,
        z0: Tensor,
        *,
        steps: int,
        topology_at: Any,
        control_at: Any,
        generator: torch.Generator | None,
    ) -> list[Data]:
        """Advance with MAP-style dynamics; add latent noise only at decode.

        Parameters
        ----------
        z0 : Tensor
            Encoded origin latents.
        steps : int
            Forecast horizon.
        topology_at : callable
            Topology schedule (hold-last or predicted).
        control_at : callable or None
            Optional control schedule.
        generator : torch.Generator or None
            RNG for isotropic latent noise.

        Returns
        -------
        list of Data
            Decoded snapshots with aleatoric decode noise.
        """
        snaps: list[Data] = []
        latent = z0
        for step in range(steps):
            edge_t, weight_t = call_topology_at(topology_at, step, latent)
            control = None if control_at is None else control_at(step)
            latent = propagate_latent(
                self.model.koopman,
                latent,
                control=control,
                default_delta_t=self.model.time_step,
                edge_index=edge_t,
                edge_weight=weight_t,
            )
            noise = self.observation_noise * torch.randn(
                latent.shape,
                dtype=latent.dtype,
                device=latent.device,
                generator=generator,
            )
            pred = self.model.decoder(latent + noise, edge_t, weight_t)
            fields: dict[str, Tensor] = {"x": pred, "edge_index": edge_t}
            if weight_t is not None:
                fields["edge_weight"] = weight_t
            snaps.append(Data(**fields))
        return snaps

    def predict_interval(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        *args: Any,
        level: float = 0.9,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        compose_latent_gaussian: bool = False,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        **kwargs: Any,
    ) -> PredictionInterval:
        """Alias of :meth:`sample_forecast` for ``IntervalForecastModel``.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Rollout origin.
        steps : int
            Forecast horizon.
        *args, **kwargs
            Forwarded to :meth:`sample_forecast`.
        level, n_samples, generator, compose_latent_gaussian
            See :meth:`sample_forecast`.
        edge_index, edge_weight, controls, future_topologies, history
            See :meth:`sample_forecast`.

        Returns
        -------
        PredictionInterval
            Monte Carlo predictive interval from Laplace factor draws.
        """
        return self.sample_forecast(
            initial_graph,
            steps,
            *args,
            level=level,
            n_samples=n_samples,
            generator=generator,
            compose_latent_gaussian=compose_latent_gaussian,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
            **kwargs,
        )
