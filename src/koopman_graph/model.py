"""GraphKoopmanModel: encoder, Koopman operator, and decoder composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data

from koopman_graph.adaptation import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.analysis import (
    KoopmanSpectrum,
    compute_generator_spectrum,
    compute_spectrum,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.continuous import (
    ContinuousKoopmanOperator,
    GeneratorParameterization,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    WindowSampler,
    _snapshot_edge_weight,
    resolve_sequence,
)
from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder
from koopman_graph.metrics import EvaluationResult, evaluate_forecast
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    concatenate_observables,
    resolve_physics_lifting_fn,
    validate_physics_output,
)
from koopman_graph.operator import InitMode, KoopmanOperator, Parameterization
from koopman_graph.serialization import (
    load_checkpoint,
    save_checkpoint,
    snapshot_state_dict,
)
from koopman_graph.training import (
    EarlyStoppingMonitor,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
    RolloutStartIndices,
    TrainingInput,
    ValidationInput,
    eval_one_epoch,
    resolve_device,
    resolve_early_stopping_monitor,
    resolve_loss_weights_for_epoch,
    resolve_lr_scheduler,
    resolve_rollout_start_indices,
    resolve_training_sequences,
    resolve_validation_sequences,
    should_stop_early,
    train_one_epoch,
    train_windowed_epoch,
)

if TYPE_CHECKING:
    from koopman_graph.env import GraphKoopmanEnv

Encoder = GNNEncoder | GATEncoder
DynamicsMode = Literal["discrete", "continuous"]
KoopmanModule = KoopmanOperator | ContinuousKoopmanOperator


class GraphKoopmanModel(nn.Module):
    """Topology-aware Koopman dynamics model for graph snapshots.

    Composes a GNN encoder (lifting), a finite-dimensional Koopman operator
    (linear latent evolution), and a symmetric GNN decoder (reconstruction).

    Attributes
    ----------
    encoder : GNNEncoder or GATEncoder
        Topology-aware encoder for latent lifting.
    decoder : GNNDecoder
        Symmetric GNN decoder for physical reconstruction.
    latent_dim : int
        Latent space dimension shared by encoder, operator, and decoder.
    time_step : float
        Physical time increment associated with one model step. Used by
        :meth:`spectrum` to convert discrete eigenvalues into continuous-time
        growth rates and frequencies.
    koopman : KoopmanOperator or ContinuousKoopmanOperator
        Learnable linear propagator in latent space.
    dynamics_mode : {"discrete", "continuous"}
        Whether latent evolution uses a discrete step map or a continuous
        generator integrated with matrix exponentials.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: GNNDecoder,
        latent_dim: int,
        time_step: float,
        *,
        dynamics_mode: DynamicsMode = "discrete",
        koopman_init_mode: InitMode = "identity_noise",
        koopman_init_scale: float = 1e-2,
        koopman_parameterization: (
            Parameterization | GeneratorParameterization
        ) = "dense",
        koopman_max_spectral_radius: float = 1.0,
        control_dim: int = 0,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
        physics_preset: str | None = None,
        physics_dim: int = 0,
    ) -> None:
        """Initialize encoder, decoder, and Koopman operator.

        Parameters
        ----------
        encoder : GNNEncoder or GATEncoder
            Topology-aware encoder for latent lifting.
        decoder : GNNDecoder
            Symmetric GNN decoder for physical reconstruction.
        latent_dim : int
            Total latent space dimension per node. When physics-informed
            observables are enabled, ``latent_dim = physics_dim +
            encoder.latent_dim``.
        time_step : float
            Physical time increment associated with one model step when
            timestamps are absent.
        dynamics_mode : {"discrete", "continuous"}, optional
            Latent evolution mode. ``"discrete"`` preserves the v0.2 behavior;
            ``"continuous"`` learns a generator integrated via matrix
            exponentials. Default is ``"discrete"``.
        koopman_init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization strategy for the Koopman matrix. Default is
            ``"identity_noise"``.
        koopman_init_scale : float, optional
            Noise scale when ``koopman_init_mode="identity_noise"``.
            Default is ``1e-2``.
        koopman_parameterization : {"dense", "odo", "schur", "dissipative", "lyapunov"},
            optional
            Koopman matrix parameterization. ``"odo"`` enforces a spectral-radius
            bound via orthogonal-diagonal-orthogonal factors. ``"schur"``,
            ``"dissipative"``, and ``"lyapunov"`` embed structural stability
            guarantees for long-horizon rollouts. Default is ``"dense"``.
        koopman_max_spectral_radius : float, optional
            Maximum eigenvalue magnitude for bounded/structural parameterizations.
            Structurally stable modes enforce a strict interior margin below
            this value. Default is ``1.0``.
        control_dim : int, optional
            Dimension of exogenous control inputs. When ``0``, the model is
            uncontrolled. Default is ``0``.
        physics_lifting_fn : callable or None, optional
            Callable mapping a PyG ``Data`` snapshot to physics-informed node
            features with shape ``(num_nodes, physics_dim)``. When provided,
            features are **prepended** to GNN embeddings before Koopman
            propagation: ``z = [z_physics || z_gnn]``.
        physics_preset : str or None, optional
            Registered preset name (for example ``"graph_laplacian"``) used when
            ``physics_lifting_fn`` is omitted. Custom callables take precedence
            over presets.
        physics_dim : int, optional
            Number of physics-informed features per node. Must be positive when
            a physics lifting function or preset is supplied, and ``0`` otherwise.
            For ``physics_preset="graph_laplacian"``, set ``physics_dim`` equal to
            ``in_channels``.

        Raises
        ------
        ValueError
            If ``latent_dim`` is not positive, ``time_step <= 0``,
            ``control_dim < 0``, physics settings are inconsistent, or encoder/
            decoder latent dimensions do not match the effective hybrid layout.
        """
        super().__init__()
        if dynamics_mode not in {"discrete", "continuous"}:
            msg = (
                "dynamics_mode must be 'discrete' or 'continuous', "
                f"got {dynamics_mode!r}"
            )
            raise ValueError(msg)
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)
        if physics_dim < 0:
            msg = f"physics_dim must be non-negative, got {physics_dim}"
            raise ValueError(msg)

        resolved_physics_fn = resolve_physics_lifting_fn(
            physics_preset=physics_preset,
            physics_lifting_fn=physics_lifting_fn,
        )
        if (resolved_physics_fn is None) != (physics_dim == 0):
            msg = (
                "physics_dim must be positive when physics lifting is enabled "
                "and zero otherwise"
            )
            raise ValueError(msg)
        if physics_preset is not None and resolved_physics_fn is None:
            msg = "physics_preset requires a registered preset or physics_lifting_fn"
            raise ValueError(msg)

        gnn_latent_dim = encoder.latent_dim
        expected_latent_dim = gnn_latent_dim + physics_dim
        if latent_dim != expected_latent_dim:
            msg = (
                f"latent_dim ({latent_dim}) must equal encoder.latent_dim "
                f"({gnn_latent_dim}) + physics_dim ({physics_dim})"
            )
            raise ValueError(msg)
        if decoder.latent_dim != latent_dim:
            msg = (
                f"decoder.latent_dim ({decoder.latent_dim}) must match "
                f"latent_dim ({latent_dim})"
            )
            raise ValueError(msg)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.gnn_latent_dim = gnn_latent_dim
        self.physics_dim = physics_dim
        self.physics_preset = physics_preset
        self.physics_lifting_fn = resolved_physics_fn
        self.physics_position = PHYSICS_POSITION
        self.time_step = time_step
        self.control_dim = control_dim
        self.dynamics_mode = dynamics_mode
        if dynamics_mode == "continuous":
            self.koopman = ContinuousKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,  # type: ignore[arg-type]
                max_real_eigenvalue=koopman_max_spectral_radius,
                control_dim=control_dim,
            )
        else:
            self.koopman = KoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,  # type: ignore[arg-type]
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
            )

    @property
    def is_continuous(self) -> bool:
        """Return whether the model uses continuous-time generator dynamics.

        Returns
        -------
        bool
            ``True`` when :attr:`dynamics_mode` is ``"continuous"``.
        """
        return self.dynamics_mode == "continuous"

    def _resolve_delta_t(self, delta_t: float | Tensor | None) -> float | Tensor:
        """Return the integration interval for one propagation step.

        Returns
        -------
        float or Tensor
            Resolved integration interval.
        """
        if delta_t is not None:
            return delta_t
        return self.time_step

    def _advance_latent(
        self,
        z: Tensor,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> Tensor:
        """Advance latent states with the active Koopman operator.

        Returns
        -------
        Tensor
            Advanced latent states.
        """
        if self.is_continuous:
            return self.koopman.advance(
                z,
                self._resolve_delta_t(delta_t),
                control=control,
            )
        return self.koopman(z, control=control)

    def spectrum(self, *, delta_t: float | None = None) -> KoopmanSpectrum:
        """Analyze the learned Koopman operator spectrum.

        In continuous mode, returns the generator spectrum by default. Pass
        ``delta_t`` to obtain the discrete-time spectrum of ``exp(L·Δt)``.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted eigenvalues, eigenvectors, and time scales.
        """
        if self.is_continuous:
            if delta_t is None:
                return compute_generator_spectrum(self.koopman.L)
            return discrete_spectrum_at_delta_t(self.koopman.L, delta_t)
        return compute_spectrum(self.koopman.K, self.time_step)

    def encode_latent(self, snapshot: Data) -> Tensor:
        """Encode a graph snapshot into latent node features.

        Parameters
        ----------
        snapshot : Data
            Graph snapshot with node features and topology.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        return self.encode(snapshot)

    def encode(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Lift graph node features into the hybrid Koopman latent space.

        When physics-informed observables are configured, returns
        ``[z_physics || z_gnn]`` with shape ``(num_nodes, latent_dim)``.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features or a PyG ``Data`` snapshot.
        edge_index : Tensor or None, optional
            Edge index required when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights for tensor input.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        edge_index = self._resolve_edge_index(x_or_data, edge_index)
        edge_weight = self._resolve_edge_weight(x_or_data, edge_weight)
        z_gnn = self.encoder(x_or_data, edge_index, edge_weight)
        if self.physics_lifting_fn is None:
            return z_gnn

        snapshot = self._as_data(x_or_data, edge_index, edge_weight)
        physics_features = self.physics_lifting_fn(snapshot)
        validate_physics_output(
            physics_features,
            physics_dim=self.physics_dim,
            num_nodes=z_gnn.size(0),
        )
        return concatenate_observables(
            physics_features,
            z_gnn,
            position=self.physics_position,
        )

    @property
    def uses_physics_observables(self) -> bool:
        """Return whether physics-informed observables are enabled.

        Returns
        -------
        bool
            ``True`` when a physics lifting function is configured.
        """
        return self.physics_lifting_fn is not None

    def enable_online_adaptation(
        self,
        *,
        forgetting_factor: float = 0.99,
        regularization: float = 1e3,
    ) -> RecursiveKoopmanAdapter:
        """Enable recursive least-squares adaptation of the Koopman operator.

        Freezes encoder and decoder parameters so only the dense Koopman
        operator is updated by :meth:`adapt_step`. Requires
        ``koopman_parameterization="dense"``.

        Parameters
        ----------
        forgetting_factor : float, optional
            RLS forgetting factor in ``(0, 1]``. Default is ``0.99``.
        regularization : float, optional
            Initial covariance scale for the RLS regressor. Default is ``1e3``.

        Returns
        -------
        RecursiveKoopmanAdapter
            Adapter instance stored on the model.

        Raises
        ------
        ValueError
            If the Koopman operator is not densely parameterized.
        """
        if self.koopman.parameterization != "dense":
            msg = (
                "Online adaptation requires dense Koopman parameterization; "
                f"got {self.koopman.parameterization!r}."
            )
            raise ValueError(msg)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

        mode: Literal["discrete", "continuous"] = (
            "continuous" if self.is_continuous else "discrete"
        )
        adapter = RecursiveKoopmanAdapter.from_operator(
            self.koopman,
            mode=mode,
            forgetting_factor=forgetting_factor,
            regularization=regularization,
        )
        self._adaptation_adapter = adapter
        return adapter

    @property
    def online_adaptation_enabled(self) -> bool:
        """Return whether online adaptation is active.

        Returns
        -------
        bool
            ``True`` when :meth:`enable_online_adaptation` has been called and
            :meth:`disable_online_adaptation` has not.
        """
        return getattr(self, "_adaptation_adapter", None) is not None

    def adapt_step(
        self,
        snapshot_t: Data,
        snapshot_tp1: Data,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> AdaptationStepResult:
        """Apply one online RLS update from a pair of graph snapshots.

        Encodes both snapshots with the frozen encoder, updates the Koopman
        operator via recursive least squares, and writes the estimate back
        into :attr:`koopman`.

        Parameters
        ----------
        snapshot_t : Data
            Source graph snapshot at time ``t``.
        snapshot_tp1 : Data
            Target graph snapshot at time ``t+1``.
        control : Tensor or None, optional
            Control input applied during the transition. Required for
            controlled models.
        delta_t : float or Tensor or None, optional
            Integration interval for continuous models. Defaults to
            :attr:`time_step` when omitted.

        Returns
        -------
        AdaptationStepResult
            Diagnostics for the adaptation step.

        Notes
        -----
        For ``dynamics_mode="continuous"``, RLS fits a discrete propagator per
        interval and maps it to a generator via ``logm(K(Δt))/Δt`` with control
        scaling ``B(Δt)/Δt``. This is a first-order approximation that does not
        match the Van Loan integration used in forward propagation. Prefer
        discrete adaptation for uniformly sampled sequences; see
        :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` for
        breakdown conditions (large or varying ``delta_t``, controlled dynamics).

        Raises
        ------
        RuntimeError
            If :meth:`enable_online_adaptation` has not been called.
        ValueError
            If continuous mode is used without a positive ``delta_t``.
        """
        adapter = getattr(self, "_adaptation_adapter", None)
        if adapter is None:
            msg = "call enable_online_adaptation() before adapt_step()"
            raise RuntimeError(msg)

        resolved_delta = delta_t
        if self.is_continuous and resolved_delta is None:
            resolved_delta = self.time_step

        with torch.no_grad():
            z_t = self.encode_latent(snapshot_t)
            z_tp1 = self.encode_latent(snapshot_tp1)
            result = adapter.update(
                z_t,
                z_tp1,
                control=control,
                delta_t=resolved_delta,
            )
            adapter.apply_to(self.koopman)
        return result

    def disable_online_adaptation(self, *, unfreeze: bool = True) -> None:
        """Disable online adaptation and optionally unfreeze encoder/decoder.

        Parameters
        ----------
        unfreeze : bool, optional
            When ``True``, restore ``requires_grad`` on encoder and decoder
            parameters. Default is ``True``.
        """
        self._adaptation_adapter = None
        if unfreeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(True)

    def save(self, path: str | Path) -> None:
        """Persist model weights and architecture configuration to disk.

        Parameters
        ----------
        path : str or Path
            Destination checkpoint file (``.pt``). Parent directories are
            created when missing.
        """
        save_checkpoint(self, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
    ) -> GraphKoopmanModel:
        """Load a trained model from a checkpoint file.

        Reconstructs encoder, decoder, and Koopman operator architecture from
        the saved configuration and restores learned weights.

        Parameters
        ----------
        path : str or Path
            Checkpoint file produced by :meth:`save`.
        map_location : str, torch.device, or None, optional
            Device mapping forwarded to :func:`torch.load`.
        physics_lifting_fn : callable or None, optional
            Custom physics lifting function required when the checkpoint stores
            hybrid observables without a registered preset.

        Returns
        -------
        GraphKoopmanModel
            Ready-to-use model in evaluation mode.
        """
        return load_checkpoint(
            path,
            map_location=map_location,
            physics_lifting_fn=physics_lifting_fn,
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> Tensor:
        """Predict the next graph snapshot from the current one.

        Performs encode → linear Koopman advance → decode for a single step.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Either a PyG ``Data`` object or node features ``x`` of shape
            ``(num_nodes, in_channels)``.
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``x_or_data`` is a tensor; ignored for ``Data`` input.
        edge_weight : Tensor, optional
            Scalar edge weights with shape ``(num_edges,)``. Required when
            ``x_or_data`` is a tensor and weights are used; ignored for
            ``Data`` input.
        control : Tensor or None, optional
            Exogenous control input for this step. Required when
            :attr:`control_dim` is positive.
        delta_t : float, Tensor, or None, optional
            Integration interval for continuous-time dynamics. Defaults to
            :attr:`time_step` when omitted.

        Returns
        -------
        Tensor
            Predicted node features of shape ``(num_nodes, out_channels)``.
        """
        edge_index = self._resolve_edge_index(x_or_data, edge_index)
        edge_weight = self._resolve_edge_weight(x_or_data, edge_weight)
        z = self.encode(x_or_data, edge_index, edge_weight)
        z_next = self._advance_latent(z, control=control, delta_t=delta_t)
        return self.decoder(z_next, edge_index, edge_weight)

    def _rollout(
        self,
        x_or_data: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
    ) -> list[tuple[Tensor, Tensor, Tensor | None]]:
        """Autoregressively advance latent state and decode for multiple steps.

        Encodes the initial graph once, then applies the Koopman operator
        repeatedly in latent space, decoding after each step.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Either a PyG ``Data`` object or node features ``x``.
        steps : int
            Number of rollout steps (must be >= 1).
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``x_or_data`` is a tensor; ignored for ``Data`` input.
        edge_weight : Tensor, optional
            Scalar edge weights with shape ``(num_edges,)``. Required when
            ``x_or_data`` is a tensor and weights are used; ignored for
            ``Data`` input.
        controls : sequence of Tensor or None, optional
            Control inputs for each rollout step. Required with length
            ``steps`` when :attr:`control_dim` is positive.
        future_topologies : sequence of Data or None, optional
            Known graph topologies for rollout decode steps. Entry ``step`` is
            used when present; otherwise the last known topology is held
            (starting from the initial graph).
        step_deltas : sequence of float or Tensor or None, optional
            Integration interval for each rollout step. When omitted, each step
            uses :attr:`time_step`.

        Returns
        -------
        list of tuple[Tensor, Tensor, Tensor or None]
            For each step, decoded prediction, ``edge_index``, and optional
            ``edge_weight`` used for decoding.

        Raises
        ------
        ValueError
            If ``steps < 1`` or controls are missing/invalid for a controlled
            model.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)

        edge_index = self._resolve_edge_index(x_or_data, edge_index)
        edge_weight = self._resolve_edge_weight(x_or_data, edge_weight)
        self._validate_controls(controls, steps=steps)
        if step_deltas is not None and len(step_deltas) != steps:
            msg = f"expected {steps} step_deltas for rollout, got {len(step_deltas)}"
            raise ValueError(msg)
        z = self.encode(x_or_data, edge_index, edge_weight)

        current_edge_index = edge_index
        current_edge_weight = edge_weight
        outputs: list[tuple[Tensor, Tensor, Tensor | None]] = []
        for step in range(steps):
            if future_topologies is not None and step < len(future_topologies):
                current_edge_index = future_topologies[step].edge_index
                current_edge_weight = _snapshot_edge_weight(future_topologies[step])
            control = None if controls is None else controls[step]
            delta_t = None if step_deltas is None else step_deltas[step]
            z = self._advance_latent(z, control=control, delta_t=delta_t)
            prediction = self.decoder(z, current_edge_index, current_edge_weight)
            outputs.append((prediction, current_edge_index, current_edge_weight))
        return outputs

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Encodes the initial graph once, advances the latent state with the
        Koopman operator for ``steps`` iterations, and decodes after each step.
        Runs in evaluation mode without gradient tracking.

        When ``future_topologies`` is omitted, each rollout step decodes with
        the **hold-last-known** topology: the initial graph topology is used
        for step 0, and each subsequent step reuses the most recently provided
        topology. Pass one ``Data`` object per rollout step (topology only; node
        features are ignored) to supply a known future rewiring schedule.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Either a PyG ``Data`` object or node features ``x`` of shape
            ``(num_nodes, in_channels)``.
        steps : int
            Number of future snapshots to predict (must be >= 1).
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``initial_graph`` is a tensor; ignored for ``Data`` input.
        edge_weight : Tensor, optional
            Scalar edge weights with shape ``(num_edges,)``. Required when
            ``initial_graph`` is a tensor and weights are used; ignored for
            ``Data`` input.
        controls : sequence of Tensor or None, optional
            Future control inputs for each rollout step. Required with length
            ``steps`` when :attr:`control_dim` is positive.
        future_topologies : sequence of Data or None, optional
            Known topologies for rollout decode steps. Shorter sequences hold
            the last provided topology for remaining steps.

        Returns
        -------
        list of Data
            ``steps`` predicted graph snapshots. Each ``Data.x`` has shape
            ``(num_nodes, out_channels)`` and carries the ``edge_index`` (and
            optional ``edge_weight``) used for that step's decode.

        Raises
        ------
        ValueError
            If ``steps < 1`` or controls are missing/invalid for a controlled
            model.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                rollout = self._rollout(
                    initial_graph,
                    steps,
                    edge_index,
                    edge_weight,
                    controls=controls,
                    future_topologies=future_topologies,
                )
        finally:
            self.train(was_training)

        output_snapshots: list[Data] = []
        for prediction, step_edge_index, step_edge_weight in rollout:
            fields: dict[str, Tensor] = {
                "x": prediction,
                "edge_index": step_edge_index,
            }
            if step_edge_weight is not None:
                fields["edge_weight"] = step_edge_weight
            output_snapshots.append(Data(**fields))
        return output_snapshots

    def predict_at(
        self,
        initial_graph: Tensor | Data,
        *,
        query_times: Sequence[float] | Sequence[Tensor] | None = None,
        step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Forecast graph snapshots at arbitrary query times.

        Exactly one of ``query_times`` or ``step_deltas`` must be provided.
        ``query_times`` are absolute times relative to the initial snapshot at
        ``t = 0``. ``step_deltas`` are positive integration intervals applied
        sequentially from the initial state.

        In discrete mode, non-uniform ``step_deltas`` or ``query_times`` raise
        :class:`ValueError` because the learned operator is tied to a fixed
        :attr:`time_step`.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot at ``t = 0``.
        query_times : sequence of float or Tensor or None, optional
            Strictly increasing absolute query times, each positive.
        step_deltas : sequence of float or Tensor or None, optional
            Strictly positive integration intervals applied in order.
        edge_index, edge_weight, controls, future_topologies
            Same semantics as :meth:`predict`.

        Returns
        -------
        list of Data
            Predicted snapshots, one per query interval.
        """
        increments = self._resolve_time_increments(
            query_times=query_times,
            step_deltas=step_deltas,
        )
        if not self.is_continuous:
            self._validate_uniform_discrete_increments(increments)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                rollout = self._rollout(
                    initial_graph,
                    len(increments),
                    edge_index,
                    edge_weight,
                    controls=controls,
                    future_topologies=future_topologies,
                    step_deltas=increments,
                )
        finally:
            self.train(was_training)

        output_snapshots: list[Data] = []
        for prediction, step_edge_index, step_edge_weight in rollout:
            fields: dict[str, Tensor] = {
                "x": prediction,
                "edge_index": step_edge_index,
            }
            if step_edge_weight is not None:
                fields["edge_weight"] = step_edge_weight
            output_snapshots.append(Data(**fields))
        return output_snapshots

    @staticmethod
    def _resolve_time_increments(
        *,
        query_times: Sequence[float] | Sequence[Tensor] | None,
        step_deltas: Sequence[float] | Sequence[Tensor] | None,
    ) -> list[float]:
        """Convert query specification into positive integration increments.

        Returns
        -------
        list of float
            Strictly positive integration intervals.
        """
        if (query_times is None) == (step_deltas is None):
            msg = "exactly one of query_times or step_deltas must be provided"
            raise ValueError(msg)

        if step_deltas is not None:
            increments = [float(torch.as_tensor(value).item()) for value in step_deltas]
            if not increments or any(value <= 0 for value in increments):
                msg = "step_deltas must be non-empty and strictly positive"
                raise ValueError(msg)
            return increments

        assert query_times is not None
        times = [float(torch.as_tensor(value).item()) for value in query_times]
        if not times or any(value <= 0 for value in times):
            msg = "query_times must be non-empty and strictly positive"
            raise ValueError(msg)
        previous = 0.0
        increments = []
        for value in times:
            if value <= previous:
                msg = "query_times must be strictly increasing"
                raise ValueError(msg)
            increments.append(value - previous)
            previous = value
        return increments

    def _validate_uniform_discrete_increments(
        self,
        increments: Sequence[float],
    ) -> None:
        """Ensure discrete models only receive uniform time increments.

        Raises
        ------
        ValueError
            If any increment differs from :attr:`time_step`.
        """
        tolerance = max(1e-6, 1e-4 * self.time_step)
        for value in increments:
            if abs(value - self.time_step) > tolerance:
                msg = (
                    "discrete dynamics_mode requires uniform increments equal to "
                    f"time_step={self.time_step}; got {value}. Use "
                    "dynamics_mode='continuous' for irregular sampling."
                )
                raise ValueError(msg)

    def evaluate(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        horizons: Sequence[int] = (3, 6, 12),
        start_indices: Sequence[int] | None = None,
    ) -> EvaluationResult:
        """Evaluate multi-horizon forecast accuracy on a snapshot sequence.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Evaluation snapshots with shared topology.
        horizons : sequence of int, optional
            Forecast horizons to report. Default is ``(3, 6, 12)``.
        start_indices : sequence of int or None, optional
            Forecast-origin indices. When ``None``, uses every valid origin in
            ``sequence``.

        Returns
        -------
        EvaluationResult
            Per-horizon and aggregate MAE, RMSE, and MAPE.
        """
        return evaluate_forecast(
            self,
            resolve_sequence(sequence),
            horizons=horizons,
            start_indices=start_indices,
        )

    def fit(
        self,
        data_sequence: TrainingInput,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        optimizer: Callable[..., Optimizer] = torch.optim.Adam,
        device: str | torch.device | None = None,
        loss_weights: LossWeights | None = None,
        loss_weight_schedule: LossWeightSchedule | None = None,
        rollout_horizon: int | None = None,
        rollout_start_indices: RolloutStartIndices = None,
        rollout_starts_per_epoch: int | None = None,
        rollout_start_seed: int | None = None,
        lr_scheduler: LRScheduler | LRSchedulerFactory | None = None,
        window_length: int | None = None,
        batch_size: int = 8,
        windows_per_epoch: int | None = None,
        window_seed: int | None = None,
        max_grad_norm: float | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        early_stopping_monitor: EarlyStoppingMonitor = "auto",
        validation_sequence: ValidationInput = None,
        restore_best_weights: bool = False,
        checkpoint_path: str | Path | None = None,
        **optimizer_kwargs: Any,
    ) -> FitHistory:
        """Train encoder, Koopman operator, and decoder end-to-end.

        Minimizes a weighted sum of one-step MSE and optional forward and
        backward consistency terms::

            loss = w_r * recon_loss
                 + w_f * ||K z_t - z_{t+1}||^2
                 + w_b * ||z_t - z_{t+1} K^{\\dagger}||^2

        where ``z_t`` and ``z_{t+1}`` are encoder outputs for consecutive
        snapshots and weights ``(w_r, w_f, w_b)`` come from a
        :class:`~koopman_graph.training.LossWeights` object or an optional
        per-epoch schedule.

        When ``data_sequence`` is a list of
        :class:`~koopman_graph.data.GraphSnapshotSequence` objects, losses are
        averaged across trajectories before each optimizer step.

        Parameters
        ----------
        data_sequence : GraphSnapshotSequence, sequence of Data, or sequence of \
GraphSnapshotSequence
            One training trajectory or multiple trajectories of the same
            system. A plain list of ``Data`` snapshots is treated as a single
            trajectory; a list whose first element is a
            :class:`~koopman_graph.data.GraphSnapshotSequence` is treated as
            multiple trajectories.
        epochs : int, optional
            Number of training epochs. Default is ``100``.
        lr : float, optional
            Learning rate passed to the optimizer. Default is ``1e-3``.
        optimizer : callable, optional
            Optimizer class. Default is :class:`torch.optim.Adam`.
        device : str, torch.device, or None, optional
            Device for training. Defaults to the model's current device, or CPU
            if the model has no parameters.
        loss_weights : LossWeights or None, optional
            Static loss weights for all epochs. When ``None`` and no schedule is
            provided, defaults to reconstruction-only training.
        loss_weight_schedule : callable or None, optional
            Callable ``epoch -> LossWeights`` applied each epoch. Overrides
            ``loss_weights`` when set.
        rollout_horizon : int or None, optional
            Number of autoregressive rollout steps used when
            ``loss_weights.rollout`` is non-zero. Defaults to
            ``num_timesteps - 1``.
        rollout_start_indices : sequence of int, ``"all"``, or None, optional
            Rollout-loss origin indices. ``None`` uses ``[0]``; ``"all"`` uses
            every valid origin for the rollout horizon.
        rollout_starts_per_epoch : int or None, optional
            When set, randomly sample this many rollout origins each epoch.
        rollout_start_seed : int or None, optional
            Base seed for random rollout-origin sampling. The effective seed is
            ``rollout_start_seed + epoch``.
        lr_scheduler : LRScheduler or callable, optional
            Learning-rate scheduler instance or factory
            ``optimizer -> scheduler``. Stepped once per epoch after the
            optimizer update.
        window_length : int or None, optional
            Fixed number of snapshots per training window. When set, enables
            mini-batch training with multiple optimizer steps per epoch.
            ``None`` preserves full-sequence single-step training.
        batch_size : int, optional
            Number of temporal windows averaged per optimizer step. Used only
            when ``window_length`` is set. Default is ``8``.
        windows_per_epoch : int or None, optional
            Maximum sampled windows per epoch. ``None`` uses every valid
            window across all trajectories.
        window_seed : int or None, optional
            Base seed for reproducible epoch-specific window shuffling.
        max_grad_norm : float or None, optional
            When set, clip the global gradient norm before each optimizer step.
        early_stopping_patience : int or None, optional
            Stop training when training loss fails to improve for this many
            consecutive epochs. Disabled when ``None``.
        early_stopping_min_delta : float, optional
            Minimum decrease in the monitored loss to count as improvement.
            Default is ``0.0``.
        early_stopping_monitor : {"auto", "train", "val"}, optional
            Loss used for early stopping and best-epoch tracking. ``"auto"``
            monitors validation loss when ``validation_sequence`` is provided,
            otherwise training loss. Default is ``"auto"``.
        validation_sequence : GraphSnapshotSequence, sequence of Data, \
sequence of GraphSnapshotSequence, or None, optional
            Optional held-out snapshots for per-epoch validation loss. A single
            validation sequence is reused for all training trajectories; a list
            of validation sequences must match the training trajectory count.
        restore_best_weights : bool, optional
            When ``True``, reload in-memory weights from the lowest-loss epoch
            after training completes. Default is ``False``.
        checkpoint_path : str, Path, or None, optional
            When set, write a checkpoint at the lowest-loss epoch using
            :meth:`save`. Default is ``None``.
        **optimizer_kwargs
            Additional keyword arguments forwarded to the optimizer constructor.

        Returns
        -------
        :class:`~koopman_graph.training.FitHistory`
            Per-epoch training and validation losses and early-stop metadata.

        Raises
        ------
        ValueError
            If ``epochs < 1``, ``early_stopping_patience < 1`` when set,
            ``early_stopping_monitor="val"`` without ``validation_sequence``,
            validation list length mismatches training trajectories, or fewer
            than two snapshots are provided for training or validation.
        """
        if epochs < 1:
            msg = f"epochs must be >= 1, got {epochs}"
            raise ValueError(msg)
        if early_stopping_patience is not None and early_stopping_patience < 1:
            msg = (
                f"early_stopping_patience must be >= 1 when set, "
                f"got {early_stopping_patience}"
            )
            raise ValueError(msg)

        train_sequences = resolve_training_sequences(data_sequence)
        for sequence in train_sequences:
            self._validate_sequence_controls(sequence)
            if sequence.num_timesteps < 2:
                msg = "data_sequence must contain at least 2 snapshots for training"
                raise ValueError(msg)

        val_sequences = resolve_validation_sequences(
            validation_sequence,
            num_training_sequences=len(train_sequences),
        )
        if val_sequences is not None:
            for sequence in val_sequences:
                self._validate_sequence_controls(sequence)
                if sequence.num_timesteps < 2:
                    msg = (
                        "validation_sequence must contain at least 2 snapshots "
                        "for validation"
                    )
                    raise ValueError(msg)

        monitor = resolve_early_stopping_monitor(
            early_stopping_monitor,
            has_validation=val_sequences is not None,
        )

        train_device = resolve_device(self, device)
        self.to(train_device)
        train_sequences = [
            self._sequence_to_device(sequence, train_device)
            for sequence in train_sequences
        ]
        if val_sequences is not None:
            val_sequences = [
                self._sequence_to_device(sequence, train_device)
                for sequence in val_sequences
            ]

        optim = optimizer(self.parameters(), lr=lr, **optimizer_kwargs)
        scheduler = resolve_lr_scheduler(lr_scheduler, optim)
        window_sampler = (
            None
            if window_length is None
            else WindowSampler(
                train_sequences,
                window_length=window_length,
                batch_size=batch_size,
                windows_per_epoch=windows_per_epoch,
                seed=window_seed,
            )
        )
        losses: list[float] = []
        reconstruction_losses: list[float] = []
        forward_losses: list[float] = []
        backward_losses: list[float] = []
        rollout_losses: list[float] = []
        eigenvalue_losses: list[float] = []
        val_losses: list[float] | None = [] if val_sequences is not None else None
        val_reconstruction_losses: list[float] | None = (
            [] if val_sequences is not None else None
        )
        val_forward_losses: list[float] | None = (
            [] if val_sequences is not None else None
        )
        val_backward_losses: list[float] | None = (
            [] if val_sequences is not None else None
        )
        val_rollout_losses: list[float] | None = (
            [] if val_sequences is not None else None
        )
        val_eigenvalue_losses: list[float] | None = (
            [] if val_sequences is not None else None
        )
        best_loss_for_stop = float("inf")
        best_loss: float | None = None
        best_epoch: int | None = None
        best_state_dict: dict[str, Tensor] | None = None
        track_best = restore_best_weights or checkpoint_path is not None
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(epochs):
            epoch_weights = resolve_loss_weights_for_epoch(
                epoch,
                loss_weights=loss_weights,
                loss_weight_schedule=loss_weight_schedule,
            )
            epoch_rollout_starts: list[int] | None = None
            if window_sampler is None:
                rollout_horizon_for_epoch = (
                    train_sequences[0].num_timesteps - 1
                    if rollout_horizon is None
                    else rollout_horizon
                )
                epoch_rollout_starts = resolve_rollout_start_indices(
                    train_sequences[0],
                    horizon=rollout_horizon_for_epoch,
                    rollout_start_indices=rollout_start_indices,
                    rollout_starts_per_epoch=rollout_starts_per_epoch,
                    rollout_start_seed=rollout_start_seed,
                    epoch=epoch,
                )
                breakdown = train_one_epoch(
                    self,
                    train_sequences,
                    optim,
                    epoch_weights,
                    max_grad_norm=max_grad_norm,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=epoch_rollout_starts,
                )
            else:
                breakdown = train_windowed_epoch(
                    self,
                    window_sampler,
                    optim,
                    epoch_weights,
                    epoch=epoch,
                    max_grad_norm=max_grad_norm,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                    rollout_starts_per_epoch=rollout_starts_per_epoch,
                    rollout_start_seed=rollout_start_seed,
                )
            if scheduler is not None:
                scheduler.step()

            term_values = breakdown.to_floats()
            losses.append(term_values["total"])
            reconstruction_losses.append(term_values["reconstruction"])
            forward_losses.append(term_values["forward"])
            backward_losses.append(term_values["backward"])
            rollout_losses.append(term_values["rollout"])
            eigenvalue_losses.append(term_values["eigenvalue"])

            monitored_loss = term_values["total"]
            if val_sequences is not None:
                val_breakdown = eval_one_epoch(
                    self,
                    val_sequences,
                    epoch_weights,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=epoch_rollout_starts,
                )
                val_terms = val_breakdown.to_floats()
                assert val_losses is not None
                assert val_reconstruction_losses is not None
                assert val_forward_losses is not None
                assert val_backward_losses is not None
                assert val_rollout_losses is not None
                assert val_eigenvalue_losses is not None
                val_losses.append(val_terms["total"])
                val_reconstruction_losses.append(val_terms["reconstruction"])
                val_forward_losses.append(val_terms["forward"])
                val_backward_losses.append(val_terms["backward"])
                val_rollout_losses.append(val_terms["rollout"])
                val_eigenvalue_losses.append(val_terms["eigenvalue"])
                if monitor == "val":
                    monitored_loss = val_terms["total"]

            if track_best and (best_loss is None or monitored_loss < best_loss):
                best_loss = monitored_loss
                best_epoch = epoch
                best_state_dict = snapshot_state_dict(self)

            if early_stopping_patience is not None:
                stop, best_loss_for_stop, epochs_without_improvement = (
                    should_stop_early(
                        epoch_loss=monitored_loss,
                        best_loss=best_loss_for_stop,
                        epochs_without_improvement=epochs_without_improvement,
                        patience=early_stopping_patience,
                        min_delta=early_stopping_min_delta,
                    )
                )
                if stop:
                    stopped_early = True
                    break

        if track_best and best_state_dict is not None:
            last_state_dict: dict[str, Tensor] | None = None
            if not restore_best_weights:
                last_state_dict = snapshot_state_dict(self)
            self.load_state_dict(best_state_dict)
            if checkpoint_path is not None:
                save_checkpoint(self, checkpoint_path)
            if not restore_best_weights and last_state_dict is not None:
                self.load_state_dict(last_state_dict)

        return FitHistory(
            loss=losses,
            epochs=len(losses),
            reconstruction_loss=reconstruction_losses,
            forward_loss=forward_losses,
            backward_loss=backward_losses,
            rollout_loss=rollout_losses,
            eigenvalue_loss=eigenvalue_losses,
            val_loss=val_losses,
            val_reconstruction_loss=val_reconstruction_losses,
            val_forward_loss=val_forward_losses,
            val_backward_loss=val_backward_losses,
            val_rollout_loss=val_rollout_losses,
            val_eigenvalue_loss=val_eigenvalue_losses,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            best_loss=best_loss,
        )

    def _sequence_to_device(
        self,
        sequence: GraphSnapshotSequence,
        train_device: torch.device,
    ) -> GraphSnapshotSequence:
        """Move a snapshot sequence and optional controls to ``train_device``.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Sequence to move.
        train_device : torch.device
            Target device.

        Returns
        -------
        GraphSnapshotSequence
            Device-local copy of ``sequence``.
        """
        return GraphSnapshotSequence(
            [self._snapshot_to_device(snapshot, train_device) for snapshot in sequence],
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs.to(train_device)
            ),
            timestamps=(
                None
                if sequence.timestamps is None
                else sequence.timestamps.to(train_device)
            ),
        )

    @staticmethod
    def _resolve_edge_index(
        x_or_data: Tensor | Data,
        edge_index: Tensor | None,
    ) -> Tensor:
        """Extract or validate ``edge_index`` from input arguments.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Graph input; when a ``Data`` object, its ``edge_index`` is returned.
        edge_index : Tensor or None
            Explicit edge index for tensor input.

        Returns
        -------
        Tensor
            Edge index with shape ``(2, num_edges)``.

        Raises
        ------
        ValueError
            If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
        """
        if isinstance(x_or_data, Data):
            return x_or_data.edge_index
        if edge_index is None:
            msg = "edge_index is required when x_or_data is a tensor"
            raise ValueError(msg)
        return edge_index

    @staticmethod
    def _resolve_edge_weight(
        x_or_data: Tensor | Data,
        edge_weight: Tensor | None,
    ) -> Tensor | None:
        """Extract or validate optional ``edge_weight`` from input arguments.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Graph input; when a ``Data`` object, its ``edge_weight`` is returned
            when present.
        edge_weight : Tensor or None
            Explicit edge weights for tensor input.

        Returns
        -------
        Tensor or None
            Edge weights with shape ``(num_edges,)``, or ``None`` when unweighted.
        """
        if isinstance(x_or_data, Data):
            return _snapshot_edge_weight(x_or_data)
        return edge_weight

    @staticmethod
    def _as_data(
        x_or_data: Tensor | Data,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Data:
        """Build a PyG ``Data`` object from tensor or ``Data`` inputs.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features or an existing snapshot.
        edge_index : Tensor
            Edge index with shape ``(2, num_edges)``.
        edge_weight : Tensor or None
            Optional edge weights with shape ``(num_edges,)``.

        Returns
        -------
        Data
            Snapshot suitable for physics lifting callables.
        """
        if isinstance(x_or_data, Data):
            return x_or_data
        data = Data(x=x_or_data, edge_index=edge_index)
        if edge_weight is not None:
            data.edge_weight = edge_weight
        return data

    def _validate_controls(
        self,
        controls: Sequence[Tensor] | None,
        *,
        steps: int,
    ) -> None:
        """Validate rollout controls against model control settings.

        Parameters
        ----------
        controls : sequence of Tensor or None
            Control inputs for each rollout step.
        steps : int
            Number of rollout steps.

        Raises
        ------
        ValueError
            If controls are missing, surplus, or provided to an uncontrolled
            model.
        """
        if self.control_dim == 0:
            if controls is not None:
                msg = "controls provided to an uncontrolled model"
                raise ValueError(msg)
            return
        if controls is None:
            msg = "controls are required when control_dim > 0"
            raise ValueError(msg)
        if len(controls) != steps:
            msg = f"expected {steps} control inputs for rollout, got {len(controls)}"
            raise ValueError(msg)

    def _validate_sequence_controls(
        self,
        sequence: GraphSnapshotSequence,
    ) -> None:
        """Validate sequence controls against this model's control dimension.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Training or validation sequence.

        Raises
        ------
        ValueError
            If controls are missing or dimensions disagree.
        """
        if self.control_dim == 0:
            if sequence.has_controls:
                msg = "sequence contains control inputs but model control_dim is 0"
                raise ValueError(msg)
            return
        if not sequence.has_controls:
            msg = "controlled model requires sequences with control inputs"
            raise ValueError(msg)
        if sequence.control_dim != self.control_dim:
            msg = (
                f"sequence control_dim ({sequence.control_dim}) must match "
                f"model control_dim ({self.control_dim})"
            )
            raise ValueError(msg)

    @staticmethod
    def _snapshot_to_device(snapshot: Data, device: torch.device) -> Data:
        """Move a graph snapshot to a target device, preserving edge weights.

        Parameters
        ----------
        snapshot : Data
            Graph snapshot to transfer.
        device : torch.device
            Destination device.

        Returns
        -------
        Data
            Snapshot with tensors moved to ``device``.
        """
        fields: dict[str, Tensor] = {
            "x": snapshot.x.to(device),
            "edge_index": snapshot.edge_index.to(device),
        }
        edge_weight = _snapshot_edge_weight(snapshot)
        if edge_weight is not None:
            fields["edge_weight"] = edge_weight.to(device)
        return Data(**fields)

    def to_latent_env(
        self,
        sequence: GraphSnapshotSequence,
        reward_fn: Callable[[Data, int], float],
        *,
        control_low: float | Sequence[float] = -1.0,
        control_high: float | Sequence[float] = 1.0,
        max_episode_steps: int = 50,
        start_index: int | None = None,
        random_start: bool = True,
        delta_t: float | None = None,
        device: torch.device | str | None = None,
    ) -> GraphKoopmanEnv:
        """Build a Gymnasium environment for latent-space closed-loop control.

        Freezes encoder and decoder parameters so RL interacts only through the
        Koopman control input while rewards are computed on decoded physical
        graph states. Requires ``control_dim > 0`` and the optional
        ``[rl]`` install extra (``gymnasium``).

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Reference snapshots for reset states and fixed episode topology.
        reward_fn : callable
            ``reward_fn(decoded_snapshot, step_index) -> float``.
        control_low : float or sequence of float, optional
            Lower action bounds. Default is ``-1.0``.
        control_high : float or sequence of float, optional
            Upper action bounds. Default is ``1.0``.
        max_episode_steps : int, optional
            Episode horizon. Default is ``50``.
        start_index : int or None, optional
            Fixed reset index into ``sequence``. When set, ``random_start`` is
            ignored.
        random_start : bool, optional
            Sample a random snapshot on each ``reset``. Default is ``True``.
        delta_t : float or None, optional
            Integration interval for each environment step. When ``None``,
            uses :attr:`time_step`. Continuous models may use a custom
            horizon; discrete models require ``delta_t is None`` or
            ``delta_t == time_step``.
        device : torch.device or str or None, optional
            Inference device. Defaults to the model's current device.

        Returns
        -------
        GraphKoopmanEnv
            Configured Gymnasium environment with flattened latent
            observations.

        Raises
        ------
        ValueError
            If ``control_dim`` is zero or arguments are invalid.
        ImportError
            If Gymnasium is not installed.
        """
        from koopman_graph.env import GraphKoopmanEnv

        if self.control_dim <= 0:
            msg = "to_latent_env requires control_dim > 0"
            raise ValueError(msg)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

        return GraphKoopmanEnv(
            self,
            sequence,
            reward_fn,
            control_low=control_low,
            control_high=control_high,
            max_episode_steps=max_episode_steps,
            start_index=start_index,
            random_start=random_start,
            delta_t=delta_t,
            device=device,
        )
