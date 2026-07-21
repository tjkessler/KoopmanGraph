"""GraphKoopmanModel: encoder, Koopman operator, and decoder composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data

from koopman_graph.adaptation import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.data import GraphSnapshotSequence, RolloutStartIndices
from koopman_graph.graph_utils import (
    propagate_latent,
    resolve_delta_t,
    resolve_edge_index,
    resolve_edge_weight,
)
from koopman_graph.metrics import EvaluationResult
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    PhysicsPosition,
)
from koopman_graph.operators import GraphKoopmanOperator, InitMode, Parameterization
from koopman_graph.operators.control import ControlMode
from koopman_graph.protocols import DynamicsMode
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.training import (
    EarlyStoppingMonitor,
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
    TrainingInput,
    ValidationInput,
    run_fit_loop,
)

from .encoding import (
    encode_at_index,
    encode_features,
)
from .encoding import (
    encode_rollout_origin as encode_rollout_origin_helper,
)
from .factory import (
    DEFAULT_BILINEAR_RANK,
    DEFAULT_CONTROL_MODE,
    DEFAULT_KOOPMAN_INIT_MODE,
    DEFAULT_KOOPMAN_INIT_SCALE,
    DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS,
    DEFAULT_KOOPMAN_PARAMETERIZATION,
    Decoder,
    Encoder,
    KoopmanArg,
    apply_resolved_components,
    resolve_model_components,
)
from .inference import (
    compute_model_spectrum,
    evaluate_sequence,
    latent_decode_rollout,
    predict_at_snapshots,
    predict_snapshots,
)
from .online_adaptation import (
    disable_online_adaptation as disable_online_adaptation_helper,
)
from .online_adaptation import (
    enable_online_adaptation as enable_online_adaptation_helper,
)
from .online_adaptation import (
    freeze_modules,
    run_adapt_step,
)
from .validation import prepare_fit_inputs

if TYPE_CHECKING:
    from koopman_graph.env import GraphKoopmanEnv


class GraphKoopmanModel(nn.Module):
    """Topology-aware Koopman dynamics model for graph snapshots.

    Composes a GNN encoder (lifting), a finite-dimensional Koopman operator
    (linear latent evolution), and a symmetric GNN decoder (reconstruction).

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and the narrower
    :class:`~koopman_graph.protocols.UncontrolledForecastModel` peer set when
    ``control_dim == 0`` and called as ``predict(data, steps)``. ``predict``
    also accepts tensors, optional controls, and future topologies — those
    kwargs are not portable to classical DMD/EDMD baselines. See the
    architecture docs call-site matrix. Training and metrics duck-typing beyond
    the forecasting façade uses
    :class:`~koopman_graph.protocols.TrainableKoopmanModel`.

    Attributes
    ----------
    encoder
        Topology-aware encoder for latent lifting
        (``GNNEncoder`` / ``GATEncoder`` / ``SAGEEncoder`` /
        ``DiffConvEncoder`` / ``DelayEmbeddingEncoder``).
    decoder
        Symmetric GNN decoder for physical reconstruction
        (``GNNDecoder`` / ``GATDecoder`` / ``SAGEDecoder`` /
        ``DiffConvDecoder``).
    latent_dim : int
        Latent space dimension shared by encoder, operator, and decoder.
    time_step : float
        Physical time increment associated with one model step. Used by
        :meth:`spectrum` to convert discrete eigenvalues into continuous-time
        growth rates and frequencies.
    koopman : KoopmanOperatorContract
        Learnable linear propagator in latent space. Built-in discrete,
        continuous, or networked operators by default; optionally an injected
        :class:`~koopman_graph.operators.KoopmanOperatorContract` ``nn.Module``.
    koopman_kind : {"pernode", "graph"}
        Factory kind used when constructing a built-in discrete operator
        (``"graph"`` selects :class:`~koopman_graph.operators.GraphKoopmanOperator`).
    dynamics_mode : {"discrete", "continuous"}
        Whether latent evolution uses a discrete step map or a continuous
        generator integrated with matrix exponentials.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_dim: int,
        time_step: float,
        *,
        dynamics_mode: DynamicsMode = "discrete",
        koopman: KoopmanArg = None,
        koopman_init_mode: InitMode = DEFAULT_KOOPMAN_INIT_MODE,
        koopman_init_scale: float = DEFAULT_KOOPMAN_INIT_SCALE,
        koopman_parameterization: Parameterization = DEFAULT_KOOPMAN_PARAMETERIZATION,
        koopman_max_spectral_radius: float = DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS,
        koopman_auxiliary_hidden_dims: Sequence[int] | None = None,
        control_dim: int = 0,
        control_mode: ControlMode = DEFAULT_CONTROL_MODE,
        bilinear_rank: int | None = DEFAULT_BILINEAR_RANK,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
        physics_preset: str | None = None,
        physics_dim: int = 0,
        physics_position: PhysicsPosition = PHYSICS_POSITION,
        n_delays: int = 1,
    ) -> None:
        """Initialize encoder, decoder, and Koopman operator.

        Parameters
        ----------
        encoder
            Topology-aware encoder for latent lifting
            (``GNNEncoder`` / ``GATEncoder`` / ``SAGEEncoder`` /
            ``DiffConvEncoder`` / ``DelayEmbeddingEncoder``).
            When ``n_delays > 1``, pass a base encoder already sized with
            ``in_channels = n_delays * feature_dim`` (composition; layers are
            not rebuilt) or an existing
            :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`.
        decoder
            Symmetric GNN decoder for physical reconstruction
            (``GNNDecoder`` / ``GATDecoder`` / ``SAGEDecoder`` /
            ``DiffConvDecoder``).
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
            exponentials. Default is ``"discrete"``. When injecting a built-in
            operator, ``dynamics_mode`` must match its type. Networked
            ``koopman="graph"`` requires ``dynamics_mode="discrete"``.
        koopman : KoopmanOperatorContract, {"pernode", "graph"}, or None, optional
            Operator selection. Pass ``"pernode"`` (default) or ``"graph"`` to
            construct a built-in discrete operator, or inject a pre-built
            :class:`~koopman_graph.operators.KoopmanOperatorContract`
            ``nn.Module``. When injecting, factory kwargs must remain at their
            defaults. Continuous models ignore ``"graph"`` (raises).
        koopman_init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization strategy for the Koopman matrix. Default is
            ``"identity_noise"``. Ignored (and must stay default) when
            ``koopman`` is an injected module.
        koopman_init_scale : float, optional
            Noise scale when ``koopman_init_mode="identity_noise"``.
            Default is ``1e-2``.
        koopman_parameterization : {"dense", "odo", "schur", "dissipative", "lyapunov",
            "auxiliary_spectral"}, optional
            Koopman matrix parameterization. ``"odo"`` enforces a spectral-radius
            bound via orthogonal-diagonal-orthogonal factors. ``"schur"``,
            ``"dissipative"``, and ``"lyapunov"`` embed structural stability
            guarantees for long-horizon rollouts. Continuous-only
            ``"auxiliary_spectral"`` uses a state-dependent auxiliary network
            (Lusch-style locally linear spectrum). Default is ``"dense"``.
        koopman_max_spectral_radius : float, optional
            Maximum eigenvalue magnitude for bounded/structural parameterizations.
            Structurally stable modes enforce a strict interior margin below
            this value. Default is ``1.0``.
        koopman_auxiliary_hidden_dims : sequence of int or None, optional
            Hidden widths for ``koopman_parameterization="auxiliary_spectral"``
            (default ``(64, 64)``). Must stay default / ``None`` when injecting
            ``koopman=...``.
        control_dim : int, optional
            Dimension of exogenous control inputs. When ``0``, the model is
            uncontrolled. Default is ``0``. Must match ``koopman.control_dim``
            when an operator is injected.
        control_mode : {"additive", "bilinear"}, optional
            How controls enter the latent map. ``"additive"`` (default) uses
            ``z @ K.T + u @ B``. ``"bilinear"`` adds state–control couplings
            for control-affine systems. Must match an injected operator's
            ``control_mode`` when present.
        bilinear_rank : int or None, optional
            Optional low-rank size for bilinear ``N_i = P_i Q_i^T``. ``None``
            stores full-rank ``N_i``. Only valid with ``control_mode="bilinear"``.
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
        physics_position : {"prepend"}, optional
            Where physics features sit relative to GNN embeddings in the hybrid
            latent. Only ``"prepend"`` is supported today. Round-tripped via
            checkpoint ``physics.position``.
        n_delays : int, optional
            Hankel / delay-embedding window length at the encoder boundary.
            ``1`` preserves single-snapshot encoding (default). When ``> 1``,
            a bare :class:`~koopman_graph.nn.encoder.GNNEncoder` /
            :class:`~koopman_graph.nn.encoder.GATEncoder` /
            :class:`~koopman_graph.nn.encoder.SAGEEncoder` /
            :class:`~koopman_graph.nn.encoder.DiffConvEncoder` is wrapped in
            :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` without
            rebuilding layers — size ``encoder.in_channels = n_delays *
            feature_dim`` yourself. Autoregressive ``predict`` encodes the
            provided observation history **once**, then advances in latent
            space (decoded rollouts are **not** fed back as delay coordinates
            by default).

        Raises
        ------
        ValueError
            If ``latent_dim`` is not positive, ``time_step <= 0``,
            ``control_dim < 0``, ``n_delays < 1``, physics settings are
            inconsistent, encoder/decoder latent dimensions do not match the
            effective hybrid layout, an injected operator conflicts with
            factory kwargs or dimensions, ``dynamics_mode`` disagrees with a
            built-in injected operator type, or ``koopman="graph"`` is
            requested in continuous mode.
        TypeError
            If ``koopman`` is provided but is not a string kind or ``nn.Module``.
        """
        super().__init__()
        components = resolve_model_components(
            encoder,
            decoder,
            latent_dim,
            time_step,
            dynamics_mode=dynamics_mode,
            koopman=koopman,
            koopman_init_mode=koopman_init_mode,
            koopman_init_scale=koopman_init_scale,
            koopman_parameterization=koopman_parameterization,
            koopman_max_spectral_radius=koopman_max_spectral_radius,
            koopman_auxiliary_hidden_dims=koopman_auxiliary_hidden_dims,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
            physics_lifting_fn=physics_lifting_fn,
            physics_preset=physics_preset,
            physics_dim=physics_dim,
            physics_position=physics_position,
            n_delays=n_delays,
        )
        apply_resolved_components(self, components)

    @property
    def uses_graph_koopman(self) -> bool:
        """Return whether latent advance uses the networked graph operator.

        Returns
        -------
        bool
            ``True`` when :attr:`koopman` is a
            :class:`~koopman_graph.operators.GraphKoopmanOperator`.
        """
        return isinstance(self.koopman, GraphKoopmanOperator)

    @property
    def is_continuous(self) -> bool:
        """Return whether the model uses continuous-time generator dynamics.

        Returns
        -------
        bool
            ``True`` when :attr:`dynamics_mode` is ``"continuous"``.
        """
        return self.dynamics_mode == "continuous"

    def resolve_delta_t(
        self,
        delta_t: float | Tensor | None = None,
    ) -> float | Tensor:
        """Resolve the continuous integration interval for this model.

        Missing ``delta_t`` falls back to :attr:`time_step`. Training, losses,
        evaluation, and :class:`~koopman_graph.env.GraphKoopmanEnv` share this
        policy for model-backed continuous paths. Standalone operators without
        a model still default to ``1.0`` via
        :func:`~koopman_graph.graph_utils.resolve_delta_t`.

        Parameters
        ----------
        delta_t : float, Tensor, or None, optional
            Explicit interval. When ``None``, returns :attr:`time_step`.

        Returns
        -------
        float or Tensor
            Resolved integration interval.
        """
        return resolve_delta_t(delta_t, default_delta_t=self.time_step)

    def _advance_latent(
        self,
        z: Tensor,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states with the active Koopman operator.

        Returns
        -------
        Tensor
            Advanced latent states.
        """
        return propagate_latent(
            self.koopman,
            z,
            control=control,
            delta_t=self.resolve_delta_t(delta_t),
            default_delta_t=self.time_step,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def spectrum(
        self,
        *,
        delta_t: float | None = None,
        edge_index: Tensor | None = None,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
    ) -> KoopmanSpectrum:
        """Analyze the learned Koopman operator spectrum.

        For ordinary discrete / continuous / custom injected operators, uses
        :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix` (the
        per-node ``K`` / ``L``). In continuous mode, returns the generator
        spectrum by default; pass ``delta_t`` for the discrete-time spectrum of
        ``exp(L·Δt)``.

        For ``koopman="graph"``, analyzes the topology-coupled effective
        operator ``I⊗K_self + Â⊗K_nbr`` and **requires** ``edge_index`` and
        ``num_nodes`` matching the topology used for propagation. Missing
        topology raises rather than silently returning the ``K_self`` spectrum.
        Classical DMD-family baselines take no ``spectrum`` kwargs.

        Parameters
        ----------
        delta_t : float or None, optional
            Continuous integration horizon for generator → discrete spectrum.
            Ignored for discrete / graph operators.
        edge_index : Tensor or None, optional
            Topology for networked graph operators. Required when
            :attr:`uses_graph_koopman` is ``True``.
        num_nodes : int or None, optional
            Node count ``N`` for the effective ``N·d`` operator. Required when
            :attr:`uses_graph_koopman` is ``True``.
        edge_weight : Tensor or None, optional
            Optional edge weights with the same semantics as latent advance.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted eigenvalues, eigenvectors, and time scales.

        Raises
        ------
        ValueError
            If a graph operator is active and ``edge_index`` or ``num_nodes``
            is missing.
        """
        return compute_model_spectrum(
            self.koopman,
            uses_graph_koopman=self.uses_graph_koopman,
            is_continuous=self.is_continuous,
            time_step=self.time_step,
            delta_t=delta_t,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        )

    def encode(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Lift graph node features into the hybrid Koopman latent space.

        When physics-informed observables are configured, returns
        ``[z_physics || z_gnn]`` with shape ``(num_nodes, latent_dim)``.

        For ``n_delays > 1``, ``x_or_data`` may be a delay window
        ``(n_delays, num_nodes, F)``, stacked features
        ``(num_nodes, n_delays * F)``, or a ``Data`` whose ``x`` is already
        stacked. Prefer :meth:`encode_at` when lifting from a
        :class:`~koopman_graph.data.GraphSnapshotSequence` so teacher-forced
        history is assembled correctly.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features, delay window, or a PyG ``Data`` snapshot.
        edge_index : Tensor or None, optional
            Edge index required when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights for tensor input.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        return encode_features(
            self.encoder,
            x_or_data,
            edge_index,
            edge_weight,
            physics_lifting_fn=self.physics_lifting_fn,
            physics_dim=self.physics_dim,
            physics_position=self.physics_position,
        )

    def encode_rollout_origin(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        history: Sequence[Data] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Encode the initial state for an autoregressive rollout.

        Matches the encode preamble of :meth:`predict` / ``_rollout``:
        delay windows use :func:`~koopman_graph.nn.delay.history_from_snapshots`
        when ``n_delays > 1`` and ``x_or_data`` is a ``Data`` snapshot.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Initial node features or graph snapshot.
        edge_index : Tensor or None, optional
            Edge index when ``x_or_data`` is a tensor (or override).
        edge_weight : Tensor or None, optional
            Optional edge weights.
        history : sequence of Data or None, optional
            Past snapshots (oldest → newest) for delay embedding.

        Returns
        -------
        tuple of Tensor, Tensor, Tensor or None
            Encoded latent ``z``, resolved ``edge_index``, and optional
            ``edge_weight`` at the rollout origin.
        """
        return encode_rollout_origin_helper(
            self.encode,
            n_delays=self.n_delays,
            x_or_data=x_or_data,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history=history,
        )

    def encode_at(
        self,
        sequence: GraphSnapshotSequence,
        index: int,
        *,
        pad: bool = True,
        zero_unobserved: bool = True,
    ) -> Tensor:
        """Encode the delay window of ``sequence`` ending at ``index``.

        When ``n_delays == 1``, this is equivalent to ``encode(sequence[index])``
        (optionally zeroing unobserved rows). When ``n_delays > 1``, builds a
        teacher-forced Hankel window from observed history — not from decoded
        rollouts.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Source trajectory.
        index : int
            Inclusive end index of the delay window.
        pad : bool, optional
            Zero-pad missing history before the sequence start. Default is
            ``True``.
        zero_unobserved : bool, optional
            Zero unobserved node features inside the window when masks are
            present. Default is ``True``.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        return encode_at_index(
            self.encoder,
            self.encode,
            sequence,
            index,
            n_delays=self.n_delays,
            pad=pad,
            zero_unobserved=zero_unobserved,
            physics_lifting_fn=self.physics_lifting_fn,
            physics_dim=self.physics_dim,
            physics_position=self.physics_position,
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
        adapter = enable_online_adaptation_helper(
            encoder=self.encoder,
            decoder=self.decoder,
            koopman=self.koopman,
            is_continuous=self.is_continuous,
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
        interval and writes back a generator aligned with
        :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.advance`
        (matrix logarithm when uncontrolled; Van Loan block inverse when
        controlled). Prefer discrete adaptation for uniformly sampled sequences
        when a discrete operator is acceptable; see
        :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` for
        matrix-logarithm / large-``delta_t`` caveats.

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
        return run_adapt_step(
            adapter,
            encode=self.encode,
            koopman=self.koopman,
            is_continuous=self.is_continuous,
            time_step=self.time_step,
            snapshot_t=snapshot_t,
            snapshot_tp1=snapshot_tp1,
            control=control,
            delta_t=delta_t,
        )

    def disable_online_adaptation(self, *, unfreeze: bool = True) -> None:
        """Disable online adaptation and optionally unfreeze encoder/decoder.

        Parameters
        ----------
        unfreeze : bool, optional
            When ``True``, restore ``requires_grad`` on encoder and decoder
            parameters. Default is ``True``.
        """
        self._adaptation_adapter = None
        disable_online_adaptation_helper(
            encoder=self.encoder,
            decoder=self.decoder,
            unfreeze=unfreeze,
        )

    def save(self, path: str | Path) -> None:
        """Persist model weights and architecture configuration to disk.

        Parameters
        ----------
        path : str or Path
            Destination checkpoint file (``.pt``). Parent directories are
            created when missing.
        """
        from koopman_graph.serialization import save_checkpoint

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
        from koopman_graph.serialization import load_checkpoint

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
        edge_index = resolve_edge_index(x_or_data, edge_index)
        edge_weight = resolve_edge_weight(x_or_data, edge_weight)
        z = self.encode(x_or_data, edge_index, edge_weight)
        z_next = self._advance_latent(
            z,
            control=control,
            delta_t=delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
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
        history: Sequence[Data] | None = None,
    ) -> list[tuple[Tensor, Tensor, Tensor | None]]:
        """Autoregressively advance latent state and decode for multiple steps.

        Encodes the initial graph once (optionally using a delay-history
        window), then applies the Koopman operator repeatedly in latent space,
        decoding after each step. Decoded predictions are **not** appended to
        the delay buffer.

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
        history : sequence of Data or None, optional
            Past snapshots (oldest → newest) used with ``x_or_data`` to form a
            delay window when ``n_delays > 1``. When omitted, missing history
            is zero-padded.

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
        return latent_decode_rollout(
            self.koopman,
            self.decoder,
            self.encode_rollout_origin,
            x_or_data=x_or_data,
            steps=steps,
            control_dim=self.control_dim,
            default_delta_t=self.time_step,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            step_deltas=step_deltas,
            history=history,
        )

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Encodes the initial graph once, advances the latent state with the
        Koopman operator for ``steps`` iterations, and decodes after each step.
        Runs in evaluation mode without gradient tracking.

        When ``n_delays > 1``, pass prior observations via ``history``
        (oldest → newest, excluding ``initial_graph``). Missing history is
        zero-padded. Decoded forecasts are **not** recycled into the delay
        buffer.

        The uncontrolled peer call site ``predict(data, steps)`` matches
        :class:`~koopman_graph.baselines.DMDBaseline` /
        :class:`~koopman_graph.baselines.EDMDBaseline`. Tensor inputs, optional
        ``controls``, and ``future_topologies`` are GraphKoopman-only and are
        **not** interchangeable with classical baselines (DMDc always requires
        ``controls``).

        When ``future_topologies`` is omitted, each rollout step decodes with
        the **hold-last-known** topology: the initial graph topology is used
        for step 0, and each subsequent step reuses the most recently provided
        topology. Pass one ``Data`` object per rollout step (topology only; node
        features are ignored) to supply a known future rewiring schedule.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Either a PyG ``Data`` object or node features ``x`` of shape
            ``(num_nodes, in_channels)``. Classical baselines accept ``Data``
            only.
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
            ``steps`` when :attr:`control_dim` is positive; optional (default
            ``None``) for uncontrolled models.
        future_topologies : sequence of Data or None, optional
            Known topologies for rollout decode steps. Shorter sequences hold
            the last provided topology for remaining steps.
        history : sequence of Data or None, optional
            Prior observations (oldest → newest, excluding ``initial_graph``)
            for delay embedding when ``n_delays > 1``.

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
        return predict_snapshots(
            self,
            self._rollout,
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
        )

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
        return predict_at_snapshots(
            self,
            self._rollout,
            initial_graph,
            is_continuous=self.is_continuous,
            time_step=self.time_step,
            query_times=query_times,
            step_deltas=step_deltas,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
        )

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
        return evaluate_sequence(
            self,
            sequence,
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
        extra_losses: ExtraLosses | None = None,
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

        Thin façade over :func:`~koopman_graph.training.run_fit_loop`: validates
        inputs and control layouts, then delegates epoch orchestration,
        device placement, early stopping, and history assembly.

        Minimizes a weighted sum of one-step MSE plus optional forward /
        backward consistency, multi-step rollout, and eigenvalue
        regularization terms (MSE means are over tensor entries)::

            loss = w_r * recon_loss
                 + w_f * mean((z_t K^T - z_{t+1})^2)
                 + w_b * mean((z_t - z_{t+1} (K^{\\dagger})^T)^2)
                 + w_rollout * rollout_loss
                 + w_eig * eigenvalue_loss
                 + w_lie * lie_consistency_loss
                 + w_pde * pde_residual_loss

        Row-convention propagation and inverses use ``z @ K.T`` /
        ``z @ (K^{\\dagger}).T``; see
        :class:`~koopman_graph.losses.ForwardConsistencyLoss` and
        :class:`~koopman_graph.losses.BackwardConsistencyLoss`. Weights
        ``(w_r, w_f, w_b, w_rollout, w_eig, w_lie, w_pde)`` come from a
        :class:`~koopman_graph.training.LossWeights` object or an optional
        per-epoch schedule.

        When ``data_sequence`` is a :class:`~koopman_graph.data.MultiTrajectory`,
        losses are averaged across trajectories before each optimizer step.

        Parameters
        ----------
        data_sequence : GraphSnapshotSequence, MultiTrajectory, or sequence of \
Data
            One training trajectory, or multiple trajectories via
            :class:`~koopman_graph.data.MultiTrajectory`. A plain list of
            ``Data`` snapshots is treated as a single trajectory. Use
            :class:`~koopman_graph.data.MultiTrajectory` (or
            :func:`~koopman_graph.data.as_multi_trajectory`) for multi-trajectory
            input; a bare list of :class:`~koopman_graph.data.GraphSnapshotSequence`
            is rejected. Empty lists and mixed
            ``GraphSnapshotSequence`` / ``Data`` lists raise ``ValueError``.
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
        extra_losses : ExtraLosses or None, optional
            Fit-time known dynamics and PDE residual callables. Required when
            the corresponding ``lie`` or ``pde`` loss weight is non-zero.
            Callables are not stored on the model or serialized.
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
        validation_sequence : GraphSnapshotSequence, MultiTrajectory, sequence \
of Data, sequence of GraphSnapshotSequence, or None, optional
            Optional held-out snapshots for per-epoch validation loss. A single
            validation sequence is reused for all training trajectories; a
            :class:`~koopman_graph.data.MultiTrajectory` or list of validation
            sequences must match the training trajectory count.
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
            Unlike classical baselines (``fit`` → ``self``), the neural model
            returns history rather than ``self``; see the ``ForecastModel``
            call-site matrix in :doc:`architecture`.

        Raises
        ------
        ValueError
            If ``epochs < 1``, ``early_stopping_patience < 1`` when set,
            ``early_stopping_monitor="val"`` without ``validation_sequence``,
            validation list length mismatches training trajectories, or fewer
            than two snapshots are provided for training or validation.
        """
        prepared = prepare_fit_inputs(
            control_dim=self.control_dim,
            data_sequence=data_sequence,
            validation_sequence=validation_sequence,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor,
        )
        return run_fit_loop(
            self,
            prepared.train_sequences,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            loss_weight_schedule=loss_weight_schedule,
            extra_losses=extra_losses,
            rollout_horizon=rollout_horizon,
            rollout_start_indices=rollout_start_indices,
            rollout_starts_per_epoch=rollout_starts_per_epoch,
            rollout_start_seed=rollout_start_seed,
            lr_scheduler=lr_scheduler,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            window_seed=window_seed,
            max_grad_norm=max_grad_norm,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_monitor=prepared.early_stopping_monitor,
            val_sequences=prepared.val_sequences,
            restore_best_weights=restore_best_weights,
            checkpoint_path=checkpoint_path,
            **optimizer_kwargs,
        )

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

        freeze_modules((self.encoder, self.decoder))
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
