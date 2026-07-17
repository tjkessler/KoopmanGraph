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
from koopman_graph.analysis import (
    KoopmanSpectrum,
    compute_generator_spectrum,
    compute_spectrum,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    RolloutStartIndices,
    resolve_sequence,
)
from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    hold_last_topology_at,
    propagate_latent,
    resolve_delta_t,
    resolve_edge_index,
    resolve_edge_weight,
)
from koopman_graph.metrics import EvaluationResult, evaluate_forecast
from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
)
from koopman_graph.nn.delay import (
    history_from_snapshots,
    resolve_delay_encoder,
    stack_delay_features,
)
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    PhysicsPosition,
    concatenate_observables,
    resolve_physics_lifting_fn,
    resolve_physics_position,
    validate_physics_output,
)
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    InitMode,
    KoopmanKind,
    KoopmanOperator,
    KoopmanOperatorContract,
    Parameterization,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.protocols import DynamicsMode
from koopman_graph.training import (
    EarlyStoppingMonitor,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
    TrainingInput,
    ValidationInput,
    resolve_early_stopping_monitor,
    resolve_training_sequences,
    resolve_validation_sequences,
    run_fit_loop,
)

if TYPE_CHECKING:
    from koopman_graph.env import GraphKoopmanEnv

Encoder = GNNEncoder | GATEncoder | DelayEmbeddingEncoder
Decoder = GNNDecoder | GATDecoder
KoopmanModule = KoopmanOperator | ContinuousKoopmanOperator | GraphKoopmanOperator
KoopmanArg = KoopmanOperatorContract | KoopmanKind | None

_DEFAULT_KOOPMAN_INIT_MODE: InitMode = "identity_noise"
_DEFAULT_KOOPMAN_INIT_SCALE = 1e-2
_DEFAULT_KOOPMAN_PARAMETERIZATION: Parameterization = "dense"
_DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS = 1.0
_DEFAULT_CONTROL_MODE: ControlMode = "additive"
_DEFAULT_BILINEAR_RANK: int | None = None


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
    encoder : GNNEncoder or GATEncoder
        Topology-aware encoder for latent lifting.
    decoder : GNNDecoder or GATDecoder
        Symmetric GNN decoder for physical reconstruction.
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
        koopman_init_mode: InitMode = _DEFAULT_KOOPMAN_INIT_MODE,
        koopman_init_scale: float = _DEFAULT_KOOPMAN_INIT_SCALE,
        koopman_parameterization: Parameterization = _DEFAULT_KOOPMAN_PARAMETERIZATION,
        koopman_max_spectral_radius: float = _DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS,
        control_dim: int = 0,
        control_mode: ControlMode = _DEFAULT_CONTROL_MODE,
        bilinear_rank: int | None = _DEFAULT_BILINEAR_RANK,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
        physics_preset: str | None = None,
        physics_dim: int = 0,
        physics_position: PhysicsPosition = PHYSICS_POSITION,
        n_delays: int = 1,
    ) -> None:
        """Initialize encoder, decoder, and Koopman operator.

        Parameters
        ----------
        encoder : GNNEncoder, GATEncoder, or DelayEmbeddingEncoder
            Topology-aware encoder for latent lifting. When ``n_delays > 1``,
            pass a base encoder already sized with
            ``in_channels = n_delays * feature_dim`` (composition; layers are
            not rebuilt) or an existing
            :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`.
        decoder : GNNDecoder or GATDecoder
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
            :class:`~koopman_graph.nn.encoder.GATEncoder` is wrapped in
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
        if n_delays < 1:
            msg = f"n_delays must be >= 1, got {n_delays}"
            raise ValueError(msg)

        encoder, resolved_n_delays = resolve_delay_encoder(encoder, n_delays)

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
        self.physics_position = resolve_physics_position(physics_position)
        self.time_step = time_step
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank
        self.dynamics_mode = dynamics_mode
        self.n_delays = resolved_n_delays

        kind, injected = self._parse_koopman_arg(koopman)
        if injected is not None:
            self.koopman = self._resolve_injected_koopman(
                injected,
                latent_dim=latent_dim,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                dynamics_mode=dynamics_mode,
                koopman_init_mode=koopman_init_mode,
                koopman_init_scale=koopman_init_scale,
                koopman_parameterization=koopman_parameterization,
                koopman_max_spectral_radius=koopman_max_spectral_radius,
            )
            self.koopman_kind = (
                "graph" if isinstance(self.koopman, GraphKoopmanOperator) else "pernode"
            )
        elif dynamics_mode == "continuous":
            if kind == "graph":
                msg = (
                    "koopman='graph' requires dynamics_mode='discrete'; "
                    "networked continuous operators are not implemented"
                )
                raise ValueError(msg)
            self.koopman = ContinuousKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_real_eigenvalue=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            )
            self.koopman_kind = "pernode"
        elif kind == "graph":
            self.koopman = GraphKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            )
            self.koopman_kind = "graph"
        else:
            self.koopman = KoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            )
            self.koopman_kind = "pernode"

    @staticmethod
    def _parse_koopman_arg(
        koopman: KoopmanArg,
    ) -> tuple[KoopmanKind, KoopmanOperatorContract | None]:
        """Split string factory kinds from injected operator modules.

        Parameters
        ----------
        koopman : KoopmanOperatorContract, {"pernode", "graph"}, or None
            Constructor argument.

        Returns
        -------
        tuple[KoopmanKind, KoopmanOperatorContract or None]
            Resolved factory kind and optional injected module.

        Raises
        ------
        TypeError
            If ``koopman`` is neither a known string nor a contract module.
        ValueError
            If ``koopman`` is an unknown string kind.
        """
        if koopman is None:
            return "pernode", None
        if isinstance(koopman, str):
            if koopman not in {"pernode", "graph"}:
                msg = (
                    f"koopman string kind must be 'pernode' or 'graph', got {koopman!r}"
                )
                raise ValueError(msg)
            return koopman, None
        return "pernode", koopman

    @staticmethod
    def _resolve_injected_koopman(
        koopman: KoopmanOperatorContract,
        *,
        latent_dim: int,
        control_dim: int,
        control_mode: ControlMode,
        bilinear_rank: int | None,
        dynamics_mode: DynamicsMode,
        koopman_init_mode: InitMode,
        koopman_init_scale: float,
        koopman_parameterization: Parameterization,
        koopman_max_spectral_radius: float,
    ) -> KoopmanOperatorContract:
        """Validate and return an injected Koopman operator module.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Caller-supplied operator.
        latent_dim : int
            Model latent dimension.
        control_dim : int
            Model control dimension.
        control_mode : {"additive", "bilinear"}
            Model control mode (must match injected operator when set).
        bilinear_rank : int or None
            Model bilinear rank (must match injected operator when set).
        dynamics_mode : {"discrete", "continuous"}
            Requested dynamics mode.
        koopman_init_mode : InitMode
            Factory init mode (must be default when injecting).
        koopman_init_scale : float
            Factory init scale (must be default when injecting).
        koopman_parameterization : Parameterization
            Factory parameterization (must be default when injecting).
        koopman_max_spectral_radius : float
            Factory spectral bound (must be default when injecting).

        Returns
        -------
        KoopmanOperatorContract
            Validated operator module ready for assignment.

        Raises
        ------
        TypeError
            If ``koopman`` is not an ``nn.Module``.
        ValueError
            If factory kwargs conflict or dimensions / dynamics mode mismatch.
        """
        if not isinstance(koopman, nn.Module):
            msg = (
                "Injected koopman must be an nn.Module implementing "
                "KoopmanOperatorContract, "
                f"got {type(koopman).__name__}"
            )
            raise TypeError(msg)

        conflicting: list[str] = []
        if koopman_init_mode != _DEFAULT_KOOPMAN_INIT_MODE:
            conflicting.append("koopman_init_mode")
        if koopman_init_scale != _DEFAULT_KOOPMAN_INIT_SCALE:
            conflicting.append("koopman_init_scale")
        if koopman_parameterization != _DEFAULT_KOOPMAN_PARAMETERIZATION:
            conflicting.append("koopman_parameterization")
        if koopman_max_spectral_radius != _DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS:
            conflicting.append("koopman_max_spectral_radius")
        if conflicting:
            names = ", ".join(conflicting)
            msg = (
                "Injected koopman is mutually exclusive with non-default "
                f"factory kwargs ({names}); omit them or leave defaults when "
                "passing koopman=..."
            )
            raise ValueError(msg)

        if koopman.latent_dim != latent_dim:
            msg = (
                f"Injected koopman.latent_dim ({koopman.latent_dim}) must match "
                f"latent_dim ({latent_dim})"
            )
            raise ValueError(msg)
        if koopman.control_dim != control_dim:
            msg = (
                f"Injected koopman.control_dim ({koopman.control_dim}) must match "
                f"control_dim ({control_dim})"
            )
            raise ValueError(msg)

        injected_mode = getattr(koopman, "control_mode", _DEFAULT_CONTROL_MODE)
        injected_rank = getattr(koopman, "bilinear_rank", _DEFAULT_BILINEAR_RANK)
        if injected_mode != control_mode:
            msg = (
                f"Injected koopman.control_mode ({injected_mode!r}) must match "
                f"control_mode ({control_mode!r})"
            )
            raise ValueError(msg)
        if injected_rank != bilinear_rank:
            msg = (
                f"Injected koopman.bilinear_rank ({injected_rank!r}) must match "
                f"bilinear_rank ({bilinear_rank!r})"
            )
            raise ValueError(msg)

        if (
            isinstance(koopman, ContinuousKoopmanOperator)
            and dynamics_mode != "continuous"
        ):
            msg = (
                "Injected ContinuousKoopmanOperator requires dynamics_mode='continuous'"
            )
            raise ValueError(msg)
        if isinstance(koopman, KoopmanOperator) and dynamics_mode != "discrete":
            msg = "Injected KoopmanOperator requires dynamics_mode='discrete'"
            raise ValueError(msg)
        if isinstance(koopman, GraphKoopmanOperator) and dynamics_mode != "discrete":
            msg = "Injected GraphKoopmanOperator requires dynamics_mode='discrete'"
            raise ValueError(msg)

        return koopman

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

    def spectrum(self, *, delta_t: float | None = None) -> KoopmanSpectrum:
        """Analyze the learned Koopman operator spectrum.

        Uses :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix`
        (not concrete ``K`` / ``L`` aliases). In continuous mode, returns the
        generator spectrum by default. Pass ``delta_t`` to obtain the
        discrete-time spectrum of ``exp(L·Δt)``. Classical DMD-family
        baselines take no ``spectrum`` kwargs.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted eigenvalues, eigenvectors, and time scales.
        """
        if self.is_continuous:
            if delta_t is None:
                return compute_generator_spectrum(self.koopman.matrix)
            return discrete_spectrum_at_delta_t(self.koopman.matrix, delta_t)
        return compute_spectrum(self.koopman.matrix, self.time_step)

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
        if isinstance(x_or_data, Tensor) and x_or_data.ndim == 3:
            if edge_index is None:
                msg = "edge_index is required for delay-window tensor input"
                raise ValueError(msg)
            z_gnn = self.encoder(x_or_data, edge_index, edge_weight)
            if self.physics_lifting_fn is None:
                return z_gnn
            msg = (
                "physics-informed observables with raw delay-window tensors are "
                "unsupported; pass a Data snapshot or use encode_at"
            )
            raise ValueError(msg)

        edge_index = resolve_edge_index(x_or_data, edge_index)
        edge_weight = resolve_edge_weight(x_or_data, edge_weight)
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
        if self.n_delays == 1:
            snapshot = sequence[index]
            x = snapshot.x
            if zero_unobserved and sequence.has_observation_masks:
                from koopman_graph.nn.delay import apply_observation_mask_to_features

                x = apply_observation_mask_to_features(
                    x,
                    sequence.observation_mask_at(index),
                )
                snapshot = Data(
                    x=x,
                    edge_index=snapshot.edge_index,
                    edge_weight=getattr(snapshot, "edge_weight", None),
                )
            return self.encode(snapshot)

        x_window, edge_index, edge_weight, _history_mask = stack_delay_features(
            sequence,
            index,
            self.n_delays,
            pad=pad,
            zero_unobserved=zero_unobserved,
        )
        z_gnn = self.encoder(x_window, edge_index, edge_weight)
        if self.physics_lifting_fn is None:
            return z_gnn

        # Physics lifting uses the newest snapshot's topology/features (not the
        # full delay stack) to stay consistent with single-snapshot physics.
        snapshot = sequence[index]
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

        mode: DynamicsMode = "continuous" if self.is_continuous else "discrete"
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

        resolved_delta = delta_t
        if self.is_continuous and resolved_delta is None:
            resolved_delta = self.time_step

        with torch.no_grad():
            z_t = self.encode(snapshot_t)
            z_tp1 = self.encode(snapshot_tp1)
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
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)

        self._validate_controls(controls, steps=steps)
        if step_deltas is not None and len(step_deltas) != steps:
            msg = f"expected {steps} step_deltas for rollout, got {len(step_deltas)}"
            raise ValueError(msg)

        if self.n_delays > 1 and isinstance(x_or_data, Data):
            past = list(history) if history is not None else []
            x_window, edge_index, edge_weight, _ = history_from_snapshots(
                [*past, x_or_data],
                self.n_delays,
                pad=True,
            )
            z = self.encode(x_window, edge_index, edge_weight)
        else:
            edge_index = resolve_edge_index(x_or_data, edge_index)
            edge_weight = resolve_edge_weight(x_or_data, edge_weight)
            z = self.encode(x_or_data, edge_index, edge_weight)

        control_at = None if controls is None else (lambda step: controls[step])
        delta_t_at = None if step_deltas is None else (lambda step: step_deltas[step])
        return autoregressive_latent_rollout(
            self.koopman,
            self.decoder,
            z,
            steps=steps,
            topology_at=hold_last_topology_at(
                edge_index,
                edge_weight,
                future_topologies,
            ),
            control_at=control_at,
            delta_t_at=delta_t_at,
            default_delta_t=self.time_step,
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
                    history=history,
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

        Row-convention propagation and inverses use ``z @ K.T`` /
        ``z @ (K^{\\dagger}).T``; see
        :class:`~koopman_graph.losses.ForwardConsistencyLoss` and
        :class:`~koopman_graph.losses.BackwardConsistencyLoss`. Weights
        ``(w_r, w_f, w_b, w_rollout, w_eig)`` come from a
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
        return run_fit_loop(
            self,
            train_sequences,
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            loss_weight_schedule=loss_weight_schedule,
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
            early_stopping_monitor=monitor,
            val_sequences=val_sequences,
            restore_best_weights=restore_best_weights,
            checkpoint_path=checkpoint_path,
            **optimizer_kwargs,
        )

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
