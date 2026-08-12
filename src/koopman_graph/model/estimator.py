"""GraphKoopmanModel: encoder, Koopman operator, and decoder composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data, HeteroData

from koopman_graph.adaptation.rls import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    RolloutStartIndices,
    SnapshotSequence,
    WindowLikeSampler,
    mask_hetero_snapshot_features,
)
from koopman_graph.graph_utils import (
    OrbitMethod,
    autoregressive_hetero_latent_rollout,
    hold_last_relation_topology_at,
    pack_hetero_rollout_snapshots,
    propagate_latent,
    resolve_delta_t,
    resolve_edge_index,
    resolve_edge_weight,
    snapshot_head_index,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
    snapshot_tail_index,
)
from koopman_graph.metrics import EvaluationResult
from koopman_graph.nn import (
    DEFAULT_TOPOLOGY_EMBEDDING_DIM,
    AdaptiveAdjacency,
    DelayEmbeddingEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
    bind_hypergraph_decoder,
)
from koopman_graph.nn.delay import history_from_snapshots
from koopman_graph.nn.heterogeneous import (
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    PhysicsPosition,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
    InitMode,
    Parameterization,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.protocols import DynamicsMode
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.training import (
    EarlyStoppingMonitor,
    ExtraLosses,
    FitCallback,
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
    DEFAULT_KOOPMAN_LOCAL_RANK,
    DEFAULT_KOOPMAN_LOCAL_WINDOW,
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
from .timing import resolve_time_increments, validate_uniform_discrete_increments
from .validation import (
    prepare_fit_inputs,
    uses_cell_complex_modules,
    uses_hypergraph_modules,
    uses_relgraph_modules,
    uses_sheaf_modules,
    uses_simplicial_modules,
)

if TYPE_CHECKING:
    from koopman_graph.distributed import DistributedWindowSampler


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
        koopman_sparsity: str = "dense",
        koopman_adjacency: str = "symmetric",
        koopman_hypergraph_incidence_mode: str = "zhou_symmetric",
        koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
        koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
        koopman_local_hidden_dims: Sequence[int] | None = None,
        koopman_orbit_partition: Sequence[Sequence[int]] | None = None,
        koopman_auto_orbits: bool = False,
        koopman_orbit_method: OrbitMethod = "auto",
        koopman_symmetry: str | None = None,
        control_dim: int = 0,
        control_mode: ControlMode = DEFAULT_CONTROL_MODE,
        bilinear_rank: int | None = DEFAULT_BILINEAR_RANK,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
        physics_preset: str | None = None,
        physics_dim: int = 0,
        physics_position: PhysicsPosition = PHYSICS_POSITION,
        n_delays: int = 1,
        learn_topology: str | None = None,
        topology_embedding_dim: int = DEFAULT_TOPOLOGY_EMBEDDING_DIM,
        koopman_node_types: Sequence[str] | None = None,
        koopman_edge_types: Sequence[Sequence[str]] | None = None,
        koopman_relation_tying: str = "independent",
        koopman_basis_size: int | None = None,
        koopman_synthesize_reverse_relations: bool = False,
        koopman_latent_dims: Mapping[str, int] | None = None,
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
        koopman_sparsity : {"dense", "block_diagonal", "distributed"}, optional
            Realization mode for networked operators (``koopman="graph"`` /
            ``"hypergraph"`` / ``"hetero_graph"`` / continuous peers).
            ``"distributed"`` uses matrix-free inverse / spectrum helpers on
            discrete graph and multiplex hetero; it is **not** trainer DDP /
            ``[distributed]`` extras and does **not** enable multi-GPU
            training. Ignored for per-node operators (must remain
            ``"dense"``). Default is ``"dense"``.
        koopman_adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
            Neighbor-coupling normalization for ``koopman="graph"`` and
            continuous-graph peers. Default ``"symmetric"``. Rejected for
            non-networked ``koopman`` choices.
        koopman_local_window : int, optional
            Latent history length for ``koopman="global_local"`` (default
            ``4``). Must stay default when not using global/local.
        koopman_local_rank : int, optional
            Low-rank size of the local correction (default ``2``).
        koopman_local_hidden_dims : sequence of int or None, optional
            Local MLP hidden widths (default ``(32,)``).
        koopman_orbit_partition : sequence of sequence of int or None, optional
            Explicit node-orbit partition for symmetry-adapted
            ``koopman="graph"`` / ``"hypergraph"`` (ties ``K_self`` within
            each orbit). Overrides ``koopman_auto_orbits`` when set.
        koopman_auto_orbits : bool, optional
            When ``True``, bind orbits from topology on first advance
            (requires the ``[symmetry]`` extra for non-identity partitions).
            Graph/hypergraph only. Default ``False``.
        koopman_orbit_method : {"auto", "exact"}, optional
            Orbit backend when ``koopman_auto_orbits`` is enabled. Default
            ``"auto"``.
        koopman_symmetry : {None, "isotypic"}, optional
            Representation symmetry mode. ``"isotypic"`` ties ``K_self`` via
            exact automorphism orbits on ``koopman="graph"`` (mutually
            exclusive with orbit kwargs). Default ``None``.
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
        learn_topology : {"self_adaptive"} or None, optional
            When ``"self_adaptive"``, replace pairwise ``edge_index`` /
            ``edge_weight`` for encode / decode / graph advance / spectrum with
            a learned Graph WaveNet adjacency. Hyperedge incidence is unchanged.
            Default ``None`` is a numerical no-op.
        topology_embedding_dim : int, optional
            Embedding width ``k`` for ``learn_topology="self_adaptive"``.
            Default ``8``. Ignored when ``learn_topology`` is ``None``.
        koopman_node_types : sequence of str or None, optional
            Ordered node-type names for ``koopman="hetero_graph"`` (defaults
            to multiplex ``("node",)``).
        koopman_edge_types : sequence of sequence of str or None, optional
            Ordered ``(src, rel, dst)`` triples for ``koopman="hetero_graph"``.
        koopman_relation_tying : {"independent", "basis"}, optional
            Relation-factor tying for ``koopman="hetero_graph"``. Default
            ``"independent"``.
        koopman_basis_size : int or None, optional
            Basis size ``B`` when ``koopman_relation_tying="basis"``.
        koopman_synthesize_reverse_relations : bool, optional
            When ``True`` with ``koopman="hetero_graph"``, expand
            ``koopman_edge_types`` with reverse relations and rebuild
            RelGraph / operator banks to match. Default ``False``. Snapshots
            still need reverse ``edge_index`` banks (see
            :func:`~koopman_graph.graph_utils.materialize_reverse_relation_edges`).
        koopman_latent_dims : mapping of str to int or None, optional
            Opt-in per-type latent widths. When set, RelGraph peers are
            aligned and the discrete hetero operator receives the same map.
        koopman_hypergraph_incidence_mode
            See signature.
        Raises
        ------
        ValueError
            If ``latent_dim`` is not positive, ``time_step <= 0``,
            ``control_dim < 0``, ``n_delays < 1``, physics settings are
            inconsistent, encoder/decoder latent dimensions do not match the
            effective hybrid layout, an injected operator conflicts with
            factory kwargs or dimensions, ``dynamics_mode`` disagrees with a
            built-in injected operator type, or ``learn_topology`` is invalid.
        TypeError
            If ``koopman`` is provided but is not a string kind or ``nn.Module``."""
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
            koopman_sparsity=koopman_sparsity,
            koopman_adjacency=koopman_adjacency,  # type: ignore[arg-type]
            koopman_hypergraph_incidence_mode=koopman_hypergraph_incidence_mode,
            koopman_local_window=koopman_local_window,
            koopman_local_rank=koopman_local_rank,
            koopman_local_hidden_dims=koopman_local_hidden_dims,
            koopman_orbit_partition=koopman_orbit_partition,
            koopman_auto_orbits=koopman_auto_orbits,
            koopman_orbit_method=koopman_orbit_method,
            koopman_symmetry=koopman_symmetry,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
            physics_lifting_fn=physics_lifting_fn,
            physics_preset=physics_preset,
            physics_dim=physics_dim,
            physics_position=physics_position,
            n_delays=n_delays,
            koopman_node_types=koopman_node_types,
            koopman_edge_types=koopman_edge_types,
            koopman_relation_tying=koopman_relation_tying,
            koopman_basis_size=koopman_basis_size,
            koopman_synthesize_reverse_relations=(koopman_synthesize_reverse_relations),
            koopman_latent_dims=koopman_latent_dims,
        )
        apply_resolved_components(self, components)
        if learn_topology is not None and learn_topology != "self_adaptive":
            msg = (
                "learn_topology must be None or 'self_adaptive', "
                f"got {learn_topology!r}"
            )
            raise ValueError(msg)
        if topology_embedding_dim < 1:
            msg = (
                f"topology_embedding_dim must be positive, got {topology_embedding_dim}"
            )
            raise ValueError(msg)
        if learn_topology is not None and uses_relgraph_modules(
            components.encoder, components.decoder
        ):
            msg = (
                "learn_topology is unsupported with RelGraphEncoder / "
                "RelGraphDecoder (koopman='hetero_graph')"
            )
            raise ValueError(msg)
        self.learn_topology = learn_topology
        self.topology_embedding_dim = topology_embedding_dim
        self.adaptive_topology: AdaptiveAdjacency | None
        if learn_topology == "self_adaptive":
            self.adaptive_topology = AdaptiveAdjacency(topology_embedding_dim)
        else:
            self.adaptive_topology = None
        # Sequence-contract metadata for format-1 checkpoints (not presence tensors).
        self._allow_node_churn = False
        self._has_presence_masks = False
        self._entity_ids: tuple[str | int, ...] | None = None

    @property
    def allow_node_churn(self) -> bool:
        """Return whether the last stamped training sequence allowed node churn.

        Stamped from the primary training sequence during :meth:`fit` and
        restored from additive checkpoint keys. Presence mask tensors remain
        sequence data and are never stored on the model.

        Returns
        -------
        bool
            ``True`` when the stamped sequence had ``allow_node_churn=True``.
        """
        return self._allow_node_churn

    @property
    def has_presence_masks(self) -> bool:
        """Return whether the last stamped training sequence carried presence masks.

        Returns
        -------
        bool
            ``True`` when the stamped sequence had presence masks attached.
        """
        return self._has_presence_masks

    @property
    def entity_ids(self) -> tuple[str | int, ...] | None:
        """Return stable entity keys stamped from the last homogeneous fit.

        Heterogeneous sequences do not yet carry ``entity_ids``; those fits
        leave this attribute ``None``.

        Returns
        -------
        tuple of str or int, or None
            Universe keys of length ``N_max``, or ``None``.
        """
        return self._entity_ids

    def _stamp_node_churn_contract(self, sequence: SnapshotSequence) -> None:
        """Copy sequence churn-contract flags onto this model for checkpointing.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
            Primary training sequence. Homogeneous ``entity_ids`` are copied;
            hetero fits stamp flags only.
        """
        self._allow_node_churn = bool(getattr(sequence, "allow_node_churn", False))
        self._has_presence_masks = bool(getattr(sequence, "has_presence_masks", False))
        if (
            isinstance(sequence, GraphSnapshotSequence)
            and sequence.entity_ids is not None
        ):
            self._entity_ids = tuple(sequence.entity_ids)
        else:
            self._entity_ids = None

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
    def uses_hypergraph_koopman(self) -> bool:
        """Return whether latent advance uses the hyperedge-coupled operator.

        Returns
        -------
        bool
            ``True`` when :attr:`koopman` is a
            :class:`~koopman_graph.operators.HypergraphKoopmanOperator`.
        """
        return isinstance(self.koopman, HypergraphKoopmanOperator)

    @property
    def uses_hetero_koopman(self) -> bool:
        """Return whether latent advance uses a multiplex relational operator.

        Returns
        -------
        bool
            ``True`` when :attr:`koopman` is a
            :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator` or a
            :class:`~koopman_graph.operators.ContinuousHeteroGraphKoopmanOperator`.
            Use :attr:`uses_continuous_hetero_koopman` to distinguish the two.
        """
        return isinstance(
            self.koopman,
            (HeteroGraphKoopmanOperator, ContinuousHeteroGraphKoopmanOperator),
        )

    @property
    def uses_continuous_hetero_koopman(self) -> bool:
        """Return whether latent advance uses the continuous hetero generator.

        Returns
        -------
        bool
            ``True`` when :attr:`koopman` is a
            :class:`~koopman_graph.operators.ContinuousHeteroGraphKoopmanOperator`.
        """
        return isinstance(self.koopman, ContinuousHeteroGraphKoopmanOperator)

    @property
    def uses_typed_hetero(self) -> bool:
        """Return whether the encode / advance path uses typed node types.

        Returns
        -------
        bool
            ``True`` when the encoder projects per-type features onto the
            shared latent width (more than one node type).
        """
        return bool(getattr(self.encoder, "is_typed", False))

    def _resolve_hetero_relation_inputs(
        self,
        x_or_data: Tensor | Data | HeteroData,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> tuple[list[Tensor], list[Tensor | None], dict[str, int] | None]:
        """Resolve relation banks for multiplex or typed hetero input.

        Parameters
        ----------
        x_or_data : Tensor, Data, or HeteroData
            Hetero snapshot or stacked features with relation banks.
        edge_index : Tensor or None
            Relation banks for tensor input; ignored for ``HeteroData``.
        edge_weight : Tensor or None
            Optional relation weight banks for tensor input.

        Returns
        -------
        tuple
            ``(edge_indices, edge_weights, num_nodes_dict)`` where
            ``num_nodes_dict`` is ``None`` for multiplex graphs.
        """
        encoder = self.encoder
        if not isinstance(encoder, RelGraphEncoder):
            msg = "HeteroData paths require RelGraphEncoder peers"
            raise TypeError(msg)
        if encoder.is_typed:
            _, edge_indices, edge_weights, num_nodes_dict = (
                resolve_typed_relation_inputs(
                    cast("HeteroData | Mapping[str, Tensor]", x_or_data),
                    edge_index,
                    edge_weight,
                    node_types=encoder.node_types,
                    edge_types=encoder.edge_types,
                    num_relations=encoder.num_relations,
                )
            )
            return edge_indices, edge_weights, num_nodes_dict
        _, edge_indices, edge_weights = resolve_multiplex_relation_inputs(
            x_or_data,
            edge_index,
            edge_weight,
            num_relations=encoder.num_relations,
        )
        return edge_indices, edge_weights, None

    def _decode_hetero(
        self,
        z: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None],
        num_nodes_dict: Mapping[str, int] | None,
    ) -> Tensor | dict[str, Tensor]:
        """Decode a stacked latent block with the hetero decoder.

        Parameters
        ----------
        z : Tensor
            Stacked latent block with shape ``(N, latent_dim)``.
        edge_indices : sequence of Tensor
            Ordered relation banks in stacked global numbering.
        edge_weights : sequence of Tensor or None
            Optional per-relation weights.
        num_nodes_dict : mapping of str to int or None
            Per-type node counts; required for typed decoders.

        Returns
        -------
        Tensor or dict of str to Tensor
            Reconstructed features (per-type mapping for typed decoders).
        """
        decoder = self.decoder
        if isinstance(decoder, RelGraphDecoder) and decoder.is_typed:
            return decoder(
                z,
                edge_indices,
                edge_weights,
                num_nodes_dict=num_nodes_dict,
            )
        return decoder(z, edge_indices, edge_weights)

    @property
    def uses_continuous_graph_koopman(self) -> bool:
        """Return whether latent advance uses the continuous networked operator.

        Returns
        -------
        bool
            ``True`` when :attr:`koopman` is a
            :class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator`.
        """
        return isinstance(self.koopman, ContinuousGraphKoopmanOperator)

    @property
    def learns_pairwise_topology(self) -> bool:
        """Return whether pairwise edges are replaced by learned adjacency.

        Notes
        -----
        Hypergraph encode / operator paths keep exogenous hyperedge incidence;
        this flag is still ``True`` when ``learn_topology`` is enabled so the
        ``AdaptiveAdjacency`` module trains and serializes.

        Returns
        -------
        bool
            See summary line."""
        return self.adaptive_topology is not None

    def _uses_hypergraph_encode(self) -> bool:
        """Return whether the active encoder consumes hyperedge incidence.

        Returns
        -------
        bool
            See summary line."""
        encoder = self.encoder
        if isinstance(encoder, DelayEmbeddingEncoder):
            encoder = encoder.base_encoder
        return isinstance(encoder, HypergraphEncoder)

    def _uses_relgraph_encode(self) -> bool:
        """Return whether the active encoder consumes relation edge banks.

        Returns
        -------
        bool
            ``True`` for :class:`~koopman_graph.nn.RelGraphEncoder` peers.
        """
        return isinstance(self.encoder, RelGraphEncoder)

    def materialize_learned_topology(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        *,
        num_nodes: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return learned dense-COO topology, binding ``N`` on first use.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Reference graph used to infer ``N`` and device when needed.
        edge_index, edge_weight
            Unused when learning is enabled (replace semantics); accepted for
            call-site symmetry.
        num_nodes : int or None, optional
            Explicit node count. Inferred from ``x_or_data`` when omitted.

        Returns
        -------
        tuple of Tensor
            Learned ``edge_index`` ``(2, N²)`` and ``edge_weight`` ``(N²,)``.

        Raises
        ------
        RuntimeError
            If adaptive topology is not enabled.
        ValueError
            If ``num_nodes`` cannot be inferred or conflicts with a prior bind.
        """
        del edge_index, edge_weight  # replace semantics: exogenous pairwise unused
        if self.adaptive_topology is None:
            msg = "materialize_learned_topology requires learn_topology='self_adaptive'"
            raise RuntimeError(msg)
        if num_nodes is None:
            if isinstance(x_or_data, Data):
                num_nodes = (
                    int(x_or_data.num_nodes)
                    if x_or_data.num_nodes is not None
                    else int(x_or_data.x.size(0))
                )
                device = x_or_data.x.device
            elif isinstance(x_or_data, Tensor):
                num_nodes = int(x_or_data.shape[-2])
                device = x_or_data.device
            else:
                msg = "num_nodes is required when x_or_data has no node axis"
                raise ValueError(msg)
        elif isinstance(x_or_data, (Data, Tensor)):
            device = (
                x_or_data.x.device if isinstance(x_or_data, Data) else x_or_data.device
            )
        else:
            device = None
        self.adaptive_topology.set_num_nodes(num_nodes, device=device)
        return self.adaptive_topology.materialize()

    def resolve_pairwise_topology(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        *,
        num_nodes: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Resolve pairwise topology, replacing with learned Â when enabled.

        Parameters
        ----------

        x_or_data : Tensor | Data
            See the function signature / summary for ``x_or_data``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.
        num_nodes : int | None
            See the function signature / summary for ``num_nodes``.

        Returns
        -------

        tuple[Tensor, Tensor | None]
            See summary line.

        Notes
        -----

        Hypergraph encode paths keep exogenous topology (hyperedges are not
        rewritten). Otherwise ``learn_topology="self_adaptive"`` replaces the
        supplied pairwise edges."""
        if self.learns_pairwise_topology and not self._uses_hypergraph_encode():
            return self.materialize_learned_topology(
                x_or_data,
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
            )
        return (
            resolve_edge_index(x_or_data, edge_index),
            resolve_edge_weight(x_or_data, edge_weight),
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
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
        latent_window: Tensor | None = None,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Advance latent states with the active Koopman operator.

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        delta_t : float | Tensor | None
            See the function signature / summary for ``delta_t``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.
        hyperedge_index : Tensor | None
            See the function signature / summary for ``hyperedge_index``.
        hyperedge_weight : Tensor | None
            See the function signature / summary for ``hyperedge_weight``.
        tail_index, head_index : Tensor | None
            Directed-hypergraph incidence for random-walk modes.
        latent_window : Tensor | None
            See the function signature / summary for ``latent_window``.
        edge_indices : Sequence[Tensor] | None
            Per-relation banks for ``koopman='hetero_graph'``.
        edge_weights : Sequence[Tensor | None] | None
            Optional per-relation weights for hetero advance.
        num_nodes_dict : Mapping[str, int] | None
            Per-type node counts for typed hetero advance.

        Returns
        -------

        Tensor
            Advanced latent states."""
        return propagate_latent(
            self.koopman,
            z,
            control=control,
            delta_t=self.resolve_delta_t(delta_t),
            default_delta_t=self.time_step,
            edge_index=edge_index,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            tail_index=tail_index,
            head_index=head_index,
            latent_window=latent_window,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        )

    def spectrum(
        self,
        *,
        delta_t: float | None = None,
        edge_index: Tensor | None = None,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> KoopmanSpectrum:
        """Analyze the learned Koopman operator spectrum.

        For ordinary discrete / continuous / custom injected operators, uses
        :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix` (the
        per-node ``K`` / ``L``). In continuous mode, returns the generator
        spectrum by default; pass ``delta_t`` for the discrete-time spectrum of
        ``exp(L·Δt)``.

        For ``koopman="graph"``, analyzes the topology-coupled effective
        operator ``I⊗K_self + Â⊗K_nbr`` and **requires** ``edge_index`` and
        ``num_nodes``. For continuous networked models
        (``koopman="continuous_graph"`` or ``koopman="graph"`` with
        ``dynamics_mode="continuous"``), analyzes ``I⊗L_self + Â⊗L_nbr`` with
        the same topology requirements. For ``koopman="hypergraph"``, analyzes
        ``I⊗K_self + Ĥ⊗K_hedge`` and **requires** ``hyperedge_index`` and
        ``num_nodes``. For ``koopman="hetero_graph"``, analyzes the assembled
        multiplex / typed ``K_eff`` and **requires** ``edge_indices`` and
        ``num_nodes`` (plus ``num_nodes_dict`` when typed). Missing topology
        raises rather than silently returning the self-term spectrum.

        Parameters
        ----------
        delta_t : float or None, optional
            Continuous integration horizon for generator → discrete spectrum.
            Ignored for discrete / graph / hypergraph / hetero operators.
        edge_index : Tensor or None, optional
            Topology for networked graph operators. Required when
            :attr:`uses_graph_koopman` or
            :attr:`uses_continuous_graph_koopman` is ``True``.
        num_nodes : int or None, optional
            Node count ``N`` for the effective ``N·d`` operator. Required when
            a networked operator is active.
        edge_weight : Tensor or None, optional
            Optional edge weights with the same semantics as latent advance.
        hyperedge_index : Tensor or None, optional
            Incidence for hypergraph operators. Required when
            :attr:`uses_hypergraph_koopman` is ``True``.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        edge_indices : sequence of Tensor or None, optional
            Per-relation edge banks for ``koopman="hetero_graph"``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation weights for hetero operators.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts for typed hetero operators.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted eigenvalues, eigenvectors, and time scales.

        Raises
        ------
        ValueError
            If a networked operator is active and required topology arguments
            are missing.
        """
        if (
            self.learns_pairwise_topology
            and not self.uses_hypergraph_koopman
            and (self.uses_graph_koopman or self.uses_continuous_graph_koopman)
        ):
            if num_nodes is None:
                msg = (
                    "num_nodes is required for spectrum when "
                    "learn_topology='self_adaptive' (materialize learned Â)"
                )
                raise ValueError(msg)
            if self.adaptive_topology is None:
                msg = "adaptive_topology module missing"
                raise RuntimeError(msg)
            self.adaptive_topology.set_num_nodes(num_nodes)
            edge_index, edge_weight = self.adaptive_topology.materialize()
        return compute_model_spectrum(
            self.koopman,
            uses_graph_koopman=self.uses_graph_koopman,
            uses_hypergraph_koopman=self.uses_hypergraph_koopman,
            uses_continuous_graph_koopman=self.uses_continuous_graph_koopman,
            uses_hetero_koopman=self.uses_hetero_koopman,
            uses_continuous_hetero_koopman=self.uses_continuous_hetero_koopman,
            is_continuous=self.is_continuous,
            time_step=self.time_step,
            delta_t=delta_t,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        )

    def encode(
        self,
        x_or_data: Tensor | Data | HeteroData,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        *,
        resolve_learned_topology: bool = True,
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

        Multiplex ``HeteroData`` is accepted when RelGraph peers /
        ``koopman="hetero_graph"`` are active.

        Parameters
        ----------
        x_or_data : Tensor, Data, or HeteroData
            Node features, delay window, homogeneous ``Data``, or multiplex
            ``HeteroData``.
        edge_index : Tensor or None, optional
            Edge index required when ``x_or_data`` is a tensor. RelGraph
            tensor input accepts relation banks instead.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights for tensor input.
        resolve_learned_topology : bool, optional
            When ``True`` (default) and pairwise topology learning is enabled,
            materialize learned ``Â`` here. Callers that already resolved
            topology (e.g. :meth:`forward`) pass ``False`` so materialize runs
            at most once per top-level call.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        if isinstance(x_or_data, HeteroData) or self._uses_relgraph_encode():
            return encode_features(
                self.encoder,
                x_or_data,
                edge_index,
                edge_weight,
                physics_lifting_fn=self.physics_lifting_fn,
                physics_dim=self.physics_dim,
                physics_position=self.physics_position,
                prefer_explicit_topology=False,
            )
        replace = self.learns_pairwise_topology and not self._uses_hypergraph_encode()
        if replace and resolve_learned_topology:
            edge_index, edge_weight = self.materialize_learned_topology(
                x_or_data, edge_index, edge_weight
            )
        elif self.adaptive_topology is not None and not replace:
            # Bind N for hypergraph peers (incidence stays exogenous).
            self.materialize_learned_topology(x_or_data, edge_index, edge_weight)
        return encode_features(
            self.encoder,
            x_or_data,
            edge_index,
            edge_weight,
            physics_lifting_fn=self.physics_lifting_fn,
            physics_dim=self.physics_dim,
            physics_position=self.physics_position,
            prefer_explicit_topology=replace,
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
        if self.learns_pairwise_topology and not self._uses_hypergraph_encode():
            # Materialize at most once; encode reuses the resolved COO.
            if edge_index is None or edge_weight is None:
                edge_resolved, weight_resolved = self.materialize_learned_topology(
                    x_or_data, edge_index, edge_weight
                )
            else:
                edge_resolved, weight_resolved = edge_index, edge_weight
            if self.n_delays > 1 and isinstance(x_or_data, Data):
                past = list(history) if history is not None else []
                x_window, _, _, _ = history_from_snapshots(
                    [*past, x_or_data],
                    self.n_delays,
                    pad=True,
                )
                z = self.encode(
                    x_window,
                    edge_resolved,
                    weight_resolved,
                    resolve_learned_topology=False,
                )
            else:
                z = self.encode(
                    x_or_data,
                    edge_resolved,
                    weight_resolved,
                    resolve_learned_topology=False,
                )
            return z, edge_resolved, weight_resolved

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
        sequence: SnapshotSequence,
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

        :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` supports
        ``n_delays == 1`` only; observation masks zero unobserved rows
        per node type.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
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

        Raises
        ------
        ValueError
            If a hetero sequence uses delays.
        """
        if isinstance(sequence, HeteroGraphSnapshotSequence):
            if self.n_delays != 1:
                msg = (
                    "HeteroGraphSnapshotSequence encode_at requires n_delays=1 "
                    f"(got n_delays={self.n_delays})"
                )
                raise ValueError(msg)
            snapshot = sequence[index]
            if zero_unobserved and sequence.has_observation_masks:
                snapshot = mask_hetero_snapshot_features(
                    snapshot,
                    sequence.observation_mask_at(index),
                )
            return self.encode(snapshot)
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

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["safetensors_v1", "legacy_pt"] = "safetensors_v1",
    ) -> None:
        """Persist model weights and architecture configuration to disk.

        Parameters
        ----------
        path : str or Path
            Destination **directory** for the default ``safetensors_v1``
            container, a ``.kgckpt`` / ``.zip`` path for the zip bundle of
            the same layout, or a ``.pt`` file path when
            ``format="legacy_pt"``. Parent directories are created when
            missing.
        format : {"safetensors_v1", "legacy_pt"}, optional
            On-disk container. Default ``safetensors_v1``; pass
            ``legacy_pt`` for the pickle escape hatch.
        """
        import importlib

        serialization = importlib.import_module("koopman_graph.serialization")
        serialization.save_checkpoint(self, path, format=format)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        physics_lifting_fn: PhysicsLiftingFn | None = None,
    ) -> GraphKoopmanModel:
        """Load a trained model from a checkpoint file or directory.

        Reconstructs encoder, decoder, and Koopman operator architecture from
        the saved configuration and restores learned weights. Auto-detects
        ``safetensors_v1`` directories and ``.kgckpt`` / ``.zip`` bundles, then
        legacy ``.pt`` pickle files.

        Parameters
        ----------
        path : str or Path
            Checkpoint file or directory produced by :meth:`save` /
            :func:`~koopman_graph.serialization.save_checkpoint`.
        map_location : str, torch.device, or None, optional
            Device mapping forwarded to the underlying loader
            (:func:`torch.load` for ``.pt``; device string for
            ``safetensors_v1``).
        physics_lifting_fn : callable or None, optional
            Custom physics lifting function required when the checkpoint stores
            hybrid observables without a registered preset.

        Returns
        -------
        GraphKoopmanModel
            Ready-to-use model in evaluation mode.
        """
        import importlib

        serialization = importlib.import_module("koopman_graph.serialization")
        return serialization.load_checkpoint(
            path,
            map_location=map_location,
            physics_lifting_fn=physics_lifting_fn,
        )

    def forward(
        self,
        x_or_data: Tensor | Data | HeteroData,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Predict the next graph snapshot from the current one.

        Performs encode → linear Koopman advance → decode for a single step.

        Parameters
        ----------
        x_or_data : Tensor, Data, or HeteroData
            Homogeneous ``Data`` / features, or multiplex ``HeteroData`` when
            ``koopman="hetero_graph"``.
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``x_or_data`` is a tensor; ignored for ``Data`` / ``HeteroData``.
            RelGraph tensor input accepts relation banks instead.
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
        Tensor or dict of str to Tensor
            Predicted node features of shape ``(num_nodes, out_channels)``.
            Typed hetero models return one tensor per node type.
        """
        if self.uses_hetero_koopman or isinstance(x_or_data, HeteroData):
            if not isinstance(self.encoder, RelGraphEncoder):
                msg = "HeteroData forward requires RelGraphEncoder peers"
                raise TypeError(msg)
            edge_indices, edge_weights, num_nodes_dict = (
                self._resolve_hetero_relation_inputs(x_or_data, edge_index, edge_weight)
            )
            z = self.encode(
                x_or_data,
                edge_index,
                edge_weight,
                resolve_learned_topology=False,
            )
            z_next = self._advance_latent(
                z,
                control=control,
                delta_t=delta_t,
                edge_indices=edge_indices,
                edge_weights=edge_weights,
                num_nodes_dict=num_nodes_dict,
            )
            return self._decode_hetero(
                z_next, edge_indices, edge_weights, num_nodes_dict
            )

        edge_index, edge_weight = self.resolve_pairwise_topology(
            x_or_data, edge_index, edge_weight
        )
        if isinstance(x_or_data, Data):
            hyperedge_index = snapshot_hyperedge_index(x_or_data)
            hyperedge_weight = snapshot_hyperedge_weight(x_or_data)
            tail_index = snapshot_tail_index(x_or_data)
            head_index = snapshot_head_index(x_or_data)
        else:
            hyperedge_index = None
            hyperedge_weight = None
            tail_index = None
            head_index = None
        z = self.encode(
            x_or_data,
            edge_index,
            edge_weight,
            resolve_learned_topology=False,
        )
        z_next = self._advance_latent(
            z,
            control=control,
            delta_t=delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            tail_index=tail_index,
            head_index=head_index,
        )
        if isinstance(self.decoder, HypergraphDecoder):
            if hyperedge_index is None:
                msg = "HypergraphDecoder requires hyperedge_index on Data input"
                raise ValueError(msg)
            return self.decoder(z_next, hyperedge_index, hyperedge_weight)
        return self.decoder(z_next, edge_index, edge_weight)

    def _rollout_hetero(
        self,
        x_or_data: Tensor | HeteroData,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[HeteroData] | None = None,
        step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
    ) -> list[tuple[Tensor | dict[str, Tensor], list[Tensor], list[Tensor | None]]]:
        """Autoregressive hetero rollout returning relation-bank tuples.

        Parameters
        ----------
        x_or_data : Tensor or HeteroData
            Multiplex origin features or ``HeteroData`` snapshot.
        steps : int
            Number of rollout steps (must be ``>= 1``).
        edge_index : Tensor or None, optional
            Relation banks when ``x_or_data`` is a tensor; ignored for
            ``HeteroData``.
        edge_weight : Tensor or None, optional
            Optional relation weight banks for tensor input.
        controls : sequence of Tensor or None, optional
            Per-step controls when ``control_dim > 0``.
        future_topologies : sequence of HeteroData or None, optional
            Hold-last future multiplex topologies.
        step_deltas : sequence of float or Tensor or None, optional
            Optional per-step integration intervals.

        Returns
        -------
        list of tuple
            For each step: decoded prediction, relation edge banks, and
            optional relation weights.

        Raises
        ------
        ValueError
            If ``steps`` / controls / ``step_deltas`` are invalid.
        TypeError
            If RelGraph peers are missing.
        """
        from koopman_graph.model.validation import validate_controls

        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if not isinstance(self.encoder, RelGraphEncoder):
            msg = "Hetero rollout requires RelGraphEncoder peers"
            raise TypeError(msg)
        if not isinstance(self.decoder, RelGraphDecoder):
            msg = "Hetero rollout requires RelGraphDecoder peers"
            raise TypeError(msg)
        validate_controls(control_dim=self.control_dim, controls=controls, steps=steps)
        if step_deltas is not None and len(step_deltas) != steps:
            msg = f"expected {steps} step_deltas for rollout, got {len(step_deltas)}"
            raise ValueError(msg)

        z = self.encode(x_or_data, edge_index, edge_weight)
        edge_indices, edge_weights, num_nodes_dict = (
            self._resolve_hetero_relation_inputs(x_or_data, edge_index, edge_weight)
        )
        control_at = None if controls is None else (lambda step: controls[step])
        delta_t_at = None if step_deltas is None else (lambda step: step_deltas[step])

        def decode(
            latent: Tensor,
            banks: Sequence[Tensor],
            weights: Sequence[Tensor | None],
        ) -> Tensor | dict[str, Tensor]:
            """Decode one rollout step with typed-aware plumbing.

            Parameters
            ----------
            latent : Tensor
                Advanced stacked latent block.
            banks : sequence of Tensor
                Ordered relation edge banks for this step.
            weights : sequence of Tensor or None
                Optional per-relation weights.

            Returns
            -------
            Tensor or dict of str to Tensor
                Reconstructed features for this step.
            """
            return self._decode_hetero(latent, banks, weights, num_nodes_dict)

        return autoregressive_hetero_latent_rollout(
            self.koopman,
            decode,
            z,
            steps=steps,
            topology_at=hold_last_relation_topology_at(
                edge_indices,
                edge_weights,
                future_topologies,
                num_relations=self.encoder.num_relations,
                node_types=self.encoder.node_types,
                edge_types=self.encoder.edge_types,
            ),
            control_at=control_at,
            delta_t_at=delta_t_at,
            default_delta_t=self.time_step,
            num_nodes_dict=num_nodes_dict,
        )

    def _rollout(
        self,
        x_or_data: Tensor | Data | HeteroData,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | Sequence[HeteroData] | None = None,
        future_presence: Tensor | Sequence[Tensor] | None = None,
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
        future_presence : Tensor, sequence of Tensor, or None, optional
            Per-step entity presence for the inactive-node **hold last active
            state** policy. Matvecs still use ``N_max`` capacity. When omitted,
            all entities are treated as present.
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
        if self.uses_hetero_koopman or isinstance(x_or_data, HeteroData):
            msg = (
                "Use predict() for multiplex HeteroData rollouts; "
                "_rollout is homogeneous-only"
            )
            raise TypeError(msg)
        decoder_fn: Any = self.decoder
        if isinstance(self.decoder, HypergraphDecoder):
            if isinstance(x_or_data, Data):
                hyperedge_index = snapshot_hyperedge_index(x_or_data)
                hyperedge_weight = snapshot_hyperedge_weight(x_or_data)
            else:
                resolved_index = resolve_edge_index(x_or_data, edge_index)
                hyperedge_index = resolved_index
                hyperedge_weight = resolve_edge_weight(x_or_data, edge_weight)
            if hyperedge_index is None:
                msg = (
                    "HypergraphDecoder rollout requires hyperedge_index on "
                    "the origin graph"
                )
                raise ValueError(msg)
            decoder_fn = bind_hypergraph_decoder(
                self.decoder,
                hyperedge_index,
                hyperedge_weight,
            )
        # Static learned Â: ignore dynamic future pairwise topologies.
        rollout_futures = (
            None
            if self.learns_pairwise_topology and not self._uses_hypergraph_encode()
            else future_topologies
        )
        if self.learns_pairwise_topology and not self._uses_hypergraph_encode():
            edge_index, edge_weight = self.materialize_learned_topology(
                x_or_data, edge_index, edge_weight
            )
        return latent_decode_rollout(
            self.koopman,
            decoder_fn,
            self.encode_rollout_origin,
            x_or_data=x_or_data,
            steps=steps,
            control_dim=self.control_dim,
            default_delta_t=self.time_step,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=rollout_futures,
            future_presence=future_presence,
            step_deltas=step_deltas,
            history=history,
        )

    def predict(
        self,
        initial_graph: Tensor | Data | HeteroData,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | Sequence[HeteroData] | None = None,
        future_presence: Tensor | Sequence[Tensor] | None = None,
        history: Sequence[Data] | None = None,
    ) -> list[Data] | list[HeteroData]:
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

        When ``future_presence`` is provided, inactive entities follow the
        **hold last active state** policy (latent and decoded features freeze
        while absent and resume on re-entry). Operator matvecs still run at
        fixed-union ``N_max`` capacity; this is not a sparse-``N_active``
        speedup. Omitting ``future_presence`` keeps the 0.10 all-present path.

        Multiplex ``HeteroData`` origins (``koopman="hetero_graph"``) return
        ``list[HeteroData]`` preserving the origin node/edge-type schema.

        Parameters
        ----------
        initial_graph : Tensor, Data, or HeteroData
            Homogeneous snapshot / features, or multiplex ``HeteroData``.
            Classical baselines accept ``Data`` only.
        steps : int
            Number of future snapshots to predict (must be >= 1).
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``initial_graph`` is a tensor; ignored for ``Data`` /
            ``HeteroData``. RelGraph tensor input accepts relation banks.
        edge_weight : Tensor, optional
            Scalar edge weights with shape ``(num_edges,)``. Required when
            ``initial_graph`` is a tensor and weights are used; ignored for
            ``Data`` input.
        controls : sequence of Tensor or None, optional
            Future control inputs for each rollout step. Required with length
            ``steps`` when :attr:`control_dim` is positive; optional (default
            ``None``) for uncontrolled models.
        future_topologies : sequence of Data or HeteroData or None, optional
            Known topologies for rollout decode steps. Shorter sequences hold
            the last provided topology for remaining steps.
        future_presence : Tensor, sequence of Tensor, or None, optional
            Per-step presence masks ``(steps, N_max)`` or length-``steps``
            sequence of ``(N_max,)`` masks. Homogeneous-only; ignored / unused
            on hetero predict in this release.
        history : sequence of Data or None, optional
            Prior observations (oldest → newest, excluding ``initial_graph``)
            for delay embedding when ``n_delays > 1``. Homogeneous-only.

        Returns
        -------
        list of Data or list of HeteroData
            ``steps`` predicted snapshots. Homogeneous paths return ``Data``;
            multiplex paths return ``HeteroData``.

        Raises
        ------
        ValueError
            If ``steps < 1`` or controls are missing/invalid for a controlled
            model.
        """
        if self.uses_hetero_koopman or isinstance(initial_graph, HeteroData):
            if history is not None:
                msg = "history / delay embedding is unsupported for HeteroData predict"
                raise ValueError(msg)
            if future_presence is not None:
                msg = "future_presence is unsupported for HeteroData predict"
                raise ValueError(msg)
            if not isinstance(initial_graph, HeteroData):
                msg = (
                    "koopman='hetero_graph' predict requires a HeteroData origin "
                    "(tensor relation-bank packing is not implemented for "
                    "predict; use forward for tensor banks)"
                )
                raise TypeError(msg)
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    rollout = self._rollout_hetero(
                        initial_graph,
                        steps,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        controls=controls,
                        future_topologies=future_topologies,  # type: ignore[arg-type]
                    )
            finally:
                self.train(was_training)
            encoder = self.encoder
            typed_node_types = (
                encoder.node_types
                if isinstance(encoder, RelGraphEncoder) and encoder.is_typed
                else None
            )
            typed_edge_types = (
                encoder.edge_types
                if isinstance(encoder, RelGraphEncoder) and encoder.is_typed
                else None
            )
            return pack_hetero_rollout_snapshots(
                rollout,
                template=initial_graph,
                node_types=typed_node_types,
                edge_types=typed_edge_types,
            )

        return predict_snapshots(
            self,
            self._rollout,
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,  # type: ignore[arg-type]
            future_presence=future_presence,
            history=history,
        )

    def predict_at(
        self,
        initial_graph: Tensor | Data | HeteroData,
        *,
        query_times: Sequence[float] | Sequence[Tensor] | None = None,
        step_deltas: Sequence[float] | Sequence[Tensor] | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | Sequence[HeteroData] | None = None,
        future_presence: Tensor | Sequence[Tensor] | None = None,
    ) -> list[Data] | list[HeteroData]:
        """Forecast graph snapshots at arbitrary query times.

        Exactly one of ``query_times`` or ``step_deltas`` must be provided.
        ``query_times`` are absolute times relative to the initial snapshot at
        ``t = 0``. ``step_deltas`` are positive integration intervals applied
        sequentially from the initial state.

        In discrete mode, non-uniform ``step_deltas`` or ``query_times`` raise
        :class:`ValueError` because the learned operator is tied to a fixed
        :attr:`time_step`.

        Multiplex / typed ``HeteroData`` origins (``koopman="hetero_graph"``)
        return ``list[HeteroData]`` preserving the origin schema. Continuous
        hetero operators are TASK-1812; discrete uniform increments apply today.

        Parameters
        ----------
        initial_graph : Tensor, Data, or HeteroData
            Initial graph snapshot at ``t = 0``.
        query_times : sequence of float or Tensor or None, optional
            Strictly increasing absolute query times, each positive.
        step_deltas : sequence of float or Tensor or None, optional
            Strictly positive integration intervals applied in order.
        edge_index, edge_weight, controls, future_topologies, future_presence
            Same semantics as :meth:`predict` (including the inactive-node
            hold policy and ``N_max`` matvec cost note).

        Returns
        -------
        list of Data or list of HeteroData
            Predicted snapshots, one per query interval.
        """
        if self.uses_hetero_koopman or isinstance(initial_graph, HeteroData):
            if future_presence is not None:
                msg = "future_presence is unsupported for HeteroData predict_at"
                raise ValueError(msg)
            if not isinstance(initial_graph, HeteroData):
                msg = (
                    "koopman='hetero_graph' predict_at requires a HeteroData "
                    "origin (tensor relation-bank packing is not implemented "
                    "for predict_at; use forward for tensor banks)"
                )
                raise TypeError(msg)
            increments = resolve_time_increments(
                query_times=query_times,
                step_deltas=step_deltas,
            )
            if not self.is_continuous:
                validate_uniform_discrete_increments(
                    time_step=self.time_step,
                    increments=increments,
                )
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    rollout = self._rollout_hetero(
                        initial_graph,
                        len(increments),
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        controls=controls,
                        future_topologies=future_topologies,  # type: ignore[arg-type]
                        step_deltas=increments,
                    )
            finally:
                self.train(was_training)
            encoder = self.encoder
            typed_node_types = (
                encoder.node_types
                if isinstance(encoder, RelGraphEncoder) and encoder.is_typed
                else None
            )
            typed_edge_types = (
                encoder.edge_types
                if isinstance(encoder, RelGraphEncoder) and encoder.is_typed
                else None
            )
            return pack_hetero_rollout_snapshots(
                rollout,
                template=initial_graph,
                node_types=typed_node_types,
                edge_types=typed_edge_types,
            )

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
            future_topologies=future_topologies,  # type: ignore[arg-type]
            future_presence=future_presence,
        )

    def evaluate(
        self,
        sequence: SnapshotSequence | Sequence[Data] | Sequence[HeteroData],
        *,
        horizons: Sequence[int] = (3, 6, 12),
        start_indices: Sequence[int] | None = None,
    ) -> EvaluationResult:
        """Evaluate multi-horizon forecast accuracy on a snapshot sequence.

        Homogeneous sequences use per-node ``Data.x`` metrics. Hetero sequences
        use concatenated flattened per-type features in
        :attr:`~koopman_graph.data.HeteroGraphSnapshotSequence.node_type_names`
        order (stacked aggregate; not a certified per-type report).

        Parameters
        ----------
        sequence : GraphSnapshotSequence, HeteroGraphSnapshotSequence, or sequence
            Evaluation snapshots. Hetero paths require static relation banks
            without controls or observation masks.
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
        sampler: WindowLikeSampler | DistributedWindowSampler | None = None,
        max_grad_norm: float | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        early_stopping_monitor: EarlyStoppingMonitor = "auto",
        validation_sequence: ValidationInput = None,
        restore_best_weights: bool = False,
        checkpoint_path: str | Path | None = None,
        callbacks: Sequence[FitCallback] | None = None,
        strategy: Literal["ddp"] | None = None,
        find_unused_parameters: bool | None = None,
        **optimizer_kwargs: Any,
    ) -> FitHistory:
        """Train encoder, Koopman operator, and decoder end-to-end.

        Thin façade over :func:`~koopman_graph.training.run_fit_loop` (default)
        or :func:`~koopman_graph.distributed.run_ddp_fit_loop` when
        ``strategy="ddp"``: validates inputs and control layouts, then
        delegates epoch orchestration, device placement, early stopping, and
        history assembly.

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
            ``None`` preserves full-sequence single-step training. Mutually
            exclusive with ``sampler``.
        batch_size : int, optional
            Number of temporal windows averaged per optimizer step. Used only
            when ``window_length`` is set. Default is ``8``.
        windows_per_epoch : int or None, optional
            Maximum sampled windows per epoch. ``None`` uses every valid
            window across all trajectories.
        window_seed : int or None, optional
            Base seed for reproducible epoch-specific window shuffling.
        sampler : WindowSampler, NeighborWindowSampler, \
DistributedWindowSampler, or None, optional
            Pre-built temporal or neighbor-subgraph window sampler. When set,
            use instead of ``window_length``. Neighbor sampling trains on
            induced subgraphs (approximation); ``predict`` / ``evaluate`` stay
            full-graph. With ``strategy="ddp"``, pass
            :class:`~koopman_graph.distributed.DistributedWindowSampler` (or
            ``window_length``); plain :class:`~koopman_graph.data.WindowSampler`
            / :class:`~koopman_graph.data.NeighborWindowSampler` are rejected.
        max_grad_norm : float or None, optional
            When set, clip the global gradient norm before each optimizer step.
        use_amp : bool, optional
            Enable CUDA automatic mixed precision (autocast + GradScaler).
            On CPU/MPS, warns once and continues in FP32. Default is ``False``.
        amp_dtype : torch.dtype or None, optional
            Autocast dtype when AMP is active (default ``torch.float16``).
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
        callbacks : sequence of FitCallback or None, optional
            Observe-only fit hooks forwarded to
            :func:`~koopman_graph.training.run_fit_loop` when
            ``strategy=None``. Default ``None`` skips hooks. Not supported
            with ``strategy="ddp"`` yet (raises ``ValueError``); use
            single-process fit or Lightning loggers for distributed
            tracking.
        strategy : {"ddp"} or None, optional
            Training orchestration backend. ``None`` (default) uses
            :func:`~koopman_graph.training.run_fit_loop` (single-process,
            unchanged from 0.7.1). ``"ddp"`` delegates to
            :func:`~koopman_graph.distributed.run_ddp_fit_loop` for native
            PyTorch DDP / ``torchrun`` launches; at world size 1 wrapping is
            skipped. See also
            :func:`~koopman_graph.distributed.prepare_ddp_model`.
        find_unused_parameters : bool or None, optional
            DDP unused-parameter search when ``strategy="ddp"``. ``None``
            (default) resolves to ``True`` for
            ``koopman_kind="hetero_graph"`` and ``False`` otherwise. Ignored
            for single-process ``strategy=None``.
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
            validation list length mismatches training trajectories, fewer
            than two snapshots are provided for training or validation, or
            ``strategy`` is not ``None`` / ``"ddp"``, or ``callbacks`` is set
            with ``strategy="ddp"``.
        """
        uses_simplicial_modules(self.encoder, self.decoder)
        uses_sheaf_modules(self.encoder, self.decoder)
        uses_cell_complex_modules(self.encoder, self.decoder)
        prepared = prepare_fit_inputs(
            control_dim=self.control_dim,
            data_sequence=data_sequence,
            validation_sequence=validation_sequence,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_monitor=early_stopping_monitor,
            allow_hyperedges=(
                uses_hypergraph_modules(self.encoder, self.decoder)
                or self.uses_hypergraph_koopman
            ),
        )
        self._stamp_node_churn_contract(prepared.train_sequences[0])
        loop_kwargs: dict[str, Any] = {
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer,
            "device": device,
            "loss_weights": loss_weights,
            "loss_weight_schedule": loss_weight_schedule,
            "extra_losses": extra_losses,
            "rollout_horizon": rollout_horizon,
            "rollout_start_indices": rollout_start_indices,
            "rollout_starts_per_epoch": rollout_starts_per_epoch,
            "rollout_start_seed": rollout_start_seed,
            "lr_scheduler": lr_scheduler,
            "window_length": window_length,
            "batch_size": batch_size,
            "windows_per_epoch": windows_per_epoch,
            "window_seed": window_seed,
            "sampler": sampler,
            "max_grad_norm": max_grad_norm,
            "use_amp": use_amp,
            "amp_dtype": amp_dtype,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_monitor": prepared.early_stopping_monitor,
            "val_sequences": prepared.val_sequences,
            "restore_best_weights": restore_best_weights,
            "checkpoint_path": checkpoint_path,
            **optimizer_kwargs,
        }
        if strategy is None:
            return run_fit_loop(
                self,
                prepared.train_sequences,
                callbacks=callbacks,
                **loop_kwargs,
            )
        if strategy == "ddp":
            if callbacks is not None:
                msg = (
                    "fit(..., callbacks=...) is not supported with "
                    'strategy="ddp" yet; use strategy=None (single-process) '
                    "or Lightning loggers for distributed tracking"
                )
                raise ValueError(msg)
            from koopman_graph.distributed import run_ddp_fit_loop

            return run_ddp_fit_loop(
                self,
                prepared.train_sequences,
                find_unused_parameters=find_unused_parameters,
                **loop_kwargs,
            )
        msg = (
            f"unsupported fit strategy {strategy!r}; expected None or 'ddp' "
            "(see koopman_graph.distributed.run_ddp_fit_loop)"
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
    ) -> Any:
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
        import importlib

        env_mod = importlib.import_module("koopman_graph.env")
        GraphKoopmanEnv = env_mod.GraphKoopmanEnv

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
