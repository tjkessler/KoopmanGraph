"""Koopman operator factory and injection validation for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer constructing operators
through :class:`~koopman_graph.model.GraphKoopmanModel`; these helpers exist so
the estimator stays orchestration-focused without cross-module private imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from torch import nn

from koopman_graph.graph_utils.symmetry import OrbitMethod
from koopman_graph.graph_utils.topology import synthesize_reverse_edge_types
from koopman_graph.nn import (
    CellComplexGNNDecoder,
    CellComplexGNNEncoder,
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    InvariantGeometryEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
    SAGEDecoder,
    SAGEEncoder,
    SheafGNNDecoder,
    SheafGNNEncoder,
    SimplicialDecoder,
    SimplicialEncoder,
)
from koopman_graph.nn.delay import resolve_delay_encoder
from koopman_graph.nn.gnn import ActivationName
from koopman_graph.observables import (
    PhysicsLiftingFn,
    PhysicsPosition,
    resolve_physics_lifting_fn,
    resolve_physics_position,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphAdjacency,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HodgeKoopmanOperator,
    HypergraphKoopmanOperator,
    InitMode,
    KoopmanKind,
    KoopmanOperator,
    KoopmanOperatorContract,
    MixtureKoopmanOperator,
    Parameterization,
    ParametricKoopmanOperator,
    SwitchedKoopmanOperator,
)
from koopman_graph.operators.auxiliary_spectral import (
    DEFAULT_AUXILIARY_HIDDEN_DIMS,
    normalize_auxiliary_hidden_dims,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.global_local import (
    DEFAULT_LOCAL_HIDDEN_DIMS,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOCAL_WINDOW,
    normalize_local_hidden_dims,
)
from koopman_graph.operators.parametric import (
    DEFAULT_PARAMETER_DIM,
    DEFAULT_WEIGHT_KIND,
    WeightKind,
)
from koopman_graph.operators.polynomial_graph import validate_filter_degree
from koopman_graph.operators.stochastic import attach_process_noise
from koopman_graph.operators.switched import DEFAULT_NUM_MODES
from koopman_graph.protocols import DynamicsMode

Encoder = (
    GNNEncoder
    | GATEncoder
    | SAGEEncoder
    | DiffConvEncoder
    | HypergraphEncoder
    | SimplicialEncoder
    | SheafGNNEncoder
    | CellComplexGNNEncoder
    | InvariantGeometryEncoder
    | RelGraphEncoder
    | DelayEmbeddingEncoder
)
Decoder = (
    GNNDecoder
    | GATDecoder
    | SAGEDecoder
    | DiffConvDecoder
    | HypergraphDecoder
    | SimplicialDecoder
    | SheafGNNDecoder
    | CellComplexGNNDecoder
    | RelGraphDecoder
)

EncoderKind = Literal["sheaf", "cell_complex"]
KoopmanModule = (
    KoopmanOperator
    | ContinuousKoopmanOperator
    | GraphKoopmanOperator
    | HypergraphKoopmanOperator
    | HeteroGraphKoopmanOperator
    | GlobalLocalKoopmanOperator
    | ContinuousGraphKoopmanOperator
    | ContinuousHeteroGraphKoopmanOperator
)
#: Both multiplex/typed hetero operator families (discrete and continuous),
#: for isinstance checks that treat them identically (RelGraph peer wiring).
HeteroKoopmanOperator = (
    HeteroGraphKoopmanOperator | ContinuousHeteroGraphKoopmanOperator
)
KoopmanArg = KoopmanOperatorContract | KoopmanKind | None

DEFAULT_KOOPMAN_INIT_MODE: InitMode = "identity_noise"
DEFAULT_KOOPMAN_INIT_SCALE = 1e-2
DEFAULT_KOOPMAN_PARAMETERIZATION: Parameterization = "dense"
DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS = 1.0
DEFAULT_KOOPMAN_AUXILIARY_HIDDEN_DIMS: tuple[int, ...] = DEFAULT_AUXILIARY_HIDDEN_DIMS
DEFAULT_KOOPMAN_LOCAL_WINDOW = DEFAULT_LOCAL_WINDOW
DEFAULT_KOOPMAN_LOCAL_RANK = DEFAULT_LOCAL_RANK
DEFAULT_KOOPMAN_LOCAL_HIDDEN_DIMS: tuple[int, ...] = DEFAULT_LOCAL_HIDDEN_DIMS
DEFAULT_CONTROL_MODE: ControlMode = "additive"
DEFAULT_BILINEAR_RANK: int | None = None
DEFAULT_KOOPMAN_ORBIT_PARTITION: Sequence[Sequence[int]] | None = None
DEFAULT_KOOPMAN_AUTO_ORBITS = False
DEFAULT_KOOPMAN_ORBIT_METHOD: OrbitMethod = "auto"
DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE = "zhou_symmetric"
_HYPERGRAPH_INCIDENCE_MODES = frozenset(
    {"zhou_symmetric", "forward_random_walk", "dual_random_walk"}
)
DEFAULT_KOOPMAN_SYMMETRY: str | None = None
DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE = "zhou_symmetric"
_HYPERGRAPH_INCIDENCE_MODES = frozenset(
    {"zhou_symmetric", "forward_random_walk", "dual_random_walk"}
)
DEFAULT_KOOPMAN_SYMMETRY: str | None = None
DEFAULT_KOOPMAN_ADJACENCY: GraphAdjacency = "symmetric"
DEFAULT_KOOPMAN_FILTER_DEGREE = 1
DEFAULT_KOOPMAN_NUM_MODES = DEFAULT_NUM_MODES
DEFAULT_KOOPMAN_PARAMETER_DIM = DEFAULT_PARAMETER_DIM
DEFAULT_KOOPMAN_WEIGHT_KIND: WeightKind = DEFAULT_WEIGHT_KIND
_NETWORKED_ADJACENCY_KINDS: frozenset[str] = frozenset({"graph", "continuous_graph"})
_GRAPH_ADJACENCY_MODES: frozenset[str] = frozenset(
    {"symmetric", "random_walk", "dual_random_walk"}
)


def _resolve_isotypic_symmetry(
    koopman_symmetry: str | None,
    *,
    kind: str,
    dynamics_mode: DynamicsMode,
    koopman_orbit_partition: Sequence[Sequence[int]] | None,
    koopman_auto_orbits: bool,
    koopman_orbit_method: OrbitMethod,
    koopman_latent_dims: Mapping[str, int] | None,
) -> bool:
    """Validate ``koopman_symmetry`` and return whether isotypic mode is on.

    Parameters
    ----------
    koopman_symmetry
        See signature.
    kind
        See signature.
    dynamics_mode
        See signature.
    koopman_orbit_partition
        See signature.
    koopman_auto_orbits
        See signature.
    koopman_orbit_method
        See signature.
    koopman_latent_dims
        See signature.

    Returns
    -------
        See signature."""
    if koopman_symmetry is None:
        return False
    if koopman_symmetry != "isotypic":
        msg = f"koopman_symmetry must be None or 'isotypic', got {koopman_symmetry!r}"
        raise ValueError(msg)
    if koopman_orbit_partition is not None or koopman_auto_orbits:
        msg = (
            "koopman_symmetry='isotypic' is mutually exclusive with "
            "koopman_orbit_partition / koopman_auto_orbits"
        )
        raise ValueError(msg)
    if koopman_orbit_method != DEFAULT_KOOPMAN_ORBIT_METHOD:
        msg = (
            "koopman_symmetry='isotypic' forces exact automorphism orbits; "
            "omit koopman_orbit_method (or leave the default)"
        )
        raise ValueError(msg)
    if kind != "graph":
        msg = (
            "koopman_symmetry='isotypic' requires koopman='graph' "
            f"(K_self MVP); got koopman={kind!r}"
        )
        raise ValueError(msg)
    if dynamics_mode != "discrete":
        msg = (
            "koopman_symmetry='isotypic' is unsupported for "
            f"dynamics_mode={dynamics_mode!r}"
        )
        raise ValueError(msg)
    if koopman_latent_dims is not None:
        msg = (
            "koopman_symmetry='isotypic' is unsupported with rectangular "
            "koopman_latent_dims (d_τ)"
        )
        raise ValueError(msg)
    return True


@dataclass(frozen=True, slots=True)
class ResolvedModelComponents:
    """Validated encoder / decoder / physics / operator bundle for construction.

    Attributes
    ----------
    encoder : nn.Module
        Topology-aware encoder module.
    decoder : nn.Module
        Topology-aware decoder module.
    physics : Any | None
        Optional physics residual helper.
    koopman : Any
        Assembled or injected Koopman operator.
    """

    encoder: Encoder
    decoder: Decoder
    latent_dim: int
    gnn_latent_dim: int
    physics_dim: int
    physics_preset: str | None
    physics_lifting_fn: PhysicsLiftingFn | None
    physics_position: PhysicsPosition
    time_step: float
    control_dim: int
    control_mode: ControlMode
    bilinear_rank: int | None
    dynamics_mode: DynamicsMode
    n_delays: int
    koopman: KoopmanOperatorContract
    koopman_kind: KoopmanKind
    synthesize_reverse_relations: bool = False


def _relgraph_edge_types_match(
    module: RelGraphEncoder | RelGraphDecoder,
    edge_types: tuple[tuple[str, str, str], ...],
) -> bool:
    """Return whether a RelGraph peer's declared edge types match ``edge_types``.

    Parameters
    ----------
    module
        Value for ``module``.
    edge_types
        Value for ``edge_types``.

    Returns
    -------
    object
        Function result.
    """
    declared = module.edge_types
    if declared is None:
        return True
    return tuple(tuple(edge_type) for edge_type in declared) == edge_types


def _rebuild_relgraph_peers_for_edge_types(
    encoder: RelGraphEncoder,
    decoder: RelGraphDecoder,
    edge_types: tuple[tuple[str, str, str], ...],
    *,
    latent_dims: Mapping[str, int] | None = None,
) -> tuple[RelGraphEncoder, RelGraphDecoder]:
    """Rebuild RelGraph peers for an expanded edge-type schema.

    Parameters
    ----------
    encoder
        Value for ``encoder``.
    decoder
        Value for ``decoder``.
    edge_types
        Value for ``edge_types``.
    latent_dims
        Value for ``latent_dims``.

    Returns
    -------
    object
        Function result.
    """
    num_relations = len(edge_types)
    resolved_dims = latent_dims if latent_dims is not None else encoder.latent_dims
    new_encoder = RelGraphEncoder(
        encoder.in_channels,
        encoder.hidden_channels,
        encoder.latent_dim,
        num_relations,
        num_layers=encoder.num_layers,
        activation=encoder.activation_name,
        normalization=encoder.normalization,
        root_weight=encoder.root_weight,
        node_types=encoder.node_types,
        edge_types=edge_types,
        latent_dims=resolved_dims,
    )
    new_decoder = RelGraphDecoder(
        decoder.latent_dim,
        decoder.hidden_channels,
        decoder.out_channels,
        num_relations,
        num_layers=decoder.num_layers,
        activation=decoder.activation_name,
        normalization=decoder.normalization,
        root_weight=decoder.root_weight,
        node_types=decoder.node_types,
        edge_types=edge_types,
        latent_dims=resolved_dims,
    )
    return new_encoder, new_decoder


def _align_relgraph_latent_dims(
    encoder: RelGraphEncoder,
    decoder: RelGraphDecoder,
    latent_dims: Mapping[str, int] | None,
) -> tuple[RelGraphEncoder, RelGraphDecoder]:
    """Rebuild RelGraph peers when factory ``latent_dims`` disagree with them.

    Parameters
    ----------
    encoder : RelGraphEncoder
        Current encoder peer.
    decoder : RelGraphDecoder
        Current decoder peer.
    latent_dims : mapping of str to int or None
        Target opt-in widths (``None`` keeps shared-d peers).

    Returns
    -------
    tuple of RelGraphEncoder and RelGraphDecoder
        Peers whose ``latent_dims`` match ``latent_dims``.

    Raises
    ------
    ValueError
        If encoder and decoder disagree with each other on ``latent_dims``.
    """
    enc_dims = encoder.latent_dims
    dec_dims = decoder.latent_dims
    if enc_dims != dec_dims:
        msg = (
            "RelGraphEncoder.latent_dims "
            f"{enc_dims!r} must match RelGraphDecoder.latent_dims {dec_dims!r}"
        )
        raise ValueError(msg)
    target = None if latent_dims is None else dict(latent_dims)
    current = None if enc_dims is None else dict(enc_dims)
    if current == target:
        return encoder, decoder
    edge_types = encoder.edge_types
    if edge_types is None:
        msg = (
            "RelGraph peers require explicit edge_types when aligning "
            "koopman_latent_dims"
        )
        raise ValueError(msg)
    return _rebuild_relgraph_peers_for_edge_types(
        encoder,
        decoder,
        tuple(tuple(edge_type) for edge_type in edge_types),
        latent_dims=target,
    )


def validate_typed_relgraph_peers(
    encoder: RelGraphEncoder,
    decoder: Decoder,
    operator: HeteroKoopmanOperator,
) -> None:
    """Validate typed RelGraph peers against the hetero operator schema.

    Encoder and decoder node-type order must equal the operator's
    ``node_types`` so latent slices line up with per-type self blocks.
    Opt-in ``latent_dims`` / rectangular mode must agree across peers and
    the discrete or continuous hetero operator.

    Parameters
    ----------
    encoder : RelGraphEncoder
        Resolved relational encoder.
    decoder : Decoder
        Resolved decoder (validated only when it is a ``RelGraphDecoder``).
    operator : HeteroGraphKoopmanOperator or ContinuousHeteroGraphKoopmanOperator
        Resolved discrete or continuous hetero Koopman operator.

    Raises
    ------
    ValueError
        If typed node-type order mismatches the operator, if only one of the
        peers is typed, if the operator is typed while the peers are not, or
        if ``latent_dims`` / rectangular flags disagree.
    """
    operator_types = tuple(operator.node_types)
    encoder_typed = bool(encoder.is_typed)
    decoder_typed = isinstance(decoder, RelGraphDecoder) and bool(decoder.is_typed)
    if encoder_typed != decoder_typed:
        msg = (
            "RelGraphEncoder and RelGraphDecoder must agree on typed channels; "
            f"encoder typed={encoder_typed}, decoder typed={decoder_typed}"
        )
        raise ValueError(msg)
    if not encoder_typed:
        if len(operator_types) > 1:
            msg = (
                "HeteroGraphKoopmanOperator declares node types "
                f"{operator_types!r} but RelGraph peers use a single shared "
                "in_channels; pass a mapping of per-type feature widths"
            )
            raise ValueError(msg)
        return
    if tuple(encoder.node_types) != operator_types:
        msg = (
            f"RelGraphEncoder node types {tuple(encoder.node_types)!r} must "
            f"match HeteroGraphKoopmanOperator node types {operator_types!r}"
        )
        raise ValueError(msg)
    assert isinstance(decoder, RelGraphDecoder)
    if tuple(decoder.node_types) != operator_types:
        msg = (
            f"RelGraphDecoder node types {tuple(decoder.node_types)!r} must "
            f"match HeteroGraphKoopmanOperator node types {operator_types!r}"
        )
        raise ValueError(msg)
    operator_edges = tuple(tuple(edge_type) for edge_type in operator.edge_types)
    for module in (encoder, decoder):
        declared = module.edge_types
        if declared is None:
            continue
        if tuple(tuple(edge_type) for edge_type in declared) != operator_edges:
            msg = (
                f"{type(module).__name__} edge types {tuple(declared)!r} must "
                f"match HeteroGraphKoopmanOperator edge types {operator_edges!r}"
            )
            raise ValueError(msg)
    enc_dims = encoder.latent_dims
    dec_dims = decoder.latent_dims
    op_dims = operator.latent_dims
    if enc_dims != dec_dims or enc_dims != op_dims:
        msg = (
            "RelGraph / hetero operator latent_dims must match; "
            f"got encoder={enc_dims!r}, decoder={dec_dims!r}, "
            f"operator={op_dims!r}"
        )
        raise ValueError(msg)
    if bool(encoder.is_rectangular) != bool(operator.is_rectangular) or bool(
        decoder.is_rectangular
    ) != bool(operator.is_rectangular):
        msg = (
            "RelGraph / hetero operator is_rectangular flags must match; "
            f"got encoder={encoder.is_rectangular}, "
            f"decoder={decoder.is_rectangular}, "
            f"operator={operator.is_rectangular}"
        )
        raise ValueError(msg)


def resolve_model_components(
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
    koopman_adjacency: GraphAdjacency = DEFAULT_KOOPMAN_ADJACENCY,
    koopman_filter_degree: int = DEFAULT_KOOPMAN_FILTER_DEGREE,
    koopman_hypergraph_incidence_mode: str = (
        DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE
    ),
    koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
    koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
    koopman_local_hidden_dims: Sequence[int] | None = None,
    koopman_orbit_partition: Sequence[Sequence[int]] | None = (
        DEFAULT_KOOPMAN_ORBIT_PARTITION
    ),
    koopman_auto_orbits: bool = DEFAULT_KOOPMAN_AUTO_ORBITS,
    koopman_orbit_method: OrbitMethod = DEFAULT_KOOPMAN_ORBIT_METHOD,
    koopman_symmetry: str | None = DEFAULT_KOOPMAN_SYMMETRY,
    control_dim: int = 0,
    control_mode: ControlMode = DEFAULT_CONTROL_MODE,
    bilinear_rank: int | None = DEFAULT_BILINEAR_RANK,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
    physics_preset: str | None = None,
    physics_dim: int = 0,
    physics_position: PhysicsPosition,
    n_delays: int = 1,
    koopman_node_types: Sequence[str] | None = None,
    koopman_edge_types: Sequence[Sequence[str]] | None = None,
    koopman_relation_tying: str = "independent",
    koopman_basis_size: int | None = None,
    koopman_synthesize_reverse_relations: bool = False,
    koopman_latent_dims: Mapping[str, int] | None = None,
    koopman_num_modes: int = DEFAULT_KOOPMAN_NUM_MODES,
    koopman_parameter_dim: int = DEFAULT_KOOPMAN_PARAMETER_DIM,
    koopman_weight_kind: WeightKind = DEFAULT_KOOPMAN_WEIGHT_KIND,
) -> ResolvedModelComponents:
    """Validate and assemble encoder / physics / Koopman construction inputs.

    Parameters
    ----------
    encoder : Encoder
        Encoder module passed to the façade.
    decoder : Decoder
        Decoder module passed to the façade.
    latent_dim : int
        Latent state dimension.
    time_step : float
        Nominal discrete time step.
    dynamics_mode : DynamicsMode
        ``"discrete"`` or ``"continuous"`` operator family.
    koopman : KoopmanArg
        String kind or injected operator instance.
    koopman_init_mode : InitMode
        Operator initialization mode.
    koopman_init_scale : float
        Initialization scale for random operator fills.
    koopman_parameterization : Parameterization
        Dense / ODO / Schur / dissipative / Lyapunov parameterization.
    koopman_max_spectral_radius : float
        Spectral-radius clamp for discrete operators.
    koopman_auxiliary_hidden_dims : Sequence[int] or None
        Hidden widths for auxiliary-spectral operators.
    koopman_sparsity : str
        Networked sparsity mode (``dense`` / ``block_diagonal`` /
        ``distributed``). ``distributed`` selects matrix-free inverse /
        Arnoldi spectrum helpers on discrete graph and multiplex hetero; it
        is **not** :mod:`koopman_graph.distributed` trainer DDP /
        ``[distributed]`` extras and does **not** enable multi-GPU training.
    koopman_adjacency : {"symmetric", "random_walk", "dual_random_walk"}
        Neighbor-coupling normalization for ``koopman="graph"`` /
        ``"continuous_graph"``. Default ``"symmetric"``.
    koopman_filter_degree : int, optional
        Monomial hop degree for discrete ``koopman="graph"``. Default ``1``
        (one-tap). Rejected for other kinds and for continuous graph.
    koopman_hypergraph_incidence_mode : str
        Incidence normalization for ``koopman="hypergraph"``
        (``zhou_symmetric`` / ``forward_random_walk`` / ``dual_random_walk``).
        Default ``"zhou_symmetric"``.
    koopman_local_window, koopman_local_rank, koopman_local_hidden_dims
        Global/local operator hyperparameters.
    koopman_orbit_partition, koopman_auto_orbits, koopman_orbit_method
        Symmetry / orbit-tying configuration.
    koopman_symmetry : {None, "isotypic"}, optional
        Representation-theoretic symmetry mode. ``"isotypic"`` ties
        ``K_self`` via exact ``Aut(G)`` orbits on ``koopman="graph"``
        (mutually exclusive with orbit kwargs). Default ``None``.
    control_dim : int
        Control input dimension (``0`` for uncontrolled).
    control_mode : ControlMode
        ``"additive"`` or ``"bilinear"`` control.
    bilinear_rank : int or None
        Low-rank bilinear size when ``control_mode="bilinear"``.
    physics_lifting_fn : PhysicsLiftingFn or None
        Optional physics feature map.
    physics_preset : str or None
        Named physics lifting preset.
    physics_dim : int
        Physics feature dimension.
    physics_position : PhysicsPosition
        Where physics features are concatenated.
    n_delays : int
        Delay-embedding depth (``1`` = no delay).
    koopman_node_types : sequence of str or None, optional
        Ordered node-type names for ``koopman="hetero_graph"``.
    koopman_edge_types : sequence of sequence of str or None, optional
        Ordered ``(src, rel, dst)`` triples for ``koopman="hetero_graph"``.
    koopman_relation_tying : {"independent", "basis"}, optional
        Relation-factor tying for ``koopman="hetero_graph"``. Default
        ``"independent"``.
    koopman_basis_size : int or None, optional
        Basis size ``B`` when ``koopman_relation_tying="basis"``.
    koopman_synthesize_reverse_relations : bool, optional
        When ``True`` with ``koopman="hetero_graph"``, expand
        ``koopman_edge_types`` via
        :func:`~koopman_graph.graph_utils.synthesize_reverse_edge_types`
        and align RelGraph / operator banks to the expanded schema.
        Default ``False``.
    koopman_latent_dims : mapping of str to int or None, optional
        Opt-in per-type latent widths. When set, RelGraph peers are rebuilt to
        match and the discrete or continuous hetero operator receives the
        same mapping.
    koopman_num_modes : int, optional
        Interpolant mode count for ``koopman="parametric"``. Default 2.
    koopman_parameter_dim : int, optional
        Regime-coordinate width for ``koopman="parametric"``. Default 1.
    koopman_weight_kind : {"rbf", "simplex"}, optional
        Interpolant weights for ``koopman="parametric"``. Default ``"rbf"``.

    Returns
    -------
    ResolvedModelComponents
        Frozen bundle ready to assign onto the façade.

    Raises
    ------
    ValueError
        If dimensions, physics settings, delays, or operator factory inputs
        are inconsistent.
    TypeError
        If ``koopman`` is neither a string kind nor a contract module.
    """
    if dynamics_mode not in {"discrete", "continuous", "stochastic"}:
        msg = (
            "dynamics_mode must be 'discrete', 'continuous', or 'stochastic', "
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

    from koopman_graph.model.validation import uses_relgraph_modules

    uses_relgraph = uses_relgraph_modules(encoder, decoder)
    kind_preview, _ = parse_koopman_arg(koopman)
    wants_hetero = kind_preview == "hetero_graph" or isinstance(
        koopman, HeteroKoopmanOperator
    )
    if wants_hetero and not uses_relgraph:
        msg = (
            "koopman='hetero_graph' (or an injected HeteroGraphKoopmanOperator "
            "/ ContinuousHeteroGraphKoopmanOperator) requires RelGraphEncoder "
            "and RelGraphDecoder"
        )
        raise ValueError(msg)
    if uses_relgraph and not wants_hetero:
        msg = (
            "RelGraphEncoder / RelGraphDecoder require koopman='hetero_graph' "
            f"(got koopman={kind_preview!r})"
        )
        raise ValueError(msg)
    if uses_relgraph and n_delays != 1:
        msg = (
            "RelGraphEncoder / RelGraphDecoder do not support n_delays > 1 "
            f"(got n_delays={n_delays})"
        )
        raise ValueError(msg)
    if uses_relgraph and physics_dim != 0:
        msg = "physics-informed observables are unsupported with RelGraph peers"
        raise ValueError(msg)

    resolved_latent_dims: Mapping[str, int] | None = koopman_latent_dims
    if isinstance(koopman, HeteroKoopmanOperator):
        if (
            koopman_latent_dims is not None
            and koopman.latent_dims is not None
            and dict(koopman_latent_dims) != dict(koopman.latent_dims)
        ):
            msg = (
                "koopman_latent_dims "
                f"{dict(koopman_latent_dims)!r} must match injected "
                f"{type(koopman).__name__}.latent_dims "
                f"{dict(koopman.latent_dims)!r}"
            )
            raise ValueError(msg)
        if resolved_latent_dims is None:
            resolved_latent_dims = koopman.latent_dims
    if uses_relgraph:
        assert isinstance(encoder, RelGraphEncoder)
        assert isinstance(decoder, RelGraphDecoder)
        if encoder.latent_dims != decoder.latent_dims:
            msg = (
                "RelGraphEncoder.latent_dims "
                f"{encoder.latent_dims!r} must match "
                f"RelGraphDecoder.latent_dims {decoder.latent_dims!r}"
            )
            raise ValueError(msg)
        if resolved_latent_dims is None:
            resolved_latent_dims = encoder.latent_dims
        elif encoder.latent_dims is not None and dict(encoder.latent_dims) != dict(
            resolved_latent_dims
        ):
            # Rebuild peers below to match the factory / injected target.
            pass
    if resolved_latent_dims is not None and not wants_hetero:
        msg = (
            "koopman_latent_dims requires koopman='hetero_graph' "
            "(or an injected discrete/continuous hetero operator)"
        )
        raise ValueError(msg)
    if uses_relgraph:
        assert isinstance(encoder, RelGraphEncoder)
        assert isinstance(decoder, RelGraphDecoder)
        encoder, decoder = _align_relgraph_latent_dims(
            encoder,
            decoder,
            resolved_latent_dims,
        )

    synthesize_reverse_relations = bool(koopman_synthesize_reverse_relations)
    resolved_edge_types: Sequence[Sequence[str]] | None = koopman_edge_types
    if synthesize_reverse_relations:
        if not wants_hetero:
            msg = (
                "koopman_synthesize_reverse_relations=True requires "
                "koopman='hetero_graph' (or an injected HeteroGraphKoopmanOperator)"
            )
            raise ValueError(msg)
        if koopman_edge_types is None:
            msg = (
                "koopman_synthesize_reverse_relations=True requires "
                "koopman_edge_types (forward schema to expand)"
            )
            raise ValueError(msg)
        assert isinstance(encoder, RelGraphEncoder)
        assert isinstance(decoder, RelGraphDecoder)
        n_forward = len(tuple(koopman_edge_types))
        expanded = synthesize_reverse_edge_types(koopman_edge_types)
        n_expanded = len(expanded)
        if encoder.num_relations == n_expanded:
            if not _relgraph_edge_types_match(encoder, expanded):
                msg = (
                    "RelGraphEncoder.edge_types "
                    f"{tuple(encoder.edge_types)!r} must match the expanded "
                    f"schema {expanded!r} when num_relations already equals "
                    f"|expanded|={n_expanded}"
                )
                raise ValueError(msg)
            if not _relgraph_edge_types_match(decoder, expanded):
                msg = (
                    "RelGraphDecoder.edge_types "
                    f"{tuple(decoder.edge_types)!r} must match the expanded "
                    f"schema {expanded!r} when num_relations already equals "
                    f"|expanded|={n_expanded}"
                )
                raise ValueError(msg)
        elif encoder.num_relations == n_forward:
            encoder, decoder = _rebuild_relgraph_peers_for_edge_types(
                encoder,
                decoder,
                expanded,
            )
        else:
            msg = (
                "koopman_synthesize_reverse_relations=True requires RelGraph "
                f"num_relations equal to forward |R|={n_forward} (auto-rebuild) "
                f"or expanded |R|={n_expanded} (use as-is); got "
                f"{encoder.num_relations}"
            )
            raise ValueError(msg)
        resolved_edge_types = expanded

    num_relations: int | None = None
    relation_normalization = None
    if uses_relgraph:
        assert isinstance(encoder, RelGraphEncoder)
        num_relations = encoder.num_relations
        relation_normalization = encoder.normalization

    operator, koopman_kind = build_koopman(
        koopman=koopman,
        latent_dim=latent_dim,
        control_dim=control_dim,
        control_mode=control_mode,
        bilinear_rank=bilinear_rank,
        dynamics_mode=dynamics_mode,
        koopman_init_mode=koopman_init_mode,
        koopman_init_scale=koopman_init_scale,
        koopman_parameterization=koopman_parameterization,
        koopman_max_spectral_radius=koopman_max_spectral_radius,
        koopman_auxiliary_hidden_dims=koopman_auxiliary_hidden_dims,
        koopman_sparsity=koopman_sparsity,
        koopman_adjacency=koopman_adjacency,
        koopman_filter_degree=koopman_filter_degree,
        koopman_hypergraph_incidence_mode=koopman_hypergraph_incidence_mode,
        koopman_local_window=koopman_local_window,
        koopman_local_rank=koopman_local_rank,
        koopman_local_hidden_dims=koopman_local_hidden_dims,
        koopman_orbit_partition=koopman_orbit_partition,
        koopman_auto_orbits=koopman_auto_orbits,
        koopman_orbit_method=koopman_orbit_method,
        koopman_symmetry=koopman_symmetry,
        num_relations=num_relations,
        relation_normalization=relation_normalization,
        node_types=koopman_node_types,
        edge_types=resolved_edge_types,
        relation_tying=koopman_relation_tying,
        basis_size=koopman_basis_size,
        latent_dims=resolved_latent_dims,
        koopman_num_modes=koopman_num_modes,
        koopman_parameter_dim=koopman_parameter_dim,
        koopman_weight_kind=koopman_weight_kind,
    )
    if isinstance(operator, HeteroKoopmanOperator):
        if not uses_relgraph:
            msg = (
                f"{type(operator).__name__} requires RelGraphEncoder and "
                "RelGraphDecoder"
            )
            raise ValueError(msg)
        assert isinstance(encoder, RelGraphEncoder)
        if operator.num_relations != encoder.num_relations:
            msg = (
                f"{type(operator).__name__}.num_relations "
                f"({operator.num_relations}) must match "
                f"RelGraphEncoder.num_relations ({encoder.num_relations})"
            )
            raise ValueError(msg)
        if operator.normalization != encoder.normalization:
            msg = (
                f"{type(operator).__name__}.normalization "
                f"({operator.normalization!r}) must match "
                f"RelGraphEncoder.normalization ({encoder.normalization!r})"
            )
            raise ValueError(msg)
        validate_typed_relgraph_peers(encoder, decoder, operator)
        koopman_kind = "hetero_graph"
    return ResolvedModelComponents(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        gnn_latent_dim=gnn_latent_dim,
        physics_dim=physics_dim,
        physics_preset=physics_preset,
        physics_lifting_fn=resolved_physics_fn,
        physics_position=resolve_physics_position(physics_position),
        time_step=time_step,
        control_dim=control_dim,
        control_mode=control_mode,
        bilinear_rank=bilinear_rank,
        dynamics_mode=dynamics_mode,
        n_delays=resolved_n_delays,
        koopman=operator,
        koopman_kind=koopman_kind,
        synthesize_reverse_relations=synthesize_reverse_relations,
    )


def apply_resolved_components(
    model: nn.Module,
    components: ResolvedModelComponents,
) -> None:
    """Assign a resolved construction bundle onto a model instance.

    Parameters
    ----------
    model : nn.Module
        Façade receiving encoder / decoder / operator attributes.
    components : ResolvedModelComponents
        Bundle from :func:`resolve_model_components`.
    """
    model.encoder = components.encoder  # type: ignore[attr-defined]
    model.decoder = components.decoder  # type: ignore[attr-defined]
    model.latent_dim = components.latent_dim  # type: ignore[attr-defined]
    model.gnn_latent_dim = components.gnn_latent_dim  # type: ignore[attr-defined]
    model.physics_dim = components.physics_dim  # type: ignore[attr-defined]
    model.physics_preset = components.physics_preset  # type: ignore[attr-defined]
    model.physics_lifting_fn = components.physics_lifting_fn  # type: ignore[attr-defined]
    model.physics_position = components.physics_position  # type: ignore[attr-defined]
    model.time_step = components.time_step  # type: ignore[attr-defined]
    model.control_dim = components.control_dim  # type: ignore[attr-defined]
    model.control_mode = components.control_mode  # type: ignore[attr-defined]
    model.bilinear_rank = components.bilinear_rank  # type: ignore[attr-defined]
    model.dynamics_mode = components.dynamics_mode  # type: ignore[attr-defined]
    model.n_delays = components.n_delays  # type: ignore[attr-defined]
    model.koopman = components.koopman  # type: ignore[attr-defined]
    model.koopman_kind = components.koopman_kind  # type: ignore[attr-defined]
    model.synthesize_reverse_relations = (  # type: ignore[attr-defined]
        components.synthesize_reverse_relations
    )


def parse_koopman_arg(
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
        if koopman not in {
            "pernode",
            "graph",
            "hypergraph",
            "hetero_graph",
            "global_local",
            "continuous_graph",
            "switched",
            "mixture",
            "parametric",
            "hodge",
        }:
            msg = (
                "koopman string kind must be 'pernode', 'graph', "
                "'hypergraph', 'hetero_graph', 'global_local', "
                "'continuous_graph', 'switched', 'mixture', "
                "'parametric', or 'hodge', "
                f"got {koopman!r}"
            )
            raise ValueError(msg)
        return koopman, None
    return "pernode", koopman


def resolve_injected_koopman(
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
    koopman_auxiliary_hidden_dims: tuple[int, ...] | None,
    koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
    koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
    koopman_local_hidden_dims: tuple[int, ...] | None = None,
    koopman_orbit_partition: Sequence[Sequence[int]] | None = (
        DEFAULT_KOOPMAN_ORBIT_PARTITION
    ),
    koopman_auto_orbits: bool = DEFAULT_KOOPMAN_AUTO_ORBITS,
    koopman_orbit_method: OrbitMethod = DEFAULT_KOOPMAN_ORBIT_METHOD,
    koopman_symmetry: str | None = DEFAULT_KOOPMAN_SYMMETRY,
    koopman_adjacency: GraphAdjacency = DEFAULT_KOOPMAN_ADJACENCY,
    koopman_filter_degree: int = DEFAULT_KOOPMAN_FILTER_DEGREE,
    koopman_hypergraph_incidence_mode: str = (
        DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE
    ),
    koopman_num_modes: int = DEFAULT_KOOPMAN_NUM_MODES,
    koopman_parameter_dim: int = DEFAULT_KOOPMAN_PARAMETER_DIM,
    koopman_weight_kind: WeightKind = DEFAULT_KOOPMAN_WEIGHT_KIND,
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
    koopman_auxiliary_hidden_dims : tuple of int or None
        Auxiliary network widths (must be default / ``None`` when injecting).
    koopman_local_window : int
        See the function signature / summary for ``koopman_local_window``.
    koopman_local_rank : int
        See the function signature / summary for ``koopman_local_rank``.
    koopman_local_hidden_dims : tuple[int, ...] | None
        See the function signature / summary for ``koopman_local_hidden_dims``.
    koopman_orbit_partition : Sequence[Sequence[int]] | None
        See the function signature / summary for ``koopman_orbit_partition``.
    koopman_auto_orbits : bool
        See the function signature / summary for ``koopman_auto_orbits``.
    koopman_orbit_method : OrbitMethod
        See the function signature / summary for ``koopman_orbit_method``.
    koopman_adjacency : GraphAdjacency
        Factory adjacency mode (must be default when injecting).
    koopman_filter_degree : int
        Factory hop degree (must be default when injecting).
    koopman_symmetry
        See signature.

    koopman_hypergraph_incidence_mode
        See signature.
    koopman_num_modes : int, optional
        Must stay default when injecting.
    koopman_parameter_dim : int, optional
        Must stay default when injecting.
    koopman_weight_kind : str, optional
        Must stay default when injecting.
    Returns
    -------

    KoopmanOperatorContract
        Validated operator module ready for assignment.

    Raises
    ------

    TypeError
        If ``koopman`` is not an ``nn.Module``.
    ValueError
        If factory kwargs conflict or dimensions / dynamics mode mismatch."""
    if not isinstance(koopman, nn.Module):
        msg = (
            "Injected koopman must be an nn.Module implementing "
            "KoopmanOperatorContract, "
            f"got {type(koopman).__name__}"
        )
        raise TypeError(msg)

    conflicting: list[str] = []
    if koopman_init_mode != DEFAULT_KOOPMAN_INIT_MODE:
        conflicting.append("koopman_init_mode")
    if koopman_init_scale != DEFAULT_KOOPMAN_INIT_SCALE:
        conflicting.append("koopman_init_scale")
    if koopman_parameterization != DEFAULT_KOOPMAN_PARAMETERIZATION:
        conflicting.append("koopman_parameterization")
    if koopman_max_spectral_radius != DEFAULT_KOOPMAN_MAX_SPECTRAL_RADIUS:
        conflicting.append("koopman_max_spectral_radius")
    if (
        koopman_auxiliary_hidden_dims is not None
        and koopman_auxiliary_hidden_dims != DEFAULT_KOOPMAN_AUXILIARY_HIDDEN_DIMS
    ):
        conflicting.append("koopman_auxiliary_hidden_dims")
    if koopman_local_window != DEFAULT_KOOPMAN_LOCAL_WINDOW:
        conflicting.append("koopman_local_window")
    if koopman_local_rank != DEFAULT_KOOPMAN_LOCAL_RANK:
        conflicting.append("koopman_local_rank")
    if (
        koopman_local_hidden_dims is not None
        and koopman_local_hidden_dims != DEFAULT_KOOPMAN_LOCAL_HIDDEN_DIMS
    ):
        conflicting.append("koopman_local_hidden_dims")
    if koopman_orbit_partition is not DEFAULT_KOOPMAN_ORBIT_PARTITION:
        conflicting.append("koopman_orbit_partition")
    if koopman_auto_orbits != DEFAULT_KOOPMAN_AUTO_ORBITS:
        conflicting.append("koopman_auto_orbits")
    if koopman_orbit_method != DEFAULT_KOOPMAN_ORBIT_METHOD:
        conflicting.append("koopman_orbit_method")
    if koopman_symmetry != DEFAULT_KOOPMAN_SYMMETRY:
        conflicting.append("koopman_symmetry")
    if koopman_adjacency != DEFAULT_KOOPMAN_ADJACENCY:
        conflicting.append("koopman_adjacency")
    if koopman_filter_degree != DEFAULT_KOOPMAN_FILTER_DEGREE:
        conflicting.append("koopman_filter_degree")
    if koopman_hypergraph_incidence_mode != DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE:
        conflicting.append("koopman_hypergraph_incidence_mode")
    if koopman_num_modes != DEFAULT_KOOPMAN_NUM_MODES:
        conflicting.append("koopman_num_modes")
    if koopman_parameter_dim != DEFAULT_KOOPMAN_PARAMETER_DIM:
        conflicting.append("koopman_parameter_dim")
    if koopman_weight_kind != DEFAULT_KOOPMAN_WEIGHT_KIND:
        conflicting.append("koopman_weight_kind")
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

    injected_mode = getattr(koopman, "control_mode", DEFAULT_CONTROL_MODE)
    injected_rank = getattr(koopman, "bilinear_rank", DEFAULT_BILINEAR_RANK)
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

    if isinstance(koopman, ContinuousKoopmanOperator) and dynamics_mode != "continuous":
        msg = "Injected ContinuousKoopmanOperator requires dynamics_mode='continuous'"
        raise ValueError(msg)
    if isinstance(koopman, KoopmanOperator) and dynamics_mode not in {
        "discrete",
        "stochastic",
    }:
        msg = (
            "Injected KoopmanOperator requires dynamics_mode='discrete' or 'stochastic'"
        )
        raise ValueError(msg)
    if isinstance(koopman, GraphKoopmanOperator) and dynamics_mode not in {
        "discrete",
        "stochastic",
    }:
        msg = (
            "Injected GraphKoopmanOperator requires dynamics_mode='discrete' or "
            "'stochastic'"
        )
        raise ValueError(msg)
    if (
        isinstance(
            koopman,
            (
                SwitchedKoopmanOperator
                | MixtureKoopmanOperator
                | ParametricKoopmanOperator
            ),
        )
        and dynamics_mode != "discrete"
    ):
        msg = f"Injected {type(koopman).__name__} requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if isinstance(koopman, HypergraphKoopmanOperator):
        if dynamics_mode == "stochastic":
            msg = (
                "dynamics_mode='stochastic' does not support hypergraph "
                "operators; use pernode, graph, or shared-d hetero_graph"
            )
            raise ValueError(msg)
        if dynamics_mode != "discrete":
            msg = "Injected HypergraphKoopmanOperator requires dynamics_mode='discrete'"
            raise ValueError(msg)
    if isinstance(koopman, HeteroGraphKoopmanOperator):
        if dynamics_mode not in {"discrete", "stochastic"}:
            msg = (
                "Injected HeteroGraphKoopmanOperator requires "
                "dynamics_mode='discrete' or 'stochastic'"
            )
            raise ValueError(msg)
        if dynamics_mode == "stochastic" and getattr(koopman, "is_rectangular", False):
            msg = (
                "dynamics_mode='stochastic' does not support rectangular "
                "hetero latent_dims; use shared latent_dim"
            )
            raise ValueError(msg)
    if (
        isinstance(koopman, ContinuousHeteroGraphKoopmanOperator)
        and dynamics_mode != "continuous"
    ):
        msg = (
            "Injected ContinuousHeteroGraphKoopmanOperator requires "
            "dynamics_mode='continuous'"
        )
        raise ValueError(msg)
    if isinstance(koopman, GlobalLocalKoopmanOperator):
        if dynamics_mode == "stochastic":
            msg = (
                "dynamics_mode='stochastic' does not support global_local "
                "operators; use pernode, graph, or shared-d hetero_graph"
            )
            raise ValueError(msg)
        if dynamics_mode != "discrete":
            msg = (
                "Injected GlobalLocalKoopmanOperator requires dynamics_mode='discrete'"
            )
            raise ValueError(msg)
    if (
        isinstance(koopman, ContinuousGraphKoopmanOperator)
        and dynamics_mode != "continuous"
    ):
        msg = (
            "Injected ContinuousGraphKoopmanOperator requires "
            "dynamics_mode='continuous'"
        )
        raise ValueError(msg)

    if dynamics_mode == "stochastic" and not getattr(koopman, "stochastic", False):
        attach_process_noise(koopman, latent_dim=latent_dim)
    return koopman


def _finalize_built_koopman(
    operator: KoopmanOperatorContract,
    kind: KoopmanKind,
    *,
    dynamics_mode: DynamicsMode,
    latent_dim: int,
) -> tuple[KoopmanOperatorContract, KoopmanKind]:
    """Attach stochastic process noise when requested, then return the pair.

    Parameters
    ----------
    operator : KoopmanOperatorContract
        Built discrete-family operator.
    kind : KoopmanKind
        Resolved factory kind.
    dynamics_mode : DynamicsMode
        Requested dynamics mode.
    latent_dim : int
        Shared latent width for process-noise diagonal.

    Returns
    -------
    tuple of KoopmanOperatorContract and KoopmanKind
        Operator (possibly with process noise) and kind.
    """
    if dynamics_mode == "stochastic":
        attach_process_noise(operator, latent_dim=latent_dim)
    return operator, kind


def _reject_stochastic_kind(
    kind: KoopmanKind,
    *,
    latent_dim: int,
    latent_dims: Mapping[str, int] | None,
) -> None:
    """Reject unsupported ``dynamics_mode='stochastic'`` factory kinds.

    Parameters
    ----------
    kind : KoopmanKind
        Requested Koopman factory kind.
    latent_dim : int
        Shared latent width.
    latent_dims : mapping of str to int or None
        Optional per-type widths (rectangular hetero).

    Raises
    ------
    ValueError
        If the kind or rectangular hetero layout is unsupported.
    """
    if kind in {
        "hypergraph",
        "global_local",
        "continuous_graph",
        "switched",
        "mixture",
        "parametric",
    }:
        msg = (
            "dynamics_mode='stochastic' supports koopman='pernode', 'graph', "
            f"or shared-d 'hetero_graph'; got koopman={kind!r}"
        )
        raise ValueError(msg)
    if (
        kind == "hetero_graph"
        and latent_dims is not None
        and any(width != latent_dim for width in latent_dims.values())
    ):
        msg = (
            "dynamics_mode='stochastic' does not support rectangular "
            "hetero latent_dims; use shared latent_dim"
        )
        raise ValueError(msg)


def _reject_local_kwargs_unless_global_local(
    kind: KoopmanKind,
    *,
    koopman_local_window: int,
    koopman_local_rank: int,
    koopman_local_hidden_dims: Sequence[int] | None,
) -> None:
    """Reject non-default local-network kwargs for non-global_local kinds.

    Parameters
    ----------

    kind : KoopmanKind
        See the function signature / summary for ``kind``.
    koopman_local_window : int
        See the function signature / summary for ``koopman_local_window``.
    koopman_local_rank : int
        See the function signature / summary for ``koopman_local_rank``.
    koopman_local_hidden_dims : Sequence[int] | None
        See the function signature / summary for ``koopman_local_hidden_dims``.

    Returns
    -------

    None
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    non_default = (
        koopman_local_window != DEFAULT_KOOPMAN_LOCAL_WINDOW
        or koopman_local_rank != DEFAULT_KOOPMAN_LOCAL_RANK
        or (
            koopman_local_hidden_dims is not None
            and tuple(koopman_local_hidden_dims) != DEFAULT_KOOPMAN_LOCAL_HIDDEN_DIMS
        )
    )
    if kind != "global_local" and non_default:
        msg = (
            "koopman_local_window / koopman_local_rank / "
            "koopman_local_hidden_dims require koopman='global_local'"
        )
        raise ValueError(msg)


def _reject_parametric_kwargs_unless_parametric(
    kind: KoopmanKind,
    *,
    koopman_num_modes: int,
    koopman_parameter_dim: int,
    koopman_weight_kind: str,
) -> None:
    """Reject non-default interpolant kwargs for non-parametric kinds.

    Parameters
    ----------
    kind : KoopmanKind
        Resolved factory kind.
    koopman_num_modes : int
        Requested mode count.
    koopman_parameter_dim : int
        Requested :math:`d_\\mu`.
    koopman_weight_kind : str
        Requested weight kind.

    Raises
    ------
    ValueError
        If interpolant kwargs are set for another kind.
    """
    non_default = (
        koopman_num_modes != DEFAULT_KOOPMAN_NUM_MODES
        or koopman_parameter_dim != DEFAULT_KOOPMAN_PARAMETER_DIM
        or koopman_weight_kind != DEFAULT_KOOPMAN_WEIGHT_KIND
    )
    if kind != "parametric" and non_default:
        msg = (
            "koopman_num_modes / koopman_parameter_dim / "
            "koopman_weight_kind require koopman='parametric'"
        )
        raise ValueError(msg)


def build_encoder_peers(
    encoder: EncoderKind,
    *,
    in_channels: int,
    hidden_channels: int,
    latent_dim: int,
    out_channels: int,
    num_layers: int = 2,
    activation: ActivationName = "relu",
    residual: bool = False,
    restriction_maps: Literal["diagonal", "general"] = "diagonal",
) -> tuple[Encoder, Decoder]:
    """Build a matched encoder / decoder peer pair by kind string.

    Parameters
    ----------
    encoder : {"sheaf", "cell_complex"}
        Encoder family. ``\"sheaf\"`` builds sheaf peers; ``\"cell_complex\"``
        builds Hodge-``L_0`` cell-complex peers
        (:class:`~koopman_graph.nn.cell_complex.CellComplexGNNEncoder` /
        :class:`~koopman_graph.nn.cell_complex.CellComplexGNNDecoder`).
    in_channels : int
        Physical input feature width.
    hidden_channels : int
        Hidden channel width for both peers.
    latent_dim : int
        Per-node latent width (must match ``GraphKoopmanModel.latent_dim``
        when physics lifting is off).
    out_channels : int
        Physical output feature width for the decoder.
    num_layers : int, optional
        Stack depth for both peers. Default is ``2``.
    activation : str, optional
        Hidden activation name. Default is ``\"relu\"``.
    residual : bool, optional
        Residual skips when widths match. Default is ``False``.
    restriction_maps : {"diagonal", "general"}, optional
        Sheaf restriction-map parameterization (sheaf peers only). Default
        is ``\"diagonal\"``. ``\"general\"`` is opt-in and refused above the
        documented channel ceiling.

    Returns
    -------
    tuple of Encoder, Decoder
        Matched peer pair ready for :class:`~koopman_graph.model.GraphKoopmanModel`.

    Raises
    ------
    ValueError
        If ``encoder`` is not a registered kind.
    """
    if encoder == "sheaf":
        return (
            SheafGNNEncoder(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers=num_layers,
                activation=activation,
                residual=residual,
                restriction_maps=restriction_maps,
            ),
            SheafGNNDecoder(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers=num_layers,
                activation=activation,
                residual=residual,
                restriction_maps=restriction_maps,
            ),
        )
    if encoder == "cell_complex":
        return (
            CellComplexGNNEncoder(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers=num_layers,
                activation=activation,
                residual=residual,
            ),
            CellComplexGNNDecoder(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers=num_layers,
                activation=activation,
                residual=residual,
            ),
        )
    msg = f"Unknown encoder={encoder!r}; supported kinds: 'sheaf', 'cell_complex'"
    raise ValueError(msg)


def build_koopman(
    *,
    koopman: KoopmanArg,
    latent_dim: int,
    control_dim: int,
    control_mode: ControlMode,
    bilinear_rank: int | None,
    dynamics_mode: DynamicsMode,
    koopman_init_mode: InitMode,
    koopman_init_scale: float,
    koopman_parameterization: Parameterization,
    koopman_max_spectral_radius: float,
    koopman_auxiliary_hidden_dims: Sequence[int] | None,
    koopman_sparsity: str = "dense",
    koopman_adjacency: GraphAdjacency = DEFAULT_KOOPMAN_ADJACENCY,
    koopman_filter_degree: int = DEFAULT_KOOPMAN_FILTER_DEGREE,
    koopman_hypergraph_incidence_mode: str = (
        DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE
    ),
    koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
    koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
    koopman_local_hidden_dims: Sequence[int] | None = None,
    koopman_orbit_partition: Sequence[Sequence[int]] | None = (
        DEFAULT_KOOPMAN_ORBIT_PARTITION
    ),
    koopman_auto_orbits: bool = DEFAULT_KOOPMAN_AUTO_ORBITS,
    koopman_orbit_method: OrbitMethod = DEFAULT_KOOPMAN_ORBIT_METHOD,
    koopman_symmetry: str | None = DEFAULT_KOOPMAN_SYMMETRY,
    num_relations: int | None = None,
    relation_normalization: str | None = None,
    node_types: Sequence[str] | None = None,
    edge_types: Sequence[Sequence[str]] | None = None,
    relation_tying: str = "independent",
    basis_size: int | None = None,
    latent_dims: Mapping[str, int] | None = None,
    koopman_num_modes: int = DEFAULT_KOOPMAN_NUM_MODES,
    koopman_parameter_dim: int = DEFAULT_KOOPMAN_PARAMETER_DIM,
    koopman_weight_kind: WeightKind = DEFAULT_KOOPMAN_WEIGHT_KIND,
) -> tuple[KoopmanOperatorContract, KoopmanKind]:
    """Construct or validate the model Koopman operator.

    Parameters
    ----------

    koopman : KoopmanOperatorContract, string kind, or None
        Factory kind or injected operator module.
    latent_dim : int
        Latent dimension per node.
    control_dim : int
        Control input dimension.
    control_mode : {"additive", "bilinear"}
        Control entry mode.
    bilinear_rank : int or None
        Optional low-rank bilinear size.
    dynamics_mode : {"discrete", "continuous"}
        Latent evolution mode.
    koopman_init_mode : InitMode
        Built-in operator initialization mode.
    koopman_init_scale : float
        Initialization scale for noisy identity mode.
    koopman_parameterization : Parameterization
        Built-in operator parameterization string.
    koopman_max_spectral_radius : float
        Spectral / real-part bound for structural modes.
    koopman_auxiliary_hidden_dims : sequence of int or None
        Auxiliary network widths for continuous ``auxiliary_spectral``.
    koopman_local_window, koopman_local_rank, koopman_local_hidden_dims
        Local-network config for ``koopman="global_local"``.
    koopman_sparsity : str
        Networked sparsity mode (``dense`` / ``block_diagonal`` /
        ``distributed``). ``distributed`` is matrix-free operator math — **not**
        trainer DDP / multi-GPU training (see :func:`build_koopman_model`).
    koopman_adjacency : GraphAdjacency
        Neighbor-coupling normalization for networked graph operators.
    koopman_filter_degree : int
        Monomial hop degree for discrete ``koopman="graph"``. Default ``1``.
    koopman_orbit_partition : Sequence[Sequence[int]] | None
        See the function signature / summary for ``koopman_orbit_partition``.
    koopman_auto_orbits : bool
        See the function signature / summary for ``koopman_auto_orbits``.
    koopman_orbit_method : OrbitMethod
        See the function signature / summary for ``koopman_orbit_method``.
    num_relations : int or None, optional
        Relation count for ``koopman="hetero_graph"`` (resolved from RelGraph
        peers by :func:`resolve_model_components`).
    relation_normalization : str or None, optional
        Per-relation normalization forwarded to
        :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`
        (defaults to ``"rgcn_in_degree"``).
    node_types : sequence of str or None, optional
        Ordered node-type names for ``koopman="hetero_graph"``.
    edge_types : sequence of sequence of str or None, optional
        Ordered ``(src, rel, dst)`` triples for ``koopman="hetero_graph"``.
    relation_tying : {"independent", "basis"}, optional
        Relation-factor tying for ``HeteroGraphKoopmanOperator``.
    basis_size : int or None, optional
        Basis size when ``relation_tying="basis"``.
    latent_dims : mapping of str to int or None, optional
        Opt-in per-type widths for discrete or continuous
        ``koopman="hetero_graph"``.
    koopman_num_modes : int, optional
        Interpolant mode count for ``koopman="parametric"``.
    koopman_parameter_dim : int, optional
        Regime-coordinate width for ``koopman="parametric"``.
    koopman_weight_kind : {"rbf", "simplex"}, optional
        Interpolant weights for ``koopman="parametric"``.
    koopman_hypergraph_incidence_mode
        See signature.

    koopman_symmetry
        See signature.
    Returns
    -------

    tuple[KoopmanOperatorContract, KoopmanKind]
        Operator module and resolved kind.

    Raises
    ------

    ValueError
        If factory kwargs conflict with injection, kinds, or dynamics mode.
    TypeError
        If ``koopman`` is neither a string kind nor a contract module."""
    kind, injected = parse_koopman_arg(koopman)
    resolved_aux_dims: tuple[int, ...] | None
    if koopman_auxiliary_hidden_dims is None:
        resolved_aux_dims = None
    else:
        resolved_aux_dims = normalize_auxiliary_hidden_dims(
            koopman_auxiliary_hidden_dims
        )
    resolved_local_dims = normalize_local_hidden_dims(koopman_local_hidden_dims)

    if (
        kind not in {"graph", "hypergraph", "hetero_graph", "continuous_graph"}
        and koopman_sparsity != "dense"
    ):
        msg = (
            "koopman_sparsity is only meaningful for koopman='graph', "
            "'hypergraph', 'hetero_graph', or 'continuous_graph'; got "
            f"sparsity={koopman_sparsity!r} with koopman={kind!r}"
        )
        raise ValueError(msg)

    if koopman_adjacency not in _GRAPH_ADJACENCY_MODES:
        accepted = ", ".join(sorted(_GRAPH_ADJACENCY_MODES))
        msg = (
            "koopman_adjacency must be one of "
            f"{{{accepted}}}, got {koopman_adjacency!r}"
        )
        raise ValueError(msg)
    filter_degree = validate_filter_degree(koopman_filter_degree)
    discrete_graph = (
        injected is None
        and kind == "graph"
        and dynamics_mode in {"discrete", "stochastic"}
    )
    if (
        injected is None
        and not discrete_graph
        and filter_degree != DEFAULT_KOOPMAN_FILTER_DEGREE
    ):
        msg = (
            "koopman_filter_degree is only meaningful for discrete "
            f"koopman='graph'; got filter_degree={filter_degree!r} with "
            f"koopman={kind!r} and dynamics_mode={dynamics_mode!r}"
        )
        raise ValueError(msg)
    if koopman_hypergraph_incidence_mode not in _HYPERGRAPH_INCIDENCE_MODES:
        accepted = ", ".join(sorted(_HYPERGRAPH_INCIDENCE_MODES))
        msg = (
            "koopman_hypergraph_incidence_mode must be one of "
            f"{{{accepted}}}, got {koopman_hypergraph_incidence_mode!r}"
        )
        raise ValueError(msg)
    # Injection uses kind="pernode" as a placeholder; non-default adjacency is
    # rejected later as a conflicting factory kwarg.
    if (
        injected is None
        and kind not in _NETWORKED_ADJACENCY_KINDS
        and koopman_adjacency != DEFAULT_KOOPMAN_ADJACENCY
    ):
        msg = (
            "koopman_adjacency is only meaningful for koopman='graph' or "
            f"'continuous_graph'; got adjacency={koopman_adjacency!r} with "
            f"koopman={kind!r}"
        )
        raise ValueError(msg)
    if (
        injected is None
        and kind != "hypergraph"
        and koopman_hypergraph_incidence_mode
        != DEFAULT_KOOPMAN_HYPERGRAPH_INCIDENCE_MODE
    ):
        msg = (
            "koopman_hypergraph_incidence_mode is only meaningful for "
            f"koopman='hypergraph'; got "
            f"incidence_mode={koopman_hypergraph_incidence_mode!r} with "
            f"koopman={kind!r}"
        )
        raise ValueError(msg)

    # Injection conflicts for koopman_symmetry are checked in
    # resolve_injected_koopman; skip kind-specific isotypic validation here.
    isotypic_symmetry = (
        False
        if injected is not None
        else _resolve_isotypic_symmetry(
            koopman_symmetry,
            kind=kind,
            dynamics_mode=dynamics_mode,
            koopman_orbit_partition=koopman_orbit_partition,
            koopman_auto_orbits=koopman_auto_orbits,
            koopman_orbit_method=koopman_orbit_method,
            koopman_latent_dims=latent_dims,
        )
    )
    symmetry_requested = (
        koopman_orbit_partition is not None or koopman_auto_orbits or isotypic_symmetry
    )
    if symmetry_requested and kind not in {"graph", "hypergraph", "hetero_graph"}:
        msg = (
            "koopman_orbit_partition / koopman_auto_orbits / "
            "koopman_symmetry require koopman='graph', 'hypergraph', or "
            f"multiplex 'hetero_graph', got koopman={kind!r}"
        )
        raise ValueError(msg)
    if symmetry_requested and kind == "hetero_graph" and dynamics_mode == "continuous":
        msg = (
            "koopman_orbit_partition / koopman_auto_orbits are unsupported "
            "for continuous hetero_graph operators"
        )
        raise ValueError(msg)
    if koopman_orbit_method not in {"auto", "exact"}:
        msg = (
            "koopman_orbit_method must be 'auto' or 'exact', "
            f"got {koopman_orbit_method!r}"
        )
        raise ValueError(msg)

    _reject_local_kwargs_unless_global_local(
        kind
        if injected is None
        else (
            "global_local"
            if isinstance(injected, GlobalLocalKoopmanOperator)
            else "pernode"
        ),
        koopman_local_window=koopman_local_window,
        koopman_local_rank=koopman_local_rank,
        koopman_local_hidden_dims=koopman_local_hidden_dims,
    )
    _reject_parametric_kwargs_unless_parametric(
        kind
        if injected is None
        else (
            "parametric"
            if isinstance(injected, ParametricKoopmanOperator)
            else "pernode"
        ),
        koopman_num_modes=koopman_num_modes,
        koopman_parameter_dim=koopman_parameter_dim,
        koopman_weight_kind=koopman_weight_kind,
    )

    if injected is not None:
        # Injection path: local kwargs must stay default (checked above for
        # non-global_local injections via kind fallback).
        if not isinstance(injected, GlobalLocalKoopmanOperator):
            _reject_local_kwargs_unless_global_local(
                "pernode",
                koopman_local_window=koopman_local_window,
                koopman_local_rank=koopman_local_rank,
                koopman_local_hidden_dims=koopman_local_hidden_dims,
            )
        operator = resolve_injected_koopman(
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
            koopman_auxiliary_hidden_dims=resolved_aux_dims,
            koopman_local_window=koopman_local_window,
            koopman_local_rank=koopman_local_rank,
            koopman_local_hidden_dims=(
                None if koopman_local_hidden_dims is None else resolved_local_dims
            ),
            koopman_orbit_partition=koopman_orbit_partition,
            koopman_auto_orbits=koopman_auto_orbits,
            koopman_orbit_method=koopman_orbit_method,
            koopman_symmetry=koopman_symmetry,
            koopman_adjacency=koopman_adjacency,
            koopman_filter_degree=koopman_filter_degree,
            koopman_hypergraph_incidence_mode=koopman_hypergraph_incidence_mode,
            koopman_num_modes=koopman_num_modes,
            koopman_parameter_dim=koopman_parameter_dim,
            koopman_weight_kind=koopman_weight_kind,
        )
        if isinstance(operator, ContinuousGraphKoopmanOperator):
            resolved_kind: KoopmanKind = "continuous_graph"
        elif isinstance(operator, HeteroKoopmanOperator):
            resolved_kind = "hetero_graph"
        elif isinstance(operator, HypergraphKoopmanOperator):
            resolved_kind = "hypergraph"
        elif isinstance(operator, HodgeKoopmanOperator):
            resolved_kind = "hodge"
        elif isinstance(operator, GraphKoopmanOperator):
            resolved_kind = "graph"
        elif isinstance(operator, GlobalLocalKoopmanOperator):
            resolved_kind = "global_local"
        elif isinstance(operator, SwitchedKoopmanOperator):
            resolved_kind = "switched"
        elif isinstance(operator, MixtureKoopmanOperator):
            resolved_kind = "mixture"
        elif isinstance(operator, ParametricKoopmanOperator):
            resolved_kind = "parametric"
        else:
            resolved_kind = "pernode"
        return operator, resolved_kind

    if dynamics_mode == "stochastic":
        _reject_stochastic_kind(
            kind,
            latent_dim=latent_dim,
            latent_dims=latent_dims,
        )

    if dynamics_mode == "continuous":
        if kind in {
            "hypergraph",
            "global_local",
            "switched",
            "mixture",
            "parametric",
            "hodge",
        }:
            msg = (
                f"koopman={kind!r} requires dynamics_mode='discrete'; "
                "continuous hypergraph / global_local / switched / mixture / "
                "parametric / hodge operators are not implemented"
            )
            raise ValueError(msg)
        if kind == "hetero_graph":
            if resolved_aux_dims is not None:
                msg = (
                    "koopman_auxiliary_hidden_dims is not supported for "
                    "ContinuousHeteroGraphKoopmanOperator"
                )
                raise ValueError(msg)
            if num_relations is None:
                msg = (
                    "koopman='hetero_graph' requires RelGraphEncoder peers so "
                    "num_relations can be resolved"
                )
                raise ValueError(msg)
            normalization = (
                "rgcn_in_degree"
                if relation_normalization is None
                else relation_normalization
            )
            return (
                ContinuousHeteroGraphKoopmanOperator(
                    latent_dim,
                    num_relations,
                    init_mode=koopman_init_mode,
                    init_scale=koopman_init_scale,
                    parameterization=koopman_parameterization,
                    max_real_eigenvalue=koopman_max_spectral_radius,
                    control_dim=control_dim,
                    control_mode=control_mode,
                    bilinear_rank=bilinear_rank,
                    sparsity=koopman_sparsity,  # type: ignore[arg-type]
                    normalization=normalization,  # type: ignore[arg-type]
                    node_types=node_types,
                    edge_types=edge_types,
                    relation_tying=relation_tying,  # type: ignore[arg-type]
                    basis_size=basis_size,
                    latent_dims=latent_dims,
                ),
                "hetero_graph",
            )
        if kind in {"graph", "continuous_graph"}:
            if (
                resolved_aux_dims is not None
                and koopman_parameterization != "auxiliary_spectral"
            ):
                msg = (
                    "koopman_auxiliary_hidden_dims requires "
                    "koopman_parameterization='auxiliary_spectral'"
                )
                raise ValueError(msg)
            if resolved_aux_dims is not None:
                msg = (
                    "koopman_auxiliary_hidden_dims is not supported for "
                    "ContinuousGraphKoopmanOperator"
                )
                raise ValueError(msg)
            return (
                ContinuousGraphKoopmanOperator(
                    latent_dim,
                    init_mode=koopman_init_mode,
                    init_scale=koopman_init_scale,
                    parameterization=koopman_parameterization,
                    max_real_eigenvalue=koopman_max_spectral_radius,
                    control_dim=control_dim,
                    control_mode=control_mode,
                    bilinear_rank=bilinear_rank,
                    sparsity=koopman_sparsity,  # type: ignore[arg-type]
                    adjacency=koopman_adjacency,
                ),
                "continuous_graph",
            )
        if (
            resolved_aux_dims is not None
            and koopman_parameterization != "auxiliary_spectral"
        ):
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return (
            ContinuousKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_real_eigenvalue=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                auxiliary_hidden_dims=resolved_aux_dims,
            ),
            "pernode",
        )

    if kind == "continuous_graph":
        msg = "koopman='continuous_graph' requires dynamics_mode='continuous'"
        raise ValueError(msg)

    if kind == "graph":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            GraphKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                sparsity=koopman_sparsity,  # type: ignore[arg-type]
                adjacency=koopman_adjacency,
                filter_degree=filter_degree,
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
                isotypic_symmetry=isotypic_symmetry,
            ),
            "graph",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "hypergraph":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            HypergraphKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                sparsity=koopman_sparsity,  # type: ignore[arg-type]
                incidence_mode=koopman_hypergraph_incidence_mode,  # type: ignore[arg-type]
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
            ),
            "hypergraph",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "hetero_graph":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        if num_relations is None:
            msg = (
                "koopman='hetero_graph' requires RelGraphEncoder peers so "
                "num_relations can be resolved"
            )
            raise ValueError(msg)
        normalization = (
            "rgcn_in_degree"
            if relation_normalization is None
            else relation_normalization
        )
        return _finalize_built_koopman(
            HeteroGraphKoopmanOperator(
                latent_dim,
                num_relations,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                sparsity=koopman_sparsity,  # type: ignore[arg-type]
                normalization=normalization,  # type: ignore[arg-type]
                node_types=node_types,
                edge_types=edge_types,
                relation_tying=relation_tying,  # type: ignore[arg-type]
                basis_size=basis_size,
                latent_dims=latent_dims,
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
            ),
            "hetero_graph",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "switched":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            SwitchedKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            ),
            "switched",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "mixture":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            MixtureKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                local_window=koopman_local_window,
            ),
            "mixture",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "parametric":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            ParametricKoopmanOperator(
                latent_dim,
                num_modes=koopman_num_modes,
                parameter_dim=koopman_parameter_dim,
                weight_kind=koopman_weight_kind,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            ),
            "parametric",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "hodge":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            HodgeKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                sparsity=koopman_sparsity,  # type: ignore[arg-type]
                adjacency=koopman_adjacency,
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
                isotypic_symmetry=isotypic_symmetry,
            ),
            "hodge",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if kind == "global_local":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return _finalize_built_koopman(
            GlobalLocalKoopmanOperator(
                latent_dim,
                init_mode=koopman_init_mode,
                init_scale=koopman_init_scale,
                parameterization=koopman_parameterization,
                max_spectral_radius=koopman_max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
                local_window=koopman_local_window,
                local_rank=koopman_local_rank,
                local_hidden_dims=resolved_local_dims,
            ),
            "global_local",
            dynamics_mode=dynamics_mode,
            latent_dim=latent_dim,
        )

    if resolved_aux_dims is not None:
        msg = (
            "koopman_auxiliary_hidden_dims requires "
            "dynamics_mode='continuous' and "
            "koopman_parameterization='auxiliary_spectral'"
        )
        raise ValueError(msg)
    return _finalize_built_koopman(
        KoopmanOperator(
            latent_dim,
            init_mode=koopman_init_mode,
            init_scale=koopman_init_scale,
            parameterization=koopman_parameterization,
            max_spectral_radius=koopman_max_spectral_radius,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
        ),
        "pernode",
        dynamics_mode=dynamics_mode,
        latent_dim=latent_dim,
    )
