"""Koopman operator factory and injection validation for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer constructing operators
through :class:`~koopman_graph.model.GraphKoopmanModel`; these helpers exist so
the estimator stays orchestration-focused without cross-module private imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

from koopman_graph.graph_utils.symmetry import OrbitMethod
from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    SAGEDecoder,
    SAGEEncoder,
)
from koopman_graph.nn.delay import resolve_delay_encoder
from koopman_graph.observables import (
    PhysicsLiftingFn,
    PhysicsPosition,
    resolve_physics_lifting_fn,
    resolve_physics_position,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HypergraphKoopmanOperator,
    InitMode,
    KoopmanKind,
    KoopmanOperator,
    KoopmanOperatorContract,
    Parameterization,
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
from koopman_graph.protocols import DynamicsMode

Encoder = (
    GNNEncoder
    | GATEncoder
    | SAGEEncoder
    | DiffConvEncoder
    | HypergraphEncoder
    | DelayEmbeddingEncoder
)
Decoder = GNNDecoder | GATDecoder | SAGEDecoder | DiffConvDecoder | HypergraphDecoder
KoopmanModule = (
    KoopmanOperator
    | ContinuousKoopmanOperator
    | GraphKoopmanOperator
    | HypergraphKoopmanOperator
    | GlobalLocalKoopmanOperator
    | ContinuousGraphKoopmanOperator
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
    koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
    koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
    koopman_local_hidden_dims: Sequence[int] | None = None,
    koopman_orbit_partition: Sequence[Sequence[int]] | None = (
        DEFAULT_KOOPMAN_ORBIT_PARTITION
    ),
    koopman_auto_orbits: bool = DEFAULT_KOOPMAN_AUTO_ORBITS,
    koopman_orbit_method: OrbitMethod = DEFAULT_KOOPMAN_ORBIT_METHOD,
    control_dim: int = 0,
    control_mode: ControlMode = DEFAULT_CONTROL_MODE,
    bilinear_rank: int | None = DEFAULT_BILINEAR_RANK,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
    physics_preset: str | None = None,
    physics_dim: int = 0,
    physics_position: PhysicsPosition,
    n_delays: int = 1,
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
        Graph / hypergraph / continuous-graph sparsity mode
        (``dense`` / ``block_diagonal``).
    koopman_local_window, koopman_local_rank, koopman_local_hidden_dims
        Global/local operator hyperparameters.
    koopman_orbit_partition, koopman_auto_orbits, koopman_orbit_method
        Symmetry / orbit-tying configuration.
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
    if dynamics_mode not in {"discrete", "continuous"}:
        msg = f"dynamics_mode must be 'discrete' or 'continuous', got {dynamics_mode!r}"
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
        koopman_local_window=koopman_local_window,
        koopman_local_rank=koopman_local_rank,
        koopman_local_hidden_dims=koopman_local_hidden_dims,
        koopman_orbit_partition=koopman_orbit_partition,
        koopman_auto_orbits=koopman_auto_orbits,
        koopman_orbit_method=koopman_orbit_method,
    )
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
            "global_local",
            "continuous_graph",
        }:
            msg = (
                "koopman string kind must be 'pernode', 'graph', "
                "'hypergraph', 'global_local', or 'continuous_graph', "
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
    if isinstance(koopman, KoopmanOperator) and dynamics_mode != "discrete":
        msg = "Injected KoopmanOperator requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if isinstance(koopman, GraphKoopmanOperator) and dynamics_mode != "discrete":
        msg = "Injected GraphKoopmanOperator requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if isinstance(koopman, HypergraphKoopmanOperator) and dynamics_mode != "discrete":
        msg = "Injected HypergraphKoopmanOperator requires dynamics_mode='discrete'"
        raise ValueError(msg)
    if isinstance(koopman, GlobalLocalKoopmanOperator) and dynamics_mode != "discrete":
        msg = "Injected GlobalLocalKoopmanOperator requires dynamics_mode='discrete'"
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

    return koopman


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
    koopman_local_window: int = DEFAULT_KOOPMAN_LOCAL_WINDOW,
    koopman_local_rank: int = DEFAULT_KOOPMAN_LOCAL_RANK,
    koopman_local_hidden_dims: Sequence[int] | None = None,
    koopman_orbit_partition: Sequence[Sequence[int]] | None = (
        DEFAULT_KOOPMAN_ORBIT_PARTITION
    ),
    koopman_auto_orbits: bool = DEFAULT_KOOPMAN_AUTO_ORBITS,
    koopman_orbit_method: OrbitMethod = DEFAULT_KOOPMAN_ORBIT_METHOD,
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
        See the function signature / summary for ``koopman_sparsity``.
    koopman_orbit_partition : Sequence[Sequence[int]] | None
        See the function signature / summary for ``koopman_orbit_partition``.
    koopman_auto_orbits : bool
        See the function signature / summary for ``koopman_auto_orbits``.
    koopman_orbit_method : OrbitMethod
        See the function signature / summary for ``koopman_orbit_method``.

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
        kind not in {"graph", "hypergraph", "continuous_graph"}
        and koopman_sparsity != "dense"
    ):
        msg = (
            "koopman_sparsity is only meaningful for koopman='graph', "
            f"'hypergraph', or 'continuous_graph'; got sparsity="
            f"{koopman_sparsity!r} with koopman={kind!r}"
        )
        raise ValueError(msg)

    symmetry_requested = koopman_orbit_partition is not None or koopman_auto_orbits
    if symmetry_requested and kind not in {"graph", "hypergraph"}:
        msg = (
            "koopman_orbit_partition / koopman_auto_orbits require "
            f"koopman='graph' or 'hypergraph', got koopman={kind!r}"
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
        )
        if isinstance(operator, ContinuousGraphKoopmanOperator):
            resolved_kind: KoopmanKind = "continuous_graph"
        elif isinstance(operator, HypergraphKoopmanOperator):
            resolved_kind = "hypergraph"
        elif isinstance(operator, GraphKoopmanOperator):
            resolved_kind = "graph"
        elif isinstance(operator, GlobalLocalKoopmanOperator):
            resolved_kind = "global_local"
        else:
            resolved_kind = "pernode"
        return operator, resolved_kind

    if dynamics_mode == "continuous":
        if kind in {"hypergraph", "global_local"}:
            msg = (
                f"koopman={kind!r} requires dynamics_mode='discrete'; "
                "continuous hypergraph / global_local operators are not "
                "implemented"
            )
            raise ValueError(msg)
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
        return (
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
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
            ),
            "graph",
        )

    if kind == "hypergraph":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return (
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
                orbit_partition=koopman_orbit_partition,
                auto_orbits=koopman_auto_orbits,
                orbit_method=koopman_orbit_method,
            ),
            "hypergraph",
        )

    if kind == "global_local":
        if resolved_aux_dims is not None:
            msg = (
                "koopman_auxiliary_hidden_dims requires "
                "dynamics_mode='continuous' and "
                "koopman_parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)
        return (
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
        )

    if resolved_aux_dims is not None:
        msg = (
            "koopman_auxiliary_hidden_dims requires "
            "dynamics_mode='continuous' and "
            "koopman_parameterization='auxiliary_spectral'"
        )
        raise ValueError(msg)
    return (
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
    )
