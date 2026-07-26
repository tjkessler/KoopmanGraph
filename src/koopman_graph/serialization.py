"""Checkpoint serialization for :class:`~koopman_graph.model.GraphKoopmanModel`.

Checkpoint format versions
--------------------------
``format_version`` 1 (current baseline; beta through 0.x)
    Full architecture config for discrete and continuous dynamics, hybrid
    physics observables, control (including bilinear metadata), delay
    embeddings, and built-in operator kinds (per-node / graph / hypergraph /
    global_local / continuous_graph). Encoder/decoder ``type`` strings include
    ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
    ``"hyper_enc"``, and ``"hyper_dec"``; missing decoder ``type`` defaults to
    ``"gcn"``. Hybrid ``physics`` blocks own ``dim``, ``preset``, and
    ``position``; ``position`` is round-tripped and validated on load
    (currently only ``"prepend"``). Missing ``position`` defaults to
    ``"prepend"``. Optional ``n_delays`` records Hankel delay embedding; the
    stored encoder block is always the base encoder config with
    ``in_channels = n_delays * feature_dim``. Format-1 also stores placeholder
    keys ``sparsity`` (operator realization; ``"dense"`` or
    ``"block_diagonal"`` for supported networked kinds),
    ``adjacency`` (``"symmetric"`` / ``"random_walk"`` /
    ``"dual_random_walk"`` for graph / continuous-graph operators, else
    ``None``), ``learn_topology`` (``None`` or ``"self_adaptive"``) with
    ``topology_embedding_dim``, and ``symmetry`` (``None`` or a dict with
    ``auto_orbits``, ``orbit_partition``, and ``method`` for orbit-tied
    graph / hypergraph operators).

Beta policy
    While the package is pre-1.0, ``FORMAT_VERSION`` stays at ``1``. Incomplete
    or previously published incompatible checkpoints are **deprecated** and
    rejected with a clear re-save error rather than migrated. Formal
    multi-version checkpoint tracking begins at 1.0. Loaders accept only
    ``{1}``; ``format_version`` 2 and other lineages remain unsupported.

Custom injected operators (anything other than the built-in serializable
operator classes registered in this module) are **not** round-trippable:
:func:`build_model_config` / :meth:`GraphKoopmanModel.save` raise rather than
silently writing incomplete factory metadata.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from torch import nn

from koopman_graph.nn import (
    DEFAULT_TOPOLOGY_EMBEDDING_DIM,
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphTransformerDecoder,
    GraphTransformerEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    SAGEDecoder,
    SAGEEncoder,
)
from koopman_graph.observables import (
    PHYSICS_POSITION,
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
    KoopmanOperator,
    resolve_factory_stability_bound,
)
from koopman_graph.protocols import ModeShapeModel

FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = frozenset({1})

# Keys always written by :func:`build_model_config` for the current format-1
# baseline. Sparse historical payloads that omit these are rejected on load.
_FORMAT_1_REQUIRED_KEYS = frozenset(
    {
        "latent_dim",
        "time_step",
        "dynamics_mode",
        "koopman_kind",
        "koopman_init_mode",
        "koopman_init_scale",
        "koopman_parameterization",
        "koopman_max_spectral_radius",
        "control_dim",
        "control_mode",
        "bilinear_rank",
        "n_delays",
        "physics",
        "encoder",
        "decoder",
        "sparsity",
        "adjacency",
        "learn_topology",
        "topology_embedding_dim",
        "symmetry",
        "local_window",
        "local_rank",
        "local_hidden_dims",
    }
)

Decoder = (
    GNNDecoder
    | GATDecoder
    | SAGEDecoder
    | DiffConvDecoder
    | GraphTransformerDecoder
    | HypergraphDecoder
)
BaseEncoder = (
    GNNEncoder
    | GATEncoder
    | SAGEEncoder
    | DiffConvEncoder
    | GraphTransformerEncoder
    | HypergraphEncoder
)
_SERIALIZABLE_KOOPMAN_TYPES = (
    KoopmanOperator,
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    HypergraphKoopmanOperator,
    GlobalLocalKoopmanOperator,
    ContinuousGraphKoopmanOperator,
)
_RESERVED_KOOPMAN_KINDS: dict[str, str] = {}


_NETWORKED_ADJACENCY_KINDS = frozenset({"graph", "continuous_graph"})
_GRAPH_ADJACENCY_MODES = frozenset({"symmetric", "random_walk", "dual_random_walk"})


def _require_format1_schema(config: dict[str, Any]) -> None:
    """Reject incomplete configs that lack the current format-1 schema keys.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a checkpoint.

    Raises
    ------
    ValueError
        If any required current-schema key is missing.
    """
    missing = sorted(_FORMAT_1_REQUIRED_KEYS - config.keys())
    if missing:
        msg = (
            "Deprecated checkpoint schema: missing required format_version 1 "
            f"fields: {', '.join(missing)}. Pre-1.0 checkpoints are not "
            "migrated; re-save the model with the current package or "
            "reconstruct the architecture explicitly."
        )
        raise ValueError(msg)


def _resolve_checkpoint_adjacency(
    adjacency: Any,
    *,
    koopman_kind: str,
) -> str:
    """Validate checkpoint ``adjacency`` and return a factory keyword value.

    Parameters
    ----------
    adjacency : Any
        Checkpoint ``config.adjacency`` value (``None`` for non-networked
        kinds, or a mode string for graph / continuous-graph).
    koopman_kind : str
        Serialized operator kind.

    Returns
    -------
    str
        Factory ``koopman_adjacency`` value (defaults to ``"symmetric"`` for
        non-networked kinds when ``adjacency`` is ``None``).

    Raises
    ------
    ValueError
        If a networked kind is missing / invalid ``adjacency``, or a
        non-networked kind supplies a non-null value.
    """
    accepted = ", ".join(sorted(_GRAPH_ADJACENCY_MODES))
    if koopman_kind in _NETWORKED_ADJACENCY_KINDS:
        if adjacency is None:
            msg = (
                "Checkpoint config.adjacency is required for networked "
                f"koopman_kind={koopman_kind!r}; accepted values: "
                f"{{{accepted}}}"
            )
            raise ValueError(msg)
        if adjacency not in _GRAPH_ADJACENCY_MODES:
            msg = (
                "Checkpoint config.adjacency must be one of "
                f"{{{accepted}}}, got {adjacency!r}"
            )
            raise ValueError(msg)
        return str(adjacency)
    if adjacency is not None:
        msg = (
            "Checkpoint config.adjacency must be null for non-networked "
            f"koopman_kind={koopman_kind!r}, got {adjacency!r}"
        )
        raise ValueError(msg)
    return "symmetric"


def _migrate_config(config: dict[str, Any], *, format_version: int) -> dict[str, Any]:
    """Validate checkpoint config before reconstruct (beta: no migrations).

    Format 1 is the only supported baseline through 0.x. Incomplete schemas are
    rejected as deprecated rather than backfilled. Multi-version migration
    branches are deferred until the 1.0 checkpoint policy.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a saved checkpoint.
    format_version : int
        Checkpoint ``format_version`` after supported-version validation.

    Returns
    -------
    dict
        Config ready for :func:`reconstruct_model`.

    Raises
    ------
    ValueError
        If the format version is unsupported or the config fails schema
        validation for the active version.
    """
    if format_version == 1:
        _require_format1_schema(config)
        return config

    msg = (
        f"No migration path for checkpoint format_version {format_version}; "
        f"supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
    )
    raise ValueError(msg)


def _parse_symmetry_config(
    symmetry: Any,
) -> tuple[list[list[int]] | None, bool, str]:
    """Parse the format-1 ``symmetry`` config block into factory kwargs.

    Parameters
    ----------
    symmetry : Any
        ``None`` or a dict with ``auto_orbits``, ``orbit_partition``, and
        ``method``.

    Returns
    -------
    tuple
        ``(orbit_partition, auto_orbits, orbit_method)``.

    Raises
    ------
    ValueError
        If the block shape or field types are invalid.
    """
    if symmetry is None:
        return None, False, "auto"
    if not isinstance(symmetry, dict):
        msg = f"symmetry config must be a dict or None, got {type(symmetry).__name__}"
        raise ValueError(msg)
    auto_orbits = bool(symmetry.get("auto_orbits", False))
    method = symmetry.get("method", "auto")
    if method not in {"auto", "exact"}:
        msg = f"symmetry.method must be 'auto' or 'exact', got {method!r}"
        raise ValueError(msg)
    raw_partition = symmetry.get("orbit_partition")
    if raw_partition is None:
        return None, auto_orbits, method
    if not isinstance(raw_partition, (list, tuple)):
        msg = "symmetry.orbit_partition must be a sequence of orbits or None"
        raise ValueError(msg)
    partition: list[list[int]] = []
    for orbit in raw_partition:
        if not isinstance(orbit, (list, tuple)):
            msg = "each symmetry.orbit_partition orbit must be a sequence of ints"
            raise ValueError(msg)
        partition.append([int(node) for node in orbit])
    return partition, auto_orbits, method


def _package_version() -> str:
    """Return the installed package version for checkpoint metadata.

    Returns
    -------
    str
        Installed ``koopman-graph`` version, or ``"0.0.0"`` when running from
        source without package metadata.
    """
    try:
        return version("koopman-graph")
    except PackageNotFoundError:
        return "0.0.0"


_SUPPORTED_ENCODER_TYPES: dict[str, type[BaseEncoder]] = {
    "gcn": GNNEncoder,
    "gat": GATEncoder,
    "sage": SAGEEncoder,
    "diffconv": DiffConvEncoder,
    "transformer": GraphTransformerEncoder,
    "hyper_enc": HypergraphEncoder,
}

_SUPPORTED_DECODER_TYPES: dict[str, type[Decoder]] = {
    "gcn": GNNDecoder,
    "gat": GATDecoder,
    "sage": SAGEDecoder,
    "diffconv": DiffConvDecoder,
    "transformer": GraphTransformerDecoder,
    "hyper_dec": HypergraphDecoder,
}


def _encoder_type(encoder: BaseEncoder) -> str:
    """Return the checkpoint encoder type string for an encoder instance.

    Parameters
    ----------
    encoder : GNNEncoder, GATEncoder, SAGEEncoder, DiffConvEncoder, or
        GraphTransformerEncoder
        Encoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
        or ``"hyper_enc"``.

    Raises
    ------
    TypeError
        If ``encoder`` is not a supported encoder class.
    """
    if isinstance(encoder, GraphTransformerEncoder):
        return "transformer"
    if isinstance(encoder, DiffConvEncoder):
        return "diffconv"
    if isinstance(encoder, SAGEEncoder):
        return "sage"
    if isinstance(encoder, GATEncoder):
        return "gat"
    if isinstance(encoder, GNNEncoder):
        return "gcn"
    if isinstance(encoder, HypergraphEncoder):
        return "hyper_enc"
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _unwrap_base_encoder(
    encoder: nn.Module,
) -> tuple[BaseEncoder, int]:
    """Return the serializable base encoder and delay count.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder, possibly wrapped in :class:`DelayEmbeddingEncoder`.

    Returns
    -------
    base_encoder : GNNEncoder, GATEncoder, SAGEEncoder, DiffConvEncoder, or
        GraphTransformerEncoder
        Checkpoint-rebuildable encoder.
    n_delays : int
        Delay window length (``1`` when unwrapped).
    """
    if isinstance(encoder, DelayEmbeddingEncoder):
        base = encoder.base_encoder
        if not isinstance(
            base,
            (
                GNNEncoder,
                GATEncoder,
                SAGEEncoder,
                DiffConvEncoder,
                GraphTransformerEncoder,
            ),
        ):
            msg = (
                "DelayEmbeddingEncoder.base_encoder must be GNNEncoder, "
                "GATEncoder, SAGEEncoder, DiffConvEncoder, or "
                "GraphTransformerEncoder for "
                f"checkpoints; got {type(base).__name__}"
            )
            raise TypeError(msg)
        return base, encoder.n_delays
    if isinstance(
        encoder,
        (
            GNNEncoder,
            GATEncoder,
            SAGEEncoder,
            DiffConvEncoder,
            GraphTransformerEncoder,
            HypergraphEncoder,
        ),
    ):
        return encoder, 1
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _decoder_type(decoder: Decoder) -> str:
    """Return the checkpoint decoder type string for a decoder instance.

    Parameters
    ----------
    decoder : GNNDecoder, GATDecoder, SAGEDecoder, DiffConvDecoder, or
        GraphTransformerDecoder
        Decoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
        or ``"hyper_dec"``.

    Raises
    ------
    TypeError
        If ``decoder`` is not a supported decoder class.
    """
    if isinstance(decoder, GraphTransformerDecoder):
        return "transformer"
    if isinstance(decoder, DiffConvDecoder):
        return "diffconv"
    if isinstance(decoder, SAGEDecoder):
        return "sage"
    if isinstance(decoder, GATDecoder):
        return "gat"
    if isinstance(decoder, GNNDecoder):
        return "gcn"
    if isinstance(decoder, HypergraphDecoder):
        return "hyper_dec"
    msg = f"Unsupported decoder type: {type(decoder).__name__}"
    raise TypeError(msg)


def _require_serializable_koopman(model: ModeShapeModel) -> None:
    """Reject custom injected operators that lack checkpoint factory metadata.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model whose ``koopman`` submodule will be serialized.

    Raises
    ------
    TypeError
        If ``model.koopman`` is not a built-in
        :class:`~koopman_graph.operators.KoopmanOperator`,
        :class:`~koopman_graph.operators.ContinuousKoopmanOperator`, or
        :class:`~koopman_graph.operators.GraphKoopmanOperator`.
    """
    if isinstance(model.koopman, _SERIALIZABLE_KOOPMAN_TYPES):
        return
    msg = (
        "Checkpoint serialization supports only built-in KoopmanOperator, "
        "ContinuousKoopmanOperator, GraphKoopmanOperator, "
        "HypergraphKoopmanOperator, GlobalLocalKoopmanOperator, and "
        "ContinuousGraphKoopmanOperator instances. Custom injected operators "
        "are not round-trippable; save the operator state separately or "
        "reconstruct the model with koopman=... after load. "
        f"Got {type(model.koopman).__name__}."
    )
    raise TypeError(msg)


def build_model_config(model: ModeShapeModel) -> dict[str, Any]:
    """Extract architecture configuration from a :class:`GraphKoopmanModel`.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose encoder, decoder, and Koopman settings will be serialized.

    Returns
    -------
    dict
        JSON-serializable architecture configuration.

    Raises
    ------
    TypeError
        If ``model.koopman`` is a custom injected operator (not a built-in
        :class:`~koopman_graph.operators.KoopmanOperator`,
        :class:`~koopman_graph.operators.ContinuousKoopmanOperator`, or
        :class:`~koopman_graph.operators.GraphKoopmanOperator`).
    """
    _require_serializable_koopman(model)
    encoder, n_delays = _unwrap_base_encoder(model.encoder)
    decoder = model.decoder
    encoder_config: dict[str, Any] = {
        "type": _encoder_type(encoder),
        "in_channels": encoder.in_channels,
        "hidden_channels": encoder.hidden_channels,
        "latent_dim": encoder.latent_dim,
        "num_layers": encoder.num_layers,
        "activation": encoder.activation_name,
    }
    if isinstance(encoder, (GATEncoder, GraphTransformerEncoder)):
        encoder_config["heads"] = encoder.heads
        encoder_config["dropout"] = encoder.dropout
    if isinstance(encoder, GraphTransformerEncoder):
        encoder_config["edge_dim"] = encoder.edge_dim
    if isinstance(encoder, DiffConvEncoder):
        encoder_config["diffusion_steps"] = encoder.diffusion_steps

    decoder_config: dict[str, Any] = {
        "type": _decoder_type(decoder),
        "latent_dim": decoder.latent_dim,
        "hidden_channels": decoder.hidden_channels,
        "out_channels": decoder.out_channels,
        "num_layers": decoder.num_layers,
        "activation": decoder.activation_name,
    }
    if isinstance(decoder, (GATDecoder, GraphTransformerDecoder)):
        decoder_config["heads"] = decoder.heads
        decoder_config["dropout"] = decoder.dropout
    if isinstance(decoder, GraphTransformerDecoder):
        decoder_config["edge_dim"] = decoder.edge_dim
    if isinstance(decoder, DiffConvDecoder):
        decoder_config["diffusion_steps"] = decoder.diffusion_steps

    physics_config: dict[str, Any] | None = None
    if model.physics_dim > 0:
        physics_config = {
            "dim": model.physics_dim,
            "preset": model.physics_preset,
            "position": model.physics_position,
        }

    sparsity = getattr(model.koopman, "sparsity", "dense")
    adjacency = getattr(model.koopman, "adjacency", None)
    return {
        "latent_dim": model.latent_dim,
        "time_step": model.time_step,
        "dynamics_mode": model.dynamics_mode,
        "koopman_kind": getattr(model, "koopman_kind", "pernode"),
        "koopman_init_mode": model.koopman.init_mode,
        "koopman_init_scale": model.koopman.init_scale,
        "koopman_parameterization": model.koopman.parameterization,
        "koopman_max_spectral_radius": resolve_factory_stability_bound(
            model.koopman,
            dynamics_mode=model.dynamics_mode,
        ),
        "koopman_auxiliary_hidden_dims": (
            list(model.koopman.auxiliary_hidden_dims)
            if isinstance(model.koopman, ContinuousKoopmanOperator)
            and model.koopman.parameterization == "auxiliary_spectral"
            else None
        ),
        "local_window": (
            int(model.koopman.local_window)
            if isinstance(model.koopman, GlobalLocalKoopmanOperator)
            else None
        ),
        "local_rank": (
            int(model.koopman.local_rank)
            if isinstance(model.koopman, GlobalLocalKoopmanOperator)
            else None
        ),
        "local_hidden_dims": (
            list(model.koopman.local_hidden_dims)
            if isinstance(model.koopman, GlobalLocalKoopmanOperator)
            else None
        ),
        "control_dim": model.control_dim,
        "control_mode": getattr(model, "control_mode", "additive"),
        "bilinear_rank": getattr(model, "bilinear_rank", None),
        "n_delays": n_delays,
        "physics": physics_config,
        "encoder": encoder_config,
        "decoder": decoder_config,
        "sparsity": sparsity,
        "adjacency": adjacency,
        "learn_topology": getattr(model, "learn_topology", None),
        "topology_embedding_dim": (
            int(model.topology_embedding_dim)
            if getattr(model, "learn_topology", None) is not None
            else None
        ),
        "symmetry": (
            model.koopman.symmetry_config()
            if hasattr(model.koopman, "symmetry_config")
            else None
        ),
    }


def _build_encoder(config: dict[str, Any]) -> BaseEncoder:
    """Instantiate an encoder from a checkpoint configuration block.

    Parameters
    ----------
    config : dict
        Encoder configuration block from a saved checkpoint.

    Returns
    -------
    GNNEncoder, GATEncoder, SAGEEncoder, DiffConvEncoder, or
        GraphTransformerEncoder
        Reconstructed encoder matching the saved architecture.

    Raises
    ------
    ValueError
        If the encoder ``type`` field is unsupported.
    """
    encoder_type = config["type"]
    encoder_cls = _SUPPORTED_ENCODER_TYPES.get(encoder_type)
    if encoder_cls is None:
        msg = f"Unsupported encoder type in checkpoint: {encoder_type!r}"
        raise ValueError(msg)

    common_kwargs = {
        "in_channels": config["in_channels"],
        "hidden_channels": config["hidden_channels"],
        "latent_dim": config["latent_dim"],
        "num_layers": config["num_layers"],
        "activation": config["activation"],
    }
    if encoder_type == "gat":
        return GATEncoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
        )
    if encoder_type == "transformer":
        return GraphTransformerEncoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
            edge_dim=config.get("edge_dim"),
        )
    if encoder_type == "sage":
        return SAGEEncoder(**common_kwargs)
    if encoder_type == "diffconv":
        return DiffConvEncoder(
            **common_kwargs,
            diffusion_steps=config.get("diffusion_steps", 2),
        )
    if encoder_type == "hyper_enc":
        return HypergraphEncoder(**common_kwargs)
    return GNNEncoder(**common_kwargs)


def _build_decoder(config: dict[str, Any]) -> Decoder:
    """Instantiate a decoder from a checkpoint configuration block.

    Parameters
    ----------
    config : dict
        Decoder configuration block from a saved checkpoint. Missing ``type``
        defaults to ``"gcn"`` for checkpoints written before GAT decoder
        support.

    Returns
    -------
    GNNDecoder, GATDecoder, SAGEDecoder, DiffConvDecoder, or
        GraphTransformerDecoder
        Reconstructed decoder matching the saved architecture.

    Raises
    ------
    ValueError
        If the decoder ``type`` field is unsupported.
    """
    decoder_type = config.get("type", "gcn")
    decoder_cls = _SUPPORTED_DECODER_TYPES.get(decoder_type)
    if decoder_cls is None:
        msg = f"Unsupported decoder type in checkpoint: {decoder_type!r}"
        raise ValueError(msg)

    common_kwargs = {
        "latent_dim": config["latent_dim"],
        "hidden_channels": config["hidden_channels"],
        "out_channels": config["out_channels"],
        "num_layers": config["num_layers"],
        "activation": config["activation"],
    }
    if decoder_type == "gat":
        return GATDecoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
        )
    if decoder_type == "transformer":
        return GraphTransformerDecoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
            edge_dim=config.get("edge_dim"),
        )
    if decoder_type == "sage":
        return SAGEDecoder(**common_kwargs)
    if decoder_type == "diffconv":
        return DiffConvDecoder(
            **common_kwargs,
            diffusion_steps=config.get("diffusion_steps", 2),
        )
    if decoder_type == "hyper_dec":
        return HypergraphDecoder(**common_kwargs)
    return GNNDecoder(**common_kwargs)


def reconstruct_model(
    config: dict[str, Any],
    *,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> ModeShapeModel:
    """Reconstruct a :class:`GraphKoopmanModel` from a checkpoint configuration.

    Parameters
    ----------
    config : dict
        Architecture configuration produced by :func:`build_model_config`.
    physics_lifting_fn : callable or None, optional
        Custom physics lifting function for hybrid checkpoints that do not store
        a registered preset.

    Returns
    -------
    GraphKoopmanModel
        Uninitialized-weight model matching the saved architecture.

    Raises
    ------
    ValueError
        If a hybrid checkpoint requires a physics lifting function that is not
        provided and cannot be resolved from a preset, or if
        ``physics.position`` is unsupported.
    """
    import importlib

    estimator_mod = importlib.import_module("koopman_graph.model.estimator")
    GraphKoopmanModel = estimator_mod.GraphKoopmanModel

    decoder = _build_decoder(config["decoder"])
    encoder = _build_encoder(config["encoder"])

    physics_config = config.get("physics")
    physics_dim = 0
    physics_preset: str | None = None
    physics_position: PhysicsPosition = PHYSICS_POSITION
    resolved_physics_fn: PhysicsLiftingFn | None = None
    if isinstance(physics_config, dict):
        physics_dim = int(physics_config.get("dim", 0))
        physics_preset = physics_config.get("preset")
        if physics_dim > 0:
            physics_position = resolve_physics_position(physics_config.get("position"))
            resolved_physics_fn = resolve_physics_lifting_fn(
                physics_preset=physics_preset,
                physics_lifting_fn=physics_lifting_fn,
            )
            if resolved_physics_fn is None:
                msg = (
                    "Checkpoint uses hybrid physics observables but no preset is "
                    "stored; pass physics_lifting_fn to load_checkpoint"
                )
                raise ValueError(msg)

    koopman_kind = config.get("koopman_kind", "pernode")
    if koopman_kind in _RESERVED_KOOPMAN_KINDS:
        task = _RESERVED_KOOPMAN_KINDS[koopman_kind]
        msg = f"koopman_kind={koopman_kind!r} is planned; lands in {task}"
        raise ValueError(msg)
    orbit_partition, auto_orbits, orbit_method = _parse_symmetry_config(
        config.get("symmetry")
    )
    koopman_adjacency = _resolve_checkpoint_adjacency(
        config["adjacency"],
        koopman_kind=str(koopman_kind),
    )

    learn_topology = config.get("learn_topology")
    topology_embedding_dim = config.get("topology_embedding_dim")
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config["latent_dim"],
        time_step=config["time_step"],
        dynamics_mode=config.get("dynamics_mode", "discrete"),
        koopman=koopman_kind,
        koopman_init_mode=config["koopman_init_mode"],
        koopman_init_scale=config["koopman_init_scale"],
        koopman_parameterization=config.get("koopman_parameterization", "dense"),
        koopman_max_spectral_radius=config.get("koopman_max_spectral_radius", 1.0),
        koopman_auxiliary_hidden_dims=config.get("koopman_auxiliary_hidden_dims"),
        koopman_sparsity=config.get("sparsity", "dense"),
        koopman_adjacency=koopman_adjacency,
        koopman_local_window=(
            int(config["local_window"]) if config.get("local_window") is not None else 4
        ),
        koopman_local_rank=(
            int(config["local_rank"]) if config.get("local_rank") is not None else 2
        ),
        koopman_local_hidden_dims=config.get("local_hidden_dims"),
        koopman_orbit_partition=orbit_partition,
        koopman_auto_orbits=auto_orbits,
        koopman_orbit_method=orbit_method,
        learn_topology=learn_topology,
        topology_embedding_dim=(
            int(topology_embedding_dim)
            if topology_embedding_dim is not None
            else DEFAULT_TOPOLOGY_EMBEDDING_DIM
        ),
        control_dim=config.get("control_dim", 0),
        control_mode=config.get("control_mode", "additive"),
        bilinear_rank=config.get("bilinear_rank"),
        physics_lifting_fn=resolved_physics_fn,
        physics_preset=physics_preset,
        physics_dim=physics_dim,
        physics_position=physics_position,
        n_delays=int(config.get("n_delays", 1)),
    )


def build_checkpoint(model: ModeShapeModel) -> dict[str, Any]:
    """Build a versioned checkpoint dictionary for a model.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose weights and architecture will be serialized.

    Returns
    -------
    dict
        Checkpoint payload suitable for :func:`torch.save`.
    """
    return {
        "format_version": FORMAT_VERSION,
        "package_version": _package_version(),
        "config": build_model_config(model),
        "state_dict": model.state_dict(),
    }


def save_checkpoint(model: ModeShapeModel, path: str | Path) -> None:
    """Persist a trained model checkpoint to disk.

    Parameters
    ----------
    model : ModeShapeModel
        Model to serialize.
    path : str or Path
        Destination ``.pt`` file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(model), destination)


def _allocate_adaptive_topology_from_state(
    model: ModeShapeModel,
    state_dict: dict[str, Any],
) -> None:
    """Bind ``AdaptiveAdjacency`` embeddings before ``load_state_dict``.

    Parameters
    ----------

    model : ModeShapeModel
        See the function signature / summary for ``model``.
    state_dict : dict[str, Any]
        See the function signature / summary for ``state_dict``.

    Returns
    -------

    None
        See summary line.

    Notes
    -----

    Lazy allocation means reconstructed models have no embedding parameters
    until ``set_num_nodes``; checkpoint tensors provide ``N``."""
    adaptive = getattr(model, "adaptive_topology", None)
    if adaptive is None:
        return
    source = state_dict.get("adaptive_topology.source_embedding")
    if source is None:
        return
    adaptive.set_num_nodes(int(source.shape[0]), device=source.device)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> ModeShapeModel:
    """Load a trained model from a checkpoint file.

    Parameters
    ----------
    path : str or Path
        Checkpoint ``.pt`` file produced by :func:`save_checkpoint`.
    map_location : str, torch.device, or None, optional
        Device mapping forwarded to :func:`torch.load`.
    physics_lifting_fn : callable or None, optional
        Custom physics lifting function for hybrid checkpoints without a stored
        preset.

    Returns
    -------
    GraphKoopmanModel
        Reconstructed model with restored weights in evaluation mode.

    Raises
    ------
    ValueError
        If the checkpoint format version is unsupported or the payload is invalid.
    FileNotFoundError
        If ``path`` does not exist.
    """
    destination = Path(path)
    if not destination.is_file():
        msg = f"Checkpoint file not found: {destination}"
        raise FileNotFoundError(msg)

    payload = torch.load(destination, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        msg = "Checkpoint must be a dictionary payload"
        raise ValueError(msg)

    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_FORMAT_VERSIONS)
        )
        msg = (
            f"Unsupported checkpoint format_version {format_version!r}; "
            f"supported versions: {supported}"
        )
        raise ValueError(msg)

    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        msg = "Checkpoint must contain 'config' and 'state_dict' dictionaries"
        raise ValueError(msg)

    migrated_config = _migrate_config(config, format_version=int(format_version))
    model = reconstruct_model(migrated_config, physics_lifting_fn=physics_lifting_fn)
    _allocate_adaptive_topology_from_state(model, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of a module's ``state_dict`` for checkpointing.

    Parameters
    ----------
    module : nn.Module
        Module whose parameters will be copied.

    Returns
    -------
    dict
        Deep copy of :meth:`nn.Module.state_dict` with detached tensors.
    """
    state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    return deepcopy(state)
