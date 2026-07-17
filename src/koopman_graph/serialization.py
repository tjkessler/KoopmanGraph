"""Checkpoint serialization for :class:`~koopman_graph.model.GraphKoopmanModel`.

Checkpoint format versions
--------------------------
``format_version`` 1 (v0.2.x)
    Discrete dynamics only; config omits ``dynamics_mode``, ``physics``,
    ``control_dim``, and structured-operator fields added in v0.3.0.

``format_version`` 2 (v0.3.0+)
    Full config including continuous-time mode, hybrid physics observables,
    control dimension, and Koopman parameterization metadata. Decoder configs
    may include ``type`` (``"gcn"`` or ``"gat"``); missing ``type`` defaults to
    ``"gcn"`` for backward compatibility. Hybrid ``physics`` blocks own
    ``dim``, ``preset``, and ``position``; ``position`` is round-tripped and
    validated on load (currently only ``"prepend"``). Missing ``position``
    defaults to ``"prepend"``. Optional ``n_delays`` (default ``1``) records
    Hankel delay embedding; the stored encoder block is always the base
    GCN/GAT config with ``in_channels = n_delays * feature_dim``.
    v0.4.0 keeps this format: ``koopman_kind``, ``control_mode``, and
    ``bilinear_rank`` are additive optional fields with safe defaults.

Loaders accept both versions. v1 checkpoints are migrated in memory by filling
missing optional fields with defaults (``dynamics_mode="discrete"``, no physics,
``control_dim=0``, ``koopman_parameterization="dense"``).

Custom injected operators (anything other than
:class:`~koopman_graph.operators.KoopmanOperator`,
:class:`~koopman_graph.operators.ContinuousKoopmanOperator`, or
:class:`~koopman_graph.operators.GraphKoopmanOperator`) are **not**
round-trippable: :func:`build_model_config` / :meth:`GraphKoopmanModel.save`
raise rather than silently writing incomplete factory metadata.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
)
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    PhysicsPosition,
    resolve_physics_lifting_fn,
    resolve_physics_position,
)
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
    resolve_factory_stability_bound,
)

if TYPE_CHECKING:
    from koopman_graph.model import GraphKoopmanModel

FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})

Decoder = GNNDecoder | GATDecoder
_SERIALIZABLE_KOOPMAN_TYPES = (
    KoopmanOperator,
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
)


def _migrate_config(config: dict[str, Any], *, format_version: int) -> dict[str, Any]:
    """Fill v0.2.x checkpoint config defaults for v0.3.0 fields.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a saved checkpoint.
    format_version : int
        Checkpoint ``format_version`` before migration.

    Returns
    -------
    dict
        Config with v0.3.0 optional fields populated when absent.
    """
    if format_version >= 2:
        return config

    migrated = dict(config)
    migrated.setdefault("dynamics_mode", "discrete")
    migrated.setdefault("koopman_parameterization", "dense")
    migrated.setdefault("koopman_max_spectral_radius", 1.0)
    migrated.setdefault("control_dim", 0)
    migrated.setdefault("physics", None)
    migrated.setdefault("n_delays", 1)
    return migrated


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


_SUPPORTED_ENCODER_TYPES: dict[str, type[GNNEncoder] | type[GATEncoder]] = {
    "gcn": GNNEncoder,
    "gat": GATEncoder,
}

_SUPPORTED_DECODER_TYPES: dict[str, type[GNNDecoder] | type[GATDecoder]] = {
    "gcn": GNNDecoder,
    "gat": GATDecoder,
}


def _encoder_type(encoder: GNNEncoder | GATEncoder) -> str:
    """Return the checkpoint encoder type string for an encoder instance.

    Parameters
    ----------
    encoder : GNNEncoder or GATEncoder
        Encoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"`` for :class:`~koopman_graph.nn.encoder.GNNEncoder` and ``"gat"``
        for :class:`~koopman_graph.nn.encoder.GATEncoder`.

    Raises
    ------
    TypeError
        If ``encoder`` is not a supported encoder class.
    """
    if isinstance(encoder, GATEncoder):
        return "gat"
    if isinstance(encoder, GNNEncoder):
        return "gcn"
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _unwrap_base_encoder(
    encoder: nn.Module,
) -> tuple[GNNEncoder | GATEncoder, int]:
    """Return the serializable base encoder and delay count.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder, possibly wrapped in :class:`DelayEmbeddingEncoder`.

    Returns
    -------
    base_encoder : GNNEncoder or GATEncoder
        Checkpoint-rebuildable encoder.
    n_delays : int
        Delay window length (``1`` when unwrapped).
    """
    if isinstance(encoder, DelayEmbeddingEncoder):
        base = encoder.base_encoder
        if not isinstance(base, (GNNEncoder, GATEncoder)):
            msg = (
                "DelayEmbeddingEncoder.base_encoder must be GNNEncoder or "
                f"GATEncoder for checkpoints; got {type(base).__name__}"
            )
            raise TypeError(msg)
        return base, encoder.n_delays
    if isinstance(encoder, (GNNEncoder, GATEncoder)):
        return encoder, 1
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _decoder_type(decoder: Decoder) -> str:
    """Return the checkpoint decoder type string for a decoder instance.

    Parameters
    ----------
    decoder : GNNDecoder or GATDecoder
        Decoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"`` for :class:`~koopman_graph.nn.decoder.GNNDecoder` and ``"gat"``
        for :class:`~koopman_graph.nn.decoder.GATDecoder`.

    Raises
    ------
    TypeError
        If ``decoder`` is not a supported decoder class.
    """
    if isinstance(decoder, GATDecoder):
        return "gat"
    if isinstance(decoder, GNNDecoder):
        return "gcn"
    msg = f"Unsupported decoder type: {type(decoder).__name__}"
    raise TypeError(msg)


def _require_serializable_koopman(model: GraphKoopmanModel) -> None:
    """Reject custom injected operators that lack checkpoint factory metadata.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model whose ``koopman`` submodule will be serialized.

    Raises
    ------
    TypeError
        If ``model.koopman`` is not a built-in discrete or continuous operator.
    """
    if isinstance(model.koopman, _SERIALIZABLE_KOOPMAN_TYPES):
        return
    msg = (
        "Checkpoint serialization supports only built-in KoopmanOperator, "
        "ContinuousKoopmanOperator, and GraphKoopmanOperator instances. "
        "Custom injected operators are not round-trippable; save the operator "
        "state separately or reconstruct the model with koopman=... after load. "
        f"Got {type(model.koopman).__name__}."
    )
    raise TypeError(msg)


def build_model_config(model: GraphKoopmanModel) -> dict[str, Any]:
    """Extract architecture configuration from a :class:`GraphKoopmanModel`.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model whose encoder, decoder, and Koopman settings will be serialized.

    Returns
    -------
    dict
        JSON-serializable architecture configuration.

    Raises
    ------
    TypeError
        If ``model.koopman`` is a custom injected operator (not a built-in
        :class:`~koopman_graph.operators.KoopmanOperator` or
        :class:`~koopman_graph.operators.ContinuousKoopmanOperator`).
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
    if isinstance(encoder, GATEncoder):
        encoder_config["heads"] = encoder.heads
        encoder_config["dropout"] = encoder.dropout

    decoder_config: dict[str, Any] = {
        "type": _decoder_type(decoder),
        "latent_dim": decoder.latent_dim,
        "hidden_channels": decoder.hidden_channels,
        "out_channels": decoder.out_channels,
        "num_layers": decoder.num_layers,
        "activation": decoder.activation_name,
    }
    if isinstance(decoder, GATDecoder):
        decoder_config["heads"] = decoder.heads
        decoder_config["dropout"] = decoder.dropout

    physics_config: dict[str, Any] | None = None
    if model.physics_dim > 0:
        physics_config = {
            "dim": model.physics_dim,
            "preset": model.physics_preset,
            "position": model.physics_position,
        }

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
        "control_dim": model.control_dim,
        "control_mode": getattr(model, "control_mode", "additive"),
        "bilinear_rank": getattr(model, "bilinear_rank", None),
        "n_delays": n_delays,
        "physics": physics_config,
        "encoder": encoder_config,
        "decoder": decoder_config,
    }


def _build_encoder(config: dict[str, Any]) -> GNNEncoder | GATEncoder:
    """Instantiate an encoder from a checkpoint configuration block.

    Parameters
    ----------
    config : dict
        Encoder configuration block from a saved checkpoint.

    Returns
    -------
    GNNEncoder or GATEncoder
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
    GNNDecoder or GATDecoder
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
    return GNNDecoder(**common_kwargs)


def reconstruct_model(
    config: dict[str, Any],
    *,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> GraphKoopmanModel:
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
    from koopman_graph.model import GraphKoopmanModel

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

    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config["latent_dim"],
        time_step=config["time_step"],
        dynamics_mode=config.get("dynamics_mode", "discrete"),
        koopman=config.get("koopman_kind", "pernode"),
        koopman_init_mode=config["koopman_init_mode"],
        koopman_init_scale=config["koopman_init_scale"],
        koopman_parameterization=config.get("koopman_parameterization", "dense"),
        koopman_max_spectral_radius=config.get("koopman_max_spectral_radius", 1.0),
        control_dim=config.get("control_dim", 0),
        control_mode=config.get("control_mode", "additive"),
        bilinear_rank=config.get("bilinear_rank"),
        physics_lifting_fn=resolved_physics_fn,
        physics_preset=physics_preset,
        physics_dim=physics_dim,
        physics_position=physics_position,
        n_delays=int(config.get("n_delays", 1)),
    )


def build_checkpoint(model: GraphKoopmanModel) -> dict[str, Any]:
    """Build a versioned checkpoint dictionary for a model.

    Parameters
    ----------
    model : GraphKoopmanModel
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


def save_checkpoint(model: GraphKoopmanModel, path: str | Path) -> None:
    """Persist a trained model checkpoint to disk.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model to serialize.
    path : str or Path
        Destination ``.pt`` file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(model), destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> GraphKoopmanModel:
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
