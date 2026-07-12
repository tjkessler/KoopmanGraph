"""Checkpoint serialization for :class:`~koopman_graph.model.GraphKoopmanModel`."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder

if TYPE_CHECKING:
    from koopman_graph.model import GraphKoopmanModel

FORMAT_VERSION = 1


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


def _encoder_type(encoder: GNNEncoder | GATEncoder) -> str:
    """Return the checkpoint encoder type string for an encoder instance.

    Parameters
    ----------
    encoder : GNNEncoder or GATEncoder
        Encoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"`` for :class:`~koopman_graph.encoder.GNNEncoder` and ``"gat"``
        for :class:`~koopman_graph.encoder.GATEncoder`.

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
    """
    encoder = model.encoder
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

    return {
        "latent_dim": model.latent_dim,
        "time_step": model.time_step,
        "koopman_init_mode": model.koopman.init_mode,
        "koopman_init_scale": model.koopman.init_scale,
        "koopman_parameterization": model.koopman.parameterization,
        "koopman_max_spectral_radius": model.koopman.max_spectral_radius,
        "control_dim": model.control_dim,
        "encoder": encoder_config,
        "decoder": {
            "latent_dim": decoder.latent_dim,
            "hidden_channels": decoder.hidden_channels,
            "out_channels": decoder.out_channels,
            "num_layers": decoder.num_layers,
            "activation": decoder.activation_name,
        },
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


def reconstruct_model(config: dict[str, Any]) -> GraphKoopmanModel:
    """Reconstruct a :class:`GraphKoopmanModel` from a checkpoint configuration.

    Parameters
    ----------
    config : dict
        Architecture configuration produced by :func:`build_model_config`.

    Returns
    -------
    GraphKoopmanModel
        Uninitialized-weight model matching the saved architecture.
    """
    from koopman_graph.model import GraphKoopmanModel

    decoder_config = config["decoder"]
    decoder = GNNDecoder(
        latent_dim=decoder_config["latent_dim"],
        hidden_channels=decoder_config["hidden_channels"],
        out_channels=decoder_config["out_channels"],
        num_layers=decoder_config["num_layers"],
        activation=decoder_config["activation"],
    )
    encoder = _build_encoder(config["encoder"])
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config["latent_dim"],
        time_step=config["time_step"],
        koopman_init_mode=config["koopman_init_mode"],
        koopman_init_scale=config["koopman_init_scale"],
        koopman_parameterization=config.get("koopman_parameterization", "dense"),
        koopman_max_spectral_radius=config.get("koopman_max_spectral_radius", 1.0),
        control_dim=config.get("control_dim", 0),
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
) -> GraphKoopmanModel:
    """Load a trained model from a checkpoint file.

    Parameters
    ----------
    path : str or Path
        Checkpoint ``.pt`` file produced by :func:`save_checkpoint`.
    map_location : str, torch.device, or None, optional
        Device mapping forwarded to :func:`torch.load`.

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
    if format_version != FORMAT_VERSION:
        msg = (
            f"Unsupported checkpoint format_version {format_version!r}; "
            f"expected {FORMAT_VERSION}"
        )
        raise ValueError(msg)

    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        msg = "Checkpoint must contain 'config' and 'state_dict' dictionaries"
        raise ValueError(msg)

    model = reconstruct_model(config)
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
