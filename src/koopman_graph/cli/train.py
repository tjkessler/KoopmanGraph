"""``koopman-graph train`` — config-driven fit + safe checkpoint save."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from koopman_graph import (
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    SAGEDecoder,
    SAGEEncoder,
)
from koopman_graph.cli.config import ConfigError, load_train_config
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.topology import path_edge_index

_ENCODER_KINDS = frozenset({"gcn", "gat", "sage"})
_MODEL_PASSTHROUGH = frozenset(
    {
        "time_step",
        "dynamics_mode",
        "koopman",
        "koopman_parameterization",
        "koopman_init_mode",
        "koopman_init_scale",
        "n_delays",
        "control_dim",
        "control_mode",
    }
)
_FIT_PASSTHROUGH = frozenset(
    {
        "epochs",
        "lr",
        "device",
        "batch_size",
        "window_length",
        "max_grad_norm",
        "use_amp",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "restore_best_weights",
    }
)


def _require_int(value: object, *, path: str) -> int:
    """Coerce a config value to ``int`` or raise :class:`ConfigError`.

    Parameters
    ----------
    value : object
        Candidate integer from the config.
    path : str
        Dotted config path for error messages.

    Returns
    -------
    int
        Validated integer (bool rejected).

    Raises
    ------
    ConfigError
        If ``value`` is not an ``int`` (or is a ``bool``).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path} must be an int, got {type(value).__name__}"
        raise ConfigError(msg)
    return value


def _require_float(value: object, *, path: str) -> float:
    """Coerce a config value to ``float`` or raise :class:`ConfigError`.

    Parameters
    ----------
    value : object
        Candidate number from the config.
    path : str
        Dotted config path for error messages.

    Returns
    -------
    float
        Validated float (bool rejected).

    Raises
    ------
    ConfigError
        If ``value`` is not an ``int`` or ``float`` (or is a ``bool``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path} must be a number, got {type(value).__name__}"
        raise ConfigError(msg)
    return float(value)


def _build_encoder_decoder(
    model_cfg: dict[str, Any],
) -> tuple[Any, Any, int]:
    """Build matched encoder/decoder peers and return ``latent_dim``.

    Parameters
    ----------
    model_cfg : dict
        Validated ``model`` section from a train config.

    Returns
    -------
    encoder, decoder, latent_dim
        Constructed peers and latent width.

    Raises
    ------
    ConfigError
        If encoder/decoder kinds or channel sizes are invalid.
    """
    encoder_kind = model_cfg.get("encoder")
    if not isinstance(encoder_kind, str):
        raise ConfigError("model.encoder is required (string kind)")
    encoder_kind = encoder_kind.lower()
    if encoder_kind not in _ENCODER_KINDS:
        allowed = ", ".join(sorted(_ENCODER_KINDS))
        msg = f"Unsupported model.encoder {encoder_kind!r}; CLI MVP allows: {allowed}"
        raise ConfigError(msg)

    decoder_kind = model_cfg.get("decoder", encoder_kind)
    if not isinstance(decoder_kind, str):
        raise ConfigError("model.decoder must be a string when set")
    decoder_kind = decoder_kind.lower()
    if decoder_kind != encoder_kind:
        msg = (
            f"model.decoder {decoder_kind!r} must match model.encoder "
            f"{encoder_kind!r} in the CLI MVP"
        )
        raise ConfigError(msg)

    for key in ("in_channels", "hidden_channels", "latent_dim"):
        if key not in model_cfg:
            raise ConfigError(f"Missing required key: model.{key}")

    in_channels = _require_int(model_cfg["in_channels"], path="model.in_channels")
    hidden_channels = _require_int(
        model_cfg["hidden_channels"], path="model.hidden_channels"
    )
    latent_dim = _require_int(model_cfg["latent_dim"], path="model.latent_dim")
    num_layers = _require_int(model_cfg.get("num_layers", 2), path="model.num_layers")
    if in_channels < 1 or hidden_channels < 1 or latent_dim < 1 or num_layers < 1:
        raise ConfigError("model channel / layer sizes must be positive")

    peer_kwargs = {
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "latent_dim": latent_dim,
        "num_layers": num_layers,
    }
    if encoder_kind == "gcn":
        encoder = GNNEncoder(**peer_kwargs)
        decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
        )
    elif encoder_kind == "gat":
        encoder = GATEncoder(**peer_kwargs)
        decoder = GATDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
        )
    else:
        encoder = SAGEEncoder(**peer_kwargs)
        decoder = SAGEDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_layers=num_layers,
        )
    return encoder, decoder, latent_dim


def build_model_from_config(model_cfg: dict[str, Any]) -> GraphKoopmanModel:
    """Construct :class:`~koopman_graph.GraphKoopmanModel` from CLI ``model``.

    Parameters
    ----------
    model_cfg : dict
        Validated ``model`` section.

    Returns
    -------
    GraphKoopmanModel
        Homogeneous model ready for ``fit``.
    """
    encoder, decoder, latent_dim = _build_encoder_decoder(model_cfg)
    kwargs: dict[str, Any] = {}
    for key in _MODEL_PASSTHROUGH:
        if key in model_cfg:
            kwargs[key] = model_cfg[key]
    if "time_step" not in kwargs:
        kwargs["time_step"] = 1.0
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        **kwargs,
    )


def _build_synthetic_path_sequence(data_cfg: dict[str, Any]) -> GraphSnapshotSequence:
    """Build a seeded decay trajectory on a path graph.

    Parameters
    ----------
    data_cfg : dict
        ``data`` section with ``kind='synthetic_path'`` fields.

    Returns
    -------
    GraphSnapshotSequence
        Synthetic trajectory for smoke training.
    """
    num_nodes = _require_int(data_cfg.get("num_nodes", 8), path="data.num_nodes")
    num_timesteps = _require_int(
        data_cfg.get("num_timesteps", 40), path="data.num_timesteps"
    )
    seed = _require_int(data_cfg.get("seed", 0), path="data.seed")
    feature_dim = _require_int(
        data_cfg.get("feature_dim", data_cfg.get("in_channels", 3)),
        path="data.feature_dim",
    )
    if num_nodes < 2 or num_timesteps < 2 or feature_dim < 1:
        raise ConfigError(
            "synthetic_path requires num_nodes>=2, num_timesteps>=2, feature_dim>=1"
        )

    generator = torch.Generator().manual_seed(seed)
    edge_index = path_edge_index(num_nodes)
    x0 = torch.randn(num_nodes, feature_dim, generator=generator)
    snapshots = [
        Data(x=x0 * (0.9**t), edge_index=edge_index) for t in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def load_cached_sequence(data_cfg: dict[str, Any]) -> GraphSnapshotSequence:
    """Load a trusted ``GraphSnapshotSequence`` (or list of ``Data``) from disk.

    Parameters
    ----------
    data_cfg : dict
        ``data`` section with ``kind='cached_sequence'`` and ``path``.

    Returns
    -------
    GraphSnapshotSequence
        Loaded trajectory.

    Notes
    -----
    Uses ``torch.load`` pickle — trusted-source only (see ``SECURITY.md``).
    """
    path = Path(str(data_cfg["path"]))
    if not path.is_file():
        raise ConfigError(f"Cached sequence not found: {path}")
    # Trust boundary: pickle load — same as teaching caches (see SECURITY.md).
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, GraphSnapshotSequence):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], Data):
        return GraphSnapshotSequence(payload)
    msg = (
        f"Cached sequence at {path} must be a GraphSnapshotSequence "
        "or a non-empty list of Data snapshots"
    )
    raise ConfigError(msg)


def build_sequence_from_config(data_cfg: dict[str, Any]) -> GraphSnapshotSequence:
    """Build the training sequence for an allowlisted ``data.kind``.

    Parameters
    ----------
    data_cfg : dict
        Validated ``data`` section.

    Returns
    -------
    GraphSnapshotSequence
        Synthetic or cached trajectory.
    """
    kind = data_cfg["kind"]
    if kind == "synthetic_path":
        return _build_synthetic_path_sequence(data_cfg)
    if kind == "cached_sequence":
        return load_cached_sequence(data_cfg)
    msg = f"Unsupported data.kind {kind!r}"
    raise ConfigError(msg)


def _resolve_checkpoint_path(
    config: dict[str, Any],
    *,
    out_dir: Path | None,
) -> Path:
    """Resolve the checkpoint destination path (directory or ``.kgckpt``).

    Parameters
    ----------
    config : dict
        Validated train config (may omit ``checkpoint``).
    out_dir : Path or None
        Optional base directory for relative ``checkpoint.path`` values.

    Returns
    -------
    Path
        Absolute or resolved destination path (parent dirs created).
    """
    checkpoint = config.get("checkpoint") or {}
    raw = checkpoint.get("path", "artifacts/model.kgckpt")
    path = Path(str(raw))
    if out_dir is not None and not path.is_absolute():
        path = out_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fit_kwargs(fit_cfg: dict[str, Any]) -> dict[str, Any]:
    """Filter allowlisted fit kwargs for :meth:`GraphKoopmanModel.fit`.

    Parameters
    ----------
    fit_cfg : dict
        ``fit`` section from the train config.

    Returns
    -------
    dict
        Keyword arguments forwarded to ``model.fit``.
    """
    kwargs: dict[str, Any] = {}
    for key in _FIT_PASSTHROUGH:
        if key in fit_cfg:
            kwargs[key] = fit_cfg[key]
    return kwargs


def run_train(
    config: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """Train from a validated config and save a safetensors checkpoint.

    Parameters
    ----------
    config : dict
        Validated train config (see :func:`load_train_config`).
    out_dir : str, Path, or None, optional
        When set, relative ``checkpoint.path`` values are resolved under this
        directory.

    Returns
    -------
    Path
        Checkpoint path written by :meth:`GraphKoopmanModel.save`.
    """
    model = build_model_from_config(config["model"])
    sequence = build_sequence_from_config(config["data"])
    fit_cfg = config.get("fit") or {}
    model.fit(sequence, **_fit_kwargs(fit_cfg))

    checkpoint_cfg = config.get("checkpoint") or {}
    fmt = checkpoint_cfg.get("format", "safetensors_v1")
    if fmt not in {"safetensors_v1", "legacy_pt"}:
        raise ConfigError(f"Unsupported checkpoint.format {fmt!r}")

    destination = _resolve_checkpoint_path(
        config,
        out_dir=None if out_dir is None else Path(out_dir),
    )
    model.save(destination, format=fmt)  # type: ignore[arg-type]
    return destination


def handle_train(args: argparse.Namespace) -> int:
    """Argparse handler for the ``train`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace with ``config`` and optional ``out``.

    Returns
    -------
    int
        ``0`` on success; ``1`` on config / I/O / type errors.
    """
    try:
        config = load_train_config(args.config)
        out_dir = Path(args.out) if args.out is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        destination = run_train(config, out_dir=out_dir)
    except (ConfigError, ValueError, FileNotFoundError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote checkpoint: {destination}")
    return 0
