"""Load and validate CLI train configs (JSON always; YAML via ``[cli]``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Top-level sections for train configs (Appendix B).
_TRAIN_TOP_LEVEL = frozenset({"model", "data", "fit", "checkpoint"})

# MVP allowlists — extend in code + docs when new CLI surfaces ship.
_MODEL_KEYS = frozenset(
    {
        "encoder",
        "decoder",
        "in_channels",
        "hidden_channels",
        "latent_dim",
        "num_layers",
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

_DATA_KINDS = frozenset({"synthetic_path", "cached_sequence"})

_DATA_KEYS = frozenset(
    {
        "kind",
        "num_nodes",
        "num_timesteps",
        "seed",
        "in_channels",
        "path",
        "feature_dim",
    }
)

_FIT_KEYS = frozenset(
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

_CHECKPOINT_KEYS = frozenset({"path", "format"})


class ConfigError(ValueError):
    """Invalid CLI config (unknown key, bad type, or missing required field).

    Notes
    -----
    Raised by config loaders and train/predict handlers for actionable CLI
    failures (exit code 1 at the boundary).
    """


def _require_mapping(value: object, *, path: str) -> dict[str, Any]:
    """Return ``value`` as a ``dict`` or raise :class:`ConfigError`.

    Parameters
    ----------
    value : object
        Candidate mapping from a parsed config.
    path : str
        Dotted config path used in error messages (for example ``"model"``).

    Returns
    -------
    dict of str to Any
        The same mapping after key-type checks.

    Raises
    ------
    ConfigError
        If ``value`` is not a ``dict`` or contains non-string keys.
    """
    if not isinstance(value, dict):
        msg = f"{path} must be a mapping, got {type(value).__name__}"
        raise ConfigError(msg)
    # JSON object keys are str; reject non-str keys early.
    for key in value:
        if not isinstance(key, str):
            msg = f"{path} keys must be strings, got {type(key).__name__}"
            raise ConfigError(msg)
    return value


def _reject_unknown_keys(
    section: dict[str, Any],
    *,
    allowed: frozenset[str],
    path: str,
) -> None:
    """Raise if ``section`` contains keys outside ``allowed``.

    Parameters
    ----------
    section : dict
        Config subsection to check.
    allowed : frozenset of str
        Allowlisted keys for this subsection.
    path : str
        Dotted config path prefix for error messages.

    Raises
    ------
    ConfigError
        If any key is not in ``allowed``.
    """
    unknown = sorted(set(section) - allowed)
    if unknown:
        dotted = ", ".join(f"{path}.{key}" for key in unknown)
        msg = f"Unknown config key(s): {dotted}"
        raise ConfigError(msg)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a plain ``dict``.

    Parameters
    ----------
    path : str or Path
        Config file path. Suffix ``.json`` uses the stdlib; ``.yaml`` /
        ``.yml`` require PyYAML (``pip install 'koopman-graph[cli]'``).

    Returns
    -------
    dict
        Parsed mapping (not yet schema-validated).

    Raises
    ------
    ConfigError
        If the file is missing, has an unsupported suffix, fails to parse,
        or does not contain a top-level mapping.
    ImportError
        If a YAML file is requested but PyYAML is not installed.
    """
    destination = Path(path)
    if not destination.is_file():
        msg = f"Config file not found: {destination}"
        raise ConfigError(msg)

    suffix = destination.suffix.lower()
    text = destination.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {destination}: {exc.msg}"
            raise ConfigError(msg) from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            msg = (
                "YAML configs require PyYAML. Install with: "
                "pip install 'koopman-graph[cli]' (or: pip install 'pyyaml>=6')"
            )
            raise ImportError(msg) from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {destination}: {exc}"
            raise ConfigError(msg) from exc
    else:
        msg = (
            f"Unsupported config suffix {suffix!r} for {destination}; "
            "use .json, .yaml, or .yml"
        )
        raise ConfigError(msg)

    return _require_mapping(payload, path="config")


def validate_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a train config against the MVP allowlisted schema.

    Parameters
    ----------
    config : dict
        Parsed config mapping.

    Returns
    -------
    dict
        The same mapping after validation (mutated only by type checks;
        callers may treat the return as the validated config).

    Raises
    ------
    ConfigError
        On unknown keys, missing required sections, or invalid ``data.kind``.
    """
    _reject_unknown_keys(config, allowed=_TRAIN_TOP_LEVEL, path="config")

    if "model" not in config:
        raise ConfigError("Missing required section: model")
    if "data" not in config:
        raise ConfigError("Missing required section: data")

    model = _require_mapping(config["model"], path="model")
    _reject_unknown_keys(model, allowed=_MODEL_KEYS, path="model")

    data = _require_mapping(config["data"], path="data")
    _reject_unknown_keys(data, allowed=_DATA_KEYS, path="data")
    kind = data.get("kind")
    if kind is None:
        raise ConfigError("Missing required key: data.kind")
    if kind not in _DATA_KINDS:
        allowed = ", ".join(sorted(_DATA_KINDS))
        msg = f"Unsupported data.kind {kind!r}; allowed: {allowed}"
        raise ConfigError(msg)
    if kind == "cached_sequence" and "path" not in data:
        raise ConfigError("data.kind='cached_sequence' requires data.path")

    if "fit" in config:
        fit = _require_mapping(config["fit"], path="fit")
        _reject_unknown_keys(fit, allowed=_FIT_KEYS, path="fit")

    if "checkpoint" in config:
        checkpoint = _require_mapping(config["checkpoint"], path="checkpoint")
        _reject_unknown_keys(checkpoint, allowed=_CHECKPOINT_KEYS, path="checkpoint")
        if "path" not in checkpoint:
            raise ConfigError("checkpoint section requires checkpoint.path")

    return config


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a train config file.

    Parameters
    ----------
    path : str or Path
        JSON or YAML config path.

    Returns
    -------
    dict
        Validated train config.
    """
    return validate_train_config(load_config(path))
