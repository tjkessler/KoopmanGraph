"""Coverage and error-path tests for :mod:`koopman_graph.cli`."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import koopman_graph.cli.config as config_mod


def test_config_mapping_validation_errors() -> None:
    """Mapping validation rejects non-mappings and non-string keys."""
    with pytest.raises(config_mod.ConfigError, match="must be a mapping"):
        config_mod._require_mapping([], path="config")
    with pytest.raises(config_mod.ConfigError, match="keys must be strings"):
        config_mod._require_mapping({1: "bad"}, path="config")


def test_config_load_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    """Missing files and malformed JSON produce ConfigError."""
    with pytest.raises(config_mod.ConfigError, match="Config file not found"):
        config_mod.load_config(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="Invalid JSON"):
        config_mod.load_config(invalid)


def test_config_load_yaml_import_and_parse_errors(tmp_path: Path) -> None:
    """YAML loading reports missing PyYAML and parser failures."""
    path = tmp_path / "invalid.yaml"
    path.write_text("invalid: [", encoding="utf-8")
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "yaml":
            raise ImportError("simulated missing yaml")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=blocked_import),
        pytest.raises(ImportError, match="YAML configs require PyYAML"),
    ):
        config_mod.load_config(path)

    class FakeYamlError(Exception):
        """Synthetic YAML parser error."""

    fake_yaml = ModuleType("yaml")
    fake_yaml.YAMLError = FakeYamlError  # type: ignore[attr-defined]
    fake_yaml.safe_load = MagicMock(side_effect=FakeYamlError("bad yaml"))  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"yaml": fake_yaml}),
        pytest.raises(config_mod.ConfigError, match="Invalid YAML"),
    ):
        config_mod.load_config(path)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"model": {}}, "Missing required section: data"),
        (
            {"model": {}, "data": {}},
            "Missing required key: data.kind",
        ),
        (
            {"model": {}, "data": {"kind": "cached_sequence"}},
            "requires data.path",
        ),
        (
            {
                "model": {},
                "data": {"kind": "synthetic_path"},
                "checkpoint": {},
            },
            "checkpoint.path",
        ),
    ],
)
def test_config_validate_required_data_and_checkpoint_fields(
    config: dict[str, object],
    message: str,
) -> None:
    """Train config validation covers missing required fields."""
    with pytest.raises(config_mod.ConfigError, match=message):
        config_mod.validate_train_config(config)
