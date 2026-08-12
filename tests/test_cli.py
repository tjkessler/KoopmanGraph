"""Smoke tests for the ``koopman-graph`` CLI skeleton."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from koopman_graph import __version__
from koopman_graph.cli import (
    ConfigError,
    load_config,
    load_train_config,
    main,
    validate_train_config,
)


def test_main_version_flag_prints_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` prints ``koopman-graph {version}`` and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"koopman-graph {__version__}"


def test_main_version_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``version`` subcommand matches the ``--version`` string."""
    assert main(["version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"koopman-graph {__version__}"


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits 0 and mentions the program name."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "koopman-graph" in captured.out


def test_module_invocation_version() -> None:
    """``python -m koopman_graph.cli --version`` succeeds."""
    result = subprocess.run(
        [sys.executable, "-m", "koopman_graph.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"koopman-graph {__version__}"


def test_console_script_version() -> None:
    """Installed ``koopman-graph --version`` succeeds (editable install)."""
    result = subprocess.run(
        ["koopman-graph", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"koopman-graph {__version__}"


def test_library_modules_do_not_import_cli() -> None:
    """Layer rule: non-cli package sources must not import ``cli``."""
    package_root = Path(__file__).resolve().parents[1] / "src" / "koopman_graph"
    cli_root = package_root / "cli"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if cli_root in path.parents or path.parent == cli_root:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            from_cli = stripped.startswith("from koopman_graph.cli")
            import_cli = stripped.startswith("import koopman_graph.cli")
            if from_cli or import_cli:
                offenders.append(str(path.relative_to(package_root.parent)))
                break
    assert offenders == []


def _minimal_train_config() -> dict[str, object]:
    """Return a valid MVP train config matching the design sketch."""
    return {
        "model": {
            "encoder": "gcn",
            "in_channels": 3,
            "hidden_channels": 32,
            "latent_dim": 16,
            "num_layers": 2,
            "dynamics_mode": "discrete",
        },
        "data": {
            "kind": "synthetic_path",
            "num_nodes": 8,
            "num_timesteps": 40,
            "seed": 0,
        },
        "fit": {"epochs": 20, "lr": 1e-3, "device": "cpu"},
        "checkpoint": {"path": "artifacts/model.kgckpt"},
    }


def test_load_train_config_json(tmp_path: Path) -> None:
    """JSON train configs load without requiring PyYAML."""
    import json

    path = tmp_path / "train.json"
    path.write_text(json.dumps(_minimal_train_config()), encoding="utf-8")
    loaded = load_train_config(path)
    assert loaded["data"]["kind"] == "synthetic_path"
    assert loaded["model"]["latent_dim"] == 16


def test_validate_rejects_unknown_model_key() -> None:
    """Unknown keys report a dotted path."""
    config = _minimal_train_config()
    assert isinstance(config["model"], dict)
    config["model"]["not_a_real_kwarg"] = 1
    with pytest.raises(ConfigError, match=r"model\.not_a_real_kwarg"):
        validate_train_config(config)  # type: ignore[arg-type]


def test_validate_rejects_unknown_data_kind() -> None:
    """``data.kind`` must be on the MVP allowlist."""
    config = _minimal_train_config()
    assert isinstance(config["data"], dict)
    config["data"]["kind"] = "arbitrary_import"
    with pytest.raises(ConfigError, match="data.kind"):
        validate_train_config(config)  # type: ignore[arg-type]


def test_validate_requires_model_and_data() -> None:
    """Train configs require ``model`` and ``data`` sections."""
    with pytest.raises(ConfigError, match="model"):
        validate_train_config({"data": {"kind": "synthetic_path"}})


def test_load_config_yaml(tmp_path: Path) -> None:
    """YAML loads when PyYAML is installed (``[cli]`` extra)."""
    pytest.importorskip("yaml")
    path = tmp_path / "train.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  encoder: gcn",
                "  in_channels: 3",
                "  hidden_channels: 16",
                "  latent_dim: 8",
                "  num_layers: 2",
                "data:",
                "  kind: synthetic_path",
                "  num_nodes: 5",
                "  num_timesteps: 10",
                "  seed: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_train_config(path)
    assert loaded["model"]["encoder"] == "gcn"
    assert loaded["data"]["num_nodes"] == 5


def test_load_config_unsupported_suffix(tmp_path: Path) -> None:
    """Non JSON/YAML suffixes raise ConfigError."""
    path = tmp_path / "train.toml"
    path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported config suffix"):
        load_config(path)


def test_cli_extra_declares_pyyaml() -> None:
    """``[cli]`` optional dependency pins PyYAML."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ncli = [\n" in text
    assert '"pyyaml>=6"' in text


def test_train_synthetic_writes_safetensors_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``train`` on the bundled synthetic config writes a loadable checkpoint."""
    import zipfile

    from koopman_graph.serialization import (
        SAFE_ZIP_MEMBER_NAMES,
        load_checkpoint,
    )

    root = Path(__file__).resolve().parents[1]
    config_path = root / "examples" / "cli" / "synthetic_train.json"
    assert config_path.is_file()

    code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "wrote checkpoint:" in captured.out

    checkpoint = tmp_path / "model.kgckpt"
    assert checkpoint.is_file()
    assert zipfile.is_zipfile(checkpoint)
    with zipfile.ZipFile(checkpoint, mode="r") as archive:
        assert SAFE_ZIP_MEMBER_NAMES.issubset(set(archive.namelist()))

    loaded = load_checkpoint(checkpoint)
    assert loaded.latent_dim == 8


def test_train_bad_config_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown config keys exit with code 1 and a dotted-path message."""
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "model": {
                    "encoder": "gcn",
                    "in_channels": 3,
                    "hidden_channels": 8,
                    "latent_dim": 4,
                    "not_allowed": True,
                },
                "data": {"kind": "synthetic_path", "num_nodes": 5, "num_timesteps": 8},
            }
        ),
        encoding="utf-8",
    )
    code = main(["train", "--config", str(path), "--out", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "model.not_allowed" in err


def test_cli_e2e_train_predict_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CI smoke: train → predict on the bundled synthetic config (CPU, tiny)."""
    import json
    import time

    import torch

    root = Path(__file__).resolve().parents[1]
    config_path = root / "examples" / "cli" / "synthetic_train.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["fit"]["device"] == "cpu"
    assert int(config["fit"]["epochs"]) <= 5

    train_out = tmp_path / "train"
    forecast_path = tmp_path / "forecast.pt"
    started = time.perf_counter()

    train_code = main(["train", "--config", str(config_path), "--out", str(train_out)])
    assert train_code == 0
    checkpoint = train_out / "model.kgckpt"
    assert checkpoint.is_file()

    predict_code = main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(config_path),
            "--steps",
            "4",
            "--out",
            str(forecast_path),
        ]
    )
    assert predict_code == 0
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"e2e smoke too slow for default CI: {elapsed:.1f}s"

    out = capsys.readouterr().out
    assert "wrote checkpoint:" in out
    assert "wrote forecast:" in out
    assert forecast_path.is_file()

    payload = torch.load(forecast_path, map_location="cpu", weights_only=False)
    assert payload["steps"] == 4
    assert len(payload["forecasts"]) == 4
    assert payload["summary"]["num_nodes"] == 8
    assert payload["summary"]["feature_dim"] == 3


def test_predict_missing_checkpoint_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing checkpoint path exits with code 1."""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "examples" / "cli" / "synthetic_train.json"
    code = main(
        [
            "predict",
            "--checkpoint",
            str(tmp_path / "missing.kgckpt"),
            "--data",
            str(config_path),
            "--out",
            str(tmp_path / "out.pt"),
        ]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_predict_help_documents_io(capsys: pytest.CaptureFixture[str]) -> None:
    """``predict --help`` documents checkpoint, data, and output paths."""
    with pytest.raises(SystemExit) as exc_info:
        main(["predict", "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text
    assert "--data" in help_text
    assert "--out" in help_text
    assert "forecast" in help_text.lower()
