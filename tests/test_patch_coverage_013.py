"""Patch-coverage guards for the 0.13.0 Codecov patch gate (target >= 90%)."""

from __future__ import annotations

import builtins
import importlib
import json
import runpy
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import koopman_graph.analysis.explain as explain_mod
import koopman_graph.cli.config as config_mod
import koopman_graph.cli.predict as predict_mod
import koopman_graph.cli.train as train_mod
import koopman_graph.distributed.lightning_module as lightning_mod
import koopman_graph.serialization as serialization_mod
import koopman_graph.tracking.tensorboard as tensorboard_mod
import koopman_graph.tuning.ray_tune as ray_tune_mod
import koopman_graph.tuning.search_spaces as search_spaces_mod
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.tracking.csv import CsvFitLogger
from koopman_graph.training.history import TrainingLossBreakdown

cli_main_mod = importlib.import_module("koopman_graph.cli.main")


def _minimal_model_config(*, encoder: str = "gcn") -> dict[str, object]:
    """Return the smallest valid CLI model config."""
    return {
        "encoder": encoder,
        "in_channels": 2,
        "hidden_channels": 3,
        "latent_dim": 2,
    }


def _minimal_train_config() -> dict[str, object]:
    """Return a valid train config for validation branch tests."""
    return {
        "model": _minimal_model_config(),
        "data": {"kind": "synthetic_path"},
    }


def _snapshot_sequence() -> GraphSnapshotSequence:
    """Return a deterministic two-snapshot homogeneous sequence."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.full((2, 2), float(step)), edge_index=edge_index)
        for step in range(2)
    ]
    return GraphSnapshotSequence(snapshots)


def _safe_meta(*, format_version: int = 1) -> dict[str, object]:
    """Return valid safetensors metadata."""
    return {
        "container": serialization_mod.SAFE_CONTAINER,
        "format_version": format_version,
        "package_version": "0.13.0",
    }


def _write_safe_directory(
    path: Path,
    *,
    meta_text: str | None = None,
    config_text: str | None = None,
    include_config: bool = True,
) -> None:
    """Write enough safe-checkpoint members to reach parser validation."""
    path.mkdir()
    (path / serialization_mod.SAFE_META_FILENAME).write_text(
        meta_text if meta_text is not None else json.dumps(_safe_meta()),
        encoding="utf-8",
    )
    if include_config:
        (path / serialization_mod.SAFE_CONFIG_FILENAME).write_text(
            config_text if config_text is not None else "{}",
            encoding="utf-8",
        )
    (path / serialization_mod.SAFE_WEIGHTS_FILENAME).write_bytes(b"placeholder")


def _breakdown() -> TrainingLossBreakdown:
    """Return a deterministic all-zero loss breakdown."""
    return TrainingLossBreakdown.zeros(torch.device("cpu"))


def _unsupported_explain_model(**overrides: object) -> SimpleNamespace:
    """Return a model-shaped object for explain-MVP validation."""
    attributes: dict[str, object] = {
        "n_delays": 1,
        "control_dim": 0,
        "uses_hetero_koopman": False,
        "_uses_relgraph_encode": lambda: False,
        "uses_hypergraph_koopman": False,
        "_uses_hypergraph_encode": lambda: False,
        "learns_pairwise_topology": False,
        "adaptive_topology": None,
    }
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


@pytest.mark.parametrize("value", [True, 1.2, "1"])
def test_train_require_int_rejects_non_integer_values(value: object) -> None:
    """The CLI integer validator rejects bool and non-integer values."""
    with pytest.raises(config_mod.ConfigError, match="model.value must be an int"):
        train_mod._require_int(value, path="model.value")


@pytest.mark.parametrize("value", [False, "1.0", None])
def test_train_require_float_rejects_non_numeric_values(value: object) -> None:
    """The CLI float validator rejects bool and non-numeric values."""
    with pytest.raises(config_mod.ConfigError, match="must be a number"):
        train_mod._require_float(value, path="fit.lr")


def test_train_require_float_accepts_integer() -> None:
    """The CLI float validator converts an integer to float."""
    assert train_mod._require_float(2, path="fit.lr") == pytest.approx(
        2.0, rel=0.0, abs=0.0
    )


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({}, "model.encoder is required"),
        ({"encoder": "transformer"}, "Unsupported model.encoder"),
        ({"encoder": "gcn", "decoder": 1}, "model.decoder must be a string"),
        (
            {"encoder": "gcn", "decoder": "gat"},
            "must match model.encoder",
        ),
        ({"encoder": "gcn"}, "Missing required key"),
        (
            {
                "encoder": "gcn",
                "in_channels": 0,
                "hidden_channels": 2,
                "latent_dim": 2,
            },
            "must be positive",
        ),
    ],
)
def test_train_encoder_decoder_rejects_invalid_configs(
    model_config: dict[str, object],
    message: str,
) -> None:
    """Encoder/decoder config guards raise actionable errors."""
    with pytest.raises(config_mod.ConfigError, match=message):
        train_mod._build_encoder_decoder(model_config)


@pytest.mark.parametrize("kind", ["gat", "sage"])
def test_train_builds_non_gcn_encoder_decoder_branches(kind: str) -> None:
    """GAT and GraphSAGE CLI model branches construct matched peers."""
    encoder, decoder, latent_dim = train_mod._build_encoder_decoder(
        _minimal_model_config(encoder=kind)
    )
    assert encoder is not None
    assert decoder is not None
    assert latent_dim == 2


def test_train_model_defaults_time_step() -> None:
    """An omitted time step is forwarded to the model as 1.0."""
    sentinel = object()
    with patch.object(
        train_mod,
        "GraphKoopmanModel",
        return_value=sentinel,
    ) as model_cls:
        result = train_mod.build_model_from_config(_minimal_model_config())
    assert result is sentinel
    assert model_cls.call_args.kwargs["time_step"] == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_train_synthetic_sequence_rejects_invalid_dimensions() -> None:
    """Synthetic path data enforces its node, time, and feature minima."""
    with pytest.raises(config_mod.ConfigError, match="num_nodes>=2"):
        train_mod._build_synthetic_path_sequence(
            {"num_nodes": 1, "num_timesteps": 2, "feature_dim": 1}
        )


def test_train_cached_sequence_loads_supported_payloads(tmp_path: Path) -> None:
    """Cached loaders accept sequence objects and non-empty Data lists."""
    path = tmp_path / "sequence.pt"
    sequence = _snapshot_sequence()
    torch.save(sequence, path)
    loaded_sequence = train_mod.load_cached_sequence({"path": path})
    assert len(loaded_sequence) == 2

    torch.save(list(sequence), path)
    loaded_list = train_mod.load_cached_sequence({"path": path})
    assert len(loaded_list) == 2


def test_train_cached_sequence_rejects_missing_and_invalid_payloads(
    tmp_path: Path,
) -> None:
    """Cached loaders reject missing paths and unsupported payload shapes."""
    missing = tmp_path / "missing.pt"
    with pytest.raises(config_mod.ConfigError, match="not found"):
        train_mod.load_cached_sequence({"path": missing})

    invalid = tmp_path / "invalid.pt"
    torch.save({"not": "a sequence"}, invalid)
    with pytest.raises(config_mod.ConfigError, match="must be a GraphSnapshotSequence"):
        train_mod.load_cached_sequence({"path": invalid})


def test_train_sequence_dispatches_cached_and_rejects_unknown_kind() -> None:
    """Data-kind dispatch covers the cached and unknown branches."""
    sentinel = _snapshot_sequence()
    with patch.object(train_mod, "load_cached_sequence", return_value=sentinel):
        result = train_mod.build_sequence_from_config(
            {"kind": "cached_sequence", "path": "ignored.pt"}
        )
    assert result is sentinel

    with pytest.raises(config_mod.ConfigError, match="Unsupported data.kind"):
        train_mod.build_sequence_from_config({"kind": "remote_code"})


def test_train_rejects_unknown_checkpoint_format() -> None:
    """Training refuses checkpoint formats outside the explicit allowlist."""
    fake_model = MagicMock()
    with (
        patch.object(train_mod, "build_model_from_config", return_value=fake_model),
        patch.object(
            train_mod,
            "build_sequence_from_config",
            return_value=_snapshot_sequence(),
        ),
        pytest.raises(config_mod.ConfigError, match="checkpoint.format"),
    ):
        train_mod.run_train(
            {
                "model": {},
                "data": {},
                "checkpoint": {"format": "unsafe_pickle"},
            }
        )


def test_serialization_json_and_device_helpers_cover_tuple_paths() -> None:
    """Tuple configs become lists and explicit devices become strings."""
    assert serialization_mod._json_ready({"shape": (1, (2, 3))}) == {
        "shape": [1, [2, 3]]
    }
    assert serialization_mod._json_ready([{"nested": (4,)}]) == [{"nested": [4]}]
    assert serialization_mod._resolve_safetensors_device(torch.device("cpu")) == "cpu"


def test_estimator_ddp_strategy_rejects_callbacks() -> None:
    """Distributed ``strategy='ddp'`` refuses ``callbacks=`` until supported."""
    from koopman_graph.model.estimator import GraphKoopmanModel
    from koopman_graph.nn import GNNDecoder, GNNEncoder
    from koopman_graph.training.callbacks import NoOpFitCallback

    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 3, num_layers=1),
        decoder=GNNDecoder(3, 4, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(2, 2), edge_index=edge_index) for _ in range(4)]
    )
    with pytest.raises(ValueError, match="callbacks"):
        model.fit(
            sequence,
            epochs=1,
            strategy="ddp",
            callbacks=[NoOpFitCallback()],
        )


def test_serialization_zip_writer_rejects_existing_directory(
    tmp_path: Path,
) -> None:
    """A zip checkpoint cannot overwrite an existing directory."""
    destination = tmp_path / "model.kgckpt"
    destination.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        serialization_mod._save_safetensors_v1_zip(object(), destination)


def test_serialization_zip_probe_covers_false_and_bad_zip(tmp_path: Path) -> None:
    """The safe-zip probe rejects absent files and malformed archives."""
    missing = tmp_path / "missing.kgckpt"
    assert not serialization_mod._is_safetensors_v1_zip(missing)

    candidate = tmp_path / "candidate.kgckpt"
    candidate.write_bytes(b"not a zip")
    with (
        patch.object(serialization_mod.zipfile, "is_zipfile", return_value=True),
        patch.object(
            serialization_mod.zipfile,
            "ZipFile",
            side_effect=zipfile.BadZipFile("bad"),
        ),
    ):
        assert not serialization_mod._is_safetensors_v1_zip(candidate)


def test_serialization_save_rejects_unknown_format(tmp_path: Path) -> None:
    """The checkpoint writer rejects unknown format tokens."""
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        serialization_mod.save_checkpoint(
            object(),
            tmp_path / "model",
            format="unknown",  # type: ignore[arg-type]
        )


def test_serialization_directory_loader_rejects_missing_config(
    tmp_path: Path,
) -> None:
    """A safe directory must contain config.json."""
    checkpoint = tmp_path / "missing_config"
    _write_safe_directory(checkpoint, include_config=False)
    with pytest.raises(ValueError, match=serialization_mod.SAFE_CONFIG_FILENAME):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


@pytest.mark.parametrize(
    ("meta_text", "message"),
    [
        ("{", "Invalid meta.json"),
        ("[]", "meta.json must be a JSON object"),
        (json.dumps(_safe_meta(format_version=999)), "format_version"),
    ],
)
def test_serialization_directory_loader_rejects_invalid_metadata(
    tmp_path: Path,
    meta_text: str,
    message: str,
) -> None:
    """Malformed, non-object, and unsupported metadata are rejected."""
    checkpoint = tmp_path / "bad_meta"
    _write_safe_directory(checkpoint, meta_text=meta_text)
    with pytest.raises(ValueError, match=message):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


def test_serialization_directory_loader_rejects_non_object_config(
    tmp_path: Path,
) -> None:
    """Safe config JSON must decode to an object."""
    checkpoint = tmp_path / "bad_config"
    _write_safe_directory(checkpoint, config_text="[]")
    with pytest.raises(ValueError, match="config.json must be a JSON object"):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


def test_serialization_zip_loader_rejects_missing_members_and_bad_zip(
    tmp_path: Path,
) -> None:
    """The direct zip loader reports missing members and malformed archives."""
    incomplete = tmp_path / "incomplete.kgckpt"
    with zipfile.ZipFile(incomplete, mode="w") as archive:
        archive.writestr(serialization_mod.SAFE_META_FILENAME, "{}")
    with pytest.raises(ValueError, match="missing members"):
        serialization_mod._load_safetensors_v1_zip(
            incomplete,
            map_location=None,
        )

    malformed = tmp_path / "malformed.kgckpt"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid safetensors_v1 zip"):
        serialization_mod._load_safetensors_v1_zip(
            malformed,
            map_location=None,
        )


def test_serialization_load_rejects_unsupported_path_type() -> None:
    """An existing path that is neither file nor directory is rejected."""
    destination = MagicMock()
    destination.exists.return_value = True
    destination.is_dir.return_value = False
    destination.is_file.return_value = False
    with (
        patch.object(serialization_mod, "Path", return_value=destination),
        pytest.raises(ValueError, match="Unsupported checkpoint path type"),
    ):
        serialization_mod.load_checkpoint("special-device")


def test_tensorboard_summary_writer_import_success() -> None:
    """The TensorBoard peer import returns SummaryWriter when available."""
    writer_cls = object()
    fake_module = ModuleType("torch.utils.tensorboard")
    fake_module.SummaryWriter = writer_cls  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"torch.utils.tensorboard": fake_module}):
        assert tensorboard_mod._require_summary_writer() is writer_cls


def test_tensorboard_logger_lifecycle_with_mocked_writer(tmp_path: Path) -> None:
    """Mocked TensorBoard logging covers start, train/val, flush, and close."""
    writer = MagicMock()
    writer_cls = MagicMock(return_value=writer)
    logger = tensorboard_mod.TensorBoardFitLogger(
        tmp_path / "events",
        comment="coverage",
    )

    with pytest.raises(RuntimeError, match="before on_fit_start"):
        logger.on_epoch_end(
            epoch=0,
            train_breakdown=_breakdown(),
            val_breakdown=None,
            history_so_far=None,
        )

    with patch.object(
        tensorboard_mod,
        "_require_summary_writer",
        return_value=writer_cls,
    ):
        logger.on_fit_start(model=object(), fit_kwargs={})
    logger.on_epoch_end(
        epoch=3,
        train_breakdown=_breakdown(),
        val_breakdown=_breakdown(),
        history_so_far=None,
    )
    logger.on_fit_end(history=MagicMock())

    writer_cls.assert_called_once_with(
        log_dir=str(tmp_path / "events"),
        comment="coverage",
    )
    assert writer.add_scalar.call_count == 22
    assert writer.flush.call_count == 2
    writer.close.assert_called_once_with()
    assert logger._writer is None


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


class _NonTensorForecast:
    """Model-shaped callable that returns a non-tensor forecast."""

    def __call__(self, data: Data) -> dict[str, Data]:
        return {"decoded": data}


def test_explain_wrapper_rejects_non_tensor_forecast() -> None:
    """One-step explanation requires a tensor model decode."""
    wrapper = explain_mod._HomogeneousExplainModule(
        _NonTensorForecast(),  # type: ignore[arg-type]
        target="one_step_forecast",
    )
    with pytest.raises(TypeError, match="requires a tensor decode"):
        wrapper(
            torch.ones(2, 2),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_unsupported_explain_model(control_dim=1), "control_dim"),
        (
            _unsupported_explain_model(uses_hetero_koopman=True),
            "RelGraph / hetero_graph",
        ),
        (
            _unsupported_explain_model(uses_hypergraph_koopman=True),
            "hypergraph",
        ),
        (
            _unsupported_explain_model(adaptive_topology=object()),
            "adaptive / learned",
        ),
    ],
)
def test_explain_rejects_unsupported_model_modes(
    model: SimpleNamespace,
    message: str,
) -> None:
    """Explain-MVP validation covers control, hetero, hypergraph, and adaptive."""
    with pytest.raises(ValueError, match=message):
        explain_mod._reject_unsupported_model(model)  # type: ignore[arg-type]


def test_explain_rejects_invalid_data_inputs() -> None:
    """Explain data validation rejects non-Data and missing graph fields."""
    with pytest.raises(TypeError, match="torch_geometric.data.Data"):
        explain_mod._reject_unsupported_data(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Data.x and Data.edge_index"):
        explain_mod._reject_unsupported_data(Data())


@pytest.mark.parametrize(
    ("epochs", "lr", "message"),
    [(0, 0.01, "epochs"), (1, 0.0, "lr")],
)
def test_explain_gnn_explainer_validates_hyperparameters(
    epochs: int,
    lr: float,
    message: str,
) -> None:
    """GNNExplainer rejects invalid epochs and learning rates."""
    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match=message):
        explain_mod._run_gnn_explainer(
            MagicMock(),
            data,
            target="latent",
            index=None,
            epochs=epochs,
            lr=lr,
        )


def test_explain_captum_import_error_hint() -> None:
    """The direct Captum importer adds the optional-extra install hint."""
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "captum":
            raise ImportError("simulated missing captum")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=blocked_import),
        pytest.raises(ImportError, match=r"koopman-graph\[explain\]"),
    ):
        explain_mod._import_captum_stack()


def test_explain_integrated_gradients_rejects_negative_index() -> None:
    """Integrated gradients validates the selected node index."""
    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with (
        patch.object(
            explain_mod,
            "_import_captum_stack",
            return_value=(MagicMock(), MagicMock(), MagicMock()),
        ),
        pytest.raises(ValueError, match="index must be >= 0"),
    ):
        explain_mod._run_integrated_gradients(
            nn.Identity(),  # type: ignore[arg-type]
            data,
            target="latent",
            index=-1,
        )


def test_explain_integrated_gradients_discards_non_tensor_edge_mask() -> None:
    """Integrated gradients normalizes a non-tensor edge mask to None."""

    class FakeExplainer:
        """Return a deterministic explanation without invoking the wrapper."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            *,
            index: int,
        ) -> SimpleNamespace:
            del edge_index, index
            return SimpleNamespace(node_mask=torch.ones_like(x), edge_mask=object())

    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with patch.object(
        explain_mod,
        "_import_captum_stack",
        return_value=(MagicMock(), FakeExplainer, MagicMock()),
    ):
        result = explain_mod._run_integrated_gradients(
            nn.Identity(),  # type: ignore[arg-type]
            data,
            target="latent",
            index=1,
        )
    assert result.edge_mask is None
    assert result.node_mask is not None


def test_predict_sequence_loader_covers_config_and_file_branches(
    tmp_path: Path,
) -> None:
    """Prediction data loading covers config, cache, and suffix validation."""
    config_path = tmp_path / "data.json"
    config_path.write_text("{}", encoding="utf-8")
    sentinel = _snapshot_sequence()

    with (
        patch.object(predict_mod, "load_config", return_value={"data": []}),
        pytest.raises(config_mod.ConfigError, match="config.data must be a mapping"),
    ):
        predict_mod.load_sequence_from_path(config_path)

    with (
        patch.object(
            predict_mod,
            "load_config",
            return_value={"kind": "synthetic_path"},
        ),
        patch.object(
            predict_mod,
            "build_sequence_from_config",
            return_value=sentinel,
        ),
    ):
        assert predict_mod.load_sequence_from_path(config_path) is sentinel

    with (
        patch.object(predict_mod, "load_config", return_value={"other": "value"}),
        pytest.raises(config_mod.ConfigError, match="must contain a 'data' section"),
    ):
        predict_mod.load_sequence_from_path(config_path)

    cache_path = tmp_path / "data.pt"
    cache_path.write_bytes(b"placeholder")
    with patch.object(
        predict_mod,
        "load_cached_sequence",
        return_value=sentinel,
    ) as cached_loader:
        assert predict_mod.load_sequence_from_path(cache_path) is sentinel
    cached_loader.assert_called_once()

    unsupported = tmp_path / "data.csv"
    unsupported.write_text("x\n", encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="Unsupported data path suffix"):
        predict_mod.load_sequence_from_path(unsupported)

    with pytest.raises(config_mod.ConfigError, match="Data path not found"):
        predict_mod.load_sequence_from_path(tmp_path / "missing.pt")


def test_predict_run_rejects_invalid_steps_empty_data_and_empty_forecast(
    tmp_path: Path,
) -> None:
    """Prediction validates horizon, input sequence, and forecast output."""
    with pytest.raises(config_mod.ConfigError, match="--steps must be >= 1"):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=0,
            out=tmp_path / "out.pt",
        )

    with (
        patch.object(predict_mod, "load_checkpoint", return_value=MagicMock()),
        patch.object(predict_mod, "load_sequence_from_path", return_value=[]),
        pytest.raises(config_mod.ConfigError, match="Data sequence is empty"),
    ):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=1,
            out=tmp_path / "out.pt",
        )

    model = MagicMock()
    model.predict.return_value = []
    with (
        patch.object(predict_mod, "load_checkpoint", return_value=model),
        patch.object(
            predict_mod,
            "load_sequence_from_path",
            return_value=[Data(x=torch.ones(1, 1))],
        ),
        pytest.raises(config_mod.ConfigError, match="empty forecast"),
    ):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=1,
            out=tmp_path / "out.pt",
        )


def test_predict_handler_prints_path_without_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The predict handler reports successful forecasts without shape metadata."""
    output = tmp_path / "forecast.pt"
    args = SimpleNamespace(
        checkpoint="model",
        data="data",
        steps=2,
        out=output,
    )
    with patch.object(predict_mod, "run_predict", return_value=(output, None)):
        assert predict_mod.handle_predict(args) == 0  # type: ignore[arg-type]
    assert f"wrote forecast: {output} (steps=2)" in capsys.readouterr().out


def test_ray_import_helpers_cover_success_failure_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray helpers set the uv flag and wrap lazy-import failures."""
    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)
    ray_tune_mod._prefer_driver_interpreter_for_uv_run()
    assert sys.modules["os"].environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"

    fake_ray = ModuleType("ray")
    fake_tune = ModuleType("ray.tune")
    fake_run_config = object()
    fake_train = SimpleNamespace(RunConfig=fake_run_config)

    def import_ray(name: str) -> object:
        return {
            "ray": fake_ray,
            "ray.tune": fake_tune,
            "ray.train": fake_train,
        }[name]

    with patch.object(
        ray_tune_mod.importlib,
        "import_module",
        side_effect=import_ray,
    ):
        assert ray_tune_mod._import_ray_tune() == (
            fake_ray,
            fake_tune,
            fake_run_config,
        )

    with (
        patch.object(
            ray_tune_mod.importlib,
            "import_module",
            side_effect=ImportError("missing ray"),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        ray_tune_mod._import_ray_tune()


def test_ray_tune_initializes_uninitialized_runtime() -> None:
    """The wrapper initializes Ray with caller-provided kwargs when needed."""
    fake_ray = ModuleType("ray")
    fake_ray.is_initialized = MagicMock(return_value=False)  # type: ignore[attr-defined]
    fake_ray.init = MagicMock()  # type: ignore[attr-defined]
    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = "results"
    fake_tune = ModuleType("ray.tune")
    fake_tune.TuneConfig = MagicMock(return_value=object())  # type: ignore[attr-defined]
    fake_tune.Tuner = MagicMock(return_value=fake_tuner)  # type: ignore[attr-defined]

    with patch.object(
        ray_tune_mod,
        "_import_ray_tune",
        return_value=(fake_ray, fake_tune, MagicMock()),
    ):
        result = ray_tune_mod.run_ray_tune(
            lambda config: config,
            {"lr": 0.01},
            ray_init_kwargs={"local_mode": True},
        )
    assert result == "results"
    fake_ray.init.assert_called_once_with(local_mode=True)


def test_search_space_import_helper_success_and_failure() -> None:
    """The Ray Tune search-space importer handles both lazy-import outcomes."""
    fake_tune = ModuleType("ray.tune")
    with patch.object(
        search_spaces_mod.importlib,
        "import_module",
        return_value=fake_tune,
    ):
        assert search_spaces_mod._import_tune() is fake_tune

    with (
        patch.object(
            search_spaces_mod.importlib,
            "import_module",
            side_effect=ImportError("missing ray"),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        search_spaces_mod._import_tune()


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            search_spaces_mod.example_lr_loguniform_space,
            {"lr_low": 0.0},
            "must be positive",
        ),
        (
            search_spaces_mod.example_lr_latent_dim_space,
            {"latent_dims": ()},
            "at least one",
        ),
        (
            search_spaces_mod.example_lr_latent_dim_space,
            {"latent_dims": (2, 0)},
            "entries must be >= 1",
        ),
    ],
)
def test_search_space_validates_positive_domains(
    function: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Example search spaces reject non-positive and empty domains."""
    with pytest.raises(ValueError, match=message):
        function(**kwargs)  # type: ignore[operator]


def test_cli_entry_points_cover_help_and_main_guards(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Direct and module entry points print help and exit successfully."""
    assert cli_main_mod.main([]) == 0
    assert "usage: koopman-graph" in capsys.readouterr().out

    main_path = Path(cli_main_mod.__file__)
    with (
        patch.object(sys, "argv", [str(main_path)]),
        pytest.raises(SystemExit) as main_exit,
    ):
        runpy.run_path(str(main_path), run_name="__main__")
    assert main_exit.value.code == 0

    module_path = main_path.with_name("__main__.py")
    with (
        patch.object(sys, "argv", [str(module_path)]),
        pytest.raises(SystemExit) as module_exit,
    ):
        runpy.run_module("koopman_graph.cli", run_name="__main__", alter_sys=True)
    assert module_exit.value.code == 0


def test_csv_logger_rejects_epoch_before_start(tmp_path: Path) -> None:
    """CSV logging requires on_fit_start before the first epoch."""
    logger = CsvFitLogger(tmp_path / "fit.csv")
    with pytest.raises(RuntimeError, match="before on_fit_start"):
        logger.on_epoch_end(
            epoch=0,
            train_breakdown=_breakdown(),
            val_breakdown=None,
            history_so_far=None,
        )


def test_lightning_export_forces_legacy_checkpoint_format(tmp_path: Path) -> None:
    """Lightning's Trainer hook exports an unwrapped legacy checkpoint."""
    fake_lightning = SimpleNamespace(LightningModule=nn.Module)
    with patch.object(lightning_mod, "_import_lightning", return_value=fake_lightning):
        module_cls = lightning_mod._build_koopman_lightning_module()
    model = nn.Linear(2, 2)
    module = module_cls(model)
    destination = tmp_path / "lightning.pt"

    with patch.object(serialization_mod, "save_checkpoint") as save_checkpoint:
        module.export_format1_checkpoint(destination)
    save_checkpoint.assert_called_once_with(
        model,
        destination,
        format="legacy_pt",
    )
