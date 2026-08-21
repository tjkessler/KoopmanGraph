"""Tests for GraphKoopmanModel checkpoint serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphTransformerDecoder,
    GraphTransformerEncoder,
    SAGEDecoder,
    SAGEEncoder,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.serialization import (
    FORMAT_VERSION,
    SAFE_CONFIG_FILENAME,
    SAFE_CONTAINER,
    SAFE_META_FILENAME,
    SAFE_WEIGHTS_FILENAME,
    build_checkpoint,
    build_model_config,
    load_checkpoint,
    reconstruct_model,
    save_checkpoint,
)


def _format1_config(
    *,
    encoder: dict,
    decoder: dict,
    **overrides: object,
) -> dict:
    """Build a minimal current format-1 config for hand-written checkpoint fixtures."""
    config = {
        "latent_dim": 4,
        "time_step": 0.1,
        "dynamics_mode": "discrete",
        "koopman_kind": "pernode",
        "koopman_init_mode": "identity_noise",
        "koopman_init_scale": 1e-2,
        "koopman_parameterization": "dense",
        "koopman_max_spectral_radius": 1.0,
        "control_dim": 0,
        "control_mode": "additive",
        "bilinear_rank": None,
        "n_delays": 1,
        "physics": None,
        "encoder": encoder,
        "decoder": decoder,
        "sparsity": "dense",
        "adjacency": None,
        "learn_topology": None,
        "topology_embedding_dim": None,
        "symmetry": None,
        "local_window": None,
        "local_rank": None,
        "local_hidden_dims": None,
    }
    config.update(overrides)
    return config


@pytest.fixture
def graph_koopman_model() -> GraphKoopmanModel:
    """Provide a small GraphKoopmanModel for serialization tests."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )


@pytest.fixture
def trained_gcn_model(
    scaling_sequence: GraphSnapshotSequence,
) -> GraphKoopmanModel:
    """Provide a briefly trained GCN GraphKoopmanModel."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )
    torch.manual_seed(0)
    model.fit(scaling_sequence, epochs=3, lr=1e-2)
    return model


def _predictions(model: GraphKoopmanModel, initial_graph: Data) -> list[torch.Tensor]:
    """Return detached prediction tensors for comparison."""
    return [graph.x.detach().clone() for graph in model.predict(initial_graph, steps=3)]


def test_build_model_config_captures_gcn_architecture(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Verify GCN encoder hyperparameters are serialized."""
    config = build_model_config(graph_koopman_model)
    assert config["encoder"]["type"] == "gcn"
    assert config["encoder"]["in_channels"] == 3
    assert config["encoder"]["hidden_channels"] == 8
    assert config["decoder"]["out_channels"] == 3
    assert config["latent_dim"] == 4
    assert config["time_step"] == 0.1
    assert config["koopman_parameterization"] == "dense"
    assert config["koopman_max_spectral_radius"] == 1.0


def test_build_model_config_captures_gat_hyperparameters() -> None:
    """Verify GAT-only settings are included in encoder config."""
    encoder = GATEncoder(
        in_channels=2,
        hidden_channels=16,
        latent_dim=8,
        heads=2,
        dropout=0.1,
    )
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.05,
    )
    config = build_model_config(model)
    assert config["encoder"]["type"] == "gat"
    assert config["encoder"]["heads"] == 2
    assert config["encoder"]["dropout"] == 0.1
    assert config["decoder"]["type"] == "gcn"


def test_build_model_config_captures_gat_decoder_hyperparameters() -> None:
    """Verify GAT decoder settings are included in decoder config."""
    encoder = GATEncoder(in_channels=2, hidden_channels=16, latent_dim=8, heads=2)
    decoder = GATDecoder(
        latent_dim=8,
        hidden_channels=16,
        out_channels=2,
        heads=2,
        dropout=0.1,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.05,
    )
    config = build_model_config(model)
    assert config["decoder"]["type"] == "gat"
    assert config["decoder"]["heads"] == 2
    assert config["decoder"]["dropout"] == 0.1
    rebuilt = reconstruct_model(config)
    assert isinstance(rebuilt.decoder, GATDecoder)
    assert rebuilt.decoder.heads == 2
    assert rebuilt.decoder.dropout == 0.1


def test_reconstruct_model_defaults_missing_decoder_type_to_gcn(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Verify checkpoints without decoder type still rebuild a GCN decoder."""
    config = build_model_config(graph_koopman_model)
    del config["decoder"]["type"]
    rebuilt = reconstruct_model(config)
    assert isinstance(rebuilt.decoder, GNNDecoder)


def test_reconstruct_model_matches_original_architecture(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Verify config reconstruction preserves architecture attributes."""
    config = build_model_config(graph_koopman_model)
    rebuilt = reconstruct_model(config)
    assert isinstance(rebuilt.encoder, GNNEncoder)
    assert rebuilt.encoder.in_channels == graph_koopman_model.encoder.in_channels
    assert rebuilt.decoder.out_channels == graph_koopman_model.decoder.out_channels
    assert rebuilt.latent_dim == graph_koopman_model.latent_dim
    assert rebuilt.time_step == graph_koopman_model.time_step


def test_save_load_round_trip_predictions_match(
    trained_gcn_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify save → load preserves prediction outputs."""
    initial_graph = scaling_sequence[0]
    original_preds = _predictions(trained_gcn_model, initial_graph)

    checkpoint = tmp_path / "model.pt"
    trained_gcn_model.save(checkpoint)
    loaded = GraphKoopmanModel.load(checkpoint)
    loaded_preds = _predictions(loaded, initial_graph)

    assert len(loaded_preds) == len(original_preds)
    for original, loaded_pred in zip(original_preds, loaded_preds, strict=True):
        torch.testing.assert_close(original, loaded_pred)


def test_gat_save_load_round_trip(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify GAT encoder models serialize and reload correctly."""
    encoder = GATEncoder(in_channels=3, hidden_channels=16, latent_dim=8, heads=2)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )
    torch.manual_seed(1)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)

    initial_graph = scaling_sequence[0]
    original_preds = _predictions(model, initial_graph)
    checkpoint = tmp_path / "gat_model.pt"
    model.save(checkpoint)
    loaded = GraphKoopmanModel.load(checkpoint)
    loaded_preds = _predictions(loaded, initial_graph)

    for original, loaded_pred in zip(original_preds, loaded_preds, strict=True):
        torch.testing.assert_close(original, loaded_pred)


def test_gat_encoder_decoder_save_load_round_trip(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify paired GAT encoder/decoder models serialize and reload."""
    encoder = GATEncoder(in_channels=3, hidden_channels=16, latent_dim=8, heads=2)
    decoder = GATDecoder(latent_dim=8, hidden_channels=16, out_channels=3, heads=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )
    torch.manual_seed(2)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)

    initial_graph = scaling_sequence[0]
    original_preds = _predictions(model, initial_graph)
    checkpoint = tmp_path / "gat_pair_model.pt"
    model.save(checkpoint)
    loaded = GraphKoopmanModel.load(checkpoint)
    assert isinstance(loaded.decoder, GATDecoder)
    assert loaded.decoder.heads == 2
    loaded_preds = _predictions(loaded, initial_graph)

    for original, loaded_pred in zip(original_preds, loaded_preds, strict=True):
        torch.testing.assert_close(original, loaded_pred)


def test_save_checkpoint_uses_current_format_version(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Verify new checkpoints are saved with the current format version."""
    checkpoint = build_checkpoint(graph_koopman_model)
    assert checkpoint["format_version"] == FORMAT_VERSION
    assert FORMAT_VERSION == 1
    assert set(checkpoint["config"]) >= {
        "dynamics_mode",
        "koopman_kind",
        "control_dim",
        "control_mode",
        "bilinear_rank",
        "n_delays",
        "physics",
        "sparsity",
        "adjacency",
        "learn_topology",
        "topology_embedding_dim",
        "graph_dynamics",
        "symmetry",
        "local_window",
        "local_rank",
        "local_hidden_dims",
    }


def test_save_checkpoint_safetensors_v1_directory_layout(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify safetensors_v1 writes meta/config/weights as a directory."""
    assert FORMAT_VERSION == 1
    destination = tmp_path / "safe_ckpt"
    save_checkpoint(graph_koopman_model, destination, format="safetensors_v1")

    meta_path = destination / SAFE_META_FILENAME
    config_path = destination / SAFE_CONFIG_FILENAME
    weights_path = destination / SAFE_WEIGHTS_FILENAME
    assert meta_path.is_file()
    assert config_path.is_file()
    assert weights_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["container"] == SAFE_CONTAINER
    assert meta["format_version"] == FORMAT_VERSION
    assert isinstance(meta["package_version"], str)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config == json.loads(json.dumps(config))
    assert config["latent_dim"] == graph_koopman_model.latent_dim
    assert config["encoder"]["type"] == "gcn"
    assert config["decoder"]["type"] == "gcn"


def test_save_checkpoint_safetensors_v1_rejects_existing_file(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify safetensors_v1 refuses to treat an existing file as a directory."""
    file_path = tmp_path / "not_a_dir.pt"
    file_path.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="must be a directory"):
        save_checkpoint(graph_koopman_model, file_path, format="safetensors_v1")


def test_save_checkpoint_default_is_safetensors_v1(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify default save_checkpoint writes a safetensors_v1 directory."""
    destination = tmp_path / "default_safe"
    save_checkpoint(graph_koopman_model, destination)
    assert destination.is_dir()
    assert (destination / SAFE_META_FILENAME).is_file()
    assert (destination / SAFE_CONFIG_FILENAME).is_file()
    assert (destination / SAFE_WEIGHTS_FILENAME).is_file()
    loaded = load_checkpoint(destination)
    assert not loaded.training


def test_save_checkpoint_legacy_pt_escape_hatch(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify format='legacy_pt' still writes a pickle .pt file."""
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(graph_koopman_model, checkpoint, format="legacy_pt")
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == FORMAT_VERSION
    assert isinstance(payload["config"], dict)
    assert isinstance(payload["state_dict"], dict)


def test_model_save_default_is_safetensors_v1(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify GraphKoopmanModel.save defaults to safetensors_v1."""
    destination = tmp_path / "model_safe"
    graph_koopman_model.save(destination)
    assert destination.is_dir()
    assert (destination / SAFE_META_FILENAME).is_file()
    loaded = GraphKoopmanModel.load(destination)
    assert isinstance(loaded, GraphKoopmanModel)
    assert not loaded.training


def test_model_save_legacy_pt_escape_and_load(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify Model.save(..., format='legacy_pt') round-trips via load."""
    checkpoint = tmp_path / "model_legacy.pt"
    graph_koopman_model.save(checkpoint, format="legacy_pt")
    assert checkpoint.is_file()
    loaded = GraphKoopmanModel.load(checkpoint)
    assert isinstance(loaded, GraphKoopmanModel)


def test_load_checkpoint_safetensors_v1_round_trip(
    trained_gcn_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify safetensors_v1 save → load preserves weights and predictions."""
    destination = tmp_path / "safe_ckpt"
    save_checkpoint(trained_gcn_model, destination, format="safetensors_v1")
    loaded = load_checkpoint(destination)
    assert not loaded.training

    original = trained_gcn_model.state_dict()
    restored = loaded.state_dict()
    assert set(original) == set(restored)
    for key in original:
        torch.testing.assert_close(restored[key], original[key])

    initial_graph = scaling_sequence[0]
    for original_pred, loaded_pred in zip(
        _predictions(trained_gcn_model, initial_graph),
        _predictions(loaded, initial_graph),
        strict=True,
    ):
        torch.testing.assert_close(original_pred, loaded_pred)


def test_load_checkpoint_safetensors_v1_missing_meta_raises(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify directories without meta.json are rejected."""
    destination = tmp_path / "broken"
    save_checkpoint(graph_koopman_model, destination, format="safetensors_v1")
    (destination / SAFE_META_FILENAME).unlink()
    with pytest.raises(ValueError, match="missing meta.json"):
        load_checkpoint(destination)


def test_load_checkpoint_safetensors_v1_missing_weights_raises(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify directories without model.safetensors are rejected."""
    destination = tmp_path / "broken_weights"
    save_checkpoint(graph_koopman_model, destination, format="safetensors_v1")
    (destination / SAFE_WEIGHTS_FILENAME).unlink()
    with pytest.raises(ValueError, match="missing model.safetensors"):
        load_checkpoint(destination)


def test_load_checkpoint_safetensors_v1_bad_container_raises(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify unsupported container markers are rejected."""
    destination = tmp_path / "bad_container"
    save_checkpoint(graph_koopman_model, destination, format="safetensors_v1")
    meta_path = destination / SAFE_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["container"] = "not_a_real_container"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported safetensors container"):
        load_checkpoint(destination)


def test_load_checkpoint_safetensors_v1_invalid_config_raises(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify truncated config.json raises an actionable ValueError."""
    destination = tmp_path / "bad_config"
    save_checkpoint(graph_koopman_model, destination, format="safetensors_v1")
    (destination / SAFE_CONFIG_FILENAME).write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid config.json"):
        load_checkpoint(destination)


def test_load_retired_format2_checkpoint_rejected(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify format_version 2 stays unsupported in the beta (format 1 only)."""
    path = tmp_path / "retired_v2.pt"
    torch.save(
        {
            "format_version": 2,
            "package_version": "0.4.0",
            "config": build_model_config(graph_koopman_model),
            "state_dict": graph_koopman_model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
        load_checkpoint(path)


def test_load_sparse_historical_format1_checkpoint_rejected(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify legacy sparse format_version 1 payloads are rejected."""
    sparse_config = {
        "latent_dim": 4,
        "time_step": 0.1,
        "koopman_init_mode": "identity",
        "koopman_init_scale": 1e-2,
        "encoder": {
            "type": "gcn",
            "in_channels": 3,
            "hidden_channels": 8,
            "latent_dim": 4,
            "num_layers": 2,
            "activation": "relu",
        },
        "decoder": {
            "latent_dim": 4,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "activation": "relu",
        },
    }
    path = tmp_path / "sparse_v1.pt"
    torch.save(
        {
            "format_version": 1,
            "package_version": "0.2.0",
            "config": sparse_config,
            "state_dict": graph_koopman_model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="Deprecated checkpoint schema"):
        load_checkpoint(path)


def test_load_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    """Verify missing checkpoint paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_invalid_format_version(tmp_path: Path) -> None:
    """Verify unsupported format versions raise ValueError."""
    path = tmp_path / "bad_version.pt"
    torch.save({"format_version": 99, "config": {}, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
        load_checkpoint(path)


def test_load_checkpoint_unsupported_encoder_type(tmp_path: Path) -> None:
    """Verify unknown encoder types raise ValueError."""
    config = _format1_config(
        encoder={
            "type": "unknown",
            "in_channels": 3,
            "hidden_channels": 8,
            "latent_dim": 4,
            "num_layers": 2,
            "activation": "relu",
        },
        decoder={
            "latent_dim": 4,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "activation": "relu",
        },
    )
    path = tmp_path / "bad_encoder.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "package_version": "0.1.0",
            "config": config,
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported encoder type"):
        load_checkpoint(path)


def test_load_checkpoint_unsupported_decoder_type(tmp_path: Path) -> None:
    """Verify unknown decoder types raise ValueError."""
    config = _format1_config(
        encoder={
            "type": "gcn",
            "in_channels": 3,
            "hidden_channels": 8,
            "latent_dim": 4,
            "num_layers": 2,
            "activation": "relu",
        },
        decoder={
            "type": "unknown",
            "latent_dim": 4,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "activation": "relu",
        },
    )
    path = tmp_path / "bad_decoder.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "package_version": "0.1.0",
            "config": config,
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported decoder type"):
        load_checkpoint(path)


def test_fit_restore_best_weights_reloads_lowest_loss_epoch(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify restore_best_weights keeps the best-epoch parameters."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )

    from koopman_graph.training.history import TrainingLossBreakdown

    best_state: dict[str, torch.Tensor] | None = None
    losses: list[float] = []

    def fake_train_one_epoch(*_args, **_kwargs) -> TrainingLossBreakdown:
        nonlocal best_state
        loss = float(len(losses) + 1)
        losses.append(loss)
        for param in model.parameters():
            param.data.add_(0.01)
        if loss == 1.0:
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        scalar = torch.tensor(loss)
        return TrainingLossBreakdown(
            reconstruction=scalar,
            forward=scalar,
            backward=scalar,
            rollout=scalar,
            eigenvalue=scalar,
            total=scalar,
        )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "koopman_graph.training.loop.train_one_epoch",
            fake_train_one_epoch,
        )
        history = model.fit(
            scaling_sequence,
            epochs=3,
            restore_best_weights=True,
        )

    assert history.best_epoch == 0
    assert history.best_loss == 1.0
    assert best_state is not None
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, best_state[key])


def test_fit_checkpoint_path_writes_loadable_best_epoch_checkpoint(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify checkpoint_path writes a loadable best-epoch checkpoint."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )
    torch.manual_seed(2)
    checkpoint = tmp_path / "best.pt"
    history = model.fit(
        scaling_sequence,
        epochs=4,
        lr=1e-2,
        checkpoint_path=checkpoint,
    )

    assert checkpoint.is_file()
    assert history.best_epoch is not None
    loaded = GraphKoopmanModel.load(checkpoint)
    assert isinstance(loaded.encoder, GNNEncoder)


def test_load_checkpoint_invalid_payload_raises(tmp_path: Path) -> None:
    """Verify non-dictionary checkpoints raise ValueError."""
    path = tmp_path / "bad_payload.pt"
    torch.save(["not", "a", "dict"], path)
    with pytest.raises(ValueError, match="dictionary payload"):
        load_checkpoint(path)


def test_load_checkpoint_missing_config_or_state_dict_raises(tmp_path: Path) -> None:
    """Verify checkpoints missing config/state_dict raise ValueError."""
    path = tmp_path / "missing_keys.pt"
    torch.save({"format_version": FORMAT_VERSION, "config": "bad"}, path)
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(path)


def test_save_checkpoint_creates_parent_directories(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Verify save creates nested parent directories."""
    path = tmp_path / "nested" / "dir" / "model_ckpt"
    save_checkpoint(graph_koopman_model, path)
    assert path.is_dir()
    assert (path / SAFE_WEIGHTS_FILENAME).is_file()


def test_odo_model_round_trip_preserves_predictions(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify ODO operator settings survive save/load round trip."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
        koopman_parameterization="odo",
        koopman_max_spectral_radius=0.85,
    )
    torch.manual_seed(1)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)
    before = _predictions(model, scaling_sequence[0])

    path = tmp_path / "odo_model.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.parameterization == "odo"
    assert loaded.koopman.max_spectral_radius == 0.85
    after = _predictions(loaded, scaling_sequence[0])
    for pred_before, pred_after in zip(before, after, strict=True):
        assert torch.allclose(pred_before, pred_after)


def test_lyapunov_model_round_trip_preserves_predictions(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Verify Lyapunov structural parameterization survives save/load."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
        koopman_parameterization="lyapunov",
    )
    torch.manual_seed(2)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)
    before = _predictions(model, scaling_sequence[0])

    path = tmp_path / "lyapunov_model.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.parameterization == "lyapunov"
    after = _predictions(loaded, scaling_sequence[0])
    for pred_before, pred_after in zip(before, after, strict=True):
        assert torch.allclose(pred_before, pred_after)


def test_package_version_falls_back_when_metadata_missing() -> None:
    """Verify the version helper returns a fallback without package metadata."""
    from importlib.metadata import PackageNotFoundError
    from unittest.mock import patch

    from koopman_graph.serialization import _package_version

    with patch(
        "koopman_graph.serialization.version",
        side_effect=PackageNotFoundError,
    ):
        assert _package_version() == "0.0.0"


def test_encoder_type_rejects_unsupported_encoder() -> None:
    """Verify unsupported encoder instances raise ``TypeError``."""
    from torch import nn

    from koopman_graph.serialization import _encoder_type

    with pytest.raises(TypeError, match="Unsupported encoder type"):
        _encoder_type(nn.Linear(3, 4))  # type: ignore[arg-type]


def test_decoder_type_rejects_unsupported_decoder() -> None:
    """Verify unsupported decoder instances raise ``TypeError``."""
    from torch import nn

    from koopman_graph.serialization import _decoder_type

    with pytest.raises(TypeError, match="Unsupported decoder type"):
        _decoder_type(nn.Linear(3, 4))  # type: ignore[arg-type]


def test_build_model_config_rejects_custom_injected_operator() -> None:
    """Verify custom injected operators are not checkpoint-serializable."""
    from torch import nn

    class _CustomOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = 4
            self.control_dim = 0
            self.parameterization = "dense"
            self._matrix = nn.Parameter(torch.eye(4))

        @property
        def matrix(self) -> torch.Tensor:
            return self._matrix

        def advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del delta_t, control
            return z @ self._matrix.T

        def inverse_advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
            inverse_matrix: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del delta_t, control, inverse_matrix
            return z

        def bound_metric(self) -> torch.Tensor:
            return torch.tensor(1.0)

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=_CustomOperator(),
    )
    with pytest.raises(TypeError, match="not round-trippable"):
        build_model_config(model)


def test_injected_builtin_operator_remains_serializable() -> None:
    """Verify injecting a built-in operator still serializes factory metadata."""
    from koopman_graph.operators import KoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = KoopmanOperator(4, parameterization="dense")
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=operator,
    )
    config = build_model_config(model)
    assert config["koopman_parameterization"] == "dense"
    assert config["dynamics_mode"] == "discrete"


def test_sage_encoder_decoder_checkpoint_round_trip(tmp_path: Path) -> None:
    """Verify SAGE encoder/decoder type strings round-trip through save/load."""
    model = GraphKoopmanModel(
        encoder=SAGEEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=SAGEDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
    )
    config = build_model_config(model)
    assert config["encoder"]["type"] == "sage"
    assert config["decoder"]["type"] == "sage"
    path = tmp_path / "sage.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, SAGEEncoder)
    assert isinstance(loaded.decoder, SAGEDecoder)


def test_diffconv_encoder_decoder_checkpoint_round_trip(tmp_path: Path) -> None:
    """Verify DiffConv type strings and diffusion_steps round-trip."""
    model = GraphKoopmanModel(
        encoder=DiffConvEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=4,
            diffusion_steps=3,
        ),
        decoder=DiffConvDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            diffusion_steps=3,
        ),
        latent_dim=4,
        time_step=0.1,
    )
    config = build_model_config(model)
    assert config["encoder"]["type"] == "diffconv"
    assert config["encoder"]["diffusion_steps"] == 3
    assert config["decoder"]["type"] == "diffconv"
    assert config["decoder"]["diffusion_steps"] == 3
    path = tmp_path / "diffconv.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, DiffConvEncoder)
    assert isinstance(loaded.decoder, DiffConvDecoder)
    assert loaded.encoder.diffusion_steps == 3
    assert loaded.decoder.diffusion_steps == 3


def test_transformer_encoder_decoder_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    """Verify Transformer type strings and attention knobs round-trip."""
    model = GraphKoopmanModel(
        encoder=GraphTransformerEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=4,
            heads=2,
            dropout=0.1,
            edge_dim=1,
        ),
        decoder=GraphTransformerDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            heads=2,
            dropout=0.1,
            edge_dim=1,
        ),
        latent_dim=4,
        time_step=0.1,
    )
    config = build_model_config(model)
    assert config["encoder"]["type"] == "transformer"
    assert config["encoder"]["heads"] == 2
    assert config["encoder"]["dropout"] == 0.1
    assert config["encoder"]["edge_dim"] == 1
    assert config["decoder"]["type"] == "transformer"
    assert config["decoder"]["edge_dim"] == 1
    path = tmp_path / "transformer.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, GraphTransformerEncoder)
    assert isinstance(loaded.decoder, GraphTransformerDecoder)
    assert loaded.encoder.heads == 2
    assert loaded.encoder.edge_dim == 1
    assert loaded.decoder.edge_dim == 1


def test_build_model_config_includes_placeholder_keys(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """New format-1 saves always write sparsity / topology / symmetry keys."""
    config = build_model_config(graph_koopman_model)
    assert config["sparsity"] == "dense"
    assert config["adjacency"] is None
    assert config["filter_degree"] is None
    assert config["learn_topology"] is None
    assert config["graph_dynamics"] is None
    assert config["symmetry"] is None
    assert config["local_window"] is None
    assert config["local_rank"] is None
    assert config["local_hidden_dims"] is None


def test_load_rejects_pre_placeholder_format1_schema(
    graph_koopman_model: GraphKoopmanModel,
    tmp_path: Path,
) -> None:
    """Older complete format-1 payloads missing new keys are deprecated."""
    config = build_model_config(graph_koopman_model)
    del config["sparsity"]
    del config["adjacency"]
    del config["learn_topology"]
    del config["symmetry"]
    path = tmp_path / "pre_placeholder.pt"
    torch.save(
        {
            "format_version": 1,
            "package_version": "0.5.0",
            "config": config,
            "state_dict": graph_koopman_model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="Deprecated checkpoint schema"):
        load_checkpoint(path)


def test_continuous_graph_kind_reconstructs_under_format1() -> None:
    """Format-1 reconstruct accepts kind continuous_graph (no longer reserved)."""
    config = _format1_config(
        encoder={
            "type": "gcn",
            "in_channels": 3,
            "hidden_channels": 8,
            "latent_dim": 4,
            "num_layers": 2,
            "activation": "relu",
        },
        decoder={
            "type": "gcn",
            "latent_dim": 4,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "activation": "relu",
        },
        koopman_kind="continuous_graph",
        dynamics_mode="continuous",
        adjacency="symmetric",
    )
    model = reconstruct_model(config)
    assert model.koopman_kind == "continuous_graph"
    assert model.dynamics_mode == "continuous"
    assert model.koopman.adjacency == "symmetric"


def test_hypergraph_format1_round_trip(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Hypergraph encoder/decoder/operator round-trip under format 1."""
    from koopman_graph.nn import HypergraphDecoder, HypergraphEncoder

    # Build a tiny hyperedge-carrying sequence on the scaling fixture topology.
    edge_index = scaling_sequence[0].edge_index
    hyperedge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    x = torch.stack([snap.x for snap in scaling_sequence.snapshots[:4]])
    sequence = GraphSnapshotSequence.from_arrays(
        x,
        edge_index,
        hyperedge_index=hyperedge_index,
    )
    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(3, 8, 4),
        decoder=HypergraphDecoder(4, 8, 3),
        latent_dim=4,
        time_step=0.1,
        koopman="hypergraph",
        control_dim=1,
        control_mode="bilinear",
        bilinear_rank=1,
    )
    # Attach dummy controls for fit.
    controls = torch.randn(sequence.num_timesteps, 1)
    sequence = GraphSnapshotSequence.from_arrays(
        x,
        edge_index,
        hyperedge_index=hyperedge_index,
        control_inputs=controls,
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["koopman_kind"] == "hypergraph"
    assert config["encoder"]["type"] == "hyper_enc"
    assert config["decoder"]["type"] == "hyper_dec"
    assert config["control_mode"] == "bilinear"
    assert config["bilinear_rank"] == 1
    assert config["sparsity"] == "dense"
    assert config["adjacency"] is None

    path = tmp_path / "hypergraph.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "hypergraph"
    assert isinstance(loaded.encoder, HypergraphEncoder)
    assert isinstance(loaded.decoder, HypergraphDecoder)
    assert loaded.control_mode == "bilinear"
    assert loaded.bilinear_rank == 1
    preds = model.predict(sequence[0], steps=2, controls=[controls[0], controls[1]])
    loaded_preds = loaded.predict(
        sequence[0], steps=2, controls=[controls[0], controls[1]]
    )
    for a, b in zip(preds, loaded_preds, strict=True):
        torch.testing.assert_close(a.x, b.x)


def test_hypergraph_operator_delay_round_trip(tmp_path: Path) -> None:
    """Format-1 round-trip keeps n_delays with GCN + hypergraph operator."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long)
    x = torch.randn(5, 3, 2)
    sequence = GraphSnapshotSequence.from_arrays(
        x,
        edge_index,
        hyperedge_index=hyperedge_index,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=4, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
        koopman="hypergraph",
        n_delays=2,
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["n_delays"] == 2
    assert config["koopman_kind"] == "hypergraph"
    path = tmp_path / "hyper_delay.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.n_delays == 2
    assert loaded.koopman_kind == "hypergraph"


def _tiny_networked_model(
    *,
    adjacency: str,
    dynamics_mode: str = "discrete",
    seed: int = 0,
) -> GraphKoopmanModel:
    """Build a small networked model for adjacency serialization tests."""
    torch.manual_seed(seed)
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=3, num_layers=1)
    decoder = GNNDecoder(latent_dim=3, hidden_channels=4, out_channels=2, num_layers=1)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=0.1,
        dynamics_mode=dynamics_mode,  # type: ignore[arg-type]
        koopman="graph",
        koopman_adjacency=adjacency,  # type: ignore[arg-type]
    )


def test_adjacency_modes_round_trip_preserve_factors_and_predictions(
    tmp_path: Path,
) -> None:
    """Save/load preserves adjacency mode, factors, and predictions for all modes."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(x=torch.randn(3, 2), edge_index=edge_index)
    for adjacency in ("symmetric", "random_walk", "dual_random_walk"):
        for dynamics_mode in ("discrete", "continuous"):
            model = _tiny_networked_model(
                adjacency=adjacency,
                dynamics_mode=dynamics_mode,
                seed=7,
            )
            if adjacency == "dual_random_walk":
                if dynamics_mode == "discrete":
                    model.koopman.set_dense_matrices(
                        0.5 * torch.eye(3),
                        0.1 * torch.eye(3),
                        k_bwd=0.05 * torch.eye(3),
                    )
                else:
                    model.koopman.set_dense_matrices(
                        -0.5 * torch.eye(3),
                        0.1 * torch.eye(3),
                        l_bwd=0.05 * torch.eye(3),
                    )
            path = tmp_path / f"{dynamics_mode}_{adjacency}.pt"
            model.eval()
            with torch.no_grad():
                before = model.predict(data, steps=2)
            config = build_model_config(model)
            assert config["adjacency"] == adjacency
            assert FORMAT_VERSION == 1
            model.save(path)
            loaded = GraphKoopmanModel.load(path)
            assert loaded.koopman.adjacency == adjacency
            if dynamics_mode == "discrete":
                assert torch.allclose(loaded.koopman.K_self, model.koopman.K_self)
                assert torch.allclose(loaded.koopman.K_nbr, model.koopman.K_nbr)
                if adjacency == "dual_random_walk":
                    assert torch.allclose(loaded.koopman.K_bwd, model.koopman.K_bwd)
                    assert any("._bwd." in key for key in loaded.state_dict())
                else:
                    assert not any("._bwd." in key for key in loaded.state_dict())
            else:
                assert torch.allclose(loaded.koopman.L_self, model.koopman.L_self)
                assert torch.allclose(loaded.koopman.L_nbr, model.koopman.L_nbr)
                if adjacency == "dual_random_walk":
                    assert torch.allclose(loaded.koopman.L_bwd, model.koopman.L_bwd)
                    assert any("._bwd." in key for key in loaded.state_dict())
                else:
                    assert not any("._bwd." in key for key in loaded.state_dict())
            loaded.eval()
            with torch.no_grad():
                after = loaded.predict(data, steps=2)
            for left, right in zip(before, after, strict=True):
                assert torch.allclose(left.x, right.x, atol=1e-5)


def test_load_rejects_networked_checkpoint_missing_adjacency(
    tmp_path: Path,
) -> None:
    """Networked payloads without ``adjacency`` raise a clear ValueError."""
    model = _tiny_networked_model(adjacency="random_walk")
    config = build_model_config(model)
    del config["adjacency"]
    path = tmp_path / "missing_adjacency.pt"
    torch.save(
        {
            "format_version": 1,
            "package_version": "0.6.0",
            "config": config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="adjacency"):
        load_checkpoint(path)


def test_factory_rejects_adjacency_for_non_networked_koopman() -> None:
    """Non-default ``koopman_adjacency`` is rejected for non-networked kinds."""
    with pytest.raises(ValueError, match="koopman_adjacency is only meaningful"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 3, num_layers=1),
            decoder=GNNDecoder(3, 4, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            koopman="pernode",
            koopman_adjacency="random_walk",
        )
    from koopman_graph.nn import HypergraphDecoder, HypergraphEncoder

    with pytest.raises(ValueError, match="koopman_adjacency is only meaningful"):
        GraphKoopmanModel(
            encoder=HypergraphEncoder(2, 4, 3),
            decoder=HypergraphDecoder(3, 4, 2),
            latent_dim=3,
            time_step=0.1,
            koopman="hypergraph",
            koopman_adjacency="dual_random_walk",
        )


def test_injection_rejects_non_default_koopman_adjacency() -> None:
    """Injected operators reject conflicting non-default ``koopman_adjacency``."""
    from koopman_graph import GraphKoopmanOperator

    injected = GraphKoopmanOperator(3, adjacency="random_walk")
    with pytest.raises(ValueError, match="koopman_adjacency"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 3, num_layers=1),
            decoder=GNNDecoder(3, 4, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            koopman=injected,
            koopman_adjacency="random_walk",
        )
