"""Tests for GraphKoopmanModel checkpoint serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GATEncoder, GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_model_config,
    load_checkpoint,
    reconstruct_model,
    save_checkpoint,
)


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


def test_load_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    """Verify missing checkpoint paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_invalid_format_version(tmp_path: Path) -> None:
    """Verify unsupported format versions raise ValueError."""
    path = tmp_path / "bad_version.pt"
    torch.save({"format_version": 99, "config": {}, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format_version"):
        load_checkpoint(path)


def test_load_checkpoint_unsupported_encoder_type(tmp_path: Path) -> None:
    """Verify unknown encoder types raise ValueError."""
    config = {
        "latent_dim": 4,
        "time_step": 0.1,
        "koopman_init_mode": "identity_noise",
        "koopman_init_scale": 1e-2,
        "encoder": {
            "type": "unknown",
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

    from koopman_graph.training import TrainingLossBreakdown

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
            "koopman_graph.model.train_one_epoch",
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
    path = tmp_path / "nested" / "dir" / "model.pt"
    save_checkpoint(graph_koopman_model, path)
    assert path.is_file()


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
