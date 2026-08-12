"""Parity tests: safetensors_v1 directory / .kgckpt ≡ legacy_pt pickle."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.serialization import (
    SAFE_CONFIG_FILENAME,
    SAFE_META_FILENAME,
    SAFE_WEIGHTS_FILENAME,
    SAFE_ZIP_MEMBER_NAMES,
    build_model_config,
    load_checkpoint,
    save_checkpoint,
)

# Safe path detaches/clones to contiguous CPU tensors; legacy pickle preserves
# the same float32 values. Bit-identical agreement is required — any drift is a
# serialization bug, not float noise (rtol=0, atol=0).
_PARITY_RTOL = 0.0
_PARITY_ATOL = 0.0


def _json_ready(value: object) -> object:
    """Mirror serialization's tuple→list coercion for config comparison."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _assert_state_dict_close(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> None:
    """Assert two state dicts match exactly on CPU float tensors."""
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(
            left[key].detach().cpu(),
            right[key].detach().cpu(),
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )


def _assert_safe_legacy_parity(
    model: GraphKoopmanModel,
    tmp_path: Path,
    *,
    predict: Callable[[GraphKoopmanModel], object],
) -> None:
    """Save both containers, reload, and compare config / weights / predict."""
    model.eval()
    with torch.no_grad():
        before = predict(model)

    safe_dir = tmp_path / "safe_ckpt"
    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(model, safe_dir, format="safetensors_v1")
    save_checkpoint(model, legacy_path, format="legacy_pt")

    assert safe_dir.is_dir()
    assert (safe_dir / SAFE_META_FILENAME).is_file()
    assert (safe_dir / SAFE_CONFIG_FILENAME).is_file()
    assert (safe_dir / SAFE_WEIGHTS_FILENAME).is_file()
    assert legacy_path.is_file()

    safe_loaded = load_checkpoint(safe_dir)
    legacy_loaded = load_checkpoint(legacy_path)
    assert not safe_loaded.training
    assert not legacy_loaded.training

    original_config = _json_ready(build_model_config(model))
    assert _json_ready(build_model_config(safe_loaded)) == original_config
    assert _json_ready(build_model_config(legacy_loaded)) == original_config

    _assert_state_dict_close(model.state_dict(), safe_loaded.state_dict())
    _assert_state_dict_close(model.state_dict(), legacy_loaded.state_dict())
    _assert_state_dict_close(safe_loaded.state_dict(), legacy_loaded.state_dict())

    with torch.no_grad():
        safe_pred = predict(safe_loaded)
        legacy_pred = predict(legacy_loaded)

    _assert_predictions_close(before, safe_pred)
    _assert_predictions_close(before, legacy_pred)
    _assert_predictions_close(safe_pred, legacy_pred)


def _assert_predictions_close(left: object, right: object) -> None:
    """Compare nested prediction tensors / Data.x values exactly."""
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_predictions_close(left_item, right_item)
        return
    if isinstance(left, Data) and isinstance(right, Data):
        torch.testing.assert_close(
            left.x,
            right.x,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )
        return
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(
            left,
            right,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )
        return
    msg = f"Unsupported prediction types: {type(left)!r} vs {type(right)!r}"
    raise TypeError(msg)


def test_safe_vs_legacy_parity_discrete_gcn(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Discrete GCN: safetensors_v1 and legacy_pt agree after short fit."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )
    torch.manual_seed(0)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)

    def predict(loaded: GraphKoopmanModel) -> list[torch.Tensor]:
        return [
            graph.x.detach().clone()
            for graph in loaded.predict(scaling_sequence[0], steps=3)
        ]

    _assert_safe_legacy_parity(model, tmp_path, predict=predict)


def test_safe_vs_legacy_parity_continuous(
    synthetic_edge_index: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Continuous dissipative model: both containers preserve predict_at."""
    torch.manual_seed(1)
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman_parameterization="dissipative",
    )
    graph = Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index)

    def predict(loaded: GraphKoopmanModel) -> list[Data]:
        return list(loaded.predict_at(graph, step_deltas=[0.1, 0.1]))

    _assert_safe_legacy_parity(model, tmp_path, predict=predict)


def test_safe_vs_legacy_parity_delay_embedding(
    synthetic_edge_index: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Delay-embedding GCN: both containers preserve n_delays and predict."""
    torch.manual_seed(2)
    # in_channels = n_delays * feature_dim = 2 * 3
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=6, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
        n_delays=2,
    )
    x0 = torch.randn(5, 3)
    snapshots = [
        Data(x=x0 * (0.95**t), edge_index=synthetic_edge_index) for t in range(6)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    model.fit(sequence, epochs=1, lr=1e-2)
    assert model.n_delays == 2

    def predict(loaded: GraphKoopmanModel) -> list[torch.Tensor]:
        assert loaded.n_delays == 2
        return [
            graph.x.detach().clone() for graph in loaded.predict(sequence[0], steps=2)
        ]

    _assert_safe_legacy_parity(model, tmp_path, predict=predict)


def _small_discrete_model() -> GraphKoopmanModel:
    """Untrained tiny GCN for zip-container unit tests."""
    torch.manual_seed(3)
    return GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2
        ),
        decoder=GNNDecoder(
            latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2
        ),
        latent_dim=4,
        time_step=0.1,
    )


def test_kgckpt_zip_round_trip_and_directory_parity(
    synthetic_edge_index: torch.Tensor,
    tmp_path: Path,
) -> None:
    """``.kgckpt`` zip matches directory safetensors_v1 (exact tensors)."""
    model = _small_discrete_model()
    model.eval()
    graph = Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index)
    with torch.no_grad():
        before = [g.x.detach().clone() for g in model.predict(graph, steps=2)]

    safe_dir = tmp_path / "safe_dir"
    kgckpt_path = tmp_path / "model.kgckpt"
    save_checkpoint(model, safe_dir, format="safetensors_v1")
    save_checkpoint(model, kgckpt_path, format="safetensors_v1")

    assert kgckpt_path.is_file()
    assert zipfile.is_zipfile(kgckpt_path)
    with zipfile.ZipFile(kgckpt_path, mode="r") as archive:
        names = set(archive.namelist())
    assert SAFE_ZIP_MEMBER_NAMES.issubset(names)
    assert not any(name.endswith(".pt") for name in names)

    dir_loaded = load_checkpoint(safe_dir)
    zip_loaded = load_checkpoint(kgckpt_path)
    assert not zip_loaded.training

    original_config = _json_ready(build_model_config(model))
    assert _json_ready(build_model_config(dir_loaded)) == original_config
    assert _json_ready(build_model_config(zip_loaded)) == original_config
    _assert_state_dict_close(model.state_dict(), zip_loaded.state_dict())
    _assert_state_dict_close(dir_loaded.state_dict(), zip_loaded.state_dict())

    with torch.no_grad():
        zip_pred = [g.x.detach().clone() for g in zip_loaded.predict(graph, steps=2)]
    _assert_predictions_close(before, zip_pred)


def test_kgckpt_suffix_rejects_non_safe_zip(tmp_path: Path) -> None:
    """``.kgckpt`` without safetensors members must not fall through to pickle."""
    bad_path = tmp_path / "broken.kgckpt"
    with zipfile.ZipFile(bad_path, mode="w") as archive:
        archive.writestr("readme.txt", "not a checkpoint\n")

    with pytest.raises(ValueError, match="bundle suffix"):
        load_checkpoint(bad_path)


def test_zip_suffix_round_trip(tmp_path: Path) -> None:
    """Explicit ``.zip`` suffix also selects the safetensors_v1 bundle writer."""
    model = _small_discrete_model()
    zip_path = tmp_path / "model.zip"
    save_checkpoint(model, zip_path, format="safetensors_v1")
    loaded = load_checkpoint(zip_path)
    _assert_state_dict_close(model.state_dict(), loaded.state_dict())
