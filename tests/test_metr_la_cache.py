"""Tests for METR-LA cache construction and helper utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import MetrLaTrafficBenchmark
from koopman_graph.datasets.metr_la import (
    NUM_SENSORS,
    _default_traffic_path,
    adjacency_to_edge_index,
    adjacency_to_edge_weight,
    build_adjacency_matrix,
    build_traffic_cache_payload,
    download_distances_csv,
    download_sensor_ids,
    ensure_traffic_cache,
    load_traffic_cache,
    normalize_speeds,
    preprocess_speeds,
    read_h5_speed_window,
)


def test_default_traffic_path_uses_custom_cache_dir(tmp_path: Path) -> None:
    """Verify custom cache directories are honored."""
    path = _default_traffic_path(tmp_path)
    assert path == tmp_path / "traffic.pt"


def test_download_sensor_ids_parses_remote_list() -> None:
    """Verify remote sensor ID download returns a comma-separated list."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"a,b,c"
    mock_response.__enter__.return_value = mock_response
    with patch("koopman_graph.datasets.metr_la.urlopen", return_value=mock_response):
        sensor_ids = download_sensor_ids()
    assert sensor_ids == ["a", "b", "c"]


def test_download_sensor_ids_url_error_raises_oserror() -> None:
    """Verify network failures surface as OSError."""
    with (
        patch(
            "koopman_graph.datasets.metr_la.urlopen",
            side_effect=URLError("network down"),
        ),
        pytest.raises(OSError, match="Failed to download METR-LA sensor IDs"),
    ):
        download_sensor_ids()


def test_download_distances_csv_reads_remote_text() -> None:
    """Verify remote distance CSV download returns decoded text."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"from,to,cost\na,b,1.0\n"
    mock_response.__enter__.return_value = mock_response
    with patch("koopman_graph.datasets.metr_la.urlopen", return_value=mock_response):
        csv_text = download_distances_csv()
    assert "from,to,cost" in csv_text


def test_download_distances_csv_url_error_raises_oserror() -> None:
    """Verify distance download failures surface as OSError."""
    with (
        patch(
            "koopman_graph.datasets.metr_la.urlopen",
            side_effect=URLError("network down"),
        ),
        pytest.raises(OSError, match="Failed to download METR-LA distances"),
    ):
        download_distances_csv()


def test_build_adjacency_matrix_skips_unknown_sensor_ids() -> None:
    """Verify adjacency construction ignores rows with unknown sensor IDs."""
    distance_csv = "from,to,cost\na,a,0.0\na,b,10.0\nb,b,0.0\nx,y,5.0\n"
    adj = build_adjacency_matrix(distance_csv, ["a", "b"], normalized_k=0.0)
    assert adj[0, 1] > 0.0


def test_adjacency_to_edge_index_builds_bidirectional_edges() -> None:
    """Verify adjacency conversion yields bidirectional edge_index pairs."""
    adj = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    edge_index = adjacency_to_edge_index(adj)
    assert edge_index.shape == (2, 2)
    assert edge_index[0, 0].item() == 0
    assert edge_index[1, 0].item() == 1


def test_adjacency_to_edge_weight_aligns_with_edge_index() -> None:
    """Verify edge weights align with bidirectional edge_index ordering."""
    adj = np.array([[0.0, 2.0], [0.5, 0.0]], dtype=np.float32)
    edge_index = adjacency_to_edge_index(adj)
    edge_weight = adjacency_to_edge_weight(adj)
    assert edge_weight.shape == (edge_index.shape[1],)
    assert edge_weight[0].item() == pytest.approx(2.0)
    assert edge_weight[1].item() == pytest.approx(0.5)


def test_read_h5_speed_window_reads_requested_rows(tmp_path: Path) -> None:
    """Verify HDF5 speed windows return the requested row slice."""
    h5py = pytest.importorskip("h5py")

    h5_path = tmp_path / "metr-la.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("df")
        group.create_dataset(
            "block0_values",
            data=np.arange(15, dtype=np.float32).reshape(5, 3),
        )

    speeds = read_h5_speed_window(h5_path, num_timesteps=2, offset=1)
    assert speeds.shape == (2, 3)
    assert speeds[0, 0] == pytest.approx(3.0)


def test_preprocess_speeds_requires_two_dimensions() -> None:
    """Verify preprocess_speeds rejects non-matrix inputs."""
    with pytest.raises(ValueError, match="speeds must have shape"):
        preprocess_speeds(np.array([1.0, 2.0], dtype=np.float32))


def test_preprocess_speeds_fills_all_missing_sensor_column() -> None:
    """Verify an all-missing sensor column is zero-filled."""
    speeds = np.zeros((4, 2), dtype=np.float32)
    cleaned = preprocess_speeds(speeds)
    assert np.all(cleaned[:, 1] == 0.0)


def test_preprocess_speeds_fills_leading_missing_values_with_backward_pass() -> None:
    """Verify backward fill imputes leading missing readings per sensor."""
    speeds = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [5.0, 3.0],
        ],
        dtype=np.float32,
    )
    cleaned = preprocess_speeds(speeds)
    assert cleaned[0, 0] == pytest.approx(5.0)
    assert cleaned[1, 0] == pytest.approx(5.0)


def test_preprocess_speeds_backward_fill_skips_when_no_future_valid_value() -> None:
    """Verify backward fill does not write when no future valid value exists."""
    speeds = np.array(
        [
            [5.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    cleaned = preprocess_speeds(speeds)
    assert cleaned[0, 0] == pytest.approx(5.0)
    assert cleaned[1, 0] == pytest.approx(5.0)
    assert cleaned[2, 0] == pytest.approx(5.0)


def test_read_h5_speed_window_requires_h5py(tmp_path: Path, monkeypatch) -> None:
    """Verify missing h5py dependency raises ImportError with guidance."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="h5py is required"):
        read_h5_speed_window(tmp_path / "missing.h5", num_timesteps=1)


def test_normalize_speeds_handles_constant_sensor_column() -> None:
    """Verify constant sensor columns do not divide by zero."""
    speeds = np.ones((5, 2), dtype=np.float32)
    normalized = normalize_speeds(speeds)
    assert normalized.shape == speeds.shape
    assert np.all(np.isfinite(normalized))


def test_build_traffic_cache_payload_assembles_cache(tmp_path: Path) -> None:
    """Verify cache payload construction stitches graph and speed tensors."""
    speeds = np.arange(6, dtype=np.float32).reshape(2, 3)
    sensor_ids = ["a", "b", "c"]
    distance_csv = "from,to,cost\na,a,0.0\na,b,10.0\nb,b,0.0\n"
    with patch(
        "koopman_graph.datasets.metr_la.download_distances_csv",
        return_value=distance_csv,
    ):
        payload = build_traffic_cache_payload(speeds, sensor_ids, normalized_k=0.0)
    assert payload["num_nodes"] == 3
    assert payload["speeds"].shape == (2, 3, 1)
    assert payload["edge_index"].dtype == torch.long
    assert payload["edge_weight"].shape == (payload["edge_index"].shape[1],)
    assert payload["edge_weight"].max().item() > 0.0


def test_build_traffic_cache_payload_validates_inputs() -> None:
    """Verify cache payload validation catches shape mismatches."""
    with pytest.raises(ValueError, match="speeds must have shape"):
        build_traffic_cache_payload(np.array([1.0, 2.0]), ["a"])
    with pytest.raises(ValueError, match="sensor_ids"):
        build_traffic_cache_payload(np.ones((2, 3)), ["a", "b"])


def test_ensure_traffic_cache_reuses_existing_file(tmp_path: Path) -> None:
    """Verify existing traffic.pt files are reused without rebuilding."""
    path = _default_traffic_path(tmp_path)
    torch.save({"edge_index": torch.zeros((2, 0), dtype=torch.long)}, path)
    result = ensure_traffic_cache(tmp_path)
    assert result == path


def test_ensure_traffic_cache_force_true_reuses_without_h5(tmp_path: Path) -> None:
    """Verify force=True without HDF5 reuses an existing cache file."""
    path = _default_traffic_path(tmp_path)
    torch.save({"edge_index": torch.zeros((2, 0), dtype=torch.long)}, path)
    result = ensure_traffic_cache(tmp_path, force=True)
    assert result == path


def test_ensure_traffic_cache_missing_without_h5_raises(tmp_path: Path) -> None:
    """Verify missing cache without HDF5 input raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="METR-LA cache is missing"):
        ensure_traffic_cache(tmp_path, force=True)


def test_ensure_traffic_cache_builds_from_h5(tmp_path: Path) -> None:
    """Verify cache creation reads HDF5 speeds and writes traffic.pt."""
    h5py = pytest.importorskip("h5py")

    h5_path = tmp_path / "metr-la.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("df")
        group.create_dataset(
            "block0_values",
            data=np.full((4, 2), 55.0, dtype=np.float32),
        )

    distance_csv = "from,to,cost\ns0,s0,0.0\ns0,s1,10.0\ns1,s1,0.0\n"
    with (
        patch(
            "koopman_graph.datasets.metr_la.download_sensor_ids",
            return_value=["s0", "s1"],
        ),
        patch(
            "koopman_graph.datasets.metr_la.download_distances_csv",
            return_value=distance_csv,
        ),
    ):
        path = ensure_traffic_cache(
            tmp_path,
            force=True,
            h5_path=h5_path,
            num_timesteps=3,
        )
    assert path.exists()
    payload = torch.load(path, weights_only=False)
    assert payload["speeds"].shape[0] == 3


def test_load_traffic_cache_casts_tensor_dtypes(tmp_path: Path) -> None:
    """Verify load_traffic_cache restores tensors with requested dtypes."""
    path = _default_traffic_path(tmp_path)
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "speeds": torch.ones(2, 1, 1, dtype=torch.float32),
        },
        path,
    )
    payload = load_traffic_cache(tmp_path, dtype=torch.float64)
    assert payload["speeds"].dtype == torch.float64
    assert payload["edge_index"].dtype == torch.long


def test_metr_la_load_sequence_rejects_invalid_speed_shape(tmp_path: Path) -> None:
    """Verify load_sequence validates cached speed tensor rank."""
    path = _default_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": ["a"],
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "speeds": torch.ones(2, 1),
            "num_nodes": NUM_SENSORS,
            "source_h5_url": "test",
            "normalized_k": 0.1,
        },
        path,
    )
    with pytest.raises(ValueError, match="Expected speeds shape"):
        MetrLaTrafficBenchmark.load_sequence(tmp_path)


def test_metr_la_load_sequence_rejects_wrong_sensor_count(tmp_path: Path) -> None:
    """Verify load_sequence validates the cached sensor count."""
    path = _default_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": ["a"],
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "speeds": torch.ones(2, 1, 1),
            "num_nodes": 1,
            "source_h5_url": "test",
            "normalized_k": 0.1,
        },
        path,
    )
    with pytest.raises(ValueError, match=str(NUM_SENSORS)):
        MetrLaTrafficBenchmark.load_sequence(tmp_path)


def test_metr_la_load_sequence_success_with_valid_cache(tmp_path: Path) -> None:
    """Verify load_sequence returns a validated snapshot sequence."""
    path = _default_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": ["a"] * NUM_SENSORS,
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "speeds": torch.ones(3, NUM_SENSORS, 1),
            "num_nodes": NUM_SENSORS,
            "source_h5_url": "test",
            "normalized_k": 0.1,
        },
        path,
    )
    sequence = MetrLaTrafficBenchmark.load_sequence(tmp_path)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == NUM_SENSORS
    assert sequence.num_timesteps == 3


def test_load_traffic_cache_recomputes_legacy_edge_weight(tmp_path: Path) -> None:
    """Verify legacy caches without edge_weight recompute Gaussian weights."""
    path = _default_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": ["a", "b", "c"],
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "speeds": torch.ones(2, 3, 1),
            "num_nodes": 3,
            "source_h5_url": "test",
            "normalized_k": 0.0,
        },
        path,
    )
    distance_csv = "from,to,cost\na,a,0.0\na,b,10.0\nb,b,0.0\n"
    with patch(
        "koopman_graph.datasets.metr_la.download_distances_csv",
        return_value=distance_csv,
    ):
        payload = load_traffic_cache(tmp_path)
    assert payload["edge_weight"].shape == (payload["edge_index"].shape[1],)


def test_metr_la_load_sequence_preserves_edge_weight(tmp_path: Path) -> None:
    """Verify METR-LA sequences expose non-uniform Gaussian-kernel weights."""
    path = _default_traffic_path(tmp_path)
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    edge_weight = torch.tensor([0.9, 0.4, 0.4, 0.9], dtype=torch.float32)
    torch.save(
        {
            "sensor_ids": ["a"] * NUM_SENSORS,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "speeds": torch.ones(3, NUM_SENSORS, 1),
            "num_nodes": NUM_SENSORS,
            "source_h5_url": "test",
            "normalized_k": 0.1,
        },
        path,
    )
    sequence = MetrLaTrafficBenchmark.load_sequence(tmp_path)
    assert sequence.edge_weight is not None
    assert torch.equal(sequence.edge_weight, edge_weight)
    assert not torch.allclose(sequence.edge_weight, torch.ones_like(edge_weight))


def test_ensure_edge_weight_returns_existing_payload_weight() -> None:
    """Verify payloads that already carry edge weights skip recomputation."""
    from koopman_graph.datasets.metr_la import _ensure_edge_weight

    edge_weight = torch.tensor([0.5, 0.25], dtype=torch.float32)
    payload = {
        "edge_weight": edge_weight,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "sensor_ids": ["a", "b"],
    }
    assert _ensure_edge_weight(payload) is edge_weight
