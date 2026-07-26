"""Tests for PEMS-BAY / PEMS0X cache construction and helpers (TASK-1316)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import PemsBayTrafficBenchmark, PemsTrafficBenchmark
from koopman_graph.datasets.download import verify_sha256
from koopman_graph.datasets.pems import (
    BAY_DISTANCES_SHA256,
    BAY_NUM_SENSORS,
    BAY_SENSOR_LOCATIONS_SHA256,
    VARIANT_NUM_SENSORS,
    _default_bay_traffic_path,
    _default_variant_traffic_path,
    build_bay_cache_payload,
    build_variant_cache_payload,
    download_bay_distances_csv,
    download_bay_sensor_ids,
    ensure_bay_traffic_cache,
    ensure_variant_traffic_cache,
    load_bay_traffic_cache,
    normalize_pems_variant,
    read_adjacency_csv,
    read_bay_h5_speed_window,
    read_npz_flow_window,
)


def test_pems_reexports_verify_sha256() -> None:
    """Historical ``pems.verify_sha256`` import path still resolves."""
    from koopman_graph.datasets import pems

    assert pems.verify_sha256 is verify_sha256


def test_normalize_pems_variant_accepts_aliases() -> None:
    """Variant strings normalize to two-digit codes."""
    assert normalize_pems_variant("04") == "04"
    assert normalize_pems_variant("PEMS08") == "08"
    with pytest.raises(ValueError, match="variant must be one of"):
        normalize_pems_variant("99")


def test_download_bay_sensor_ids_parses_and_verifies() -> None:
    """Remote sensor locations are parsed and SHA256-checked."""
    lines = "\n".join(f"{400000 + i},0.0,0.0" for i in range(BAY_NUM_SENSORS))
    raw = (lines + "\n").encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    mock_response = MagicMock()
    mock_response.read.return_value = raw
    mock_response.__enter__.return_value = mock_response
    with (
        patch("koopman_graph.datasets.download.urlopen", return_value=mock_response),
        patch("koopman_graph.datasets.pems.BAY_SENSOR_LOCATIONS_SHA256", digest),
    ):
        sensor_ids = download_bay_sensor_ids()
    assert len(sensor_ids) == BAY_NUM_SENSORS
    assert sensor_ids[0] == "400000"


def test_download_bay_sensor_ids_checksum_mismatch_raises() -> None:
    """Tampered sensor-location downloads raise ValueError."""
    raw = b"1,0,0\n"
    mock_response = MagicMock()
    mock_response.read.return_value = raw
    mock_response.__enter__.return_value = mock_response
    with (
        patch("koopman_graph.datasets.download.urlopen", return_value=mock_response),
        pytest.raises(ValueError, match="SHA256 mismatch"),
    ):
        download_bay_sensor_ids(verify=True)


def test_download_bay_distances_url_error_raises_oserror() -> None:
    """Network failures surface as OSError."""
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            side_effect=URLError("network down"),
        ),
        pytest.raises(OSError, match="Failed to download PEMS-BAY distances"),
    ):
        download_bay_distances_csv()


def test_download_bay_distances_checksum_constant_documented() -> None:
    """Documented distance digest is a 64-char hex string."""
    assert len(BAY_DISTANCES_SHA256) == 64
    assert len(BAY_SENSOR_LOCATIONS_SHA256) == 64


def test_read_bay_h5_requires_h5py(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing h5py raises guided ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="h5py is required"):
        read_bay_h5_speed_window(tmp_path / "missing.h5", num_timesteps=1)


def test_build_bay_cache_payload_assembles_cache() -> None:
    """BAY payload stitches graph and normalized speeds."""
    speeds = np.arange(6, dtype=np.float32).reshape(2, 3)
    sensor_ids = ["a", "b", "c"]
    distance_csv = "from,to,cost\na,a,0.0\na,b,10.0\nb,b,0.0\n"
    payload = build_bay_cache_payload(
        speeds,
        sensor_ids,
        distance_csv=distance_csv,
        normalized_k=0.0,
    )
    assert payload["num_nodes"] == 3
    assert payload["speeds"].shape == (2, 3, 1)
    assert payload["benchmark"] == "pems_bay"
    assert payload["feature"] == "speed"


def test_ensure_bay_cache_reuses_existing(tmp_path: Path) -> None:
    """Existing traffic.pt is reused without rebuilding."""
    path = _default_bay_traffic_path(tmp_path)
    torch.save({"edge_index": torch.zeros((2, 0), dtype=torch.long)}, path)
    assert ensure_bay_traffic_cache(tmp_path) == path


def test_ensure_bay_cache_missing_without_h5_raises(tmp_path: Path) -> None:
    """Missing BAY cache without HDF5 raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PEMS-BAY cache is missing"):
        ensure_bay_traffic_cache(tmp_path, force=True)


@pytest.mark.parametrize("h5_group", ("df", "speed"))
def test_ensure_bay_cache_builds_from_h5(tmp_path: Path, h5_group: str) -> None:
    """Cache creation reads HDF5 speeds (DCRNN ``df/`` or mirror ``speed/``)."""
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "pems-bay.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group(h5_group)
        group.create_dataset(
            "block0_values",
            data=np.full((4, BAY_NUM_SENSORS), 55.0, dtype=np.float32),
        )
    sensor_ids = [f"s{i}" for i in range(BAY_NUM_SENSORS)]
    # Minimal distance CSV: self-loops only (valid for build_adjacency_matrix).
    rows = ["from,to,cost"]
    for sensor_id in sensor_ids[:3]:
        rows.append(f"{sensor_id},{sensor_id},0.0")
    rows.append(f"{sensor_ids[0]},{sensor_ids[1]},10.0")
    distance_csv = "\n".join(rows) + "\n"
    with (
        patch(
            "koopman_graph.datasets.pems.download_bay_sensor_ids",
            return_value=sensor_ids,
        ),
        patch(
            "koopman_graph.datasets.pems.download_bay_distances_csv",
            return_value=distance_csv,
        ),
    ):
        path = ensure_bay_traffic_cache(
            tmp_path,
            force=True,
            h5_path=h5_path,
            num_timesteps=3,
            offset=0,
        )
    assert path.exists()
    payload = torch.load(path, weights_only=False)
    assert payload["speeds"].shape == (3, BAY_NUM_SENSORS, 1)


def test_ensure_bay_cache_rejects_bad_h5_checksum(tmp_path: Path) -> None:
    """Provided H5 SHA256 mismatches raise before cache build."""
    h5_path = tmp_path / "pems-bay.h5"
    h5_path.write_bytes(b"not-an-h5")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ensure_bay_traffic_cache(
            tmp_path,
            force=True,
            h5_path=h5_path,
            expected_h5_sha256="0" * 64,
        )


def test_pems_bay_load_sequence_success(tmp_path: Path) -> None:
    """Valid BAY cache loads as a GraphSnapshotSequence."""
    path = _default_bay_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": [f"s{i}" for i in range(BAY_NUM_SENSORS)],
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "speeds": torch.ones(3, BAY_NUM_SENSORS, 1),
            "num_nodes": BAY_NUM_SENSORS,
            "source_h5_url": "test",
            "normalized_k": 0.1,
        },
        path,
    )
    sequence = PemsBayTrafficBenchmark.load_sequence(tmp_path)
    topology = PemsBayTrafficBenchmark.load_topology(tmp_path)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == BAY_NUM_SENSORS
    assert topology.num_nodes == BAY_NUM_SENSORS
    assert topology.sensor_ids is not None


def test_pems_bay_load_sequence_rejects_corrupt_shape(tmp_path: Path) -> None:
    """Corrupt speed rank raises ValueError."""
    path = _default_bay_traffic_path(tmp_path)
    torch.save(
        {
            "sensor_ids": ["a"],
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "speeds": torch.ones(2, 1),
            "num_nodes": BAY_NUM_SENSORS,
        },
        path,
    )
    with pytest.raises(ValueError, match="Expected speeds shape"):
        PemsBayTrafficBenchmark.load_sequence(tmp_path)


def test_load_bay_traffic_cache_casts_dtypes(tmp_path: Path) -> None:
    """load_bay_traffic_cache honors dtype."""
    path = _default_bay_traffic_path(tmp_path)
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "speeds": torch.ones(2, 1, 1, dtype=torch.float32),
            "edge_weight": torch.ones(2, dtype=torch.float32),
            "num_nodes": 1,
            "sensor_ids": ["a"],
        },
        path,
    )
    payload = load_bay_traffic_cache(tmp_path, dtype=torch.float64)
    assert payload["speeds"].dtype == torch.float64


def test_read_npz_and_adjacency_round_trip(tmp_path: Path) -> None:
    """NPZ flow windows and dense adjacency CSV load correctly."""
    npz_path = tmp_path / "PEMS04.npz"
    adj_path = tmp_path / "PEMS04.csv"
    data = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    np.savez(npz_path, data=data)
    adj = np.eye(4, dtype=np.float32)
    adj[0, 1] = 0.5
    np.savetxt(adj_path, adj, delimiter=",")
    flows = read_npz_flow_window(npz_path, num_timesteps=2, offset=0, channel=0)
    assert flows.shape == (2, 4)
    loaded_adj = read_adjacency_csv(adj_path)
    assert loaded_adj.shape == (4, 4)
    assert loaded_adj[0, 1] == pytest.approx(0.5)


def test_build_variant_cache_payload_and_load(tmp_path: Path) -> None:
    """PEMS04 payload builds and loads through the public API."""
    variant = "04"
    n = VARIANT_NUM_SENSORS[variant]
    flows = np.ones((5, n), dtype=np.float32)
    adj = np.eye(n, dtype=np.float32)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    payload = build_variant_cache_payload(
        flows,
        adj,
        variant=variant,
        source_url="test://pems04",
    )
    path = _default_variant_traffic_path(variant, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    sequence = PemsTrafficBenchmark.load_sequence(variant, tmp_path)
    topology = PemsTrafficBenchmark.load_topology(variant, tmp_path)
    assert sequence.num_nodes == n
    assert sequence.num_timesteps == 5
    assert topology.num_nodes == n
    assert n == PemsTrafficBenchmark(variant).NUM_SENSORS


def test_ensure_variant_cache_builds_from_npz(tmp_path: Path) -> None:
    """Variant cache creation from NPZ + CSV writes traffic.pt."""
    variant = "08"
    n = VARIANT_NUM_SENSORS[variant]
    npz_path = tmp_path / "PEMS08.npz"
    adj_path = tmp_path / "PEMS08.csv"
    data = np.ones((6, n, 3), dtype=np.float32)
    np.savez(npz_path, data=data)
    np.savetxt(adj_path, np.eye(n, dtype=np.float32), delimiter=",")
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    path = ensure_variant_traffic_cache(
        variant,
        tmp_path,
        force=True,
        npz_path=npz_path,
        adj_csv_path=adj_path,
        num_timesteps=4,
        offset=0,
        expected_npz_sha256=digest,
    )
    assert path.exists()
    payload = torch.load(path, weights_only=False)
    assert payload["speeds"].shape == (4, n, 1)
    assert payload["variant"] == "08"


def test_ensure_variant_cache_missing_raises(tmp_path: Path) -> None:
    """Missing PEMS0X cache without NPZ/CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PEMS04 cache is missing"):
        ensure_variant_traffic_cache("04", tmp_path, force=True)


def test_ensure_variant_rejects_bad_npz_checksum(tmp_path: Path) -> None:
    """NPZ SHA256 mismatches raise before cache build."""
    npz_path = tmp_path / "PEMS04.npz"
    adj_path = tmp_path / "PEMS04.csv"
    np.savez(npz_path, data=np.ones((2, VARIANT_NUM_SENSORS["04"], 1)))
    np.savetxt(adj_path, np.eye(VARIANT_NUM_SENSORS["04"]), delimiter=",")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ensure_variant_traffic_cache(
            "04",
            tmp_path,
            force=True,
            npz_path=npz_path,
            adj_csv_path=adj_path,
            expected_npz_sha256="0" * 64,
        )
