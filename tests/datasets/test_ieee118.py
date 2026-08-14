"""Tests for IEEE 118-bus MATPOWER parsing and cache helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
import torch

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.datasets import (
    IEEE118DynamicBenchmark,
    TopologyPayload,
    TypedIEEE118Topology,
)
from koopman_graph.datasets.ieee118 import (
    NUM_BUSES,
    SIMULATED_DYNAMICS_DISCLAIMER,
    _build_edge_index,
    _bus_id_map,
    _default_topology_path,
    _extract_matrix_block,
    _initial_bus_features,
    _parse_numeric_rows,
    bus_type_name,
    download_matpower_case118,
    ensure_topology_cache,
    homogeneous_features_to_typed_hetero,
    load_topology,
    parse_matpower_case,
    partition_buses_by_type,
    topology_from_matpower_text,
)

MINIMAL_MATPOWER = """
function mpc = case_test
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
    1  3  10  5  0  0  1  1.0  0  135  1  1.1  0.9;
    2  2  20  10  0  0  1  1.0  0  135  1  1.1  0.9;
];
mpc.branch = [
    1  2  0.01  0.05  0  250  250  250  0  0  1  -360  360;
    1  2  0.01  0.05  0  250  250  250  0  0  0  -360  360;
];
"""


def test_default_topology_path_uses_custom_cache_dir(tmp_path: Path) -> None:
    """Verify custom cache directories are honored."""
    path = _default_topology_path(tmp_path)
    assert path == tmp_path / "topology.pt"


def test_parse_numeric_rows_skips_comments_and_blank_lines() -> None:
    """Verify semicolon-separated rows ignore comments and blanks."""
    block = "1 2 3;\n% comment\n;\n4 5 6;"
    rows = _parse_numeric_rows(block)
    assert rows == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_extract_matrix_block_parses_nested_brackets() -> None:
    """Verify matrix extraction handles nested bracket literals."""
    text = "mpc.bus = [[1 2]; [3 4]];"
    block = _extract_matrix_block(text, "bus")
    assert "1 2" in block
    assert "3 4" in block


def test_extract_matrix_block_missing_field_raises() -> None:
    """Verify missing MATPOWER fields raise a clear error."""
    with pytest.raises(ValueError, match="Could not find mpc.bus"):
        _extract_matrix_block("mpc.baseMVA = 100;", "bus")


def test_extract_matrix_block_unterminated_matrix_raises() -> None:
    """Verify unterminated matrix literals raise a clear error."""
    text = "mpc.bus = [1 2 3;"
    with pytest.raises(ValueError, match="Unterminated"):
        _extract_matrix_block(text, "bus")


def test_parse_matpower_case_success() -> None:
    """Verify a minimal MATPOWER case parses bus and branch tables."""
    parsed = parse_matpower_case(MINIMAL_MATPOWER)
    assert parsed["baseMVA"] == 100.0
    assert len(parsed["bus"]) == 2
    assert len(parsed["branch"]) == 2


def test_parse_matpower_case_missing_base_mva_raises() -> None:
    """Verify missing baseMVA raises a clear error."""
    with pytest.raises(ValueError, match="baseMVA"):
        parse_matpower_case("mpc.bus = [1 2 3];")


def test_parse_matpower_case_empty_bus_rows_raises() -> None:
    """Verify empty bus tables raise a clear error."""
    text = """
    mpc.baseMVA = 100;
    mpc.bus = [];
    mpc.branch = [1 2 0.01 0.05 0 250 250 250 0 0 1 -360 360;];
    """
    with pytest.raises(ValueError, match="no bus rows"):
        parse_matpower_case(text)


def test_parse_matpower_case_empty_branch_rows_raises() -> None:
    """Verify empty branch tables raise a clear error."""
    text = """
    mpc.baseMVA = 100;
    mpc.bus = [1 3 10 5 0 0 1 1.0 0 135 1 1.1 0.9;];
    mpc.branch = [];
    """
    with pytest.raises(ValueError, match="no branch rows"):
        parse_matpower_case(text)


def test_topology_from_matpower_text_builds_tensors() -> None:
    """Verify parsed MATPOWER text yields topology tensors."""
    topology = topology_from_matpower_text(MINIMAL_MATPOWER)
    assert isinstance(topology, TopologyPayload)
    assert topology.num_nodes == 2
    assert topology["num_nodes"] == 2
    assert topology.edge_index.shape[0] == 2
    assert topology.initial_features is not None
    assert topology.initial_features.shape == (2, 4)
    assert topology.bus_types is not None
    assert topology.bus_types.tolist() == [3, 2]


def test_build_edge_index_skips_inactive_branches() -> None:
    """Verify inactive MATPOWER branches are excluded from edge_index."""
    branch_rows = parse_matpower_case(MINIMAL_MATPOWER)["branch"]
    bus_map = _bus_id_map(parse_matpower_case(MINIMAL_MATPOWER)["bus"])
    edge_index = _build_edge_index(branch_rows, bus_map)
    assert edge_index.shape[1] == 2


def test_initial_bus_features_normalizes_loads() -> None:
    """Verify bus features include per-unit voltage, angle, and loads."""
    bus_rows = parse_matpower_case(MINIMAL_MATPOWER)["bus"]
    features = _initial_bus_features(bus_rows, base_mva=100.0, dtype=torch.float32)
    assert features[0, 0] == pytest.approx(1.0)
    assert features[0, 2] == pytest.approx(0.1)


def test_download_matpower_case118_reads_remote_text() -> None:
    """Verify remote MATPOWER download returns decoded text."""
    mock_response = MagicMock()
    mock_response.read.return_value = MINIMAL_MATPOWER.encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    with patch("koopman_graph.datasets.download.urlopen", return_value=mock_response):
        text = download_matpower_case118()
    assert "mpc.baseMVA" in text


def test_download_matpower_case118_url_error_raises_oserror() -> None:
    """Verify network failures surface as OSError."""
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            side_effect=URLError("network down"),
        ),
        pytest.raises(OSError, match="Failed to download MATPOWER"),
    ):
        download_matpower_case118()


def test_ensure_topology_cache_creates_file(tmp_path: Path) -> None:
    """Verify cache creation writes topology.pt when missing."""
    with patch(
        "koopman_graph.datasets.ieee118.download_matpower_case118",
        return_value=MINIMAL_MATPOWER,
    ):
        path = ensure_topology_cache(tmp_path, force=True)
    assert path.exists()
    payload = torch.load(path, weights_only=False)
    assert payload["num_nodes"] == 2


def test_ensure_topology_cache_reuses_existing_file(tmp_path: Path) -> None:
    """Verify existing cache files are reused without rebuilding."""
    path = _default_topology_path(tmp_path)
    torch.save({"num_nodes": 2}, path)
    with patch(
        "koopman_graph.datasets.ieee118.download_matpower_case118",
    ) as download_mock:
        result = ensure_topology_cache(tmp_path)
    assert result == path
    download_mock.assert_not_called()


def test_load_topology_casts_tensor_dtypes(tmp_path: Path) -> None:
    """Verify load_topology restores tensors with requested dtypes."""
    topology = topology_from_matpower_text(MINIMAL_MATPOWER, dtype=torch.float64)
    path = _default_topology_path(tmp_path)
    torch.save(topology.to_dict(), path)
    loaded = load_topology(tmp_path, dtype=torch.float64)
    assert isinstance(loaded, TopologyPayload)
    assert loaded.edge_index.dtype == torch.long
    assert loaded.initial_features is not None
    assert loaded.initial_features.dtype == torch.float64
    assert loaded["edge_index"].dtype == torch.long


def test_ieee118_generate_rejects_wrong_bus_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify generation fails when cached topology has the wrong node count."""
    bad_topology = topology_from_matpower_text(MINIMAL_MATPOWER)
    monkeypatch.setattr(
        IEEE118DynamicBenchmark,
        "load_topology",
        classmethod(lambda cls, *args, **kwargs: bad_topology),
    )
    with pytest.raises(ValueError, match=str(NUM_BUSES)):
        IEEE118DynamicBenchmark.generate(num_timesteps=2)


def test_bus_type_name_and_partition() -> None:
    """MATPOWER codes map to typed names and tile every bus index."""
    assert bus_type_name(1) == "load"
    assert bus_type_name(2) == "generator"
    assert bus_type_name(3) == "slack"
    with pytest.raises(ValueError, match="Unsupported"):
        bus_type_name(4)

    codes = torch.tensor([2, 1, 3, 1], dtype=torch.long)
    partitions = partition_buses_by_type(codes)
    assert partitions["generator"].tolist() == [0]
    assert partitions["load"].tolist() == [1, 3]
    assert partitions["slack"].tolist() == [2]
    covered = sorted(
        int(index) for indices in partitions.values() for index in indices.tolist()
    )
    assert covered == [0, 1, 2, 3]


def test_homogeneous_features_to_typed_hetero_minimal() -> None:
    """Tiny MATPOWER fixture converts to typed HeteroData with local edges."""
    topology = topology_from_matpower_text(MINIMAL_MATPOWER)
    assert topology.initial_features is not None
    assert topology.bus_types is not None
    hetero = homogeneous_features_to_typed_hetero(
        topology.initial_features,
        topology.edge_index,
        topology.bus_types,
    )
    assert set(hetero.node_types) == {"generator", "slack"}
    assert hetero["generator"].x.shape == (1, 4)
    assert hetero["slack"].x.shape == (1, 4)
    assert ("slack", "branch", "generator") in hetero.edge_types
    assert ("generator", "branch", "slack") in hetero.edge_types


def test_load_typed_topology_partitions_all_buses() -> None:
    """Cached IEEE 118 typed topology covers every bus exactly once."""
    typed = IEEE118DynamicBenchmark.load_typed_topology()
    assert isinstance(typed, TypedIEEE118Topology)
    assert typed.node_type_names == ("generator", "load", "slack")
    assert sum(typed.num_nodes_dict.values()) == NUM_BUSES
    assert typed.num_nodes_dict == {"generator": 53, "load": 64, "slack": 1}
    assert all(width == 4 for width in typed.feature_dims.values())
    assert "simulated" in typed.dynamics_disclaimer.lower()
    assert "simulated" in SIMULATED_DYNAMICS_DISCLAIMER.lower()


def test_generate_typed_seeded_smoke() -> None:
    """Seeded typed generate returns a HeteroGraphSnapshotSequence."""
    first = IEEE118DynamicBenchmark.generate_typed(num_timesteps=4, seed=11)
    second = IEEE118DynamicBenchmark.generate_typed(num_timesteps=4, seed=11)
    assert isinstance(first, HeteroGraphSnapshotSequence)
    assert first.num_timesteps == 4
    assert first.num_nodes_total == NUM_BUSES
    assert first.node_type_names == ("generator", "load", "slack")
    for name in first.node_type_names:
        assert torch.allclose(first[0][name].x, second[0][name].x)
        assert first[0][name].x.shape[-1] == 4
