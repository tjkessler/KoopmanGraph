"""Tests for SocioPatterns contact-epidemic cache helpers (TASK-1317)."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import ContactEpidemicBenchmark
from koopman_graph.datasets.contact_epidemic import (
    CONTACTS_SHA256,
    METADATA_SHA256,
    NUM_NODES,
    _default_cache_path,
    build_contact_cache_payload,
    download_contacts_bytes,
    download_metadata_text,
    ensure_contact_cache,
    load_contact_cache,
    parse_contact_events,
    parse_metadata,
)
from koopman_graph.datasets.download import verify_sha256


def _tiny_metadata(num_nodes: int = 3) -> str:
    """Return SocioPatterns-style metadata for a tiny synthetic school."""
    lines = [f"{1000 + i}\t1A\tM" for i in range(num_nodes)]
    return "\n".join(lines) + "\n"


def _tiny_contacts_tsv(
    *,
    t0: int = 10_000,
    duration_s: int = 7_200,
    num_nodes: int = 3,
) -> bytes:
    """Gzipped contact events spanning ``duration_s`` from ``t0``."""
    lines: list[str] = []
    for time_s in range(t0, t0 + duration_s, 20):
        lines.append(f"{time_s}\t1000\t1001\t1A\t1A")
        if num_nodes >= 3:
            lines.append(f"{time_s}\t1001\t1002\t1A\t1A")
    return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    """Checksum mismatches raise ValueError."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"actual")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_documented_digests_are_hex() -> None:
    """Published SocioPatterns digests are 64-char hex strings."""
    assert len(CONTACTS_SHA256) == 64
    assert len(METADATA_SHA256) == 64
    int(CONTACTS_SHA256, 16)
    int(METADATA_SHA256, 16)


def test_parse_metadata_and_events_round_trip() -> None:
    """Tiny TSV fixtures parse to expected counts."""
    meta = _tiny_metadata(3)
    with patch("koopman_graph.datasets.contact_epidemic.NUM_NODES", 3):
        node_ids, classes, genders = parse_metadata(meta)
    assert node_ids == [1000, 1001, 1002]
    assert classes == ["1A", "1A", "1A"]
    assert genders == ["M", "M", "M"]
    events = parse_contact_events(_tiny_contacts_tsv())
    assert events[0] == (10_000, 1000, 1001)
    assert len(events) > 0


def test_build_contact_cache_payload_shapes() -> None:
    """Teaching payload yields (T, N, 1) intensities and weighted edges."""
    meta = _tiny_metadata(3)
    contacts = _tiny_contacts_tsv()
    with patch("koopman_graph.datasets.contact_epidemic.NUM_NODES", 3):
        payload = build_contact_cache_payload(
            contacts,
            meta,
            bin_seconds=3600,
            num_bins=2,
        )
    assert payload["num_nodes"] == 3
    assert payload["speeds"].shape == (2, 3, 1)
    assert payload["speeds"].dtype == torch.float32
    assert payload["edge_index"].shape[0] == 2
    assert payload["edge_weight"] is not None
    assert payload["edge_weight"].numel() == payload["edge_index"].shape[1]
    assert payload["benchmark"] == "contact_epidemic"
    assert payload["license"] == "CC-BY-NC-SA"
    assert payload["feature"] == "contact_intensity"


def test_ensure_contact_cache_reuses_existing(tmp_path: Path) -> None:
    """Existing contact.pt is reused without rebuilding."""
    path = _default_cache_path(tmp_path)
    torch.save({"edge_index": torch.zeros((2, 0), dtype=torch.long)}, path)
    assert ensure_contact_cache(tmp_path) == path


def test_ensure_contact_cache_missing_without_sources_raises(tmp_path: Path) -> None:
    """Missing cache without paths or fetch raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Contact-epidemic cache is missing"):
        ensure_contact_cache(tmp_path, force=True)


def test_ensure_contact_cache_fetch_persists_raw_assets(tmp_path: Path) -> None:
    """``fetch=True`` downloads assets, persists them, and builds contact.pt."""
    contacts = _tiny_contacts_tsv()
    metadata = _tiny_metadata(3)
    with (
        patch("koopman_graph.datasets.contact_epidemic.NUM_NODES", 3),
        patch(
            "koopman_graph.datasets.contact_epidemic.download_contacts_bytes",
            return_value=contacts,
        ),
        patch(
            "koopman_graph.datasets.contact_epidemic.download_metadata_text",
            return_value=metadata,
        ),
    ):
        path = ensure_contact_cache(
            tmp_path,
            force=True,
            fetch=True,
            bin_seconds=3600,
            num_bins=2,
        )
    assert path.exists()
    assert (tmp_path / "primaryschool.csv.gz").read_bytes() == contacts
    assert (tmp_path / "primaryschool_metadata.txt").read_text(
        encoding="utf-8"
    ) == metadata


def test_ensure_contact_cache_builds_from_local_paths(tmp_path: Path) -> None:
    """Local contacts + metadata build a loadable teaching cache."""
    contacts_path = tmp_path / "primaryschool.csv.gz"
    metadata_path = tmp_path / "primaryschool_metadata.txt"
    contacts = _tiny_contacts_tsv()
    contacts_path.write_bytes(contacts)
    metadata_path.write_text(_tiny_metadata(3), encoding="utf-8")
    digest = hashlib.sha256(contacts).hexdigest()
    with patch("koopman_graph.datasets.contact_epidemic.NUM_NODES", 3):
        path = ensure_contact_cache(
            tmp_path,
            force=True,
            contacts_path=contacts_path,
            metadata_path=metadata_path,
            bin_seconds=3600,
            num_bins=2,
            expected_contacts_sha256=digest,
        )
        payload = load_contact_cache(tmp_path)
        sequence = ContactEpidemicBenchmark.load_sequence(tmp_path)
        topology = ContactEpidemicBenchmark.load_topology(tmp_path)
    assert path.exists()
    assert payload["speeds"].shape == (2, 3, 1)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == 3
    assert sequence.in_channels == 1
    assert topology.num_nodes == 3
    assert topology.source_url is not None
    assert topology.sensor_ids is not None
    assert len(topology.sensor_ids) == 3


def test_ensure_contact_cache_sha256_mismatch_raises(tmp_path: Path) -> None:
    """Tampered local contacts fail checksum before cache write."""
    contacts_path = tmp_path / "primaryschool.csv.gz"
    metadata_path = tmp_path / "meta.txt"
    contacts_path.write_bytes(_tiny_contacts_tsv())
    metadata_path.write_text(_tiny_metadata(3), encoding="utf-8")
    with (
        patch("koopman_graph.datasets.contact_epidemic.NUM_NODES", 3),
        pytest.raises(ValueError, match="SHA256 mismatch"),
    ):
        ensure_contact_cache(
            tmp_path,
            force=True,
            contacts_path=contacts_path,
            metadata_path=metadata_path,
            expected_contacts_sha256="0" * 64,
        )


def test_download_contacts_checksum_mismatch_raises() -> None:
    """Tampered contact downloads raise ValueError."""
    raw = b"\x1f\x8bnot-real"
    mock_response = MagicMock()
    mock_response.read.return_value = raw
    mock_response.__enter__.return_value = mock_response
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            return_value=mock_response,
        ),
        pytest.raises(ValueError, match="SHA256 mismatch"),
    ):
        download_contacts_bytes(verify=True)


def test_download_metadata_url_error_raises_oserror() -> None:
    """Network failures surface as OSError."""
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            side_effect=URLError("network down"),
        ),
        pytest.raises(OSError, match="Failed to download"),
    ):
        download_metadata_text()


def test_load_sequence_rejects_wrong_channel_shape(tmp_path: Path) -> None:
    """Cached speeds with wrong channel dim raise ValueError."""
    torch.save(
        {
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "edge_weight": torch.ones(1),
            "speeds": torch.zeros(2, NUM_NODES, 2),
            "num_nodes": NUM_NODES,
            "sensor_ids": [str(i) for i in range(NUM_NODES)],
            "source_url": "https://example.test",
        },
        _default_cache_path(tmp_path),
    )
    with pytest.raises(ValueError, match="Expected speeds shape"):
        ContactEpidemicBenchmark.load_sequence(tmp_path)


def test_contact_epidemic_export_smoke() -> None:
    """Datasets façade exports ContactEpidemicBenchmark."""
    assert ContactEpidemicBenchmark.NUM_NODES == NUM_NODES
    assert ContactEpidemicBenchmark.IN_CHANNELS == 1
    import koopman_graph.datasets as datasets_pkg

    assert "ContactEpidemicBenchmark" in datasets_pkg.__all__
