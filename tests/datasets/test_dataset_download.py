"""Unit tests for shared dataset HTTP download helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from koopman_graph.datasets.download import (
    download_url_bytes,
    download_url_text,
    download_url_to_path,
    resolve_cache_path,
    resolve_fetch_sha256,
    verify_sha256,
    verify_sha256_bytes,
)


def test_download_url_bytes_and_text_round_trip() -> None:
    """Successful fetches return raw bytes / decoded text."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"hello"
    mock_response.__enter__.return_value = mock_response
    with patch(
        "koopman_graph.datasets.download.urlopen",
        return_value=mock_response,
    ):
        assert download_url_bytes("https://example.test/a", label="demo") == b"hello"
        assert download_url_text("https://example.test/a", label="demo") == "hello"


def test_download_url_bytes_wraps_urlerror() -> None:
    """URLError becomes OSError with the caller label."""
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            side_effect=URLError("down"),
        ),
        pytest.raises(OSError, match="Failed to download demo"),
    ):
        download_url_bytes("https://example.test/a", label="demo")


def test_resolve_cache_path_defaults_and_override(tmp_path: Path) -> None:
    """Cache path uses default_dir when cache_dir is None."""
    default_dir = tmp_path / "default"
    assert (
        resolve_cache_path(
            None,
            default_dir=default_dir,
            filename="x.pt",
        )
        == default_dir / "x.pt"
    )
    custom = tmp_path / "custom"
    assert (
        resolve_cache_path(
            custom,
            default_dir=default_dir,
            filename="x.pt",
        )
        == custom / "x.pt"
    )


def test_verify_sha256_bytes_accepts_and_rejects() -> None:
    """In-memory digests match or raise ValueError."""
    payload = b"koopman-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    verify_sha256_bytes(payload, digest, label="demo")
    with pytest.raises(ValueError, match="SHA256 mismatch for demo"):
        verify_sha256_bytes(payload, "0" * 64, label="demo")


def test_verify_sha256_path_helpers(tmp_path: Path) -> None:
    """Path-based checksum helper accepts matches and rejects mismatches."""
    path = tmp_path / "blob.bin"
    payload = b"koopman-path"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    verify_sha256(path, digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256(path, "0" * 64)
    with pytest.raises(FileNotFoundError, match="checksum target is missing"):
        verify_sha256(tmp_path / "missing.bin", "0" * 64)


def test_download_url_to_path_streams_and_verifies(tmp_path: Path) -> None:
    """File downloads stream chunks and optionally verify SHA256."""
    payload = b"abcdefghij" * 100
    digest = hashlib.sha256(payload).hexdigest()
    chunks = [payload[i : i + 16] for i in range(0, len(payload), 16)] + [b""]

    mock_response = MagicMock()
    mock_response.read.side_effect = chunks
    mock_response.__enter__.return_value = mock_response

    destination = tmp_path / "nested" / "artifact.bin"
    with patch(
        "koopman_graph.datasets.download.urlopen",
        return_value=mock_response,
    ):
        path = download_url_to_path(
            "https://example.test/a.bin",
            destination,
            label="demo artifact",
            expected_sha256=digest,
            chunk_size=16,
        )
    assert path == destination
    assert path.read_bytes() == payload


def test_download_url_to_path_checksum_mismatch_raises(tmp_path: Path) -> None:
    """Mismatched stream digests raise ValueError."""
    mock_response = MagicMock()
    mock_response.read.side_effect = [b"wrong", b""]
    mock_response.__enter__.return_value = mock_response
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            return_value=mock_response,
        ),
        pytest.raises(ValueError, match="SHA256 mismatch for demo artifact"),
    ):
        download_url_to_path(
            "https://example.test/a.bin",
            tmp_path / "a.bin",
            label="demo artifact",
            expected_sha256="0" * 64,
        )


def test_resolve_fetch_sha256_uses_pin_override_and_requires_custom() -> None:
    """Default mirror uses the pin; overrides win; custom URLs need a digest."""
    default_url = "https://example.test/default.h5"
    default_sha = "a" * 64
    assert (
        resolve_fetch_sha256(
            expected_sha256=None,
            url=default_url,
            default_url=default_url,
            default_sha256=default_sha,
            label="demo",
        )
        == default_sha
    )
    override = "b" * 64
    assert (
        resolve_fetch_sha256(
            expected_sha256=override,
            url="https://example.test/other.h5",
            default_url=default_url,
            default_sha256=default_sha,
            label="demo",
        )
        == override
    )
    with pytest.raises(ValueError, match="--expected-sha256 is required"):
        resolve_fetch_sha256(
            expected_sha256=None,
            url="https://example.test/other.h5",
            default_url=default_url,
            default_sha256=default_sha,
            label="demo",
        )


def test_download_url_to_path_wraps_urlerror(tmp_path: Path) -> None:
    """File-download URLError becomes OSError."""
    with (
        patch(
            "koopman_graph.datasets.download.urlopen",
            side_effect=URLError("down"),
        ),
        pytest.raises(OSError, match="Failed to download demo artifact"),
    ):
        download_url_to_path(
            "https://example.test/a.bin",
            tmp_path / "a.bin",
            label="demo artifact",
        )
