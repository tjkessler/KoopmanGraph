"""Shared HTTP download helpers for cached benchmark datasets.

Power-user module used by METR-LA, PEMS, IEEE 118, contact-epidemic, and
teaching-cache acquisition scripts. Prefer the public benchmark classmethods
(``load_topology`` / ``load_sequence``); call these helpers only when writing
fetch scripts or new downloaders.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_DOWNLOAD_TIMEOUT = 60.0
DEFAULT_FILE_DOWNLOAD_TIMEOUT = 600.0
DEFAULT_DOWNLOAD_CHUNK_SIZE = 1 << 20


def verify_sha256_bytes(data: bytes, expected: str, *, label: str) -> None:
    """Verify that ``data`` matches a hex SHA256 digest.

    Parameters
    ----------
    data : bytes
        Bytes to hash.
    expected : str
        Expected lowercase or uppercase hex digest.
    label : str
        Human-readable name used in error messages.

    Raises
    ------
    ValueError
        If the digest does not match ``expected``.
    """
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != expected.lower():
        msg = f"SHA256 mismatch for {label}: got {digest}, expected {expected.lower()}"
        raise ValueError(msg)


def verify_sha256(path: Path, expected: str) -> None:
    """Verify that ``path`` matches a hex SHA256 digest.

    Parameters
    ----------
    path : Path
        File to hash.
    expected : str
        Expected lowercase or uppercase hex digest.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the digest does not match ``expected``.
    """
    if not path.is_file():
        msg = f"checksum target is missing: {path}"
        raise FileNotFoundError(msg)
    verify_sha256_bytes(path.read_bytes(), expected, label=str(path))


def download_url_bytes(
    url: str,
    *,
    label: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> bytes:
    """Download raw bytes from ``url`` or raise ``OSError``.

    Parameters
    ----------
    url : str
        Absolute HTTP(S) URL of the artifact.
    label : str
        Human-readable name used in error messages.
    timeout : float, optional
        Socket timeout in seconds. Default is
        :data:`DEFAULT_DOWNLOAD_TIMEOUT`.

    Returns
    -------
    bytes
        Response body.

    Raises
    ------
    OSError
        If the request fails (wraps ``URLError``).
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read()
    except URLError as exc:
        msg = f"Failed to download {label} from {url}"
        raise OSError(msg) from exc


def download_url_text(
    url: str,
    *,
    label: str,
    encoding: str = "utf-8",
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> str:
    """Download text from ``url`` or raise ``OSError``.

    Parameters
    ----------
    url : str
        Absolute HTTP(S) URL of the artifact.
    label : str
        Human-readable name used in error messages.
    encoding : str, optional
        Text decoding. Default is ``"utf-8"``.
    timeout : float, optional
        Socket timeout in seconds. Default is
        :data:`DEFAULT_DOWNLOAD_TIMEOUT`.

    Returns
    -------
    str
        Decoded response body.

    Raises
    ------
    OSError
        If the request fails (wraps ``URLError``).
    """
    return download_url_bytes(url, label=label, timeout=timeout).decode(encoding)


def download_url_to_path(
    url: str,
    destination: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    timeout: float = DEFAULT_FILE_DOWNLOAD_TIMEOUT,
    chunk_size: int = DEFAULT_DOWNLOAD_CHUNK_SIZE,
) -> Path:
    """Stream ``url`` to ``destination``, optionally verifying SHA256.

    Parameters
    ----------
    url : str
        Absolute HTTP(S) URL of the artifact.
    destination : Path
        Local path where the downloaded file is written.
    label : str
        Human-readable name used in error messages.
    expected_sha256 : str, optional
        When provided, hash the stream and raise ``ValueError`` on mismatch.
    timeout : float, optional
        Socket timeout in seconds. Default is
        :data:`DEFAULT_FILE_DOWNLOAD_TIMEOUT` (longer than in-memory fetches
        for large teaching archives).
    chunk_size : int, optional
        Read size in bytes. Default is :data:`DEFAULT_DOWNLOAD_CHUNK_SIZE`.

    Returns
    -------
    Path
        ``destination`` after a successful download.

    Raises
    ------
    OSError
        If the request fails (wraps ``URLError``).
    ValueError
        If ``expected_sha256`` is set and the digest does not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256() if expected_sha256 is not None else None
    try:
        with urlopen(url, timeout=timeout) as response, destination.open("wb") as out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
    except URLError as exc:
        msg = f"Failed to download {label} from {url}"
        raise OSError(msg) from exc

    if hasher is not None and expected_sha256 is not None:
        digest = hasher.hexdigest()
        if digest.lower() != expected_sha256.lower():
            msg = (
                f"SHA256 mismatch for {label}: got {digest}, "
                f"expected {expected_sha256.lower()}"
            )
            raise ValueError(msg)
    return destination


def resolve_fetch_sha256(
    *,
    expected_sha256: str | None,
    url: str,
    default_url: str,
    default_sha256: str,
    label: str,
) -> str:
    """Resolve the SHA256 digest required for a remote ``--fetch`` download.

    Parameters
    ----------
    expected_sha256 : str or None
        Caller-supplied digest override (e.g. ``--expected-sha256``).
    url : str
        Download URL in use.
    default_url : str
        Canonical mirror URL whose content matches ``default_sha256``.
    default_sha256 : str
        Pinned hex digest for ``default_url``.
    label : str
        Human-readable artifact name used in error messages.

    Returns
    -------
    str
        Hex digest that must be verified for the download.

    Raises
    ------
    ValueError
        If ``expected_sha256`` is omitted and ``url`` is not ``default_url``.
    """
    if expected_sha256 is not None:
        return expected_sha256
    if url == default_url:
        return default_sha256
    msg = (
        f"--expected-sha256 is required when downloading {label} from a "
        f"non-default URL (got {url!r}; default mirror is {default_url!r})."
    )
    raise ValueError(msg)


def resolve_cache_path(
    cache_dir: Path | None,
    *,
    default_dir: Path,
    filename: str,
) -> Path:
    """Resolve ``cache_dir / filename``, defaulting ``cache_dir`` when ``None``.

    Parameters
    ----------
    cache_dir : Path or None
        Caller-supplied cache root, or ``None`` for ``default_dir``.
    default_dir : Path
        Fallback cache directory when ``cache_dir`` is ``None``.
    filename : str
        Artifact filename inside the cache root.

    Returns
    -------
    Path
        Absolute or relative path to the cached file.
    """
    root = default_dir if cache_dir is None else cache_dir
    return root / filename
