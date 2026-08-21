"""Load and dump :class:`ExperimentManifest` documents.

JSON is always available. YAML requires PyYAML
(``pip install 'koopman-graph[cli]'``). Dataset SHA-256 checks use
:func:`~koopman_graph.datasets.download.verify_sha256_bytes` and do
**not** hash this manifest document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from koopman_graph.benchmark.schema import (
    DatasetRef,
    ExperimentManifest,
    ManifestError,
    manifest_from_mapping,
    manifest_to_mapping,
)
from koopman_graph.datasets.download import verify_sha256, verify_sha256_bytes

__all__ = [
    "dump_manifest",
    "load_manifest",
    "verify_dataset_hash",
]


def _read_mapping(path: Path) -> dict[str, Any]:
    """Parse a JSON or YAML file into a mapping.

    Parameters
    ----------
    path : Path
        Document path.

    Returns
    -------
    dict
        Top-level mapping.

    Raises
    ------
    ManifestError
        If the file is missing, the suffix is unsupported, or parse fails.
    ImportError
        If a YAML file is requested but PyYAML is not installed.
    """
    if not path.is_file():
        msg = f"manifest file not found: {path}"
        raise ManifestError(msg)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path}: {exc.msg}"
            raise ManifestError(msg) from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            msg = (
                "YAML manifests require PyYAML. Install with: "
                "pip install 'koopman-graph[cli]' (or: pip install 'pyyaml>=6')"
            )
            raise ImportError(msg) from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {path}: {exc}"
            raise ManifestError(msg) from exc
    else:
        msg = (
            f"Unsupported manifest suffix {suffix!r} for {path}; "
            "use .json, .yaml, or .yml"
        )
        raise ManifestError(msg)
    if not isinstance(payload, dict):
        msg = f"manifest root must be a mapping, got {type(payload).__name__}"
        raise ManifestError(msg)
    return payload


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate a manifest file.

    Parameters
    ----------
    path : str or Path
        ``.json``, ``.yaml``, or ``.yml`` path.

    Returns
    -------
    ExperimentManifest
        Frozen validated record.

    Raises
    ------
    ManifestError
        If the file or schema is invalid.
    ImportError
        If YAML is requested without PyYAML.
    """
    return manifest_from_mapping(_read_mapping(Path(path)))


def dump_manifest(manifest: ExperimentManifest, path: str | Path) -> None:
    """Write a manifest as JSON or YAML.

    Parameters
    ----------
    manifest : ExperimentManifest
        Validated record.
    path : str or Path
        Destination; suffix selects JSON vs YAML.

    Raises
    ------
    ManifestError
        If the suffix is unsupported.
    ImportError
        If YAML is requested without PyYAML.
    """
    destination = Path(path)
    payload = manifest_to_mapping(manifest)
    suffix = destination.suffix.lower()
    if suffix == ".json":
        destination.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            msg = (
                "YAML manifests require PyYAML. Install with: "
                "pip install 'koopman-graph[cli]' (or: pip install 'pyyaml>=6')"
            )
            raise ImportError(msg) from exc
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return
    msg = (
        f"Unsupported manifest suffix {suffix!r} for {destination}; "
        "use .json, .yaml, or .yml"
    )
    raise ManifestError(msg)


def verify_dataset_hash(
    dataset: DatasetRef,
    payload: bytes | str | Path,
) -> None:
    """Reject a SHA-256 mismatch against ``dataset.sha256``.

    This checks dataset payload bytes, not the manifest document.

    Parameters
    ----------
    dataset : DatasetRef
        Declared digest and label.
    payload : bytes or path-like
        Raw bytes or a file to hash.

    Raises
    ------
    ManifestError
        If the digest does not match.
    """
    label = dataset.name
    try:
        if isinstance(payload, bytes):
            verify_sha256_bytes(payload, dataset.sha256, label=label)
        else:
            verify_sha256(Path(payload), dataset.sha256)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
