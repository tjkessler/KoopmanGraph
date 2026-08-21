"""Identity-bound benchmark summaries (not a training loop).

``run_manifest`` verifies the dataset payload digest, records protocol
identity, and writes a canonical SHA-256. It does **not** fit
:class:`~koopman_graph.model.GraphKoopmanModel` or GNN teaching ports
and does not invent forecast metrics. Method execution is a later
increment.

``verify_summary`` recomputes that digest and binds the summary to the
loaded manifest. Tampering a field or the stored hash fails.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from koopman_graph import __version__
from koopman_graph.benchmark.io import load_manifest, verify_dataset_hash
from koopman_graph.benchmark.schema import (
    ExperimentManifest,
    ManifestError,
    _as_int_tuple,
    _as_str_tuple,
    _nonempty_str,
    _reject_unknown,
    _require_bool,
    _require_mapping,
    manifest_to_mapping,
)

__all__ = [
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "BenchmarkSummary",
    "SummaryError",
    "SummaryMethodRef",
    "build_summary",
    "canonical_sha256",
    "dump_summary",
    "load_summary",
    "resolve_summary_path",
    "run_manifest",
    "summary_from_mapping",
    "summary_to_mapping",
    "verify_summary",
]

SUMMARY_SCHEMA_VERSION = "benchmark_summary_v1"
SUMMARY_FILENAME = "summary.json"
_SHA256_HEX = 64

SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "dataset_sha256",
        "track",
        "methods",
        "seeds",
        "horizons",
        "metrics",
        "controls",
        "package_version",
        "executed",
        "summary_sha256",
    }
)
_METHOD_REF_KEYS: frozenset[str] = frozenset({"name", "role"})


class SummaryError(ValueError):
    """Tampered, unbound, or invalid benchmark summary.

    Notes
    -----
    Raised by summary load and :func:`verify_summary`.
    """


@dataclass(frozen=True)
class SummaryMethodRef:
    """Method identity copied onto a summary.

    Attributes
    ----------
    name : str
        Method identifier.
    role : str
        Honesty class from the manifest.
    """

    name: str
    role: str

    def __post_init__(self) -> None:
        """Validate name and role strings.

        Raises
        ------
        SummaryError
            If a field is empty.
        """
        try:
            object.__setattr__(
                self, "name", _nonempty_str(self.name, name="summary.method.name")
            )
            object.__setattr__(
                self, "role", _nonempty_str(self.role, name="summary.method.role")
            )
        except ManifestError as exc:
            raise SummaryError(str(exc)) from exc


@dataclass(frozen=True)
class BenchmarkSummary:
    """Hashed protocol identity for ``benchmark run`` / ``verify``.

    ``executed`` is ``False`` in this increment: the summary does not
    contain fitted forecast metrics.

    Attributes
    ----------
    schema_version : {"benchmark_summary_v1"}
        Summary schema name.
    manifest_id : str
        Copied from the manifest.
    manifest_sha256, dataset_sha256 : str
        Hex SHA-256 of the canonical manifest mapping and dataset payload.
    track : str
        Evidence track.
    methods : tuple of SummaryMethodRef
        Method names and roles.
    seeds, horizons : tuple of int
        Copied grids.
    metrics, controls : tuple of str
        Copied names.
    package_version : str
        ``koopman_graph.__version__`` at run time.
    executed : bool
        Whether method metrics were computed. Default ``False``.
    summary_sha256 : str
        Hex SHA-256 of the canonical mapping with this field omitted.
    """

    schema_version: Literal["benchmark_summary_v1"]
    manifest_id: str
    manifest_sha256: str
    dataset_sha256: str
    track: str
    methods: tuple[SummaryMethodRef, ...]
    seeds: tuple[int, ...]
    horizons: tuple[int, ...]
    metrics: tuple[str, ...]
    controls: tuple[str, ...]
    package_version: str
    executed: bool
    summary_sha256: str

    def __post_init__(self) -> None:
        """Validate identity fields and digests.

        Raises
        ------
        SummaryError
            If a required field is invalid.
        """
        if self.schema_version != SUMMARY_SCHEMA_VERSION:
            msg = (
                f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
            raise SummaryError(msg)
        try:
            object.__setattr__(
                self,
                "manifest_id",
                _nonempty_str(self.manifest_id, name="summary.manifest_id"),
            )
            object.__setattr__(
                self, "track", _nonempty_str(self.track, name="summary.track")
            )
            object.__setattr__(
                self,
                "package_version",
                _nonempty_str(self.package_version, name="summary.package_version"),
            )
            object.__setattr__(
                self, "executed", _require_bool(self.executed, name="summary.executed")
            )
            object.__setattr__(
                self,
                "seeds",
                _as_int_tuple(self.seeds, name="summary.seeds", minimum=0),
            )
            object.__setattr__(
                self,
                "horizons",
                _as_int_tuple(self.horizons, name="summary.horizons", minimum=1),
            )
            object.__setattr__(
                self,
                "metrics",
                _as_str_tuple(self.metrics, name="summary.metrics"),
            )
            object.__setattr__(
                self,
                "controls",
                _as_str_tuple(self.controls, name="summary.controls"),
            )
        except ManifestError as exc:
            raise SummaryError(str(exc)) from exc
        object.__setattr__(
            self, "manifest_sha256", _require_sha256(self.manifest_sha256)
        )
        object.__setattr__(self, "dataset_sha256", _require_sha256(self.dataset_sha256))
        object.__setattr__(self, "summary_sha256", _require_sha256(self.summary_sha256))
        if not self.methods:
            msg = "summary.methods must be non-empty"
            raise SummaryError(msg)
        for method in self.methods:
            if not isinstance(method, SummaryMethodRef):
                msg = (
                    "summary.methods entries must be SummaryMethodRef, "
                    f"got {type(method).__name__}"
                )
                raise SummaryError(msg)


def _require_sha256(value: object) -> str:
    """Return a lowercase 64-character hex digest.

    Parameters
    ----------
    value : object
        Candidate digest.

    Returns
    -------
    str
        Lowercase hex SHA-256.

    Raises
    ------
    SummaryError
        If ``value`` is not a 64-character hex string.
    """
    if not isinstance(value, str) or len(value) != _SHA256_HEX:
        msg = "digest must be a 64-character hex SHA-256"
        raise SummaryError(msg)
    try:
        int(value, 16)
    except ValueError as exc:
        msg = "digest must be a 64-character hex SHA-256"
        raise SummaryError(msg) from exc
    return value.lower()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 of canonical UTF-8 JSON.

    Keys are sorted. Separators are compact ``(",", ":")``.

    Parameters
    ----------
    payload : mapping
        JSON-friendly mapping.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summary_to_mapping(summary: BenchmarkSummary) -> dict[str, Any]:
    """Return a JSON-friendly nested mapping.

    Parameters
    ----------
    summary : BenchmarkSummary
        Validated record.

    Returns
    -------
    dict
        Nested mapping suitable for JSON dump.
    """
    return {
        "schema_version": summary.schema_version,
        "manifest_id": summary.manifest_id,
        "manifest_sha256": summary.manifest_sha256,
        "dataset_sha256": summary.dataset_sha256,
        "track": summary.track,
        "methods": [
            {"name": method.name, "role": method.role} for method in summary.methods
        ],
        "seeds": list(summary.seeds),
        "horizons": list(summary.horizons),
        "metrics": list(summary.metrics),
        "controls": list(summary.controls),
        "package_version": summary.package_version,
        "executed": summary.executed,
        "summary_sha256": summary.summary_sha256,
    }


def _body_mapping(summary: BenchmarkSummary) -> dict[str, Any]:
    """Return the canonical mapping without ``summary_sha256``.

    Parameters
    ----------
    summary : BenchmarkSummary
        Record whose digest is computed.

    Returns
    -------
    dict
        JSON-friendly mapping.
    """
    payload = summary_to_mapping(summary)
    payload.pop("summary_sha256", None)
    return payload


def _method_ref_from_mapping(payload: Mapping[str, Any]) -> SummaryMethodRef:
    """Build :class:`SummaryMethodRef` from a mapping.

    Parameters
    ----------
    payload : mapping
        Method identity.

    Returns
    -------
    SummaryMethodRef
        Validated reference.

    Raises
    ------
    SummaryError
        If keys are missing or unknown.
    """
    try:
        mapping = _require_mapping(payload, name="summary.method")
        _reject_unknown(mapping, _METHOD_REF_KEYS, name="summary.method")
    except ManifestError as exc:
        raise SummaryError(str(exc)) from exc
    missing = sorted(_METHOD_REF_KEYS - set(mapping))
    if missing:
        msg = f"summary.method missing required keys: {', '.join(missing)}"
        raise SummaryError(msg)
    return SummaryMethodRef(name=mapping["name"], role=mapping["role"])


def summary_from_mapping(payload: Mapping[str, Any]) -> BenchmarkSummary:
    """Build :class:`BenchmarkSummary` from a JSON mapping.

    Parameters
    ----------
    payload : mapping
        Summary document.

    Returns
    -------
    BenchmarkSummary
        Validated record.

    Raises
    ------
    SummaryError
        If keys or nested objects are invalid.
    """
    try:
        mapping = _require_mapping(payload, name="summary")
        _reject_unknown(mapping, SUMMARY_KEYS, name="summary")
    except ManifestError as exc:
        raise SummaryError(str(exc)) from exc
    missing = sorted(SUMMARY_KEYS - set(mapping))
    if missing:
        msg = f"summary missing required keys: {', '.join(missing)}"
        raise SummaryError(msg)
    methods_raw = mapping["methods"]
    if not isinstance(methods_raw, Sequence) or isinstance(methods_raw, (str, bytes)):
        msg = "summary.methods must be a sequence of mappings"
        raise SummaryError(msg)
    return BenchmarkSummary(
        schema_version=mapping["schema_version"],
        manifest_id=mapping["manifest_id"],
        manifest_sha256=mapping["manifest_sha256"],
        dataset_sha256=mapping["dataset_sha256"],
        track=mapping["track"],
        methods=tuple(_method_ref_from_mapping(item) for item in methods_raw),
        seeds=tuple(mapping["seeds"]),
        horizons=tuple(mapping["horizons"]),
        metrics=tuple(mapping["metrics"]),
        controls=tuple(mapping["controls"]),
        package_version=mapping["package_version"],
        executed=mapping["executed"],
        summary_sha256=mapping["summary_sha256"],
    )


def _attach_digest(summary: BenchmarkSummary) -> BenchmarkSummary:
    """Return ``summary`` with ``summary_sha256`` filled.

    Parameters
    ----------
    summary : BenchmarkSummary
        Record whose hash field is ignored when hashing the body.

    Returns
    -------
    BenchmarkSummary
        Record with a matching digest.
    """
    digest = canonical_sha256(_body_mapping(summary))
    return replace(summary, summary_sha256=digest)


def build_summary(manifest: ExperimentManifest) -> BenchmarkSummary:
    """Build an identity-bound summary from a validated manifest.

    Parameters
    ----------
    manifest : ExperimentManifest
        Protocol record.

    Returns
    -------
    BenchmarkSummary
        Summary with ``executed=False`` and a matching digest.
    """
    methods = tuple(
        SummaryMethodRef(name=method.name, role=method.role)
        for method in manifest.methods
    )
    draft = BenchmarkSummary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        manifest_id=manifest.manifest_id,
        manifest_sha256=canonical_sha256(manifest_to_mapping(manifest)),
        dataset_sha256=manifest.dataset.sha256,
        track=manifest.track,
        methods=methods,
        seeds=manifest.seeds,
        horizons=manifest.horizons,
        metrics=manifest.metrics,
        controls=manifest.controls,
        package_version=__version__,
        executed=False,
        summary_sha256="0" * _SHA256_HEX,
    )
    return _attach_digest(draft)


def dump_summary(summary: BenchmarkSummary, path: str | Path) -> None:
    """Write a summary JSON document.

    Parameters
    ----------
    summary : BenchmarkSummary
        Validated record.
    path : str or Path
        Destination ``.json`` path.
    """
    destination = Path(path)
    destination.write_text(
        json.dumps(summary_to_mapping(summary), indent=2) + "\n",
        encoding="utf-8",
    )


def load_summary(path: str | Path) -> BenchmarkSummary:
    """Load a summary JSON document.

    Parameters
    ----------
    path : str or Path
        ``summary.json`` path.

    Returns
    -------
    BenchmarkSummary
        Validated record.

    Raises
    ------
    SummaryError
        If the file is missing or invalid.
    """
    destination = Path(path)
    if not destination.is_file():
        msg = f"summary file not found: {destination}"
        raise SummaryError(msg)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {destination}: {exc.msg}"
        raise SummaryError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"summary root must be a mapping, got {type(payload).__name__}"
        raise SummaryError(msg)
    return summary_from_mapping(payload)


def resolve_summary_path(against: str | Path) -> Path:
    """Resolve ``--against`` to a ``summary.json`` file.

    Parameters
    ----------
    against : str or Path
        Directory containing ``summary.json``, or the file itself.

    Returns
    -------
    Path
        Summary file path.
    """
    destination = Path(against)
    if destination.is_dir():
        return destination / SUMMARY_FILENAME
    return destination


def run_manifest(
    manifest_path: str | Path,
    data_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Verify the dataset digest and write an identity-bound summary.

    Parameters
    ----------
    manifest_path : str or Path
        Manifest JSON/YAML.
    data_path : str or Path
        Dataset payload whose SHA-256 must match ``dataset.sha256``.
    out_dir : str or Path
        Directory that will receive ``summary.json``.

    Returns
    -------
    Path
        Written summary path.

    Raises
    ------
    ManifestError
        If the manifest or dataset digest is invalid.
    """
    manifest = load_manifest(manifest_path)
    verify_dataset_hash(manifest.dataset, data_path)
    summary = build_summary(manifest)
    destination_dir = Path(out_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / SUMMARY_FILENAME
    dump_summary(summary, destination)
    return destination


def verify_summary(manifest_path: str | Path, against: str | Path) -> BenchmarkSummary:
    """Reject a tampered or unbound summary.

    Recomputes ``summary_sha256`` and checks identity fields against
    the loaded manifest (id, track, dataset digest, method names/roles,
    seeds, horizons, metrics, controls).

    Parameters
    ----------
    manifest_path : str or Path
        Manifest JSON/YAML.
    against : str or Path
        Summary file or a directory containing ``summary.json``.

    Returns
    -------
    BenchmarkSummary
        Loaded summary when verification succeeds.

    Raises
    ------
    SummaryError
        If the digest or binding does not match.
    ManifestError
        If the manifest cannot be loaded.
    """
    manifest = load_manifest(manifest_path)
    summary = load_summary(resolve_summary_path(against))
    expected = canonical_sha256(_body_mapping(summary))
    if summary.summary_sha256 != expected:
        msg = (
            "summary_sha256 mismatch: "
            f"got {summary.summary_sha256}, expected {expected}"
        )
        raise SummaryError(msg)
    expected_manifest = canonical_sha256(manifest_to_mapping(manifest))
    if summary.manifest_sha256 != expected_manifest:
        msg = "summary.manifest_sha256 does not match the loaded manifest"
        raise SummaryError(msg)
    if summary.manifest_id != manifest.manifest_id:
        msg = (
            f"summary.manifest_id {summary.manifest_id!r} does not match "
            f"manifest {manifest.manifest_id!r}"
        )
        raise SummaryError(msg)
    if summary.dataset_sha256 != manifest.dataset.sha256:
        msg = "summary.dataset_sha256 does not match manifest.dataset.sha256"
        raise SummaryError(msg)
    if summary.track != manifest.track:
        msg = (
            f"summary.track {summary.track!r} does not match "
            f"manifest {manifest.track!r}"
        )
        raise SummaryError(msg)
    manifest_methods = tuple((method.name, method.role) for method in manifest.methods)
    summary_methods = tuple((method.name, method.role) for method in summary.methods)
    if summary_methods != manifest_methods:
        msg = "summary.methods do not match the loaded manifest"
        raise SummaryError(msg)
    if summary.seeds != manifest.seeds:
        msg = "summary.seeds do not match the loaded manifest"
        raise SummaryError(msg)
    if summary.horizons != manifest.horizons:
        msg = "summary.horizons do not match the loaded manifest"
        raise SummaryError(msg)
    if summary.metrics != manifest.metrics:
        msg = "summary.metrics do not match the loaded manifest"
        raise SummaryError(msg)
    if summary.controls != manifest.controls:
        msg = "summary.controls do not match the loaded manifest"
        raise SummaryError(msg)
    return summary
