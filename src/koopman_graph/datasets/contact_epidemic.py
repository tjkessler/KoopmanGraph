"""SocioPatterns primary-school contact-network benchmark (cached download).

Real-telemetry factory: ``load_topology`` / ``load_sequence`` only (no
``generate``). Complements synthetic
:class:`~koopman_graph.datasets.EpidemicNetworkBenchmark` (SIR on ring /
small-world) with a **real face-to-face contact graph** and time-binned
contact-intensity series.

Licensing
---------
Upstream data are CC-BY-NC-SA (SocioPatterns). Raw TSV/GZ archives are
**not** bundled; use :mod:`scripts.download_contact_epidemic` with SHA256
verification. Users are responsible for complying with the non-commercial
ShareAlike terms.
"""

from __future__ import annotations

import gzip
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.download import (
    download_url_bytes,
    resolve_cache_path,
    verify_sha256,
    verify_sha256_bytes,
)
from koopman_graph.datasets.metr_la import normalize_speeds
from koopman_graph.datasets.topology import TopologyPayload

CONTACTS_URL = "https://sociopatterns.org/assets/data/primaryschool.csv.gz"
METADATA_URL = "https://sociopatterns.org/assets/data/primaryschool_metadata.txt"
CONTACTS_SHA256 = "5c93d9f5a61ad44b3c90fb5146d42345ae86dd44d77bc5da2461989d3d547fc9"
METADATA_SHA256 = "92844d1206ba38825d6ca801563cfc5c0f1c993f3efd012012c21aceaeff4329"
DATASET_PAGE_URL = (
    "https://sociopatterns.org/datasets/primary-school-temporal-network-data/"
)

NUM_NODES = 242
IN_CHANNELS = 1
DEFAULT_BIN_SECONDS = 3600
DEFAULT_NUM_BINS = 24
CACHE_FILENAME = "contact.pt"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "contact_epidemic"


def _default_cache_path(cache_dir: Path | None = None) -> Path:
    """Return the default on-disk path for ``contact.pt``.

    Parameters
    ----------

    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.

    Returns
    -------

    Path
        See summary line."""
    return resolve_cache_path(
        cache_dir,
        default_dir=DEFAULT_CACHE_DIR,
        filename=CACHE_FILENAME,
    )


def download_contacts_bytes(*, verify: bool = True) -> bytes:
    """Download the gzipped primary-school contact list.

    Parameters
    ----------

    verify : bool
        See the function signature / summary for ``verify``.

    Returns
    -------

    bytes
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    raw = download_url_bytes(
        CONTACTS_URL, label="SocioPatterns primary-school contacts"
    )
    if verify:
        # Hash the on-wire gzip bytes (not the decompressed TSV).
        verify_sha256_bytes(
            raw,
            CONTACTS_SHA256,
            label="SocioPatterns primary-school contacts",
        )
    return raw


def download_metadata_text(*, verify: bool = True) -> str:
    """Download primary-school node metadata (id, class, gender).

    Parameters
    ----------

    verify : bool
        See the function signature / summary for ``verify``.

    Returns
    -------

    str
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    raw = download_url_bytes(
        METADATA_URL, label="SocioPatterns primary-school metadata"
    )
    if verify:
        verify_sha256_bytes(
            raw,
            METADATA_SHA256,
            label="SocioPatterns primary-school metadata",
        )
    return raw.decode("utf-8")


def parse_metadata(text: str) -> tuple[list[int], list[str], list[str]]:
    """Parse SocioPatterns metadata into ordered node ids / classes / genders.

    Parameters
    ----------

    text : str
        See the function signature / summary for ``text``.

    Returns
    -------

    tuple
        ``(node_ids, classes, genders)`` aligned lists."""
    node_ids: list[int] = []
    classes: list[str] = []
    genders: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            msg = f"metadata line must have id class gender, got {line!r}"
            raise ValueError(msg)
        node_ids.append(int(parts[0]))
        classes.append(parts[1])
        genders.append(parts[2])
    if len(node_ids) != NUM_NODES:
        msg = f"Expected {NUM_NODES} metadata rows, got {len(node_ids)}"
        raise ValueError(msg)
    return node_ids, classes, genders


def _open_contacts_text(contacts: bytes | str | Path) -> io.TextIOBase:
    """Return a text stream for gzipped or plain contact data.

    Parameters
    ----------

    contacts : bytes | str | Path
        See the function signature / summary for ``contacts``.

    Returns
    -------

    io.TextIOBase
        See summary line."""
    if isinstance(contacts, Path):
        raw = contacts.read_bytes()
    elif isinstance(contacts, str):
        raw = contacts.encode("utf-8")
    else:
        raw = contacts
    if raw[:2] == b"\x1f\x8b":
        handle = gzip.GzipFile(fileobj=io.BytesIO(raw))
        return io.TextIOWrapper(handle, encoding="utf-8")
    if isinstance(raw, bytes):
        return io.StringIO(raw.decode("utf-8"))
    return io.StringIO(raw)


def parse_contact_events(
    contacts: bytes | str | Path,
) -> list[tuple[int, int, int]]:
    """Parse ``(t, i, j)`` contact events from SocioPatterns TSV/GZ.

    Parameters
    ----------
    contacts : bytes, str, or Path
        Gzipped or plain tab-separated contact list.

    Returns
    -------
    list of tuple
        Events as ``(time_seconds, node_i, node_j)``.
    """
    events: list[tuple[int, int, int]] = []
    with _open_contacts_text(contacts) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                msg = f"contact line must start with t i j, got {line!r}"
                raise ValueError(msg)
            events.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not events:
        msg = "contact list is empty"
        raise ValueError(msg)
    return events


def build_contact_cache_payload(
    contacts: bytes | str | Path,
    metadata_text: str,
    *,
    bin_seconds: int = DEFAULT_BIN_SECONDS,
    num_bins: int = DEFAULT_NUM_BINS,
    time_offset: int | None = None,
    source_url: str = DATASET_PAGE_URL,
) -> dict[str, Any]:
    """Assemble a teaching-cache payload from contacts + metadata.

    Parameters
    ----------

    contacts : bytes | str | Path
        See the function signature / summary for ``contacts``.
    metadata_text : str
        See the function signature / summary for ``metadata_text``.
    bin_seconds : int
        See the function signature / summary for ``bin_seconds``.
    num_bins : int
        See the function signature / summary for ``num_bins``.
    time_offset : int | None
        See the function signature / summary for ``time_offset``.
    source_url : str
        See the function signature / summary for ``source_url``.

    Returns
    -------

    dict[str, Any]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid.

    Notes
    -----

    Topology edges aggregate undirected contact counts over the teaching
    window. Sequence features are per-bin per-node contact intensities
    (event counts), z-scored along time."""
    if bin_seconds < 1:
        msg = f"bin_seconds must be >= 1, got {bin_seconds}"
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"num_bins must be >= 1, got {num_bins}"
        raise ValueError(msg)

    node_ids, classes, genders = parse_metadata(metadata_text)
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    events = parse_contact_events(contacts)
    t_min = min(event[0] for event in events)
    start = t_min if time_offset is None else int(time_offset)
    end = start + num_bins * bin_seconds

    edge_counts: dict[tuple[int, int], float] = defaultdict(float)
    intensities = np.zeros((num_bins, NUM_NODES), dtype=np.float32)

    for time_s, left, right in events:
        if time_s < start or time_s >= end:
            continue
        if left not in id_to_index or right not in id_to_index:
            continue
        i = id_to_index[left]
        j = id_to_index[right]
        if i == j:
            continue
        bin_index = (time_s - start) // bin_seconds
        intensities[bin_index, i] += 1.0
        intensities[bin_index, j] += 1.0
        key = (i, j) if i < j else (j, i)
        edge_counts[key] += 1.0

    if not edge_counts:
        msg = (
            f"No contacts in teaching window "
            f"[start={start}, bins={num_bins}, bin_seconds={bin_seconds}]"
        )
        raise ValueError(msg)

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    for (i, j), count in sorted(edge_counts.items()):
        src.extend([i, j])
        dst.extend([j, i])
        weights.extend([count, count])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    normalized = normalize_speeds(intensities)

    return {
        "benchmark": "contact_epidemic",
        "sensor_ids": [str(node_id) for node_id in node_ids],
        "node_classes": classes,
        "node_genders": genders,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "speeds": torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1),
        "num_nodes": NUM_NODES,
        "source_url": source_url,
        "bin_seconds": int(bin_seconds),
        "num_bins": int(num_bins),
        "time_offset": int(start),
        "feature": "contact_intensity",
        "license": "CC-BY-NC-SA",
    }


def ensure_contact_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    contacts_path: Path | None = None,
    metadata_path: Path | None = None,
    bin_seconds: int = DEFAULT_BIN_SECONDS,
    num_bins: int = DEFAULT_NUM_BINS,
    time_offset: int | None = None,
    expected_contacts_sha256: str | None = None,
    fetch: bool = False,
) -> Path:
    """Build the contact-epidemic teaching cache if missing.

    Parameters
    ----------

    fetch : bool, optional
        When ``True`` and local paths are omitted, download SocioPatterns
        assets (with built-in SHA256 verification).
    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.
    force : bool
        See the function signature / summary for ``force``.
    contacts_path : Path | None
        See the function signature / summary for ``contacts_path``.
    metadata_path : Path | None
        See the function signature / summary for ``metadata_path``.
    bin_seconds : int
        See the function signature / summary for ``bin_seconds``.
    num_bins : int
        See the function signature / summary for ``num_bins``.
    time_offset : int | None
        See the function signature / summary for ``time_offset``.
    expected_contacts_sha256 : str | None
        See the function signature / summary for ``expected_contacts_sha256``.

    Returns
    -------

    Path
        See summary line."""
    path = _default_cache_path(cache_dir)
    if path.exists() and not force:
        return path

    contacts_bytes: bytes | None = None
    metadata_text: str | None = None
    cache_root = path.parent

    if contacts_path is not None:
        if expected_contacts_sha256 is not None:
            verify_sha256(contacts_path, expected_contacts_sha256)
        contacts_bytes = contacts_path.read_bytes()
    if metadata_path is not None:
        metadata_text = metadata_path.read_text(encoding="utf-8")

    if contacts_bytes is None or metadata_text is None:
        if fetch:
            cache_root.mkdir(parents=True, exist_ok=True)
            if contacts_bytes is None:
                contacts_bytes = download_contacts_bytes(verify=True)
                contacts_dest = cache_root / "primaryschool.csv.gz"
                contacts_dest.write_bytes(contacts_bytes)
            if metadata_text is None:
                metadata_text = download_metadata_text(verify=True)
                metadata_dest = cache_root / "primaryschool_metadata.txt"
                metadata_dest.write_text(metadata_text, encoding="utf-8")
        elif path.exists():
            return path
        else:
            msg = (
                "Contact-epidemic cache is missing. Provide --contacts-path and "
                "--metadata-path to scripts/download_contact_epidemic.py, or "
                "pass --fetch (CC-BY-NC-SA; user responsibility)."
            )
            raise FileNotFoundError(msg)

    payload = build_contact_cache_payload(
        contacts_bytes,
        metadata_text,
        bin_seconds=bin_seconds,
        num_bins=num_bins,
        time_offset=time_offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_contact_cache(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Load cached contact topology and intensity series.

    Parameters
    ----------

    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.
    dtype : torch.dtype
        See the function signature / summary for ``dtype``.

    Returns
    -------

    dict[str, Any]
        See summary line."""
    path = ensure_contact_cache(cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["speeds"] = payload["speeds"].to(dtype=dtype)
    if payload.get("edge_weight") is not None:
        payload["edge_weight"] = payload["edge_weight"].to(dtype=dtype)
    return payload


class ContactEpidemicBenchmark:
    """SocioPatterns primary-school face-to-face contact benchmark.

    Public entry points: ``load_topology`` / ``load_sequence`` (no ``generate``).

    Attributes
    ----------
    NUM_NODES : int
        Fixed individual count (``242``).
    IN_CHANNELS : int
        Contact-intensity feature dimension (``1``).

    Notes
    -----
    Dataset card:

    * **Scope:** Primary-school face-to-face proximity contacts (children +
      teachers), 20-second resolution, used in infectious-disease contact
      studies
    * **Size:** 242 individuals; teaching cache defaults to ``24`` hourly
      bins (``bin_seconds=3600``) of per-node contact intensity
    * **Format:** SocioPatterns TSV/GZ contacts + metadata → cached
      ``contact.pt`` (``float32``, 1 channel, z-scored intensities;
      weighted undirected topology)
    * **Source:** SocioPatterns Primary School temporal network
      (https://sociopatterns.org/datasets/primary-school-temporal-network-data/);
      cite Gemmetto et al., BMC Infect Dis 2014 and Stehlé et al., PLoS ONE
      2011; acknowledge the SocioPatterns collaboration
    * **License:** CC-BY-NC-SA — **do not redistribute** raw archives in this
      repository; fetch-script + SHA256 only; non-commercial / ShareAlike
      obligations remain with the user
    * **Limitations:** Contact intensity is not SIR state; teaching window is
      a short aggregate; class/gender metadata are attributes, not dynamics;
      NC-SA restricts commercial redistribution of the upstream data
    * **Version:** ``contact_epidemic_v1`` teaching-cache schema
    """

    NUM_NODES = NUM_NODES
    IN_CHANNELS = IN_CHANNELS

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load cached contact-graph topology.

        Parameters
        ----------

        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        TopologyPayload
            See summary line.

        Notes
        -----

        Node identifiers are exposed via ``sensor_ids`` (SocioPatterns
        anonymous integer IDs as strings). Class/gender lists are stored in
        the on-disk cache and are not part of ``TopologyPayload`` fields."""
        del dtype  # topology tensors are index/weight only
        payload = load_contact_cache(cache_dir)
        return TopologyPayload(
            edge_index=payload["edge_index"],
            num_nodes=int(payload["num_nodes"]),
            edge_weight=payload.get("edge_weight"),
            sensor_ids=list(payload["sensor_ids"]),
            source_url=payload.get("source_url", DATASET_PAGE_URL),
        )

    @classmethod
    def load_sequence(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached per-node contact-intensity snapshot sequence.

        Parameters
        ----------

        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        GraphSnapshotSequence
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid."""
        payload = load_contact_cache(cache_dir, dtype=dtype)
        speeds = payload["speeds"]
        if speeds.ndim != 3 or speeds.shape[2] != IN_CHANNELS:
            msg = (
                f"Expected speeds shape (T, N, {IN_CHANNELS}), "
                f"got {tuple(speeds.shape)}"
            )
            raise ValueError(msg)
        if int(payload["num_nodes"]) != NUM_NODES:
            msg = f"Expected {NUM_NODES} nodes, got {payload['num_nodes']}"
            raise ValueError(msg)
        return GraphSnapshotSequence.from_arrays(
            speeds,
            payload["edge_index"],
            edge_weight=payload.get("edge_weight"),
            dtype=dtype,
        )
