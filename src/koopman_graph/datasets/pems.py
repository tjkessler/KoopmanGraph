"""PEMS-BAY and PEMS03/04/07/08 traffic benchmarks (cached download).

Real telemetry factories expose ``load_topology`` / ``load_sequence`` only —
there is no ``generate``. Raw Caltrans PeMS archives are **not** bundled;
use :mod:`scripts.download_pems` with SHA256 verification.

Dataset cards (FAIR)
--------------------
See module/class docstrings below and the Sphinx page ``data.rst``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.download import (
    download_url_bytes,
    resolve_cache_path,
    verify_sha256,
    verify_sha256_bytes,
)
from koopman_graph.datasets.metr_la import (
    adjacency_to_edge_index,
    adjacency_to_edge_weight,
    build_adjacency_matrix,
    normalize_speeds,
    preprocess_speeds,
)
from koopman_graph.datasets.topology import TopologyPayload

DCRNN_SENSOR_GRAPH_BASE = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
)
BAY_SENSOR_LOCATIONS_URL = f"{DCRNN_SENSOR_GRAPH_BASE}/graph_sensor_locations_bay.csv"
BAY_DISTANCES_URL = f"{DCRNN_SENSOR_GRAPH_BASE}/distances_bay_2017.csv"
BAY_SENSOR_LOCATIONS_SHA256 = (
    "276ee01059610774d4e59572507f7e32eaac21f1f5882fcd9e3d7d426a4b7a6c"
)
BAY_DISTANCES_SHA256 = (
    "e5feed06bfa1ba4c554a946d0e03d99f2018365eec5a8f28fd8504dea9d082b5"
)
# Public mirrors for the DCRNN PEMS-BAY HDF5 (user may override).
DEFAULT_BAY_H5_MIRROR_URL = (
    "https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/pems-bay.h5"
)
# Content SHA256 of DEFAULT_BAY_H5_MIRROR_URL (HF LFS oid; verified 2026-07-25).
DEFAULT_BAY_H5_SHA256 = (
    "65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f"
)
DCRNN_BAY_H5_GOOGLE_DRIVE = (
    "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
)

PemsVariant = Literal["03", "04", "07", "08"]
VALID_PEMS_VARIANTS: tuple[PemsVariant, ...] = ("03", "04", "07", "08")

VARIANT_NUM_SENSORS: dict[PemsVariant, int] = {
    "03": 358,
    "04": 307,
    "07": 883,
    "08": 170,
}
# Community packaging used by ASTGCN / STFGNN-style NPZ releases (flow channel 0).
VARIANT_COMMUNITY_NOTES: dict[PemsVariant, str] = {
    "03": "Caltrans PeMS District 03; community NPZ+CSV packaging (ASTGCN/STFGNN)",
    "04": "Caltrans PeMS District 04; community NPZ+CSV packaging (ASTGCN/STFGNN)",
    "07": "Caltrans PeMS District 07; community NPZ+CSV packaging (ASTGCN/STFGNN)",
    "08": "Caltrans PeMS District 08; community NPZ+CSV packaging (ASTGCN/STFGNN)",
}

BAY_NUM_SENSORS = 325
IN_CHANNELS = 1
DEFAULT_NUM_TIMESTEPS = 288
DEFAULT_BAY_TIMESTEP_OFFSET = 0
TRAFFIC_FILENAME = "traffic.pt"
DEFAULT_BAY_H5_FILENAME = "pems-bay.h5"
DEFAULT_BAY_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "pems_bay"


def _default_bay_traffic_path(cache_dir: Path | None = None) -> Path:
    """Return the default PEMS-BAY ``traffic.pt`` path.

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
        default_dir=DEFAULT_BAY_CACHE_DIR,
        filename=TRAFFIC_FILENAME,
    )


def default_variant_cache_dir(variant: PemsVariant) -> Path:
    """Return the default cache directory for a PEMS0X variant.

    Prefers a directory that already contains ``traffic.pt``, checking the
    package-rooted ``data/`` tree and common notebook working directories
    (repo root or ``examples/``). When no cache exists yet, returns the
    package-rooted path used by ``scripts/download_pems.py``.

    Parameters
    ----------
    variant : PemsVariant
        PEMS0X district key (``\"03\"`` / ``\"04\"`` / ``\"07\"`` / ``\"08\"``).

    Returns
    -------
    Path
        Cache directory for ``traffic.pt``.
    """
    package_rooted = Path(__file__).resolve().parents[3] / "data" / f"pems{variant}"
    candidates = (
        package_rooted,
        Path.cwd() / "data" / f"pems{variant}",
        Path.cwd().parent / "data" / f"pems{variant}",
    )
    for cache_dir in candidates:
        if (cache_dir / TRAFFIC_FILENAME).is_file():
            return cache_dir.resolve()
    return package_rooted


def _default_variant_traffic_path(
    variant: PemsVariant,
    cache_dir: Path | None = None,
) -> Path:
    """Return the default ``traffic.pt`` path for a PEMS0X variant.

    Parameters
    ----------

    variant : PemsVariant
        See the function signature / summary for ``variant``.
    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.

    Returns
    -------

    Path
        See summary line."""
    return resolve_cache_path(
        cache_dir,
        default_dir=default_variant_cache_dir(variant),
        filename=TRAFFIC_FILENAME,
    )


def normalize_pems_variant(variant: str) -> PemsVariant:
    """Validate and normalize a PEMS0X variant string.

    Parameters
    ----------

    variant : str
        See the function signature / summary for ``variant``.

    Returns
    -------

    PemsVariant
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    key = variant.strip().lower().removeprefix("pems")
    if key not in VALID_PEMS_VARIANTS:
        msg = f"variant must be one of {VALID_PEMS_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    return key  # type: ignore[return-value]


def read_bay_h5_speed_window(
    h5_path: Path,
    *,
    num_timesteps: int,
    offset: int = 0,
) -> np.ndarray:
    """Read a window of PEMS-BAY speeds from a DCRNN-format HDF5 file.

    Parameters
    ----------

    h5_path : Path
        See the function signature / summary for ``h5_path``.
    num_timesteps : int
        See the function signature / summary for ``num_timesteps``.
    offset : int
        See the function signature / summary for ``offset``.

    Returns
    -------

    np.ndarray
        See summary line.

    Raises
    ------

    ImportError
        If ``h5py`` is not installed."""
    try:
        import h5py
    except ImportError as exc:
        msg = "h5py is required to read PEMS-BAY HDF5 files (`pip install h5py`)"
        raise ImportError(msg) from exc

    # DCRNN pandas store uses ``df/``; some public mirrors use ``speed/``.
    candidate_keys = ("df/block0_values", "speed/block0_values")
    with h5py.File(h5_path, "r") as handle:
        values = None
        for key in candidate_keys:
            if key in handle:
                values = handle[key]
                break
        if values is None:
            available = ", ".join(sorted(handle.keys())) or "(empty)"
            msg = (
                "PEMS-BAY HDF5 missing speed table "
                f"(tried {', '.join(candidate_keys)}; top-level keys: {available})"
            )
            raise KeyError(msg)
        total_rows = int(values.shape[0])
        end = offset + num_timesteps
        if offset < 0 or num_timesteps < 1 or end > total_rows:
            msg = (
                f"Requested window offset={offset}, num_timesteps={num_timesteps} "
                f"exceeds available rows ({total_rows})"
            )
            raise ValueError(msg)
        speeds = values[offset:end]
    return np.asarray(speeds, dtype=np.float32)


def download_bay_sensor_ids(*, verify: bool = True) -> list[str]:
    """Download ordered PEMS-BAY sensor IDs from DCRNN locations CSV.

    Parameters
    ----------

    verify : bool, optional
        When ``True`` (default), verify the SHA256 of the downloaded bytes
        against :data:`BAY_SENSOR_LOCATIONS_SHA256`.

    Returns
    -------

    list[str]
        See summary line."""
    raw = download_url_bytes(
        BAY_SENSOR_LOCATIONS_URL, label="PEMS-BAY sensor locations"
    )
    if verify:
        verify_sha256_bytes(
            raw,
            BAY_SENSOR_LOCATIONS_SHA256,
            label="PEMS-BAY sensor locations",
        )
    sensor_ids: list[str] = []
    reader = csv.reader(io.StringIO(raw.decode("utf-8")))
    for row in reader:
        if not row:
            continue
        sensor_ids.append(row[0].strip())
    if len(sensor_ids) != BAY_NUM_SENSORS:
        msg = f"Expected {BAY_NUM_SENSORS} PEMS-BAY sensors, got {len(sensor_ids)}"
        raise ValueError(msg)
    return sensor_ids


def download_bay_distances_csv(*, verify: bool = True) -> str:
    """Download the PEMS-BAY pairwise road-distance CSV from DCRNN.

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
    raw = download_url_bytes(BAY_DISTANCES_URL, label="PEMS-BAY distances")
    if verify:
        verify_sha256_bytes(
            raw,
            BAY_DISTANCES_SHA256,
            label="PEMS-BAY distances",
        )
    text = raw.decode("utf-8")
    # Upstream BAY file is headerless; METR-LA / build_adjacency_matrix expect
    # ``from,to,cost``.
    first_line = text.splitlines()[0] if text else ""
    if first_line.strip().lower() != "from,to,cost":
        text = "from,to,cost\n" + text.lstrip("\n")
    return text


def read_npz_flow_window(
    npz_path: Path,
    *,
    num_timesteps: int,
    offset: int = 0,
    channel: int = 0,
) -> np.ndarray:
    """Read a window of traffic-flow values from a community PEMS NPZ.

    Parameters
    ----------
    npz_path : Path
        Path to ``PEMSXX.npz`` with a ``data`` array of shape ``(T, N, C)``.
    num_timesteps : int
        Number of consecutive 5-minute readings to load.
    offset : int, optional
        Starting row offset. Default is ``0``.
    channel : int, optional
        Feature channel (``0`` = flow in ASTGCN/STFGNN packaging). Default
        ``0``.

    Returns
    -------
    ndarray
        Flow array with shape ``(num_timesteps, num_sensors)``.
    """
    payload = np.load(npz_path)
    if "data" not in payload:
        msg = f"PEMS NPZ must contain a 'data' array, keys={list(payload.keys())}"
        raise ValueError(msg)
    values = np.asarray(payload["data"])
    if values.ndim != 3:
        msg = f"Expected data shape (T, N, C), got {values.shape}"
        raise ValueError(msg)
    total_rows = int(values.shape[0])
    end = offset + num_timesteps
    if offset < 0 or num_timesteps < 1 or end > total_rows:
        msg = (
            f"Requested window offset={offset}, num_timesteps={num_timesteps} "
            f"exceeds available rows ({total_rows})"
        )
        raise ValueError(msg)
    if channel < 0 or channel >= values.shape[2]:
        msg = f"channel={channel} is out of range for C={values.shape[2]}"
        raise ValueError(msg)
    return np.asarray(values[offset:end, :, channel], dtype=np.float32)


def read_adjacency_csv(csv_path: Path) -> np.ndarray:
    """Load a dense square adjacency matrix from CSV (no header).

    Parameters
    ----------
    csv_path : Path
        Path to an ``N×N`` adjacency CSV as distributed with community PEMS
        NPZ packages.

    Returns
    -------
    ndarray
        Adjacency matrix with shape ``(N, N)`` and dtype ``float32``.
    """
    adj = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        msg = f"adjacency CSV must be square, got shape {adj.shape}"
        raise ValueError(msg)
    return adj


def build_bay_cache_payload(
    speeds: np.ndarray,
    sensor_ids: list[str],
    *,
    distance_csv: str,
    normalized_k: float = 0.1,
    source_h5_url: str = DEFAULT_BAY_H5_MIRROR_URL,
    timestep_offset: int = 0,
) -> dict[str, Any]:
    """Assemble a PEMS-BAY ``traffic.pt`` payload.

    Parameters
    ----------

    speeds : np.ndarray
        See the function signature / summary for ``speeds``.
    sensor_ids : list[str]
        See the function signature / summary for ``sensor_ids``.
    distance_csv : str
        See the function signature / summary for ``distance_csv``.
    normalized_k : float
        See the function signature / summary for ``normalized_k``.
    source_h5_url : str
        See the function signature / summary for ``source_h5_url``.
    timestep_offset : int
        See the function signature / summary for ``timestep_offset``.

    Returns
    -------

    dict[str, Any]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if speeds.ndim != 2:
        msg = f"speeds must have shape (num_timesteps, num_sensors), got {speeds.shape}"
        raise ValueError(msg)
    if speeds.shape[1] != len(sensor_ids):
        msg = (
            f"speeds has {speeds.shape[1]} sensors but sensor_ids has "
            f"{len(sensor_ids)} entries"
        )
        raise ValueError(msg)
    adj_mx = build_adjacency_matrix(distance_csv, sensor_ids, normalized_k=normalized_k)
    edge_index = adjacency_to_edge_index(adj_mx)
    edge_weight = adjacency_to_edge_weight(adj_mx)
    cleaned = preprocess_speeds(speeds)
    normalized = normalize_speeds(cleaned)
    return {
        "benchmark": "pems_bay",
        "sensor_ids": sensor_ids,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "speeds": torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1),
        "num_nodes": len(sensor_ids),
        "source_h5_url": source_h5_url,
        "timestep_offset": timestep_offset,
        "num_timesteps_cached": int(speeds.shape[0]),
        "normalized_k": normalized_k,
        "feature": "speed",
    }


def build_variant_cache_payload(
    flows: np.ndarray,
    adj_mx: np.ndarray,
    *,
    variant: PemsVariant,
    source_url: str,
    timestep_offset: int = 0,
) -> dict[str, Any]:
    """Assemble a PEMS0X ``traffic.pt`` payload from flow + adjacency.

    Parameters
    ----------

    flows : np.ndarray
        See the function signature / summary for ``flows``.
    adj_mx : np.ndarray
        See the function signature / summary for ``adj_mx``.
    variant : PemsVariant
        See the function signature / summary for ``variant``.
    source_url : str
        See the function signature / summary for ``source_url``.
    timestep_offset : int
        See the function signature / summary for ``timestep_offset``.

    Returns
    -------

    dict[str, Any]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    expected = VARIANT_NUM_SENSORS[variant]
    if flows.ndim != 2:
        msg = f"flows must have shape (num_timesteps, num_sensors), got {flows.shape}"
        raise ValueError(msg)
    if flows.shape[1] != expected:
        msg = f"PEMS{variant} expects {expected} sensors, got {flows.shape[1]}"
        raise ValueError(msg)
    if adj_mx.shape != (expected, expected):
        msg = (
            f"PEMS{variant} adjacency must have shape "
            f"{(expected, expected)}, got {adj_mx.shape}"
        )
        raise ValueError(msg)
    edge_index = adjacency_to_edge_index(adj_mx)
    edge_weight = adjacency_to_edge_weight(adj_mx)
    cleaned = preprocess_speeds(flows)  # same missing-zero imputation
    normalized = normalize_speeds(cleaned)
    sensor_ids = [f"pems{variant}_{index:04d}" for index in range(expected)]
    return {
        "benchmark": f"pems{variant}",
        "variant": variant,
        "sensor_ids": sensor_ids,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "speeds": torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1),
        "num_nodes": expected,
        "source_url": source_url,
        "timestep_offset": timestep_offset,
        "num_timesteps_cached": int(flows.shape[0]),
        "feature": "flow",
    }


def ensure_bay_traffic_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    h5_path: Path | None = None,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    offset: int = DEFAULT_BAY_TIMESTEP_OFFSET,
    normalized_k: float = 0.1,
    expected_h5_sha256: str | None = None,
) -> Path:
    """Build the PEMS-BAY teaching cache if missing.

    Parameters
    ----------

    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.
    force : bool
        See the function signature / summary for ``force``.
    h5_path : Path | None
        See the function signature / summary for ``h5_path``.
    num_timesteps : int
        See the function signature / summary for ``num_timesteps``.
    offset : int
        See the function signature / summary for ``offset``.
    normalized_k : float
        See the function signature / summary for ``normalized_k``.
    expected_h5_sha256 : str | None
        See the function signature / summary for ``expected_h5_sha256``.

    Returns
    -------

    Path
        See summary line.

    Raises
    ------

    FileNotFoundError
        Raised when inputs are invalid.
    ValueError
        Raised when inputs are invalid.

    Notes
    -----

    When ``expected_h5_sha256`` is provided, the local HDF5 is verified before
    reading."""
    path = _default_bay_traffic_path(cache_dir)
    if path.exists() and not force:
        return path
    if h5_path is None:
        if path.exists():
            return path
        msg = (
            "PEMS-BAY cache is missing. Provide --h5-path or --fetch to "
            "scripts/download_pems.py after obtaining pems-bay.h5 from the "
            "DCRNN release."
        )
        raise FileNotFoundError(msg)
    if expected_h5_sha256 is not None:
        verify_sha256(h5_path, expected_h5_sha256)

    sensor_ids = download_bay_sensor_ids()
    distance_csv = download_bay_distances_csv()
    speeds = read_bay_h5_speed_window(
        h5_path, num_timesteps=num_timesteps, offset=offset
    )
    if speeds.shape[1] != BAY_NUM_SENSORS:
        msg = f"Expected {BAY_NUM_SENSORS} PEMS-BAY sensors, got {speeds.shape[1]}"
        raise ValueError(msg)
    payload = build_bay_cache_payload(
        speeds,
        sensor_ids,
        distance_csv=distance_csv,
        normalized_k=normalized_k,
        timestep_offset=offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def ensure_variant_traffic_cache(
    variant: str,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    npz_path: Path | None = None,
    adj_csv_path: Path | None = None,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    offset: int = 0,
    expected_npz_sha256: str | None = None,
    source_url: str | None = None,
) -> Path:
    """Build a PEMS0X teaching cache from local NPZ + adjacency CSV.

    Parameters
    ----------

    variant : str
        See the function signature / summary for ``variant``.
    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.
    force : bool
        See the function signature / summary for ``force``.
    npz_path : Path | None
        See the function signature / summary for ``npz_path``.
    adj_csv_path : Path | None
        See the function signature / summary for ``adj_csv_path``.
    num_timesteps : int
        See the function signature / summary for ``num_timesteps``.
    offset : int
        See the function signature / summary for ``offset``.
    expected_npz_sha256 : str | None
        See the function signature / summary for ``expected_npz_sha256``.
    source_url : str | None
        See the function signature / summary for ``source_url``.

    Returns
    -------

    Path
        See summary line.

    Raises
    ------

    FileNotFoundError
        Raised when inputs are invalid."""
    key = normalize_pems_variant(variant)
    path = _default_variant_traffic_path(key, cache_dir)
    if path.exists() and not force:
        return path
    if npz_path is None or adj_csv_path is None:
        if path.exists():
            return path
        msg = (
            f"PEMS{key} cache is missing. Provide --npz-path and --adj-csv "
            "to scripts/download_pems.py after obtaining the community NPZ "
            "package."
        )
        raise FileNotFoundError(msg)
    if expected_npz_sha256 is not None:
        verify_sha256(npz_path, expected_npz_sha256)

    flows = read_npz_flow_window(
        npz_path, num_timesteps=num_timesteps, offset=offset, channel=0
    )
    adj_mx = read_adjacency_csv(adj_csv_path)
    payload = build_variant_cache_payload(
        flows,
        adj_mx,
        variant=key,
        source_url=source_url or str(npz_path),
        timestep_offset=offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_bay_traffic_cache(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Load cached PEMS-BAY topology and speed readings.

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
    path = ensure_bay_traffic_cache(cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["speeds"] = payload["speeds"].to(dtype=dtype)
    if payload.get("edge_weight") is not None:
        payload["edge_weight"] = payload["edge_weight"].to(dtype=dtype)
    return payload


def load_variant_traffic_cache(
    variant: str,
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Load cached PEMS0X topology and flow readings.

    Parameters
    ----------

    variant : str
        See the function signature / summary for ``variant``.
    cache_dir : Path | None
        See the function signature / summary for ``cache_dir``.
    dtype : torch.dtype
        See the function signature / summary for ``dtype``.

    Returns
    -------

    dict[str, Any]
        See summary line."""
    key = normalize_pems_variant(variant)
    path = ensure_variant_traffic_cache(key, cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["speeds"] = payload["speeds"].to(dtype=dtype)
    if payload.get("edge_weight") is not None:
        payload["edge_weight"] = payload["edge_weight"].to(dtype=dtype)
    return payload


def _sequence_from_payload(
    payload: dict[str, Any],
    *,
    expected_nodes: int,
    dtype: torch.dtype,
) -> GraphSnapshotSequence:
    """Validate a cache payload and build a snapshot sequence.

    Parameters
    ----------

    payload : dict[str, Any]
        See the function signature / summary for ``payload``.
    expected_nodes : int
        See the function signature / summary for ``expected_nodes``.
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
    speeds = payload["speeds"]
    if speeds.ndim != 3 or speeds.shape[2] != IN_CHANNELS:
        msg = f"Expected speeds shape (T, N, {IN_CHANNELS}), got {tuple(speeds.shape)}"
        raise ValueError(msg)
    num_nodes = int(payload["num_nodes"])
    if num_nodes != expected_nodes:
        msg = f"Expected {expected_nodes} sensors, got {num_nodes}"
        raise ValueError(msg)
    return GraphSnapshotSequence.from_arrays(
        speeds,
        payload["edge_index"],
        edge_weight=payload.get("edge_weight"),
        dtype=dtype,
    )


class PemsBayTrafficBenchmark:
    """PEMS-BAY traffic-speed benchmark (DCRNN / Caltrans PeMS).

    Public entry points: ``load_topology`` / ``load_sequence`` (no ``generate``).

    Attributes
    ----------
    NUM_SENSORS : int
        Fixed sensor count (``325``).
    IN_CHANNELS : int
        Speed feature dimension (``1``).

    Notes
    -----
    Dataset card:

    * **Scope:** Bay Area highway loop-detector speeds (5-minute aggregation)
    * **Size:** 325 sensors; full history ~52k steps; teaching cache defaults
      to one day (``288`` steps)
    * **Format:** DCRNN HDF5 speeds + distance CSV → cached ``traffic.pt``
      (``float32``, 1 channel, z-scored)
    * **Source:** Caltrans PeMS; DCRNN release (`Li2018DCRNN`); sensor graph
      from the DCRNN GitHub ``sensor_graph`` tree
    * **License:** Upstream Caltrans PeMS data-use terms; DCRNN code MIT —
      **do not redistribute** raw HDF5 in this repository (fetch + checksum)
    * **Limitations:** Missing readings encoded as zeros (imputed); teaching
      cache is a short window, not the full six-month corpus; graph is a
      distance-kernel approximation
    * **Version:** ``pems_bay_v1`` teaching cache schema
    """

    NUM_SENSORS = BAY_NUM_SENSORS
    IN_CHANNELS = IN_CHANNELS

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load cached PEMS-BAY graph topology and metadata.

        Parameters
        ----------

        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        TopologyPayload
            See summary line."""
        payload = load_bay_traffic_cache(cache_dir, dtype=dtype)
        return TopologyPayload(
            sensor_ids=list(payload["sensor_ids"]),
            edge_index=payload["edge_index"],
            edge_weight=payload["edge_weight"],
            num_nodes=int(payload["num_nodes"]),
            source_h5_url=payload.get("source_h5_url"),
            normalized_k=float(payload.get("normalized_k", 0.1)),
        )

    @classmethod
    def load_sequence(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached PEMS-BAY speed snapshot sequence.

        Parameters
        ----------

        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        GraphSnapshotSequence
            See summary line."""
        payload = load_bay_traffic_cache(cache_dir, dtype=dtype)
        return _sequence_from_payload(
            payload, expected_nodes=cls.NUM_SENSORS, dtype=dtype
        )


class PemsTrafficBenchmark:
    """PEMS03/04/07/08 traffic-flow benchmarks (community NPZ packaging).

    Construct with ``variant="03"|"04"|"07"|"08"``. Public entry points:
    ``load_topology`` / ``load_sequence`` (no ``generate``).

    Attributes
    ----------
    IN_CHANNELS : int
        Flow feature dimension (``1``).
    variant : str
        Normalized district key (``"03"``, ``"04"``, ``"07"``, or ``"08"``).
    NUM_SENSORS : int
        Sensor count for the selected variant.

    Notes
    -----
    Dataset card:

    * **Scope:** California PeMS district highway **flow** (5-minute)
    * **Size:** sensors — 03:358, 04:307, 07:883, 08:170; teaching cache
      defaults to ``288`` steps from a configurable offset
    * **Format:** community ``PEMSXX.npz`` (``data`` → channel 0 flow) +
      dense adjacency CSV → cached ``traffic.pt`` (``float32``, 1 channel,
      z-scored)
    * **Source:** Caltrans PeMS; community packaging as used by ASTGCN /
      STFGNN-style releases (no invented DOI — cite upstream PeMS and the
      packaging repository you download from)
    * **License:** Upstream Caltrans PeMS data-use terms; **do not bundle**
      raw NPZ in this repository (fetch-script + SHA256)
    * **Limitations:** Flow (not speed); adjacency is a static packaged
      matrix; teaching cache is a short window; exchangeability across
      districts is not assumed
    * **Version:** ``pems0x_v1`` teaching cache schema
    """

    IN_CHANNELS = IN_CHANNELS

    def __init__(self, variant: str = "04") -> None:
        """Bind a PEMS0X district variant.

        Parameters
        ----------
        variant : str, optional
            District key (``"03"``, ``"04"``, ``"07"``, or ``"08"``).
            Default is ``"04"``.
        """
        self.variant = normalize_pems_variant(variant)
        self.NUM_SENSORS = VARIANT_NUM_SENSORS[self.variant]

    @classmethod
    def load_topology(
        cls,
        variant: str = "04",
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load cached PEMS0X graph topology for ``variant``.

        Parameters
        ----------

        variant : str
            See the function signature / summary for ``variant``.
        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        TopologyPayload
            See summary line."""
        key = normalize_pems_variant(variant)
        payload = load_variant_traffic_cache(key, cache_dir, dtype=dtype)
        return TopologyPayload(
            sensor_ids=list(payload["sensor_ids"]),
            edge_index=payload["edge_index"],
            edge_weight=payload["edge_weight"],
            num_nodes=int(payload["num_nodes"]),
            source_url=payload.get("source_url"),
        )

    @classmethod
    def load_sequence(
        cls,
        variant: str = "04",
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached PEMS0X flow snapshot sequence for ``variant``.

        Parameters
        ----------

        variant : str
            See the function signature / summary for ``variant``.
        cache_dir : Path | None
            See the function signature / summary for ``cache_dir``.
        dtype : torch.dtype
            See the function signature / summary for ``dtype``.

        Returns
        -------

        GraphSnapshotSequence
            See summary line."""
        key = normalize_pems_variant(variant)
        payload = load_variant_traffic_cache(key, cache_dir, dtype=dtype)
        return _sequence_from_payload(
            payload,
            expected_nodes=VARIANT_NUM_SENSORS[key],
            dtype=dtype,
        )
