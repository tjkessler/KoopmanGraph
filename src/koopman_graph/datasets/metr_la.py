"""METR-LA traffic benchmark for tutorials and tests."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence

DCRNN_SENSOR_GRAPH_BASE = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
)
SENSOR_IDS_URL = f"{DCRNN_SENSOR_GRAPH_BASE}/graph_sensor_ids.txt"
DISTANCES_URL = f"{DCRNN_SENSOR_GRAPH_BASE}/distances_la_2012.csv"
DEFAULT_H5_MIRROR_URL = (
    "https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5"
)
DCRNN_H5_GOOGLE_DRIVE = (
    "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "metr_la"
TRAFFIC_FILENAME = "traffic.pt"
NUM_SENSORS = 207
IN_CHANNELS = 1
DEFAULT_NUM_TIMESTEPS = 60


def _default_traffic_path(cache_dir: Path | None = None) -> Path:
    """Return the default on-disk path for cached METR-LA traffic data.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Root cache directory. Defaults to ``data/metr_la`` at the repository
        root.

    Returns
    -------
    Path
        Path to ``traffic.pt`` inside the cache directory.
    """
    root = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    return root / TRAFFIC_FILENAME


def download_sensor_ids() -> list[str]:
    """Download the ordered METR-LA sensor ID list from the DCRNN repository.

    Returns
    -------
    list of str
        Sensor identifiers in graph node order.
    """
    try:
        with urlopen(SENSOR_IDS_URL, timeout=60) as response:
            text = response.read().decode("utf-8").strip()
    except URLError as exc:
        msg = f"Failed to download METR-LA sensor IDs from {SENSOR_IDS_URL}"
        raise OSError(msg) from exc
    return text.split(",")


def download_distances_csv() -> str:
    """Download the METR-LA pairwise road-distance CSV from DCRNN.

    Returns
    -------
    str
        Raw CSV text with columns ``from``, ``to``, and ``cost``.
    """
    try:
        with urlopen(DISTANCES_URL, timeout=60) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        msg = f"Failed to download METR-LA distances from {DISTANCES_URL}"
        raise OSError(msg) from exc


def build_adjacency_matrix(
    distance_csv: str,
    sensor_ids: list[str],
    *,
    normalized_k: float = 0.1,
) -> np.ndarray:
    """Build a Gaussian-kernel adjacency matrix from DCRNN distance CSV data.

    Follows the preprocessing in the
    `DCRNN gen_adj_mx script <https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py>`_.

    Parameters
    ----------
    distance_csv : str
        CSV text with columns ``from``, ``to``, and ``cost``.
    sensor_ids : list of str
        Ordered sensor identifiers.
    normalized_k : float, optional
        Entries below this threshold after kernel normalization are zeroed.
        Default is ``0.1``.

    Returns
    -------
    ndarray
        Adjacency matrix with shape ``(len(sensor_ids), len(sensor_ids))``.
    """
    if not 0.0 <= normalized_k <= 1.0:
        msg = f"normalized_k must be in [0, 1], got {normalized_k}"
        raise ValueError(msg)

    num_sensors = len(sensor_ids)
    dist_mx = np.full((num_sensors, num_sensors), np.inf, dtype=np.float32)
    sensor_id_to_ind = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}

    reader = csv.DictReader(io.StringIO(distance_csv))
    for row in reader:
        from_id = row["from"]
        to_id = row["to"]
        if from_id not in sensor_id_to_ind or to_id not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[from_id], sensor_id_to_ind[to_id]] = float(row["cost"])

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = float(distances.std())
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < normalized_k] = 0.0
    return adj_mx.astype(np.float32)


def adjacency_to_edge_index(adj_mx: np.ndarray) -> Tensor:
    """Convert an adjacency matrix to a bidirectional PyG ``edge_index``.

    Parameters
    ----------
    adj_mx : ndarray
        Weighted adjacency matrix.

    Returns
    -------
    Tensor
        Long tensor with shape ``(2, num_edges)``.
    """
    src: list[int] = []
    dst: list[int] = []
    num_nodes = adj_mx.shape[0]
    for row in range(num_nodes):
        for col in range(num_nodes):
            if adj_mx[row, col] > 0.0:
                src.extend([row, col])
                dst.extend([col, row])
    return torch.tensor([src, dst], dtype=torch.long)


def adjacency_to_edge_weight(adj_mx: np.ndarray) -> Tensor:
    """Extract scalar edge weights aligned with :func:`adjacency_to_edge_index`.

    Parameters
    ----------
    adj_mx : ndarray
        Weighted adjacency matrix.

    Returns
    -------
    Tensor
        Float tensor with shape ``(num_edges,)``.
    """
    weights: list[float] = []
    num_nodes = adj_mx.shape[0]
    for row in range(num_nodes):
        for col in range(num_nodes):
            if adj_mx[row, col] > 0.0:
                weights.append(float(adj_mx[row, col]))
                weights.append(float(adj_mx[col, row]))
    return torch.tensor(weights, dtype=torch.float32)


def read_h5_speed_window(
    h5_path: Path,
    *,
    num_timesteps: int,
    offset: int = 0,
) -> np.ndarray:
    """Read a window of METR-LA speed values from an HDF5 file.

    Parameters
    ----------
    h5_path : Path
        Path to ``metr-la.h5`` in the DCRNN pandas HDF5 format.
    num_timesteps : int
        Number of consecutive 5-minute readings to load.
    offset : int, optional
        Starting row offset in the speed table. Default is ``0``.

    Returns
    -------
    ndarray
        Speed array with shape ``(num_timesteps, num_sensors)``.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If the requested window exceeds the available rows.
    """
    try:
        import h5py
    except ImportError as exc:
        msg = "h5py is required to read METR-LA HDF5 files (`pip install h5py`)"
        raise ImportError(msg) from exc

    with h5py.File(h5_path, "r") as handle:
        values = handle["df/block0_values"]
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


def preprocess_speeds(speeds: np.ndarray) -> np.ndarray:
    """Impute missing METR-LA readings before normalization.

    The DCRNN HDF5 release encodes missing loop-detector samples as ``0``.
    Non-positive values are treated as missing and filled along time per
    sensor using forward fill followed by backward fill.

    Parameters
    ----------
    speeds : ndarray
        Raw speed array with shape ``(num_timesteps, num_sensors)``.

    Returns
    -------
    ndarray
        Imputed speeds with the same shape.

    Raises
    ------
    ValueError
        If ``speeds`` is not two-dimensional.
    """
    if speeds.ndim != 2:
        msg = f"speeds must have shape (num_timesteps, num_sensors), got {speeds.shape}"
        raise ValueError(msg)

    filled = speeds.astype(np.float32, copy=True)
    filled[filled <= 0.0] = np.nan
    num_timesteps, num_sensors = filled.shape

    for sensor in range(num_sensors):
        series = filled[:, sensor]
        if np.all(np.isnan(series)):
            filled[:, sensor] = 0.0
            continue

        last_valid = np.nan
        for step in range(num_timesteps):
            if not np.isnan(series[step]):
                last_valid = series[step]
            elif not np.isnan(last_valid):
                series[step] = last_valid

        next_valid = np.nan
        for step in range(num_timesteps - 1, -1, -1):
            if not np.isnan(series[step]):
                next_valid = series[step]
            # After the forward pass, any remaining NaN is a leading gap, so a
            # valid future value has always been seen by the backward scan.
            elif not np.isnan(next_valid):  # pragma: no branch
                series[step] = next_valid

        filled[:, sensor] = np.nan_to_num(series, nan=0.0)

    return filled


def normalize_speeds(speeds: np.ndarray) -> np.ndarray:
    """Per-sensor z-score normalization along the time axis.

    Parameters
    ----------
    speeds : ndarray
        Speed array with shape ``(num_timesteps, num_sensors)``.

    Returns
    -------
    ndarray
        Normalized speeds with the same shape.
    """
    mean = speeds.mean(axis=0, keepdims=True)
    std = speeds.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((speeds - mean) / std).astype(np.float32)


def build_traffic_cache_payload(
    speeds: np.ndarray,
    sensor_ids: list[str],
    *,
    normalized_k: float = 0.1,
    source_h5_url: str = DEFAULT_H5_MIRROR_URL,
    timestep_offset: int = 0,
) -> dict[str, Any]:
    """Assemble a cache payload from speed readings and DCRNN graph metadata.

    Parameters
    ----------
    speeds : ndarray
        Raw speed readings with shape ``(num_timesteps, num_sensors)``.
    sensor_ids : list of str
        Ordered sensor identifiers.
    normalized_k : float, optional
        Adjacency sparsity threshold. Default is ``0.1``.
    source_h5_url : str, optional
        Provenance URL for the speed source file.
    timestep_offset : int, optional
        Row offset used when extracting ``speeds`` from the HDF5 table.

    Returns
    -------
    dict
        Serializable cache payload for ``traffic.pt``.
    """
    if speeds.ndim != 2:
        msg = f"speeds must have shape (num_timesteps, num_sensors), got {speeds.shape}"
        raise ValueError(msg)
    if speeds.shape[1] != len(sensor_ids):
        msg = (
            f"speeds has {speeds.shape[1]} sensors but sensor_ids has "
            f"{len(sensor_ids)} entries"
        )
        raise ValueError(msg)

    distance_csv = download_distances_csv()
    adj_mx = build_adjacency_matrix(distance_csv, sensor_ids, normalized_k=normalized_k)
    edge_index = adjacency_to_edge_index(adj_mx)
    edge_weight = adjacency_to_edge_weight(adj_mx)
    cleaned = preprocess_speeds(speeds)
    normalized = normalize_speeds(cleaned)

    return {
        "sensor_ids": sensor_ids,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "speeds": torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1),
        "num_nodes": len(sensor_ids),
        "source_h5_url": source_h5_url,
        "timestep_offset": timestep_offset,
        "num_timesteps_cached": int(speeds.shape[0]),
        "normalized_k": normalized_k,
    }


def ensure_traffic_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    h5_path: Path | None = None,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    offset: int = 0,
    normalized_k: float = 0.1,
) -> Path:
    """Download graph metadata and build the METR-LA traffic cache if needed.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory used for ``traffic.pt``. Defaults to ``data/metr_la``.
    force : bool, optional
        Rebuild the cache even when it already exists.
    h5_path : Path, optional
        Local ``metr-la.h5`` file used to refresh speed readings. When omitted
        and ``force=True``, an existing cache is reused without fetching HDF5.
    num_timesteps : int, optional
        Number of timesteps stored in the cache. Default is ``60``.
    offset : int, optional
        Starting row in the HDF5 speed table. Default is ``0``.
    normalized_k : float, optional
        Adjacency sparsity threshold. Default is ``0.1``.

    Returns
    -------
    Path
        Path to ``traffic.pt``.
    """
    path = _default_traffic_path(cache_dir)
    if path.exists() and not force:
        return path
    if h5_path is None:
        if path.exists():
            return path
        msg = (
            "METR-LA cache is missing. Provide --h5-path to "
            "scripts/download_metr_la.py after downloading metr-la.h5 "
            "from the DCRNN release."
        )
        raise FileNotFoundError(msg)

    sensor_ids = download_sensor_ids()
    speeds = read_h5_speed_window(h5_path, num_timesteps=num_timesteps, offset=offset)
    payload = build_traffic_cache_payload(
        speeds,
        sensor_ids,
        normalized_k=normalized_k,
        timestep_offset=offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _ensure_edge_weight(payload: dict[str, Any]) -> Tensor:
    """Return edge weights from cache payload, recomputing legacy caches if needed.

    Parameters
    ----------
    payload : dict
        Cached METR-LA payload containing ``edge_index`` and ``sensor_ids``.

    Returns
    -------
    Tensor
        Scalar edge weights aligned with ``payload["edge_index"]``.
    """
    edge_weight = payload.get("edge_weight")
    if edge_weight is not None:
        return edge_weight

    sensor_ids = payload["sensor_ids"]
    normalized_k = float(payload.get("normalized_k", 0.1))
    distance_csv = download_distances_csv()
    adj_mx = build_adjacency_matrix(
        distance_csv,
        sensor_ids,
        normalized_k=normalized_k,
    )
    edge_index = payload["edge_index"]
    weights = [
        float(adj_mx[int(edge_index[0, idx].item()), int(edge_index[1, idx].item())])
        for idx in range(edge_index.shape[1])
    ]
    return torch.tensor(weights, dtype=torch.float32)


def load_traffic_cache(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Load cached METR-LA topology and speed readings.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory containing ``traffic.pt``.
    dtype : torch.dtype, optional
        Floating dtype for returned speed tensors. Default is ``torch.float32``.

    Returns
    -------
    dict
        Cache payload with ``edge_index`` and ``speeds`` tensors.
    """
    path = ensure_traffic_cache(cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["speeds"] = payload["speeds"].to(dtype=dtype)
    if payload.get("edge_weight") is None and "sensor_ids" in payload:
        payload["edge_weight"] = _ensure_edge_weight(payload)
    elif payload.get("edge_weight") is not None:
        payload["edge_weight"] = payload["edge_weight"].to(dtype=dtype)
    return payload


class MetrLaTrafficBenchmark:
    """METR-LA traffic-speed benchmark built from the DCRNN sensor graph.

    Node features are per-sensor traffic speeds (mph), z-score normalized over
    the cached time window. The road-network adjacency follows the standard
    DCRNN Gaussian kernel on pairwise distances.

    Full speed history is distributed with the
    `DCRNN release <https://github.com/liyaguang/DCRNN>`_ (Google Drive /
    Baidu Yun). A public mirror is documented in ``scripts/download_metr_la.py``.

    Attributes
    ----------
    NUM_SENSORS : int
        Number of sensors in the METR-LA graph.
    IN_CHANNELS : int
        Node feature dimension (one normalized speed per sensor).
    """

    NUM_SENSORS = NUM_SENSORS
    IN_CHANNELS = IN_CHANNELS

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Any]:
        """Load cached METR-LA graph topology and metadata.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing cached traffic artifacts. Defaults to the
            package ``data/metr_la`` directory.
        dtype : torch.dtype, optional
            Floating dtype for returned tensors. Default is ``torch.float32``.

        Returns
        -------
        dict
            Metadata with keys ``sensor_ids``, ``edge_index``, ``edge_weight``,
            ``num_nodes``, ``source_h5_url``, and ``normalized_k``.
        """
        payload = load_traffic_cache(cache_dir, dtype=dtype)
        return {
            "sensor_ids": payload["sensor_ids"],
            "edge_index": payload["edge_index"],
            "edge_weight": payload["edge_weight"],
            "num_nodes": payload["num_nodes"],
            "source_h5_url": payload["source_h5_url"],
            "normalized_k": payload["normalized_k"],
        }

    @classmethod
    def load_sequence(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached METR-LA speed snapshot sequence.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing cached traffic artifacts. Defaults to the
            package ``data/metr_la`` directory.
        dtype : torch.dtype, optional
            Floating dtype for speed features. Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Time-ordered graph snapshots with one speed feature per sensor.
        """
        payload = load_traffic_cache(cache_dir, dtype=dtype)
        speeds = payload["speeds"]
        if speeds.ndim != 3 or speeds.shape[2] != IN_CHANNELS:
            msg = (
                f"Expected speeds shape (T, N, {IN_CHANNELS}), "
                f"got {tuple(speeds.shape)}"
            )
            raise ValueError(msg)
        num_nodes = int(payload["num_nodes"])
        if num_nodes != NUM_SENSORS:
            msg = f"Expected {NUM_SENSORS} sensors, got {num_nodes}"
            raise ValueError(msg)
        return GraphSnapshotSequence.from_arrays(
            speeds,
            payload["edge_index"],
            edge_weight=payload.get("edge_weight"),
            dtype=dtype,
        )
