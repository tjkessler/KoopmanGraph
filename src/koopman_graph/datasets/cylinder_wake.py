"""Cached cylinder-wake Hopf surrogate benchmark for Koopman tutorials.

Provides a small Stuart–Landau teaching cache on an unstructured wake mesh,
complementing Laplacian-diffusion and chaotic PDE/ODE graph benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.topology import TopologyPayload

DEFAULT_WAKE_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cylinder_wake"
DEFAULT_WAKE_NUM_NODES = 72
DEFAULT_WAKE_NUM_TIMESTEPS = 120
IN_CHANNELS_SCALAR = 1


def _default_wake_path(cache_dir: Path | None = None) -> Path:
    """Return the default on-disk path for the cylinder-wake cache.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Optional cache directory override.

    Returns
    -------
    Path
        Path to ``wake.pt``.
    """
    root = cache_dir if cache_dir is not None else DEFAULT_WAKE_CACHE_DIR
    return root / "wake.pt"


def _cylinder_wake_mesh(
    *,
    num_nodes: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Build a coarse unstructured wake mesh and k-NN edge index.

    Points fill a rectangular channel excluding a unit-diameter cylinder at the
    origin. Edges are symmetric k-nearest-neighbor links.

    Parameters
    ----------
    num_nodes : int
        Number of mesh nodes to keep.
    seed : int
        RNG seed for point sampling.

    Returns
    -------
    tuple of Tensor
        ``(coords, edge_index)`` with shapes ``(N, 2)`` and ``(2, E)``.
    """
    generator = torch.Generator().manual_seed(seed)
    # Oversample then reject interior points until we have num_nodes.
    coords_list: list[Tensor] = []
    radius = 0.5
    while len(coords_list) < num_nodes:
        batch = torch.rand(num_nodes * 2, 2, generator=generator)
        xs = -1.5 + batch[:, 0] * 10.0
        ys = -2.5 + batch[:, 1] * 5.0
        pts = torch.stack([xs, ys], dim=1)
        keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2) >= radius**2
        for row in pts[keep]:
            coords_list.append(row)
            if len(coords_list) >= num_nodes:
                break
    coords = torch.stack(coords_list[:num_nodes], dim=0)

    # Symmetric 6-NN graph.
    dists = torch.cdist(coords, coords)
    dists.fill_diagonal_(float("inf"))
    knn = torch.topk(dists, k=min(6, num_nodes - 1), largest=False).indices
    undirected: set[tuple[int, int]] = set()
    for i in range(num_nodes):
        for j in knn[i].tolist():
            undirected.add((min(i, j), max(i, j)))
    src: list[int] = []
    dst: list[int] = []
    for u, v in sorted(undirected):
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return coords, edge_index


def build_cylinder_wake_payload(
    *,
    num_nodes: int = DEFAULT_WAKE_NUM_NODES,
    num_timesteps: int = DEFAULT_WAKE_NUM_TIMESTEPS,
    dt: float = 0.15,
    omega: float = 0.8,
    mu: float = 0.15,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Assemble a small Hopf/POD cylinder-wake teaching cache.

    The field is a mean wake plus a complex Stuart–Landau oscillator modulating
    two spatial modes that approximate a von Kármán street. This is a
    reproducible reduced-order surrogate for tutorial and unit-test use, not
    Navier–Stokes DNS.

    Parameters
    ----------
    num_nodes : int, optional
        Mesh size. Default is ``72``.
    num_timesteps : int, optional
        Number of stored snapshots. Default is ``120``.
    dt : float, optional
        Temporal spacing of the oscillator. Default is ``0.15``.
    omega : float, optional
        Oscillation frequency. Default is ``0.8``.
    mu : float, optional
        Stuart–Landau growth parameter. Default is ``0.15``.
    seed : int, optional
        Mesh / phase seed. Default is ``0``.
    dtype : torch.dtype, optional
        Feature dtype. Default is ``torch.float32``.

    Returns
    -------
    dict
        Serializable cache payload for ``wake.pt``.
    """
    coords, edge_index = _cylinder_wake_mesh(num_nodes=num_nodes, seed=seed)
    x = coords[:, 0]
    y = coords[:, 1]
    # Mean wake deficit + two oscillatory modes (streamwise fluctuation).
    mean = -0.35 * torch.exp(-((y / 1.2) ** 2)) * torch.sigmoid(x)
    mode_r = torch.exp(-((y / 1.4) ** 2)) * torch.sin(0.9 * x) * torch.sigmoid(x + 0.5)
    mode_i = torch.exp(-((y / 1.4) ** 2)) * torch.cos(0.9 * x) * torch.sigmoid(x + 0.5)

    # Complex Stuart–Landau: ż = (μ + iω)z − |z|² z
    z = complex(0.05, 0.02)
    frames: list[Tensor] = []
    for _ in range(num_timesteps):
        amp_r = z.real
        amp_i = z.imag
        field = mean + amp_r * mode_r + amp_i * mode_i
        frames.append(field.to(dtype=dtype).unsqueeze(-1))
        mag2 = abs(z) ** 2
        dz = (mu + 1j * omega) * z - mag2 * z
        z = z + dt * dz

    features = torch.stack(frames, dim=0)
    return {
        "features": features,
        "edge_index": edge_index,
        "coords": coords.to(dtype=dtype),
        "num_nodes": int(num_nodes),
        "num_timesteps": int(num_timesteps),
        "dt": float(dt),
        "omega": float(omega),
        "mu": float(mu),
        "seed": int(seed),
        "description": (
            "Hopf/Stuart-Landau cylinder-wake surrogate on a coarse wake mesh "
            "(teaching cache; not DNS)."
        ),
    }


def ensure_wake_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Build the cylinder-wake cache if missing.

    Parameters
    ----------
    cache_dir : Path, optional
        Cache directory. Defaults to ``data/cylinder_wake``.
    force : bool, optional
        Rebuild even when the cache exists. Default is ``False``.

    Returns
    -------
    Path
        Path to ``wake.pt``.
    """
    path = _default_wake_path(cache_dir)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_cylinder_wake_payload()
    torch.save(payload, path)
    return path


def load_wake_cache(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Load the cached cylinder-wake payload, building it if needed.

    Parameters
    ----------
    cache_dir : Path, optional
        Cache directory. Defaults to ``data/cylinder_wake``.
    dtype : torch.dtype, optional
        Floating dtype for returned tensors.

    Returns
    -------
    dict
        Cache payload with ``features``, ``edge_index``, and metadata.
    """
    path = ensure_wake_cache(cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["features"] = payload["features"].to(dtype=dtype)
    if "coords" in payload:
        payload["coords"] = payload["coords"].to(dtype=dtype)
    return payload


class CylinderWakeBenchmark:
    """Cached cylinder-wake Hopf surrogate on an unstructured wake mesh.

    Public entry points mirror METR-LA: ``load_topology`` / ``load_sequence``.
    The default cache is a small on-disk teaching dataset generated by
    :func:`build_cylinder_wake_payload` (Stuart–Landau modulated spatial modes),
    not full CFD. Rebuild with :func:`ensure_wake_cache` ``force=True``.

    Notes
    -----
    This is a teaching surrogate for Hopf wake dynamics, not a CFD result.
    """

    NUM_NODES = DEFAULT_WAKE_NUM_NODES
    IN_CHANNELS = IN_CHANNELS_SCALAR

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load cached wake-mesh topology.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing ``wake.pt``.
        dtype : torch.dtype, optional
            Floating dtype for optional coordinate metadata.

        Returns
        -------
        TopologyPayload
            Frozen topology with ``edge_index`` and ``num_nodes``.
        """
        payload = load_wake_cache(cache_dir, dtype=dtype)
        return TopologyPayload(
            edge_index=payload["edge_index"],  # type: ignore[arg-type]
            num_nodes=int(payload["num_nodes"]),  # type: ignore[arg-type]
        )

    @classmethod
    def load_sequence(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached wake snapshot sequence.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing ``wake.pt``.
        dtype : torch.dtype, optional
            Feature dtype.

        Returns
        -------
        GraphSnapshotSequence
            Time-ordered streamwise-fluctuation snapshots.
        """
        payload = load_wake_cache(cache_dir, dtype=dtype)
        features = payload["features"]
        assert isinstance(features, Tensor)
        edge_index = payload["edge_index"]
        assert isinstance(edge_index, Tensor)
        if features.ndim != 3 or features.shape[2] != IN_CHANNELS_SCALAR:
            msg = (
                f"Expected features shape (T, N, {IN_CHANNELS_SCALAR}), "
                f"got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        return GraphSnapshotSequence.from_arrays(features, edge_index, dtype=dtype)
