"""IEEE 118-bus MATPOWER benchmark for tutorials and tests."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import (
    add_gaussian_noise,
    apply_laplacian_diffusion_step,
    diffusion_sequence_from_features,
    make_generator,
    normalized_step_operator,
    validate_diffusion_generation_params,
)
from koopman_graph.datasets.topology import TopologyPayload

MATPOWER_CASE118_URL = (
    "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case118.m"
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "ieee118"
TOPOLOGY_FILENAME = "topology.pt"
IN_CHANNELS = 4
NUM_BUSES = 118


def _default_topology_path(cache_dir: Path | None = None) -> Path:
    """Return the default on-disk path for cached IEEE 118 topology.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Root cache directory. Defaults to ``data/ieee118`` at the repository
        root.

    Returns
    -------
    Path
        Path to ``topology.pt`` inside the cache directory.
    """
    root = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    return root / TOPOLOGY_FILENAME


def _extract_matrix_block(text: str, field_name: str) -> str:
    """Extract the bracketed matrix literal for a MATPOWER struct field.

    Parameters
    ----------
    text : str
        Contents of a MATPOWER ``.m`` case file.
    field_name : str
        Struct field name without the ``mpc.`` prefix (for example ``"bus"``).

    Returns
    -------
    str
        Raw matrix block text between the opening and closing brackets.

    Raises
    ------
    ValueError
        If the field cannot be found or the bracketed block is unterminated.
    """
    pattern = rf"mpc\.{field_name}\s*=\s*\["
    match = re.search(pattern, text)
    if match is None:
        msg = f"Could not find mpc.{field_name} matrix in MATPOWER case file"
        raise ValueError(msg)

    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        index += 1

    if depth != 0:
        msg = f"Unterminated mpc.{field_name} matrix in MATPOWER case file"
        raise ValueError(msg)

    return text[start : index - 1]


def _parse_numeric_rows(block: str) -> list[list[float]]:
    """Parse semicolon-separated numeric rows from a MATPOWER matrix block.

    Parameters
    ----------
    block : str
        Raw matrix block text extracted from a MATPOWER case file.

    Returns
    -------
    list of list of float
        Parsed numeric rows, excluding blank lines and comments.
    """
    rows: list[list[float]] = []
    for raw_line in block.split(";"):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        values = [float(token) for token in line.split()]
        rows.append(values)
    return rows


def parse_matpower_case(text: str) -> dict[str, Any]:
    """Parse bus and branch tables from a MATPOWER version-2 case file.

    Parameters
    ----------
    text : str
        Contents of a MATPOWER ``.m`` case file.

    Returns
    -------
    dict
        Parsed fields including ``baseMVA``, ``bus``, and ``branch`` matrices.
    """
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.+-eE]+)\s*;", text)
    if base_match is None:
        msg = "Could not find mpc.baseMVA in MATPOWER case file"
        raise ValueError(msg)

    bus_rows = _parse_numeric_rows(_extract_matrix_block(text, "bus"))
    branch_rows = _parse_numeric_rows(_extract_matrix_block(text, "branch"))
    if not bus_rows:
        msg = "MATPOWER case file contains no bus rows"
        raise ValueError(msg)
    if not branch_rows:
        msg = "MATPOWER case file contains no branch rows"
        raise ValueError(msg)

    return {
        "baseMVA": float(base_match.group(1)),
        "bus": bus_rows,
        "branch": branch_rows,
    }


def _bus_id_map(bus_rows: list[list[float]]) -> dict[int, int]:
    """Map MATPOWER bus IDs to contiguous zero-based indices.

    Parameters
    ----------
    bus_rows : list of list of float
        Parsed MATPOWER bus table rows.

    Returns
    -------
    dict of int to int
        Mapping from MATPOWER bus ID to zero-based node index.
    """
    bus_ids = [int(row[0]) for row in bus_rows]
    return {bus_id: index for index, bus_id in enumerate(bus_ids)}


def _build_edge_index(
    branch_rows: list[list[float]],
    bus_map: dict[int, int],
) -> Tensor:
    """Build a bidirectional edge index from MATPOWER branch rows.

    Parameters
    ----------
    branch_rows : list of list of float
        Parsed MATPOWER branch table rows.
    bus_map : dict of int to int
        Mapping from MATPOWER bus ID to zero-based node index.

    Returns
    -------
    Tensor
        Bidirectional edge index with shape ``(2, num_edges)``.
    """
    src: list[int] = []
    dst: list[int] = []
    for row in branch_rows:
        if int(row[10]) != 1:
            continue
        from_bus = bus_map[int(row[0])]
        to_bus = bus_map[int(row[1])]
        src.extend([from_bus, to_bus])
        dst.extend([to_bus, from_bus])
    return torch.tensor([src, dst], dtype=torch.long)


def _initial_bus_features(
    bus_rows: list[list[float]],
    *,
    base_mva: float,
    dtype: torch.dtype,
) -> Tensor:
    """Build normalized initial node features ``[Vm, Va, Pd, Qd]`` per bus.

    Parameters
    ----------
    bus_rows : list of list of float
        Parsed MATPOWER bus table rows.
    base_mva : float
        System base MVA used to normalize active and reactive loads.
    dtype : torch.dtype
        Floating dtype for the returned tensor.

    Returns
    -------
    Tensor
        Initial node features with shape ``(num_buses, 4)``.
    """
    features = torch.zeros((len(bus_rows), IN_CHANNELS), dtype=dtype)
    for index, row in enumerate(bus_rows):
        pd = row[2] / base_mva
        qd = row[3] / base_mva
        vm = row[7]
        va = row[8]
        features[index] = torch.tensor([vm, va, pd, qd], dtype=dtype)
    return features


def topology_from_matpower_text(
    text: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> TopologyPayload:
    """Convert MATPOWER case text into tensors used by the benchmark.

    Parameters
    ----------
    text : str
        Contents of a MATPOWER version-2 ``.m`` case file.
    dtype : torch.dtype, optional
        Floating dtype for node features. Default is ``torch.float32``.

    Returns
    -------
    TopologyPayload
        Frozen topology with ``base_mva``, ``bus_ids``, ``edge_index``,
        ``initial_features``, ``num_nodes``, and ``source_url``.
    """
    parsed = parse_matpower_case(text)
    bus_rows = parsed["bus"]
    branch_rows = parsed["branch"]
    base_mva = parsed["baseMVA"]
    bus_map = _bus_id_map(bus_rows)
    edge_index = _build_edge_index(branch_rows, bus_map)
    initial_features = _initial_bus_features(
        bus_rows,
        base_mva=base_mva,
        dtype=dtype,
    )
    bus_ids = torch.tensor([int(row[0]) for row in bus_rows], dtype=torch.long)
    return TopologyPayload(
        base_mva=base_mva,
        bus_ids=bus_ids,
        edge_index=edge_index,
        initial_features=initial_features,
        num_nodes=len(bus_rows),
        source_url=MATPOWER_CASE118_URL,
    )


def download_matpower_case118() -> str:
    """Download the MATPOWER IEEE 118-bus case file text.

    Returns
    -------
    str
        Raw ``case118.m`` file contents.
    """
    try:
        with urlopen(MATPOWER_CASE118_URL, timeout=60) as response:
            return response.read().decode("utf-8")
    except URLError as exc:
        msg = f"Failed to download MATPOWER case118 from {MATPOWER_CASE118_URL}"
        raise OSError(msg) from exc


def ensure_topology_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """Download, parse, and cache IEEE 118 topology if needed.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory used for cached topology artifacts. Defaults to
        ``data/ieee118`` at the repository root.
    force : bool, optional
        Rebuild the cache even when it already exists.
    dtype : torch.dtype, optional
        Floating dtype stored in the cache. Default is ``torch.float32``.

    Returns
    -------
    Path
        Path to the cached ``topology.pt`` file.
    """
    path = _default_topology_path(cache_dir)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    case_text = download_matpower_case118()
    topology = topology_from_matpower_text(case_text, dtype=dtype)
    torch.save(topology.to_dict(), path)
    return path


def load_topology(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> TopologyPayload:
    """Load cached IEEE 118 topology, creating the cache on first use.

    Prefer :meth:`IEEE118DynamicBenchmark.load_topology` in application and
    notebook code. This free function remains as the shared implementation
    (and a shim for download scripts / low-level tests).

    Parameters
    ----------
    cache_dir : Path, optional
        Directory containing ``topology.pt``.
    dtype : torch.dtype, optional
        Floating dtype for returned tensors. Default is ``torch.float32``.

    Returns
    -------
    TopologyPayload
        Frozen topology with ``edge_index`` and ``initial_features`` (also
        supports mapping-style ``payload["edge_index"]`` access).
    """
    path = ensure_topology_cache(cache_dir, dtype=dtype)
    raw = torch.load(path, weights_only=False)
    topology = TopologyPayload.from_mapping(raw)
    if topology.initial_features is None:
        msg = "IEEE 118 topology cache is missing initial_features"
        raise ValueError(msg)
    return TopologyPayload(
        edge_index=topology.edge_index.to(dtype=torch.long),
        num_nodes=topology.num_nodes,
        initial_features=topology.initial_features.to(dtype=dtype),
        bus_ids=topology.bus_ids,
        base_mva=topology.base_mva,
        source_url=topology.source_url,
    )


class IEEE118DynamicBenchmark:
    """IEEE 118-bus spatiotemporal benchmark built from MATPOWER topology.

    Node features are bus quantities ``[Vm, Va, Pd, Qd]`` (per-unit loads).
    Voltages and angles evolve via graph Laplacian diffusion on the real IEEE
    118 transmission topology; loads follow a slow sinusoidal ramp to emulate
    changing grid conditions over time.

    Public entry points are the classmethods ``load_topology`` and ``generate``.
    Prefer those over the module-level ``load_topology`` free function.

    For large-scale optimal power flow snapshots, see the PowerGraph dataset
    (``https://arxiv.org/abs/2402.02827``).

    Attributes
    ----------
    NUM_BUSES : int
        Number of buses in the IEEE 118 case.
    IN_CHANNELS : int
        Node feature dimension ``[Vm, Va, Pd, Qd]``.
    """

    NUM_BUSES = NUM_BUSES
    IN_CHANNELS = IN_CHANNELS

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load the cached IEEE 118 topology tables.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing cached topology artifacts. Defaults to the
            package ``data/ieee118`` directory.
        dtype : torch.dtype, optional
            Floating dtype for node features. Default is ``torch.float32``.

        Returns
        -------
        TopologyPayload
            Frozen topology with ``base_mva``, ``bus_ids``, ``edge_index``,
            ``initial_features``, ``num_nodes``, and ``source_url``.
        """
        return load_topology(cache_dir, dtype=dtype)

    @classmethod
    def generate(
        cls,
        *,
        num_timesteps: int = 40,
        diffusion_rate: float = 0.35,
        decay_rate: float = 0.98,
        noise_std: float = 0.002,
        load_ramp_amplitude: float = 0.15,
        load_ramp_period: float = 20.0,
        expose_load_ramp_control: bool = False,
        seed: int | None = None,
        cache_dir: Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a dynamic IEEE 118-bus snapshot sequence.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``40``.
        diffusion_rate : float, optional
            Laplacian diffusion strength for ``Vm`` and ``Va``. Default is ``0.35``.
        decay_rate : float, optional
            Global amplitude decay applied each step. Default is ``0.98``.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise. Default is ``0.002``.
        load_ramp_amplitude : float, optional
            Peak fractional change applied to ``Pd`` and ``Qd``. Default is ``0.15``.
        load_ramp_period : float, optional
            Sinusoidal load ramp period in timesteps. Default is ``20.0``.
        expose_load_ramp_control : bool, optional
            When ``True``, attach the sinusoidal load-ramp multiplier as global
            control inputs with shape ``(num_timesteps, 1)``. Default is
            ``False``.
        seed : int, optional
            Random seed for noise. ``None`` uses unseeded randomness; tutorials
            should pass an explicit seed (e.g. ``42``).
        cache_dir : Path, optional
            Directory containing cached topology artifacts.
        dtype : torch.dtype, optional
            Floating dtype for generated features. Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Time-ordered snapshots on the IEEE 118-bus graph.

        Raises
        ------
        ValueError
            If any generation parameter is invalid.
        """
        if num_timesteps < 1:
            msg = f"num_timesteps must be >= 1, got {num_timesteps}"
            raise ValueError(msg)
        validate_diffusion_generation_params(
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            noise_std=noise_std,
        )
        if load_ramp_amplitude < 0.0:
            msg = f"load_ramp_amplitude must be >= 0, got {load_ramp_amplitude}"
            raise ValueError(msg)
        if load_ramp_period <= 0.0:
            msg = f"load_ramp_period must be > 0, got {load_ramp_period}"
            raise ValueError(msg)

        topology = cls.load_topology(cache_dir, dtype=dtype)
        edge_index = topology.edge_index
        if topology.initial_features is None:
            msg = "IEEE 118 topology is missing initial_features"
            raise ValueError(msg)
        initial_features = topology.initial_features
        num_nodes = int(topology.num_nodes)
        if num_nodes != NUM_BUSES:
            msg = f"Expected {NUM_BUSES} buses, got {num_nodes}"
            raise ValueError(msg)

        generator = make_generator(seed)
        step_operator = normalized_step_operator(
            edge_index,
            num_nodes,
            diffusion_rate,
            dtype=dtype,
        )
        base_loads = initial_features[:, 2:].clone()
        state = initial_features.clone()
        snapshots = [state.clone()]
        ramp_controls: list[Tensor] = [torch.ones(1, dtype=dtype, device=state.device)]

        for step in range(num_timesteps - 1):
            voltage_state = apply_laplacian_diffusion_step(
                state[:, :2],
                step_operator,
                decay_rate,
            )
            ramp = 1.0 + load_ramp_amplitude * math.sin(
                2.0 * math.pi * (step + 1) / load_ramp_period
            )
            if expose_load_ramp_control:
                ramp_controls.append(
                    torch.tensor([ramp], dtype=dtype, device=state.device)
                )
            load_state = base_loads * ramp
            state = torch.cat([voltage_state, load_state], dim=1)
            state = add_gaussian_noise(
                state,
                noise_std,
                generator=generator,
                dtype=dtype,
            )
            snapshots.append(state.clone())

        features = torch.stack(snapshots, dim=0)
        sequence = diffusion_sequence_from_features(features, edge_index, dtype=dtype)
        if expose_load_ramp_control:
            return GraphSnapshotSequence(
                sequence.snapshots,
                control_inputs=torch.stack(ramp_controls, dim=0),
            )
        return sequence
