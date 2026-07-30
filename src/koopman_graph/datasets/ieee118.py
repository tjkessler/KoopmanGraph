"""IEEE 118-bus MATPOWER benchmark for tutorials and tests.

Homogeneous entry points (``load_topology`` / ``generate``) keep the historical
``Data`` sequence API. Typed helpers (``load_typed_topology`` /
``generate_typed``) partition buses by MATPOWER ``BUS_TYPE`` into
``generator`` / ``load`` / ``slack`` node types for hetero demos.

Honesty
-------
The transmission **topology** and bus roles come from the public MATPOWER
``case118`` artifact. Feature trajectories from ``generate`` /
``generate_typed`` are **simulated** (Laplacian diffusion on voltages/angles
plus a sinusoidal load ramp) — they are **not** real grid telemetry or OPF
solutions. Do not cite this benchmark as measured power-system dynamics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.datasets.download import download_url_text, resolve_cache_path
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

# MATPOWER BUS_TYPE codes (column index 1): PQ=1, PV=2, REF=3.
MATPOWER_BUS_TYPE_TO_NAME: dict[int, str] = {
    1: "load",
    2: "generator",
    3: "slack",
}
TYPED_NODE_TYPE_ORDER: tuple[str, ...] = ("generator", "load", "slack")
TYPED_RELATION_NAME = "branch"
SIMULATED_DYNAMICS_DISCLAIMER = (
    "IEEE 118 typed / homogeneous dynamics in this package are simulated "
    "(Laplacian diffusion + load ramp) on the MATPOWER case118 topology; "
    "they are not real grid telemetry."
)

EdgeTypeTriple = tuple[str, str, str]


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
    return resolve_cache_path(
        cache_dir,
        default_dir=DEFAULT_CACHE_DIR,
        filename=TOPOLOGY_FILENAME,
    )


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


def _bus_type_codes(bus_rows: list[list[float]]) -> Tensor:
    """Extract MATPOWER ``BUS_TYPE`` codes for each bus row.

    Parameters
    ----------
    bus_rows : list of list of float
        Parsed MATPOWER bus table rows.

    Returns
    -------
    Tensor
        Integer codes with shape ``(num_buses,)``.
    """
    return torch.tensor([int(row[1]) for row in bus_rows], dtype=torch.long)


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
        Frozen topology with ``base_mva``, ``bus_ids``, ``bus_types``,
        ``edge_index``, ``initial_features``, ``num_nodes``, and
        ``source_url``.
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
        bus_types=_bus_type_codes(bus_rows),
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
    return download_url_text(
        MATPOWER_CASE118_URL,
        label="MATPOWER case118",
    )


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
    bus_types = topology.bus_types
    if bus_types is None:
        # Upgrade legacy caches that predate the bus_types field.
        path = ensure_topology_cache(cache_dir, force=True, dtype=dtype)
        raw = torch.load(path, weights_only=False)
        topology = TopologyPayload.from_mapping(raw)
        bus_types = topology.bus_types
    if bus_types is None:
        msg = "IEEE 118 topology cache is missing bus_types after rebuild"
        raise ValueError(msg)
    if topology.initial_features is None:
        msg = "IEEE 118 topology cache is missing initial_features after rebuild"
        raise ValueError(msg)
    return TopologyPayload(
        edge_index=topology.edge_index.to(dtype=torch.long),
        num_nodes=topology.num_nodes,
        initial_features=topology.initial_features.to(dtype=dtype),
        bus_ids=topology.bus_ids,
        bus_types=bus_types.to(dtype=torch.long),
        base_mva=topology.base_mva,
        source_url=topology.source_url,
    )


def bus_type_name(code: int) -> str:
    """Map a MATPOWER ``BUS_TYPE`` code to a typed node-type name.

    Parameters
    ----------
    code : int
        MATPOWER bus type (``1`` PQ / load, ``2`` PV / generator, ``3`` slack).

    Returns
    -------
    str
        One of ``"load"``, ``"generator"``, or ``"slack"``.

    Raises
    ------
    ValueError
        If ``code`` is not a supported MATPOWER bus type.
    """
    try:
        return MATPOWER_BUS_TYPE_TO_NAME[int(code)]
    except KeyError as exc:
        msg = (
            f"Unsupported MATPOWER BUS_TYPE code {code!r}; "
            f"expected one of {sorted(MATPOWER_BUS_TYPE_TO_NAME)}"
        )
        raise ValueError(msg) from exc


def partition_buses_by_type(bus_types: Tensor) -> dict[str, Tensor]:
    """Partition homogeneous bus indices by MATPOWER type name.

    Parameters
    ----------
    bus_types : Tensor
        Integer MATPOWER codes with shape ``(num_buses,)``.

    Returns
    -------
    dict of str to Tensor
        Global bus indices for each name in :data:`TYPED_NODE_TYPE_ORDER`
        (empty tensors omitted).
    """
    if bus_types.ndim != 1:
        msg = f"bus_types must be rank-1, got shape {tuple(bus_types.shape)}"
        raise ValueError(msg)
    partitions: dict[str, Tensor] = {}
    for code, name in MATPOWER_BUS_TYPE_TO_NAME.items():
        indices = torch.nonzero(bus_types == code, as_tuple=False).view(-1)
        if indices.numel() > 0:
            partitions[name] = indices.to(dtype=torch.long)
    unknown = [
        int(code)
        for code in torch.unique(bus_types).tolist()
        if int(code) not in MATPOWER_BUS_TYPE_TO_NAME
    ]
    if unknown:
        msg = f"Unsupported MATPOWER BUS_TYPE codes in bus_types: {unknown!r}"
        raise ValueError(msg)
    return partitions


def homogeneous_features_to_typed_hetero(
    features: Tensor,
    edge_index: Tensor,
    bus_types: Tensor,
    *,
    node_type_order: Sequence[str] = TYPED_NODE_TYPE_ORDER,
    relation: str = TYPED_RELATION_NAME,
) -> HeteroData:
    """Convert homogeneous bus features into a typed ``HeteroData`` snapshot.

    Every node type keeps the shared physical width ``F`` (IEEE default 4).
    Branch edges become ``(src_type, relation, dst_type)`` stores with
    **type-local** indices.

    Parameters
    ----------
    features : Tensor
        Homogeneous bus features with shape ``(num_buses, F)``.
    edge_index : Tensor
        Homogeneous bidirectional branch index with shape ``(2, E)``.
    bus_types : Tensor
        MATPOWER bus-type codes with shape ``(num_buses,)``.
    node_type_order : sequence of str, optional
        Preferred stacking / schema order. Types absent from ``bus_types``
        are skipped.
    relation : str, optional
        Relation name used in every edge-type triple. Default is ``"branch"``.

    Returns
    -------
    HeteroData
        Typed snapshot with per-type ``x`` and per-relation ``edge_index``.
    """
    if features.ndim != 2:
        msg = f"features must have shape (num_buses, F), got {tuple(features.shape)}"
        raise ValueError(msg)
    if bus_types.shape[0] != features.shape[0]:
        msg = (
            f"bus_types length ({bus_types.shape[0]}) must match "
            f"features rows ({features.shape[0]})"
        )
        raise ValueError(msg)
    partitions = partition_buses_by_type(bus_types)
    ordered_types = tuple(name for name in node_type_order if name in partitions)
    if not ordered_types:
        msg = "bus_types partition is empty after applying node_type_order"
        raise ValueError(msg)

    global_to_local = torch.full((features.shape[0],), -1, dtype=torch.long)
    for name in ordered_types:
        globals_for_type = partitions[name]
        global_to_local[globals_for_type] = torch.arange(
            globals_for_type.numel(),
            dtype=torch.long,
        )

    data = HeteroData()
    for name in ordered_types:
        data[name].x = features[partitions[name]]

    src_all = edge_index[0].tolist()
    dst_all = edge_index[1].tolist()
    buckets: dict[EdgeTypeTriple, list[list[int]]] = {}
    type_of = [bus_type_name(int(code)) for code in bus_types.tolist()]
    for src, dst in zip(src_all, dst_all, strict=True):
        edge_type = (type_of[src], relation, type_of[dst])
        local_src = int(global_to_local[src])
        local_dst = int(global_to_local[dst])
        if local_src < 0 or local_dst < 0:
            msg = f"edge ({src}, {dst}) references a bus outside the typed partition"
            raise ValueError(msg)
        bucket = buckets.setdefault(edge_type, [[], []])
        bucket[0].append(local_src)
        bucket[1].append(local_dst)

    for edge_type, (src_list, dst_list) in sorted(
        buckets.items(),
        key=lambda item: repr(item[0]),
    ):
        data[edge_type].edge_index = torch.tensor(
            [src_list, dst_list],
            dtype=torch.long,
        )
    return data


@dataclass(frozen=True)
class TypedIEEE118Topology:
    """Typed IEEE 118 topology for hetero demos (MATPOWER roles + simulated x).

    Honesty
    -------
    Bus roles come from MATPOWER ``BUS_TYPE``. Feature values on
    :attr:`snapshot` are the **cached / simulated** initial bus quantities —
    not real-time grid measurements. See
    :data:`SIMULATED_DYNAMICS_DISCLAIMER`.

    Attributes
    ----------
    node_type_names : tuple of str
        Ordered node types present in the case (subset of generator / load /
        slack).
    edge_types : tuple of (src, rel, dst)
        Relation schema discovered from branches.
    num_nodes_dict : dict of str to int
        Per-type bus counts.
    feature_dims : dict of str to int
        Per-type feature widths (shared ``F`` for all types).
    snapshot : HeteroData
        Initial typed snapshot (topology + initial features).
    bus_types : Tensor
        Homogeneous-order MATPOWER bus-type codes with shape ``(118,)``.
    homogeneous_edge_index : Tensor
        Original bidirectional branch index.
    source_url : str
        MATPOWER case URL.
    dynamics_disclaimer : str
        Simulated-dynamics disclosure for notebooks and docs.
    """

    node_type_names: tuple[str, ...]
    edge_types: tuple[EdgeTypeTriple, ...]
    num_nodes_dict: dict[str, int]
    feature_dims: dict[str, int]
    snapshot: HeteroData
    bus_types: Tensor
    homogeneous_edge_index: Tensor
    source_url: str
    dynamics_disclaimer: str = SIMULATED_DYNAMICS_DISCLAIMER


def load_typed_topology(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> TypedIEEE118Topology:
    """Load IEEE 118 topology partitioned into generator / load / slack.

    Prefer :meth:`IEEE118DynamicBenchmark.load_typed_topology` in application
    code.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory containing ``topology.pt``.
    dtype : torch.dtype, optional
        Floating dtype for node features. Default is ``torch.float32``.

    Returns
    -------
    TypedIEEE118Topology
        Typed topology payload with an initial ``HeteroData`` snapshot.
    """
    topology = load_topology(cache_dir, dtype=dtype)
    if topology.bus_types is None or topology.initial_features is None:
        msg = "IEEE 118 topology is missing bus_types or initial_features"
        raise ValueError(msg)
    snapshot = homogeneous_features_to_typed_hetero(
        topology.initial_features,
        topology.edge_index,
        topology.bus_types,
    )
    node_type_names = tuple(snapshot.node_types)
    edge_types = tuple(snapshot.edge_types)
    num_nodes_dict = {name: int(snapshot[name].num_nodes) for name in node_type_names}
    feature_dims = {name: int(snapshot[name].x.size(-1)) for name in node_type_names}
    return TypedIEEE118Topology(
        node_type_names=node_type_names,
        edge_types=edge_types,
        num_nodes_dict=num_nodes_dict,
        feature_dims=feature_dims,
        snapshot=snapshot,
        bus_types=topology.bus_types,
        homogeneous_edge_index=topology.edge_index,
        source_url=topology.source_url or MATPOWER_CASE118_URL,
    )


class IEEE118DynamicBenchmark:
    """IEEE 118-bus spatiotemporal benchmark built from MATPOWER topology.

    Node features are bus quantities ``[Vm, Va, Pd, Qd]`` (per-unit loads).
    Voltages and angles evolve via graph Laplacian diffusion on the real IEEE
    118 transmission topology; loads follow a slow sinusoidal ramp to emulate
    changing grid conditions over time. These trajectories are **simulated**
    teaching dynamics, not measured grid telemetry
    (:data:`SIMULATED_DYNAMICS_DISCLAIMER`).

    Public entry points are the classmethods ``load_topology``, ``generate``,
    ``load_typed_topology``, and ``generate_typed``. Prefer those over the
    module-level free functions. Typed helpers partition buses by MATPOWER
    ``BUS_TYPE`` (generator / load / slack) with a shared feature width
    ``F = 4`` on every type.

    For large-scale optimal power flow snapshots, see the PowerGraph dataset
    (``https://arxiv.org/abs/2402.02827``).

    Attributes
    ----------
    NUM_BUSES : int
        Number of buses in the IEEE 118 case.
    IN_CHANNELS : int
        Node feature dimension ``[Vm, Va, Pd, Qd]``.
    TYPED_NODE_TYPE_ORDER : tuple of str
        Preferred typed node-type order.
    """

    NUM_BUSES = NUM_BUSES
    IN_CHANNELS = IN_CHANNELS
    TYPED_NODE_TYPE_ORDER = TYPED_NODE_TYPE_ORDER

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
            Frozen topology with ``base_mva``, ``bus_ids``, ``bus_types``,
            ``edge_index``, ``initial_features``, ``num_nodes``, and
            ``source_url``.
        """
        return load_topology(cache_dir, dtype=dtype)

    @classmethod
    def load_typed_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TypedIEEE118Topology:
        """Load IEEE 118 topology as typed generator / load / slack stores.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing cached topology artifacts.
        dtype : torch.dtype, optional
            Floating dtype for node features. Default is ``torch.float32``.

        Returns
        -------
        TypedIEEE118Topology
            Typed topology with an initial ``HeteroData`` snapshot. Dynamics
            remain simulated when paired with :meth:`generate_typed`.
        """
        return load_typed_topology(cache_dir, dtype=dtype)

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

    @classmethod
    def generate_typed(
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
    ) -> HeteroGraphSnapshotSequence:
        """Generate a typed IEEE 118 sequence (generator / load / slack).

        Dynamics match :meth:`generate` (simulated Laplacian diffusion + load
        ramp on the MATPOWER topology) and are then partitioned into typed
        ``HeteroData`` snapshots. See :data:`SIMULATED_DYNAMICS_DISCLAIMER`.

        Parameters
        ----------
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``40``.
        diffusion_rate : float, optional
            Laplacian diffusion strength for ``Vm`` and ``Va``.
        decay_rate : float, optional
            Global amplitude decay applied each step.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise.
        load_ramp_amplitude : float, optional
            Peak fractional change applied to ``Pd`` and ``Qd``.
        load_ramp_period : float, optional
            Sinusoidal load ramp period in timesteps.
        expose_load_ramp_control : bool, optional
            When ``True``, attach global control inputs ``(T, 1)``.
        seed : int, optional
            Random seed for noise. Tutorials should pass an explicit seed.
        cache_dir : Path, optional
            Directory containing cached topology artifacts.
        dtype : torch.dtype, optional
            Floating dtype for generated features.

        Returns
        -------
        HeteroGraphSnapshotSequence
            Typed snapshots with shared feature width ``F = 4`` on every
            node type and ``(src_type, "branch", dst_type)`` edge stores.
        """
        homogeneous = cls.generate(
            num_timesteps=num_timesteps,
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            noise_std=noise_std,
            load_ramp_amplitude=load_ramp_amplitude,
            load_ramp_period=load_ramp_period,
            expose_load_ramp_control=expose_load_ramp_control,
            seed=seed,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        topology = cls.load_topology(cache_dir, dtype=dtype)
        if topology.bus_types is None:
            msg = "IEEE 118 topology is missing bus_types"
            raise ValueError(msg)
        typed_snapshots: list[HeteroData] = []
        for snapshot in homogeneous.snapshots:
            assert isinstance(snapshot, Data)
            if snapshot.x is None:
                msg = "homogeneous IEEE snapshot is missing feature matrix x"
                raise ValueError(msg)
            typed_snapshots.append(
                homogeneous_features_to_typed_hetero(
                    snapshot.x,
                    topology.edge_index,
                    topology.bus_types,
                )
            )
        return HeteroGraphSnapshotSequence(
            typed_snapshots,
            control_inputs=homogeneous.control_inputs,
        )
