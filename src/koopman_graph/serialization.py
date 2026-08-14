"""Checkpoint serialization for :class:`~koopman_graph.model.GraphKoopmanModel`.

Checkpoint format versions
--------------------------
``format_version`` 1 (current baseline; beta through 0.x)
    Full architecture config for discrete and continuous dynamics, hybrid
    physics observables, control (including bilinear metadata), delay
    embeddings, and built-in operator kinds (per-node / graph / hypergraph /
    global_local / continuous_graph / switched / mixture / hodge).
    Encoder/decoder ``type`` strings include
    ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
    ``"hyper_enc"``, ``"hyper_dec"``, ``"sim_enc"``, ``"sim_dec"``,
    ``"sheaf_enc"``, ``"sheaf_dec"``, ``"cell_enc"``, ``"cell_dec"``,
    ``"inv_geom_enc"``, ``"relgraph_enc"``, and ``"relgraph_dec"``; missing
    decoder ``type`` defaults to ``"gcn"``. Hybrid ``physics`` blocks own
    ``dim``, ``preset``, and
    ``position``; ``position`` is round-tripped and validated on load
    (currently only ``"prepend"``). Missing ``position`` defaults to
    ``"prepend"``. Optional ``n_delays`` records Hankel delay embedding; the
    stored encoder block is always the base encoder config with
    ``in_channels = n_delays * feature_dim``. Format-1 also stores placeholder
    keys ``sparsity`` (operator realization; ``"dense"``,
    ``"block_diagonal"``, or ``"distributed"`` for supported networked kinds),
    ``adjacency`` (``"symmetric"`` / ``"random_walk"`` /
    ``"dual_random_walk"`` for graph / continuous-graph operators, else
    ``None``), additive ``hypergraph_incidence_mode``
    (``"zhou_symmetric"`` / ``"forward_random_walk"`` /
    ``"dual_random_walk"``; absent ⇒ ``"zhou_symmetric"`` for hypergraph;
    ``None`` otherwise), ``learn_topology`` (``None`` or ``"self_adaptive"``)
    with ``topology_embedding_dim``, and ``symmetry`` (``None`` or a dict with
    ``auto_orbits``, ``orbit_partition``, and ``method`` for orbit-tied
    graph / hypergraph operators; additive ``symmetry`` field
    ``"orbit"`` / ``"isotypic"`` records ``koopman_symmetry`` without a
    format bump). When ``koopman_kind="hetero_graph"``,
    additive keys ``node_types``, ``edge_types`` (JSON ``(src, rel, dst)``
    triples), ``relation_tying`` (``"independent"`` / ``"basis"``),
    ``basis_size``, and
    ``relation_normalization`` (``"rgcn_in_degree"`` /
    ``"random_walk"``) are required on load; optional
    ``synthesize_reverse_relations`` (bool; absent ⇒ ``False``) records
    whether reverse relations were factory-synthesized. Optional additive
    ``latent_dims`` (``dict[str, int]``; absent ⇒ shared ``latent_dim``)
    records per-type widths ``d_τ`` for rectangular hetero (Q1=A). Incomplete
    or mismatched ``latent_dims`` are rejected — never silently coerced from
    factor shapes. Homogeneous checkpoints omit and ignore these keys.
    Additive sequence-contract keys ``allow_node_churn`` (bool; absent ⇒
    ``False``), ``has_presence_masks`` (bool; absent ⇒ ``False``), and
    optional ``entity_ids`` (JSON list of str/int; absent/null ⇒ ``None``)
    record the training universe contract. Presence mask **tensors** are
    sequence data and are **not** stored in checkpoints — reload them with
    the sequence used for evaluation / further training.
    Additive sequence-contract keys ``allow_node_churn`` (bool; absent ⇒
    ``False``), ``has_presence_masks`` (bool; absent ⇒ ``False``), and
    optional ``entity_ids`` (JSON list of str/int; absent/null ⇒ ``None``)
    record the training universe contract. Presence mask **tensors** are
    sequence data and are **not** stored in checkpoints — reload them with
    the sequence used for evaluation / further training.

Beta policy
    While the package is pre-1.0, ``FORMAT_VERSION`` stays at ``1``. Incomplete
    or previously published incompatible checkpoints are **deprecated** and
    rejected with a clear re-save error rather than migrated. Formal
    multi-version checkpoint tracking begins at 1.0. Loaders accept only
    ``{1}``; ``format_version`` 2 and other lineages remain unsupported.

On-disk containers
    ``safetensors_v1`` (default; prefer for sharing)
        Same three members — ``meta.json`` (container marker +
        ``format_version`` + ``package_version``), ``config.json`` (JSON
        architecture config from :func:`build_model_config`), and
        ``model.safetensors`` (weight tensors) — as either a **directory**
        or a **zip bundle** (``.kgckpt`` / ``.zip``). Logical architecture
        schema remains format-1; the container only changes how bytes are
        stored. Default :func:`save_checkpoint` /
        :meth:`GraphKoopmanModel.save` writes a directory unless ``path``
        ends in ``.kgckpt`` or ``.zip``, in which case it writes a zip with
        those members only (no pickle). Weights load via safetensors rather
        than pickle; see repository ``SECURITY.md`` for trust boundaries.
    ``legacy_pt``
        Single ``torch.save`` pickle file (``.pt`` / ``.pth``) holding
        ``format_version``, ``package_version``, ``config``, and
        ``state_dict``. Loading executes pickle and remains a trusted-source
        trust boundary (see ``SECURITY.md``). Pass ``format="legacy_pt"``
        for the pickle escape hatch. Training ``checkpoint_path`` writers
        keep ``legacy_pt`` explicitly so file-path best-epoch checkpoints
        remain single ``.pt`` files.

Load detection
    1. Directory with ``meta.json`` → ``safetensors_v1`` directory.
    2. File that is a zip containing root members ``meta.json``,
       ``config.json``, and ``model.safetensors`` → ``safetensors_v1`` zip
       (``.kgckpt`` / ``.zip``; also matched when those markers are present
       so pickle is not used when safe markers exist).
    3. Other files → legacy ``torch.load`` pickle path.

Custom injected operators (anything other than the built-in serializable
operator classes registered in this module) are **not** round-trippable:
:func:`build_model_config` / :meth:`GraphKoopmanModel.save` raise rather than
silently writing incomplete factory metadata.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save as safetensors_save_bytes
from safetensors.torch import save_file as safetensors_save_file
from torch import nn

from koopman_graph.data.hetero_layout import validate_latent_dims
from koopman_graph.nn import (
    DEFAULT_TOPOLOGY_EMBEDDING_DIM,
    CellComplexGNNDecoder,
    CellComplexGNNEncoder,
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphTransformerDecoder,
    GraphTransformerEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    InvariantGeometryEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
    SAGEDecoder,
    SAGEEncoder,
    SheafGNNDecoder,
    SheafGNNEncoder,
    SimplicialDecoder,
    SimplicialEncoder,
)
from koopman_graph.observables import (
    PHYSICS_POSITION,
    PhysicsLiftingFn,
    PhysicsPosition,
    resolve_physics_lifting_fn,
    resolve_physics_position,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HodgeKoopmanOperator,
    HypergraphKoopmanOperator,
    KoopmanOperator,
    MixtureKoopmanOperator,
    SwitchedKoopmanOperator,
    resolve_factory_stability_bound,
)
from koopman_graph.protocols import ModeShapeModel

FORMAT_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = frozenset({1})

CheckpointFormat = Literal["safetensors_v1", "legacy_pt"]
SAFE_CONTAINER = "safetensors_v1"
SAFE_META_FILENAME = "meta.json"
SAFE_CONFIG_FILENAME = "config.json"
SAFE_WEIGHTS_FILENAME = "model.safetensors"
SAFE_BUNDLE_SUFFIXES = frozenset({".kgckpt", ".zip"})
SAFE_ZIP_MEMBER_NAMES = frozenset(
    {
        SAFE_META_FILENAME,
        SAFE_CONFIG_FILENAME,
        SAFE_WEIGHTS_FILENAME,
    }
)

# Keys always written by :func:`build_model_config` for the current format-1
# baseline. Sparse historical payloads that omit these are rejected on load.
_FORMAT_1_REQUIRED_KEYS = frozenset(
    {
        "latent_dim",
        "time_step",
        "dynamics_mode",
        "koopman_kind",
        "koopman_init_mode",
        "koopman_init_scale",
        "koopman_parameterization",
        "koopman_max_spectral_radius",
        "control_dim",
        "control_mode",
        "bilinear_rank",
        "n_delays",
        "physics",
        "encoder",
        "decoder",
        "sparsity",
        "adjacency",
        "learn_topology",
        "topology_embedding_dim",
        "symmetry",
        "local_window",
        "local_rank",
        "local_hidden_dims",
    }
)

Decoder = (
    GNNDecoder
    | GATDecoder
    | SAGEDecoder
    | DiffConvDecoder
    | GraphTransformerDecoder
    | HypergraphDecoder
    | SimplicialDecoder
    | SheafGNNDecoder
    | CellComplexGNNDecoder
    | RelGraphDecoder
)
BaseEncoder = (
    GNNEncoder
    | GATEncoder
    | SAGEEncoder
    | DiffConvEncoder
    | GraphTransformerEncoder
    | HypergraphEncoder
    | SimplicialEncoder
    | SheafGNNEncoder
    | CellComplexGNNEncoder
    | InvariantGeometryEncoder
    | RelGraphEncoder
)
_SERIALIZABLE_KOOPMAN_TYPES = (
    KoopmanOperator,
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    HypergraphKoopmanOperator,
    GlobalLocalKoopmanOperator,
    ContinuousGraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    SwitchedKoopmanOperator,
    MixtureKoopmanOperator,
    HodgeKoopmanOperator,
)
_RESERVED_KOOPMAN_KINDS: dict[str, str] = {}

_HETERO_REQUIRED_KEYS = frozenset(
    {
        "node_types",
        "edge_types",
        "relation_tying",
        "basis_size",
        "relation_normalization",
    }
)
_RELATION_NORMALIZATION_MODES = frozenset({"rgcn_in_degree", "random_walk"})
_RELATION_TYING_MODES = frozenset({"independent", "basis"})


_NETWORKED_ADJACENCY_KINDS = frozenset({"graph", "continuous_graph", "hodge"})
_GRAPH_ADJACENCY_MODES = frozenset({"symmetric", "random_walk", "dual_random_walk"})


def _require_format1_schema(config: dict[str, Any]) -> None:
    """Reject incomplete configs that lack the current format-1 schema keys.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a checkpoint.

    Raises
    ------
    ValueError
        If any required current-schema key is missing.
    """
    missing = sorted(_FORMAT_1_REQUIRED_KEYS - config.keys())
    if missing:
        msg = (
            "Deprecated checkpoint schema: missing required format_version 1 "
            f"fields: {', '.join(missing)}. Pre-1.0 checkpoints are not "
            "migrated; re-save the model with the current package or "
            "reconstruct the architecture explicitly."
        )
        raise ValueError(msg)


def _resolve_checkpoint_adjacency(
    adjacency: Any,
    *,
    koopman_kind: str,
) -> str:
    """Validate checkpoint ``adjacency`` and return a factory keyword value.

    Parameters
    ----------
    adjacency : Any
        Checkpoint ``config.adjacency`` value (``None`` for non-networked
        kinds, or a mode string for graph / continuous-graph).
    koopman_kind : str
        Serialized operator kind.

    Returns
    -------
    str
        Factory ``koopman_adjacency`` value (defaults to ``"symmetric"`` for
        non-networked kinds when ``adjacency`` is ``None``).

    Raises
    ------
    ValueError
        If a networked kind is missing / invalid ``adjacency``, or a
        non-networked kind supplies a non-null value.
    """
    accepted = ", ".join(sorted(_GRAPH_ADJACENCY_MODES))
    if koopman_kind in _NETWORKED_ADJACENCY_KINDS:
        if adjacency is None:
            msg = (
                "Checkpoint config.adjacency is required for networked "
                f"koopman_kind={koopman_kind!r}; accepted values: "
                f"{{{accepted}}}"
            )
            raise ValueError(msg)
        if adjacency not in _GRAPH_ADJACENCY_MODES:
            msg = (
                "Checkpoint config.adjacency must be one of "
                f"{{{accepted}}}, got {adjacency!r}"
            )
            raise ValueError(msg)
        return str(adjacency)
    if adjacency is not None:
        msg = (
            "Checkpoint config.adjacency must be null for non-networked "
            f"koopman_kind={koopman_kind!r}, got {adjacency!r}"
        )
        raise ValueError(msg)
    return "symmetric"


def _migrate_config(config: dict[str, Any], *, format_version: int) -> dict[str, Any]:
    """Validate checkpoint config before reconstruct (beta: no migrations).

    Format 1 is the only supported baseline through 0.x. Incomplete schemas are
    rejected as deprecated rather than backfilled. Multi-version migration
    branches are deferred until the 1.0 checkpoint policy.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a saved checkpoint.
    format_version : int
        Checkpoint ``format_version`` after supported-version validation.

    Returns
    -------
    dict
        Config ready for :func:`reconstruct_model`.

    Raises
    ------
    ValueError
        If the format version is unsupported or the config fails schema
        validation for the active version.
    """
    if format_version == 1:
        _require_format1_schema(config)
        if config.get("koopman_kind") == "hetero_graph":
            _require_hetero_schema(config)
        return config

    msg = (
        f"No migration path for checkpoint format_version {format_version}; "
        f"supported versions: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
    )
    raise ValueError(msg)


def _require_hetero_schema(config: dict[str, Any]) -> None:
    """Reject incomplete or unsupported hetero format-1 payloads.

    Parameters
    ----------
    config : dict
        Architecture configuration block from a checkpoint.

    Raises
    ------
    ValueError
        If required hetero keys are missing or malformed, or if
        ``relation_tying`` / ``basis_size`` are inconsistent.
    """
    missing = sorted(_HETERO_REQUIRED_KEYS - config.keys())
    if missing:
        msg = (
            "Incomplete hetero checkpoint schema: missing required "
            f"fields: {', '.join(missing)}. Hetero checkpoints require "
            "node_types and edge_types; re-save with the current package."
        )
        raise ValueError(msg)

    node_types = config["node_types"]
    edge_types = config["edge_types"]
    if not isinstance(node_types, (list, tuple)) or not node_types:
        msg = "Checkpoint config.node_types must be a non-empty sequence of strings"
        raise ValueError(msg)
    if any(not isinstance(name, str) or not name for name in node_types):
        msg = "Checkpoint config.node_types entries must be non-empty strings"
        raise ValueError(msg)
    if not isinstance(edge_types, (list, tuple)) or not edge_types:
        msg = (
            "Checkpoint config.edge_types must be a non-empty sequence of "
            "(src, rel, dst) triples"
        )
        raise ValueError(msg)
    for entry in edge_types:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            msg = (
                "Checkpoint config.edge_types entries must be "
                f"(src, rel, dst) triples; got {entry!r}"
            )
            raise ValueError(msg)
        if any(not isinstance(part, str) or not part for part in entry):
            msg = "Checkpoint config.edge_types triples must use non-empty strings"
            raise ValueError(msg)

    relation_tying = config["relation_tying"]
    if relation_tying not in _RELATION_TYING_MODES:
        accepted = ", ".join(sorted(_RELATION_TYING_MODES))
        msg = (
            "Checkpoint config.relation_tying must be one of "
            f"{{{accepted}}}, got {relation_tying!r}"
        )
        raise ValueError(msg)
    basis_size = config["basis_size"]
    if relation_tying == "independent":
        if basis_size is not None:
            msg = (
                "Checkpoint config.basis_size must be null when "
                "relation_tying='independent'"
            )
            raise ValueError(msg)
    else:
        if not isinstance(basis_size, int) or isinstance(basis_size, bool):
            msg = (
                "Checkpoint config.basis_size must be a positive int when "
                f"relation_tying='basis'; got {basis_size!r}"
            )
            raise ValueError(msg)
        if basis_size < 1:
            msg = (
                "Checkpoint config.basis_size must be a positive int when "
                f"relation_tying='basis'; got {basis_size!r}"
            )
            raise ValueError(msg)

    relation_normalization = config["relation_normalization"]
    if relation_normalization not in _RELATION_NORMALIZATION_MODES:
        accepted = ", ".join(sorted(_RELATION_NORMALIZATION_MODES))
        msg = (
            "Checkpoint config.relation_normalization must be one of "
            f"{{{accepted}}}, got {relation_normalization!r}"
        )
        raise ValueError(msg)

    if "synthesize_reverse_relations" in config:
        flag = config["synthesize_reverse_relations"]
        if not isinstance(flag, bool):
            msg = (
                "Checkpoint config.synthesize_reverse_relations must be a "
                f"bool when present, got {flag!r}"
            )
            raise ValueError(msg)

    if "latent_dims" in config:
        latent_dims = config["latent_dims"]
        if not isinstance(latent_dims, dict):
            msg = (
                "Checkpoint config.latent_dims must be a mapping of node type "
                f"to positive int when present, got {type(latent_dims).__name__}"
            )
            raise ValueError(msg)
        try:
            validate_latent_dims(
                node_types,
                latent_dims,
                shared_latent_dim=int(config["latent_dim"]),
            )
        except ValueError as exc:
            msg = (
                f"Checkpoint config.latent_dims is incomplete or invalid: {exc}. "
                "Re-save with the current package (FORMAT_VERSION=1 additive "
                "latent_dims); shared-d checkpoints omit this key."
            )
            raise ValueError(msg) from exc

    encoder = config.get("encoder")
    if (
        isinstance(encoder, dict)
        and "normalization" in encoder
        and encoder["normalization"] != relation_normalization
    ):
        msg = (
            "Checkpoint config.relation_normalization "
            f"({relation_normalization!r}) must match "
            f"encoder.normalization ({encoder['normalization']!r})"
        )
        raise ValueError(msg)


def _state_dict_has_rectangular_hetero_markers(
    state_dict: dict[str, Any],
) -> bool:
    """Return whether ``state_dict`` contains rectangular hetero factor keys.

    Parameters
    ----------
    state_dict
        Value for ``state_dict``.

    Returns
    -------
    object
        Function result.
    """
    for key in state_dict:
        if key.startswith("koopman._rel_rect."):
            return True
        if ".type_latent." in key or ".type_latent_in." in key:
            return True
    return False


def _validate_hetero_latent_dims_vs_state(
    config: dict[str, Any],
    state_dict: dict[str, Any],
) -> None:
    """Reject rectangular weights without ``latent_dims`` or shape mismatches.

    Parameters
    ----------
    config : dict
        Migrated hetero architecture config.
    state_dict : dict
        Checkpoint weight dictionary.

    Raises
    ------
    ValueError
        If rectangular markers disagree with ``latent_dims``, or per-type
        self-factor shapes disagree with declared ``d_τ``.
    """
    has_rect = _state_dict_has_rectangular_hetero_markers(state_dict)
    latent_dims = config.get("latent_dims")
    if has_rect and latent_dims is None:
        msg = (
            "Checkpoint state_dict contains rectangular hetero factors "
            "(_rel_rect / type_latent*) but config.latent_dims is missing; "
            "re-save with the current package (FORMAT_VERSION=1 additive "
            "latent_dims). Rectangular mode is never inferred from weights alone."
        )
        raise ValueError(msg)
    if latent_dims is None:
        return

    validated = validate_latent_dims(
        config["node_types"],
        latent_dims,
        shared_latent_dim=int(config["latent_dim"]),
    )
    assert validated is not None
    is_rectangular = any(
        width != int(config["latent_dim"]) for width in validated.values()
    )
    if is_rectangular and not any(
        key.startswith("koopman._rel_rect.") for key in state_dict
    ):
        msg = (
            "Checkpoint config.latent_dims implies rectangular relation "
            "factors but state_dict has no koopman._rel_rect.* keys; "
            "re-save with the current package."
        )
        raise ValueError(msg)

    for name, width in validated.items():
        self_key = f"koopman._selves.{name}.K"
        if self_key not in state_dict:
            continue
        shape = tuple(state_dict[self_key].shape)
        expected = (int(width), int(width))
        if shape != expected:
            msg = (
                f"Checkpoint state_dict[{self_key!r}] has shape {shape} but "
                f"config.latent_dims[{name!r}]={width} expects {expected}; "
                "re-save with matching latent_dims / factors."
            )
            raise ValueError(msg)


def _parse_symmetry_config(
    symmetry: Any,
) -> tuple[list[list[int]] | None, bool, str, str | None]:
    """Parse the format-1 ``symmetry`` config block into factory kwargs.

    Parameters
    ----------
    symmetry : Any
        ``None`` or a dict with ``auto_orbits``, ``orbit_partition``,
        ``method``, and optional additive ``symmetry``
        (``"orbit"`` / ``"isotypic"``).

    Returns
    -------
    tuple
        ``(orbit_partition, auto_orbits, orbit_method, koopman_symmetry)``.
        ``koopman_symmetry`` is ``"isotypic"`` when the additive field says
        so; otherwise ``None`` (orbit / legacy paths use the orbit kwargs).

    Raises
    ------
    ValueError
        If the block shape or field types are invalid.
    """
    if symmetry is None:
        return None, False, "auto", None
    if not isinstance(symmetry, dict):
        msg = f"symmetry config must be a dict or None, got {type(symmetry).__name__}"
        raise ValueError(msg)
    mode = symmetry.get("symmetry")
    if mode is not None and mode not in {"orbit", "isotypic"}:
        msg = f"symmetry.symmetry must be 'orbit', 'isotypic', or absent, got {mode!r}"
        raise ValueError(msg)
    koopman_symmetry = "isotypic" if mode == "isotypic" else None
    auto_orbits = bool(symmetry.get("auto_orbits", False))
    method = symmetry.get("method", "auto")
    if koopman_symmetry == "isotypic":
        method = "exact"
        auto_orbits = False
    if method not in {"auto", "exact"}:
        msg = f"symmetry.method must be 'auto' or 'exact', got {method!r}"
        raise ValueError(msg)
    raw_partition = symmetry.get("orbit_partition")
    if raw_partition is None:
        return None, auto_orbits, method, koopman_symmetry
    if not isinstance(raw_partition, (list, tuple)):
        msg = "symmetry.orbit_partition must be a sequence of orbits or None"
        raise ValueError(msg)
    partition: list[list[int]] = []
    for orbit in raw_partition:
        if not isinstance(orbit, (list, tuple)):
            msg = "each symmetry.orbit_partition orbit must be a sequence of ints"
            raise ValueError(msg)
        partition.append([int(node) for node in orbit])
    return partition, auto_orbits, method, koopman_symmetry


def _package_version() -> str:
    """Return the installed package version for checkpoint metadata.

    Returns
    -------
    str
        Installed ``koopman-graph`` version, or ``"0.0.0"`` when running from
        source without package metadata.
    """
    try:
        return version("koopman-graph")
    except PackageNotFoundError:
        return "0.0.0"


_SUPPORTED_ENCODER_TYPES: dict[str, type[BaseEncoder]] = {
    "gcn": GNNEncoder,
    "gat": GATEncoder,
    "sage": SAGEEncoder,
    "diffconv": DiffConvEncoder,
    "transformer": GraphTransformerEncoder,
    "hyper_enc": HypergraphEncoder,
    "sim_enc": SimplicialEncoder,
    "sheaf_enc": SheafGNNEncoder,
    "cell_enc": CellComplexGNNEncoder,
    "inv_geom_enc": InvariantGeometryEncoder,
    "relgraph_enc": RelGraphEncoder,
}

_SUPPORTED_DECODER_TYPES: dict[str, type[Decoder]] = {
    "gcn": GNNDecoder,
    "gat": GATDecoder,
    "sage": SAGEDecoder,
    "diffconv": DiffConvDecoder,
    "transformer": GraphTransformerDecoder,
    "hyper_dec": HypergraphDecoder,
    "sim_dec": SimplicialDecoder,
    "sheaf_dec": SheafGNNDecoder,
    "cell_dec": CellComplexGNNDecoder,
    "relgraph_dec": RelGraphDecoder,
}


def _encoder_type(encoder: BaseEncoder) -> str:
    """Return the checkpoint encoder type string for an encoder instance.

    Parameters
    ----------
    encoder : GNNEncoder, GATEncoder, SAGEEncoder, DiffConvEncoder, or
        GraphTransformerEncoder
        Encoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
        ``"hyper_enc"``, ``"sim_enc"``, ``"sheaf_enc"``, ``"cell_enc"``,
        ``"inv_geom_enc"``, or ``"relgraph_enc"``.

    Raises
    ------
    TypeError
        If ``encoder`` is not a supported encoder class.
    """
    if isinstance(encoder, GraphTransformerEncoder):
        return "transformer"
    if isinstance(encoder, DiffConvEncoder):
        return "diffconv"
    if isinstance(encoder, SAGEEncoder):
        return "sage"
    if isinstance(encoder, GATEncoder):
        return "gat"
    if isinstance(encoder, GNNEncoder):
        return "gcn"
    if isinstance(encoder, HypergraphEncoder):
        return "hyper_enc"
    if isinstance(encoder, SimplicialEncoder):
        return "sim_enc"
    if isinstance(encoder, SheafGNNEncoder):
        return "sheaf_enc"
    if isinstance(encoder, CellComplexGNNEncoder):
        return "cell_enc"
    if isinstance(encoder, InvariantGeometryEncoder):
        return "inv_geom_enc"
    if isinstance(encoder, RelGraphEncoder):
        return "relgraph_enc"
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _unwrap_base_encoder(
    encoder: nn.Module,
) -> tuple[BaseEncoder, int]:
    """Return the serializable base encoder and delay count.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder, possibly wrapped in :class:`DelayEmbeddingEncoder`.

    Returns
    -------
    base_encoder : GNNEncoder, GATEncoder, SAGEEncoder, DiffConvEncoder, or
        GraphTransformerEncoder
        Checkpoint-rebuildable encoder.
    n_delays : int
        Delay window length (``1`` when unwrapped).
    """
    if isinstance(encoder, DelayEmbeddingEncoder):
        base = encoder.base_encoder
        if not isinstance(
            base,
            (
                GNNEncoder,
                GATEncoder,
                SAGEEncoder,
                DiffConvEncoder,
                GraphTransformerEncoder,
            ),
        ):
            msg = (
                "DelayEmbeddingEncoder.base_encoder must be GNNEncoder, "
                "GATEncoder, SAGEEncoder, DiffConvEncoder, or "
                "GraphTransformerEncoder for "
                f"checkpoints; got {type(base).__name__}"
            )
            raise TypeError(msg)
        return base, encoder.n_delays
    if isinstance(
        encoder,
        (
            GNNEncoder,
            GATEncoder,
            SAGEEncoder,
            DiffConvEncoder,
            GraphTransformerEncoder,
            HypergraphEncoder,
            SimplicialEncoder,
            SheafGNNEncoder,
            CellComplexGNNEncoder,
            InvariantGeometryEncoder,
            RelGraphEncoder,
        ),
    ):
        return encoder, 1
    msg = f"Unsupported encoder type: {type(encoder).__name__}"
    raise TypeError(msg)


def _decoder_type(decoder: Decoder) -> str:
    """Return the checkpoint decoder type string for a decoder instance.

    Parameters
    ----------
    decoder : GNNDecoder, GATDecoder, SAGEDecoder, DiffConvDecoder, or
        GraphTransformerDecoder
        Decoder whose architecture type will be serialized.

    Returns
    -------
    str
        ``"gcn"``, ``"gat"``, ``"sage"``, ``"diffconv"``, ``"transformer"``,
        ``"hyper_dec"``, ``"sim_dec"``, ``"sheaf_dec"``, ``"cell_dec"``, or
        ``"relgraph_dec"``.

    Raises
    ------
    TypeError
        If ``decoder`` is not a supported decoder class.
    """
    if isinstance(decoder, GraphTransformerDecoder):
        return "transformer"
    if isinstance(decoder, DiffConvDecoder):
        return "diffconv"
    if isinstance(decoder, SAGEDecoder):
        return "sage"
    if isinstance(decoder, GATDecoder):
        return "gat"
    if isinstance(decoder, GNNDecoder):
        return "gcn"
    if isinstance(decoder, HypergraphDecoder):
        return "hyper_dec"
    if isinstance(decoder, SimplicialDecoder):
        return "sim_dec"
    if isinstance(decoder, SheafGNNDecoder):
        return "sheaf_dec"
    if isinstance(decoder, CellComplexGNNDecoder):
        return "cell_dec"
    if isinstance(decoder, RelGraphDecoder):
        return "relgraph_dec"
    msg = f"Unsupported decoder type: {type(decoder).__name__}"
    raise TypeError(msg)


def _require_serializable_koopman(model: ModeShapeModel) -> None:
    """Reject custom injected operators that lack checkpoint factory metadata.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model whose ``koopman`` submodule will be serialized.

    Raises
    ------
    TypeError
        If ``model.koopman`` is not a built-in
        :class:`~koopman_graph.operators.KoopmanOperator`,
        :class:`~koopman_graph.operators.ContinuousKoopmanOperator`, or
        :class:`~koopman_graph.operators.GraphKoopmanOperator`.
    """
    if isinstance(model.koopman, _SERIALIZABLE_KOOPMAN_TYPES):
        return
    msg = (
        "Checkpoint serialization supports only built-in KoopmanOperator, "
        "ContinuousKoopmanOperator, GraphKoopmanOperator, "
        "HypergraphKoopmanOperator, GlobalLocalKoopmanOperator, "
        "ContinuousGraphKoopmanOperator, HeteroGraphKoopmanOperator, "
        "SwitchedKoopmanOperator, MixtureKoopmanOperator, and "
        "HodgeKoopmanOperator "
        "instances. Custom injected operators are not round-trippable; "
        "save the operator state separately or reconstruct the model with "
        f"koopman=... after load. Got {type(model.koopman).__name__}."
    )
    raise TypeError(msg)


def _operator_bound_num_nodes(model: ModeShapeModel) -> int | None:
    """Return operator / adaptive-topology bound ``N_max`` when known.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose Koopman operator or adaptive topology may bind node count.

    Returns
    -------
    int or None
        Bound universe size, or ``None`` when the operator does not fix ``N``.
    """
    koopman = getattr(model, "koopman", None)
    node_orbit = getattr(koopman, "_node_orbit", None)
    if node_orbit is not None:
        return int(node_orbit.numel())
    adaptive = getattr(model, "adaptive_topology", None)
    if adaptive is not None and getattr(adaptive, "num_nodes", None) is not None:
        return int(adaptive.num_nodes)
    return None


def _coerce_checkpoint_entity_ids(
    raw: object,
) -> tuple[str | int, ...] | None:
    """Validate optional checkpoint ``entity_ids`` into a typed tuple.

    Parameters
    ----------
    raw : object
        Checkpoint value (``None``, list, or tuple of str/int).

    Returns
    -------
    tuple of str or int, or None
        Coerced entity ids.

    Raises
    ------
    ValueError
        If the value is not a list/tuple of str/int, or is empty.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        msg = (
            "Checkpoint config.entity_ids must be a list of str/int or null; "
            f"got {type(raw).__name__}. Re-save with the current package."
        )
        raise ValueError(msg)
    if len(raw) < 1:
        msg = "Checkpoint config.entity_ids must be non-empty when provided"
        raise ValueError(msg)
    ids: list[str | int] = []
    for idx, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            msg = (
                "Checkpoint config.entity_ids entries must be str or int; "
                f"index {idx} has type {type(item).__name__}. Re-save with "
                "the current package."
            )
            raise ValueError(msg)
        ids.append(item)
    return tuple(ids)


def _reject_entity_ids_length_mismatch(
    model: ModeShapeModel,
    entity_ids: Sequence[str | int],
) -> None:
    """Reject ``entity_ids`` whose length disagrees with bound ``N_max``.

    Parameters
    ----------
    model : ModeShapeModel
        Model that may expose a bound operator node count.
    entity_ids : sequence of str or int
        Candidate universe keys.

    Raises
    ------
    ValueError
        If a bound ``N_max`` exists and ``len(entity_ids)`` differs.
    """
    bound = _operator_bound_num_nodes(model)
    if bound is None:
        return
    if len(entity_ids) == bound:
        return
    msg = (
        f"Checkpoint entity_ids length ({len(entity_ids)}) disagrees with "
        f"operator N_max ({bound}); re-save with matching entity_ids or "
        "rebuild the model. Presence mask tensors are sequence data and are "
        "not stored in checkpoints."
    )
    raise ValueError(msg)


def _apply_churn_contract_from_config(
    model: ModeShapeModel,
    config: dict[str, Any],
) -> None:
    """Restore additive churn-contract keys onto ``model``.

    Absent keys load as the 0.10 fixed-cardinality contract
    (``allow_node_churn=False``, ``has_presence_masks=False``,
    ``entity_ids=None``).

    Parameters
    ----------
    model : ModeShapeModel
        Reconstructed model receiving the contract attributes.
    config : dict
        Architecture configuration block from a checkpoint.

    Raises
    ------
    ValueError
        If ``entity_ids`` are malformed or disagree with bound ``N_max``.
    """
    allow = bool(config.get("allow_node_churn", False))
    has_presence = bool(config.get("has_presence_masks", False))
    entity_ids = _coerce_checkpoint_entity_ids(config.get("entity_ids"))
    if entity_ids is not None:
        _reject_entity_ids_length_mismatch(model, entity_ids)
    model._allow_node_churn = allow  # type: ignore[attr-defined]
    model._has_presence_masks = has_presence  # type: ignore[attr-defined]
    model._entity_ids = entity_ids  # type: ignore[attr-defined]


def build_model_config(model: ModeShapeModel) -> dict[str, Any]:
    """Extract architecture configuration from a :class:`GraphKoopmanModel`.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose encoder, decoder, and Koopman settings will be serialized.

    Returns
    -------
    dict
        JSON-serializable architecture configuration.

    Raises
    ------
    TypeError
        If ``model.koopman`` is a custom injected operator (not a built-in
        :class:`~koopman_graph.operators.KoopmanOperator`,
        :class:`~koopman_graph.operators.ContinuousKoopmanOperator`, or
        :class:`~koopman_graph.operators.GraphKoopmanOperator`).
    ValueError
        If stamped ``entity_ids`` disagree with a bound operator ``N_max``.
    """
    _require_serializable_koopman(model)
    encoder, n_delays = _unwrap_base_encoder(model.encoder)
    decoder = model.decoder
    encoder_config: dict[str, Any] = {
        "type": _encoder_type(encoder),
        "in_channels": encoder.in_channels,
        "hidden_channels": encoder.hidden_channels,
        "latent_dim": encoder.latent_dim,
        "num_layers": encoder.num_layers,
        "activation": encoder.activation_name,
    }
    if isinstance(encoder, (GATEncoder, GraphTransformerEncoder)):
        encoder_config["heads"] = encoder.heads
        encoder_config["dropout"] = encoder.dropout
    if isinstance(encoder, GraphTransformerEncoder):
        encoder_config["edge_dim"] = encoder.edge_dim
    if isinstance(encoder, DiffConvEncoder):
        encoder_config["diffusion_steps"] = encoder.diffusion_steps
    if isinstance(encoder, SimplicialEncoder):
        encoder_config["residual"] = encoder.residual
    if isinstance(encoder, SheafGNNEncoder):
        encoder_config["residual"] = encoder.residual
        encoder_config["restriction_maps"] = encoder.restriction_maps
    if isinstance(encoder, CellComplexGNNEncoder):
        encoder_config["residual"] = encoder.residual
    if isinstance(encoder, RelGraphEncoder):
        encoder_config["num_relations"] = encoder.num_relations
        encoder_config["normalization"] = encoder.normalization
        encoder_config["root_weight"] = encoder.root_weight
        if encoder.node_types is not None:
            encoder_config["node_types"] = list(encoder.node_types)
        if encoder.edge_types is not None:
            encoder_config["edge_types"] = [
                list(triple) for triple in encoder.edge_types
            ]

    decoder_config: dict[str, Any] = {
        "type": _decoder_type(decoder),
        "latent_dim": decoder.latent_dim,
        "hidden_channels": decoder.hidden_channels,
        "out_channels": decoder.out_channels,
        "num_layers": decoder.num_layers,
        "activation": decoder.activation_name,
    }
    if isinstance(decoder, (GATDecoder, GraphTransformerDecoder)):
        decoder_config["heads"] = decoder.heads
        decoder_config["dropout"] = decoder.dropout
    if isinstance(decoder, GraphTransformerDecoder):
        decoder_config["edge_dim"] = decoder.edge_dim
    if isinstance(decoder, DiffConvDecoder):
        decoder_config["diffusion_steps"] = decoder.diffusion_steps
    if isinstance(decoder, SimplicialDecoder):
        decoder_config["residual"] = decoder.residual
    if isinstance(decoder, SheafGNNDecoder):
        decoder_config["residual"] = decoder.residual
        decoder_config["restriction_maps"] = decoder.restriction_maps
    if isinstance(decoder, CellComplexGNNDecoder):
        decoder_config["residual"] = decoder.residual
    if isinstance(decoder, RelGraphDecoder):
        decoder_config["num_relations"] = decoder.num_relations
        decoder_config["normalization"] = decoder.normalization
        decoder_config["root_weight"] = decoder.root_weight
        if decoder.node_types is not None:
            decoder_config["node_types"] = list(decoder.node_types)
        if decoder.edge_types is not None:
            decoder_config["edge_types"] = [
                list(triple) for triple in decoder.edge_types
            ]

    physics_config: dict[str, Any] | None = None
    if model.physics_dim > 0:
        physics_config = {
            "dim": model.physics_dim,
            "preset": model.physics_preset,
            "position": model.physics_position,
        }

    sparsity = getattr(model.koopman, "sparsity", "dense")
    adjacency = getattr(model.koopman, "adjacency", None)
    incidence_mode = getattr(model.koopman, "incidence_mode", None)
    config: dict[str, Any] = {
        "latent_dim": model.latent_dim,
        "time_step": model.time_step,
        "dynamics_mode": model.dynamics_mode,
        "koopman_kind": getattr(model, "koopman_kind", "pernode"),
        "koopman_init_mode": model.koopman.init_mode,
        "koopman_init_scale": model.koopman.init_scale,
        "koopman_parameterization": model.koopman.parameterization,
        "koopman_max_spectral_radius": resolve_factory_stability_bound(
            model.koopman,
            dynamics_mode=model.dynamics_mode,
        ),
        "koopman_auxiliary_hidden_dims": (
            list(model.koopman.auxiliary_hidden_dims)
            if isinstance(model.koopman, ContinuousKoopmanOperator)
            and model.koopman.parameterization == "auxiliary_spectral"
            else None
        ),
        "local_window": (
            int(model.koopman.local_window)
            if isinstance(
                model.koopman,
                (GlobalLocalKoopmanOperator, MixtureKoopmanOperator),
            )
            else None
        ),
        "local_rank": (
            int(model.koopman.local_rank)
            if isinstance(model.koopman, GlobalLocalKoopmanOperator)
            else None
        ),
        "local_hidden_dims": (
            list(model.koopman.local_hidden_dims)
            if isinstance(model.koopman, GlobalLocalKoopmanOperator)
            else None
        ),
        "control_dim": model.control_dim,
        "control_mode": getattr(model, "control_mode", "additive"),
        "bilinear_rank": getattr(model, "bilinear_rank", None),
        "n_delays": n_delays,
        "physics": physics_config,
        "encoder": encoder_config,
        "decoder": decoder_config,
        "sparsity": sparsity,
        "adjacency": adjacency,
        "hypergraph_incidence_mode": (
            str(incidence_mode)
            if isinstance(model.koopman, HypergraphKoopmanOperator)
            else None
        ),
        "learn_topology": getattr(model, "learn_topology", None),
        "topology_embedding_dim": (
            int(model.topology_embedding_dim)
            if getattr(model, "learn_topology", None) is not None
            else None
        ),
        "symmetry": (
            model.koopman.symmetry_config()
            if hasattr(model.koopman, "symmetry_config")
            else None
        ),
        "allow_node_churn": bool(getattr(model, "allow_node_churn", False)),
        "has_presence_masks": bool(getattr(model, "has_presence_masks", False)),
    }
    entity_ids = getattr(model, "entity_ids", None)
    if entity_ids is not None:
        config["entity_ids"] = list(entity_ids)
        _reject_entity_ids_length_mismatch(model, entity_ids)
    if isinstance(model.koopman, HeteroGraphKoopmanOperator):
        config["node_types"] = list(model.koopman.node_types)
        config["edge_types"] = [list(triple) for triple in model.koopman.edge_types]
        config["relation_tying"] = model.koopman.relation_tying
        config["basis_size"] = model.koopman.basis_size
        config["relation_normalization"] = model.koopman.normalization
        config["synthesize_reverse_relations"] = bool(
            getattr(model, "synthesize_reverse_relations", False)
        )
        config["adjacency"] = None
        if model.koopman.latent_dims is not None:
            config["latent_dims"] = dict(model.koopman.latent_dims)
    return config


def _build_encoder(
    config: dict[str, Any],
    *,
    latent_dims: dict[str, int] | None = None,
) -> BaseEncoder:
    """Instantiate an encoder from a checkpoint configuration block.

    Parameters
    ----------
    config
        Value for ``config``.
    latent_dims
        Value for ``latent_dims``.

    Returns
    -------
    object
        Function result.
    """
    encoder_type = config["type"]
    encoder_cls = _SUPPORTED_ENCODER_TYPES.get(encoder_type)
    if encoder_cls is None:
        msg = f"Unsupported encoder type in checkpoint: {encoder_type!r}"
        raise ValueError(msg)

    common_kwargs = {
        "in_channels": config["in_channels"],
        "hidden_channels": config["hidden_channels"],
        "latent_dim": config["latent_dim"],
        "num_layers": config["num_layers"],
        "activation": config["activation"],
    }
    if encoder_type == "gat":
        return GATEncoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
        )
    if encoder_type == "transformer":
        return GraphTransformerEncoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
            edge_dim=config.get("edge_dim"),
        )
    if encoder_type == "sage":
        return SAGEEncoder(**common_kwargs)
    if encoder_type == "diffconv":
        return DiffConvEncoder(
            **common_kwargs,
            diffusion_steps=config.get("diffusion_steps", 2),
        )
    if encoder_type == "hyper_enc":
        return HypergraphEncoder(**common_kwargs)
    if encoder_type == "sim_enc":
        return SimplicialEncoder(
            **common_kwargs,
            residual=config.get("residual", False),
        )
    if encoder_type == "sheaf_enc":
        return SheafGNNEncoder(
            **common_kwargs,
            residual=config.get("residual", False),
            restriction_maps=config.get("restriction_maps", "diagonal"),
        )
    if encoder_type == "cell_enc":
        return CellComplexGNNEncoder(
            **common_kwargs,
            residual=config.get("residual", False),
        )
    if encoder_type == "inv_geom_enc":
        return InvariantGeometryEncoder(**common_kwargs)
    if encoder_type == "relgraph_enc":
        return RelGraphEncoder(
            **common_kwargs,
            num_relations=config["num_relations"],
            normalization=config.get("normalization", "rgcn_in_degree"),
            root_weight=config.get("root_weight", True),
            node_types=config.get("node_types"),
            edge_types=config.get("edge_types"),
            latent_dims=latent_dims,
        )
    return GNNEncoder(**common_kwargs)


def _build_decoder(
    config: dict[str, Any],
    *,
    latent_dims: dict[str, int] | None = None,
) -> Decoder:
    """Instantiate a decoder from a checkpoint configuration block.

    Parameters
    ----------
    config
        Value for ``config``.
    latent_dims
        Value for ``latent_dims``.

    Returns
    -------
    object
        Function result.
    """
    decoder_type = config.get("type", "gcn")
    decoder_cls = _SUPPORTED_DECODER_TYPES.get(decoder_type)
    if decoder_cls is None:
        msg = f"Unsupported decoder type in checkpoint: {decoder_type!r}"
        raise ValueError(msg)

    common_kwargs = {
        "latent_dim": config["latent_dim"],
        "hidden_channels": config["hidden_channels"],
        "out_channels": config["out_channels"],
        "num_layers": config["num_layers"],
        "activation": config["activation"],
    }
    if decoder_type == "gat":
        return GATDecoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
        )
    if decoder_type == "transformer":
        return GraphTransformerDecoder(
            **common_kwargs,
            heads=config.get("heads", 1),
            dropout=config.get("dropout", 0.0),
            edge_dim=config.get("edge_dim"),
        )
    if decoder_type == "sage":
        return SAGEDecoder(**common_kwargs)
    if decoder_type == "diffconv":
        return DiffConvDecoder(
            **common_kwargs,
            diffusion_steps=config.get("diffusion_steps", 2),
        )
    if decoder_type == "hyper_dec":
        return HypergraphDecoder(**common_kwargs)
    if decoder_type == "sim_dec":
        return SimplicialDecoder(
            **common_kwargs,
            residual=config.get("residual", False),
        )
    if decoder_type == "sheaf_dec":
        return SheafGNNDecoder(
            **common_kwargs,
            residual=config.get("residual", False),
            restriction_maps=config.get("restriction_maps", "diagonal"),
        )
    if decoder_type == "cell_dec":
        return CellComplexGNNDecoder(
            **common_kwargs,
            residual=config.get("residual", False),
        )
    if decoder_type == "relgraph_dec":
        return RelGraphDecoder(
            **common_kwargs,
            num_relations=config["num_relations"],
            normalization=config.get("normalization", "rgcn_in_degree"),
            root_weight=config.get("root_weight", True),
            node_types=config.get("node_types"),
            edge_types=config.get("edge_types"),
            latent_dims=latent_dims,
        )
    return GNNDecoder(**common_kwargs)


def reconstruct_model(
    config: dict[str, Any],
    *,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> ModeShapeModel:
    """Reconstruct a :class:`GraphKoopmanModel` from a checkpoint configuration.

    Parameters
    ----------
    config : dict
        Architecture configuration produced by :func:`build_model_config`.
    physics_lifting_fn : callable or None, optional
        Custom physics lifting function for hybrid checkpoints that do not store
        a registered preset.

    Returns
    -------
    GraphKoopmanModel
        Uninitialized-weight model matching the saved architecture.

    Raises
    ------
    ValueError
        If a hybrid checkpoint requires a physics lifting function that is not
        provided and cannot be resolved from a preset, or if
        ``physics.position`` is unsupported.
    """
    import importlib

    estimator_mod = importlib.import_module("koopman_graph.model.estimator")
    GraphKoopmanModel = estimator_mod.GraphKoopmanModel

    hetero_latent_dims = (
        dict(config["latent_dims"])
        if config.get("koopman_kind") == "hetero_graph"
        and config.get("latent_dims") is not None
        else None
    )
    decoder = _build_decoder(config["decoder"], latent_dims=hetero_latent_dims)
    encoder = _build_encoder(config["encoder"], latent_dims=hetero_latent_dims)

    physics_config = config.get("physics")
    physics_dim = 0
    physics_preset: str | None = None
    physics_position: PhysicsPosition = PHYSICS_POSITION
    resolved_physics_fn: PhysicsLiftingFn | None = None
    if isinstance(physics_config, dict):
        physics_dim = int(physics_config.get("dim", 0))
        physics_preset = physics_config.get("preset")
        if physics_dim > 0:
            physics_position = resolve_physics_position(physics_config.get("position"))
            resolved_physics_fn = resolve_physics_lifting_fn(
                physics_preset=physics_preset,
                physics_lifting_fn=physics_lifting_fn,
            )
            if resolved_physics_fn is None:
                msg = (
                    "Checkpoint uses hybrid physics observables but no preset is "
                    "stored; pass physics_lifting_fn to load_checkpoint"
                )
                raise ValueError(msg)

    koopman_kind = config.get("koopman_kind", "pernode")
    if koopman_kind in _RESERVED_KOOPMAN_KINDS:
        task = _RESERVED_KOOPMAN_KINDS[koopman_kind]
        msg = f"koopman_kind={koopman_kind!r} is planned; lands in {task}"
        raise ValueError(msg)
    orbit_partition, auto_orbits, orbit_method, koopman_symmetry = (
        _parse_symmetry_config(config.get("symmetry"))
    )
    koopman_adjacency = _resolve_checkpoint_adjacency(
        config["adjacency"],
        koopman_kind=str(koopman_kind),
    )

    learn_topology = config.get("learn_topology")
    topology_embedding_dim = config.get("topology_embedding_dim")
    hetero_node_types = (
        list(config["node_types"]) if koopman_kind == "hetero_graph" else None
    )
    hetero_edge_types = (
        [list(triple) for triple in config["edge_types"]]
        if koopman_kind == "hetero_graph"
        else None
    )
    # Isotypic is mutually exclusive with orbit kwargs at the factory; restore
    # a saved partition after construction so state_dict bank shapes match.
    # Factory requires the default orbit_method when isotypic (exact is forced
    # inside the operator).
    factory_orbit_partition = (
        None if koopman_symmetry == "isotypic" else orbit_partition
    )
    factory_auto_orbits = False if koopman_symmetry == "isotypic" else auto_orbits
    factory_orbit_method = "auto" if koopman_symmetry == "isotypic" else orbit_method
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config["latent_dim"],
        time_step=config["time_step"],
        dynamics_mode=config.get("dynamics_mode", "discrete"),
        koopman=koopman_kind,
        koopman_init_mode=config["koopman_init_mode"],
        koopman_init_scale=config["koopman_init_scale"],
        koopman_parameterization=config.get("koopman_parameterization", "dense"),
        koopman_max_spectral_radius=config.get("koopman_max_spectral_radius", 1.0),
        koopman_auxiliary_hidden_dims=config.get("koopman_auxiliary_hidden_dims"),
        koopman_sparsity=config.get("sparsity", "dense"),
        koopman_adjacency=koopman_adjacency,
        koopman_hypergraph_incidence_mode=str(
            config.get("hypergraph_incidence_mode", "zhou_symmetric")
            or "zhou_symmetric"
        ),
        koopman_local_window=(
            int(config["local_window"]) if config.get("local_window") is not None else 4
        ),
        koopman_local_rank=(
            int(config["local_rank"]) if config.get("local_rank") is not None else 2
        ),
        koopman_local_hidden_dims=config.get("local_hidden_dims"),
        koopman_orbit_partition=factory_orbit_partition,
        koopman_auto_orbits=factory_auto_orbits,
        koopman_orbit_method=factory_orbit_method,
        koopman_symmetry=koopman_symmetry,
        learn_topology=learn_topology,
        topology_embedding_dim=(
            int(topology_embedding_dim)
            if topology_embedding_dim is not None
            else DEFAULT_TOPOLOGY_EMBEDDING_DIM
        ),
        control_dim=config.get("control_dim", 0),
        control_mode=config.get("control_mode", "additive"),
        bilinear_rank=config.get("bilinear_rank"),
        physics_lifting_fn=resolved_physics_fn,
        physics_preset=physics_preset,
        physics_dim=physics_dim,
        physics_position=physics_position,
        n_delays=int(config.get("n_delays", 1)),
        koopman_node_types=hetero_node_types,
        koopman_edge_types=hetero_edge_types,
        koopman_relation_tying=(
            str(config["relation_tying"])
            if koopman_kind == "hetero_graph"
            else "independent"
        ),
        koopman_basis_size=(
            None
            if koopman_kind != "hetero_graph" or config["basis_size"] is None
            else int(config["basis_size"])
        ),
        koopman_synthesize_reverse_relations=(
            bool(config.get("synthesize_reverse_relations", False))
            if koopman_kind == "hetero_graph"
            else False
        ),
        koopman_latent_dims=hetero_latent_dims,
    )
    if (
        koopman_symmetry == "isotypic"
        and orbit_partition is not None
        and hasattr(model.koopman, "set_orbit_partition")
    ):
        num_nodes = max(max(orbit) for orbit in orbit_partition) + 1
        model.koopman.set_orbit_partition(orbit_partition, num_nodes=num_nodes)
    _apply_churn_contract_from_config(model, config)
    return model


def build_checkpoint(model: ModeShapeModel) -> dict[str, Any]:
    """Build a versioned checkpoint dictionary for a model.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose weights and architecture will be serialized.

    Returns
    -------
    dict
        Checkpoint payload suitable for :func:`torch.save`.
    """
    return {
        "format_version": FORMAT_VERSION,
        "package_version": _package_version(),
        "config": build_model_config(model),
        "state_dict": model.state_dict(),
    }


def _json_ready(value: Any) -> Any:
    """Convert nested configs to JSON-serializable values.

    Parameters
    ----------
    value : Any
        Config fragment that may contain tuples.

    Returns
    -------
    Any
        Structure with tuples replaced by lists.
    """
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _state_dict_for_safetensors(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detach, clone, and contiguous-ize tensors for safetensors I/O.

    Parameters
    ----------
    state_dict : mapping of str to Tensor
        Model ``state_dict``.

    Returns
    -------
    dict of str to Tensor
        CPU contiguous copies suitable for :func:`safetensors.torch.save_file`.
    """
    return {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in state_dict.items()
    }


def _safetensors_v1_member_payload(
    model: ModeShapeModel,
) -> tuple[str, str, dict[str, torch.Tensor]]:
    """Build ``meta.json`` / ``config.json`` text and CPU weight tensors.

    Parameters
    ----------
    model : ModeShapeModel
        Model whose config and ``state_dict`` are serialized.

    Returns
    -------
    meta_text, config_text, weights
        JSON text for ``meta.json`` and ``config.json``, plus a CPU
        contiguous weight mapping for safetensors.
    """
    config = _json_ready(build_model_config(model))
    meta = {
        "container": SAFE_CONTAINER,
        "format_version": FORMAT_VERSION,
        "package_version": _package_version(),
    }
    meta_text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    weights = _state_dict_for_safetensors(model.state_dict())
    return meta_text, config_text, weights


def _save_safetensors_v1_directory(model: ModeShapeModel, path: Path) -> None:
    """Write a ``safetensors_v1`` directory checkpoint.

    Parameters
    ----------
    model : ModeShapeModel
        Model to serialize.
    path : Path
        Destination directory (created if missing). Must not be an existing
        regular file.

    Raises
    ------
    ValueError
        If ``path`` exists as a file.
    """
    if path.exists() and path.is_file():
        msg = (
            f"safetensors_v1 checkpoint path must be a directory; "
            f"got existing file {path}"
        )
        raise ValueError(msg)

    path.mkdir(parents=True, exist_ok=True)
    meta_text, config_text, weights = _safetensors_v1_member_payload(model)
    (path / SAFE_META_FILENAME).write_text(meta_text, encoding="utf-8")
    (path / SAFE_CONFIG_FILENAME).write_text(config_text, encoding="utf-8")
    safetensors_save_file(weights, path / SAFE_WEIGHTS_FILENAME)


def _save_safetensors_v1_zip(model: ModeShapeModel, path: Path) -> None:
    """Write a ``safetensors_v1`` zip / ``.kgckpt`` checkpoint.

    Parameters
    ----------
    model : ModeShapeModel
        Model to serialize.
    path : Path
        Destination ``.kgckpt`` or ``.zip`` file path. Must not be an
        existing directory.

    Raises
    ------
    ValueError
        If ``path`` exists as a directory.
    """
    if path.exists() and path.is_dir():
        msg = (
            f"safetensors_v1 zip checkpoint path must be a file; "
            f"got existing directory {path}"
        )
        raise ValueError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    meta_text, config_text, weights = _safetensors_v1_member_payload(model)
    weight_bytes = safetensors_save_bytes(weights)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SAFE_META_FILENAME, meta_text)
        archive.writestr(SAFE_CONFIG_FILENAME, config_text)
        archive.writestr(SAFE_WEIGHTS_FILENAME, weight_bytes)


def _is_safetensors_v1_zip(path: Path) -> bool:
    """Return True when ``path`` is a zip with root ``safetensors_v1`` members.

    PyTorch ``.pt`` archives are often zip files themselves; this probe
    requires the safe-container member names so legacy pickle archives are
    not mistaken for ``safetensors_v1``.

    Parameters
    ----------
    path : Path
        Candidate file path.

    Returns
    -------
    bool
        ``True`` when the zip root contains the safe-container member set.
    """
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return SAFE_ZIP_MEMBER_NAMES.issubset(names)


def save_checkpoint(
    model: ModeShapeModel,
    path: str | Path,
    *,
    format: CheckpointFormat = "safetensors_v1",
) -> None:
    """Persist a trained model checkpoint to disk.

    Parameters
    ----------
    model : ModeShapeModel
        Model to serialize.
    path : str or Path
        When ``format="safetensors_v1"`` (default): destination **directory**,
        or a ``.kgckpt`` / ``.zip`` file for the zip bundle of the same three
        members. When ``format="legacy_pt"``: destination ``.pt`` file path.
    format : {"safetensors_v1", "legacy_pt"}, optional
        On-disk container. Default ``safetensors_v1`` writes JSON config +
        safetensors weights; pass ``legacy_pt`` for the pickle ``.pt`` escape
        hatch (still used by ``fit(..., checkpoint_path=...)`` writers).

    Raises
    ------
    ValueError
        If ``format`` is unknown, or ``safetensors_v1`` path type conflicts
        with an existing file/directory.
    """
    destination = Path(path)
    if format == "legacy_pt":
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(build_checkpoint(model), destination)
        return
    if format == "safetensors_v1":
        if destination.suffix.lower() in SAFE_BUNDLE_SUFFIXES:
            _save_safetensors_v1_zip(model, destination)
        else:
            _save_safetensors_v1_directory(model, destination)
        return
    msg = f"Unsupported checkpoint format {format!r}"
    raise ValueError(msg)


def _allocate_adaptive_topology_from_state(
    model: ModeShapeModel,
    state_dict: dict[str, Any],
) -> None:
    """Bind ``AdaptiveAdjacency`` embeddings before ``load_state_dict``.

    Parameters
    ----------

    model : ModeShapeModel
        See the function signature / summary for ``model``.
    state_dict : dict[str, Any]
        See the function signature / summary for ``state_dict``.

    Returns
    -------

    None
        See summary line.

    Notes
    -----

    Lazy allocation means reconstructed models have no embedding parameters
    until ``set_num_nodes``; checkpoint tensors provide ``N``."""
    adaptive = getattr(model, "adaptive_topology", None)
    if adaptive is None:
        return
    source = state_dict.get("adaptive_topology.source_embedding")
    if source is None:
        return
    adaptive.set_num_nodes(int(source.shape[0]), device=source.device)


def _resolve_safetensors_device(
    map_location: str | torch.device | None,
) -> str:
    """Normalize ``map_location`` for :func:`safetensors.torch.load_file`.

    Parameters
    ----------
    map_location : str, torch.device, or None
        Device mapping from :func:`load_checkpoint`. Callables are not
        supported on the safetensors path.

    Returns
    -------
    str
        Device string (``"cpu"`` when ``map_location`` is ``None``).
    """
    if map_location is None:
        return "cpu"
    return str(map_location)


def _load_safetensors_v1_directory(
    path: Path,
    *,
    map_location: str | torch.device | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    """Load config and weights from a ``safetensors_v1`` directory.

    Parameters
    ----------
    path : Path
        Checkpoint directory containing ``meta.json``, ``config.json``, and
        ``model.safetensors``.
    map_location : str, torch.device, or None
        Device for weight tensors.

    Returns
    -------
    config : dict
        Architecture configuration.
    state_dict : dict of str to Tensor
        Model weights.
    format_version : int
        Logical checkpoint schema version from ``meta.json``.

    Raises
    ------
    ValueError
        If metadata, config, or weights are missing or invalid.
    """
    meta_path = path / SAFE_META_FILENAME
    config_path = path / SAFE_CONFIG_FILENAME
    weights_path = path / SAFE_WEIGHTS_FILENAME
    if not meta_path.is_file():
        msg = f"safetensors_v1 checkpoint missing {SAFE_META_FILENAME}: {path}"
        raise ValueError(msg)
    if not config_path.is_file():
        msg = f"safetensors_v1 checkpoint missing {SAFE_CONFIG_FILENAME}: {path}"
        raise ValueError(msg)
    if not weights_path.is_file():
        msg = f"safetensors_v1 checkpoint missing {SAFE_WEIGHTS_FILENAME}: {path}"
        raise ValueError(msg)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid {SAFE_META_FILENAME} in safetensors_v1 checkpoint: {path}"
        raise ValueError(msg) from exc
    if not isinstance(meta, dict):
        msg = f"{SAFE_META_FILENAME} must be a JSON object: {path}"
        raise ValueError(msg)
    container = meta.get("container")
    if container != SAFE_CONTAINER:
        msg = (
            f"Unsupported safetensors container {container!r} in {meta_path}; "
            f"expected {SAFE_CONTAINER!r}"
        )
        raise ValueError(msg)
    format_version = meta.get("format_version", FORMAT_VERSION)
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_FORMAT_VERSIONS)
        )
        msg = (
            f"Unsupported checkpoint format_version {format_version!r}; "
            f"supported versions: {supported}"
        )
        raise ValueError(msg)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid {SAFE_CONFIG_FILENAME} in safetensors_v1 checkpoint: {path}"
        raise ValueError(msg) from exc
    if not isinstance(config, dict):
        msg = f"{SAFE_CONFIG_FILENAME} must be a JSON object: {path}"
        raise ValueError(msg)

    state_dict = safetensors_load_file(
        weights_path,
        device=_resolve_safetensors_device(map_location),
    )
    return config, state_dict, int(format_version)


def _load_safetensors_v1_zip(
    path: Path,
    *,
    map_location: str | torch.device | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    """Load config and weights from a ``safetensors_v1`` zip / ``.kgckpt``.

    Extracts the three root members into a temporary directory and reuses the
    directory loader validation path. Only exact member names are extracted
    (no nested paths).

    Parameters
    ----------
    path : Path
        Zip archive containing ``meta.json``, ``config.json``, and
        ``model.safetensors`` at the archive root.
    map_location : str, torch.device, or None
        Device for weight tensors.

    Returns
    -------
    config : dict
        Architecture configuration.
    state_dict : dict of str to Tensor
        Model weights.
    format_version : int
        Logical checkpoint schema version from ``meta.json``.

    Raises
    ------
    ValueError
        If members are missing or the archive is not a valid zip.
    """
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            names = set(archive.namelist())
            missing = sorted(SAFE_ZIP_MEMBER_NAMES - names)
            if missing:
                msg = f"safetensors_v1 zip missing members {missing} in {path}"
                raise ValueError(msg)
            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for member in SAFE_ZIP_MEMBER_NAMES:
                    archive.extract(member, path=tmp_path)
                return _load_safetensors_v1_directory(
                    tmp_path,
                    map_location=map_location,
                )
    except zipfile.BadZipFile as exc:
        msg = f"Invalid safetensors_v1 zip checkpoint: {path}"
        raise ValueError(msg) from exc


def _load_legacy_pt_file(
    path: Path,
    *,
    map_location: str | torch.device | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    """Load config and weights from a legacy pickle ``.pt`` checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    map_location : str, torch.device, or None
        Device mapping forwarded to :func:`torch.load`.

    Returns
    -------
    config : dict
        Architecture configuration.
    state_dict : dict of str to Tensor
        Model weights.
    format_version : int
        Logical checkpoint schema version.

    Raises
    ------
    ValueError
        If the payload shape or format version is invalid.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        msg = "Checkpoint must be a dictionary payload"
        raise ValueError(msg)

    format_version = payload.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_FORMAT_VERSIONS)
        )
        msg = (
            f"Unsupported checkpoint format_version {format_version!r}; "
            f"supported versions: {supported}"
        )
        raise ValueError(msg)

    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        msg = "Checkpoint must contain 'config' and 'state_dict' dictionaries"
        raise ValueError(msg)
    return config, state_dict, int(format_version)


def _assemble_model_from_checkpoint(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    format_version: int,
    physics_lifting_fn: PhysicsLiftingFn | None,
) -> ModeShapeModel:
    """Reconstruct a model and load weights from parsed checkpoint pieces.

    Parameters
    ----------
    config : dict
        Checkpoint config mapping (possibly pre-migration).
    state_dict : dict of str to Tensor
        Weight tensors to load.
    format_version : int
        On-disk format version used for config migration.
    physics_lifting_fn : callable or None
        Optional custom physics lifting for hybrid reconstructions.

    Returns
    -------
    ModeShapeModel
        Reconstructed model in evaluation mode with loaded weights.
    """
    migrated_config = _migrate_config(config, format_version=format_version)
    if migrated_config.get("koopman_kind") == "hetero_graph":
        _validate_hetero_latent_dims_vs_state(migrated_config, state_dict)
    model = reconstruct_model(migrated_config, physics_lifting_fn=physics_lifting_fn)
    _allocate_adaptive_topology_from_state(model, state_dict)
    # Re-validate entity_ids once adaptive topology binds N_max from state.
    _apply_churn_contract_from_config(model, migrated_config)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> ModeShapeModel:
    """Load a trained model from a checkpoint file or directory.

    Auto-detects on-disk containers:

    1. **Directory** with ``meta.json`` → ``safetensors_v1`` directory
       (JSON config + safetensors weights; no pickle).
    2. **Zip** whose root members include ``meta.json``, ``config.json``,
       and ``model.safetensors`` → ``safetensors_v1`` zip / ``.kgckpt``
       (same three files; no pickle). Matched by member names so safe
       markers win over legacy pickle even when the file is a zip archive.
    3. **Other files** (typically ``.pt`` / ``.pth``) → legacy
       ``torch.load`` pickle payload (trusted-source trust boundary; see
       ``SECURITY.md``).

    Parameters
    ----------
    path : str or Path
        Checkpoint ``.pt`` file, ``safetensors_v1`` directory, or
        ``.kgckpt`` / ``.zip`` bundle produced by :func:`save_checkpoint`.
    map_location : str, torch.device, or None, optional
        Device mapping. For legacy ``.pt`` files, forwarded to
        :func:`torch.load`. For ``safetensors_v1``, normalized to a device
        string (``None`` → ``"cpu"``); callable ``map_location`` values are
        not supported on that path.
    physics_lifting_fn : callable or None, optional
        Custom physics lifting function for hybrid checkpoints without a stored
        preset.

    Returns
    -------
    GraphKoopmanModel
        Reconstructed model with restored weights in evaluation mode.

    Raises
    ------
    ValueError
        If the checkpoint layout, container, format version, or payload is
        invalid.
    FileNotFoundError
        If ``path`` does not exist.
    """
    destination = Path(path)
    if not destination.exists():
        msg = f"Checkpoint path not found: {destination}"
        raise FileNotFoundError(msg)

    if destination.is_dir():
        config, state_dict, format_version = _load_safetensors_v1_directory(
            destination,
            map_location=map_location,
        )
    elif _is_safetensors_v1_zip(destination):
        config, state_dict, format_version = _load_safetensors_v1_zip(
            destination,
            map_location=map_location,
        )
    elif destination.is_file():
        if destination.suffix.lower() in SAFE_BUNDLE_SUFFIXES:
            msg = (
                f"Path {destination} has a safetensors_v1 bundle suffix but "
                f"is not a zip with {sorted(SAFE_ZIP_MEMBER_NAMES)}"
            )
            raise ValueError(msg)
        config, state_dict, format_version = _load_legacy_pt_file(
            destination,
            map_location=map_location,
        )
    else:
        msg = f"Unsupported checkpoint path type: {destination}"
        raise ValueError(msg)

    return _assemble_model_from_checkpoint(
        config,
        state_dict,
        format_version=format_version,
        physics_lifting_fn=physics_lifting_fn,
    )


def snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of a module's ``state_dict`` for checkpointing.

    Parameters
    ----------
    module : nn.Module
        Module whose parameters will be copied.

    Returns
    -------
    dict
        Deep copy of :meth:`nn.Module.state_dict` with detached tensors.
    """
    state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    return deepcopy(state)
