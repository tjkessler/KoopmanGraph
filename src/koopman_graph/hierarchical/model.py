"""Hierarchical GraphKoopman wrapper: pool → coarse Koopman → unpool.

Composes :class:`~koopman_graph.model.GraphKoopmanModel` without subclassing
or forking the shared latent rollout loop. Forecasting at the coarsest level
is faster on large graphs but can lose fine-scale accuracy — this is
**coarse-level forecasting with learned unpooling**, not P-K-GCN-style
physics-augmented spatiotemporal super-resolution.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.hierarchical.pooling import (
    PoolingKind,
    PoolSchedule,
    PoolStep,
    ScatterUnpool,
    apply_pool_layer,
    build_pool_layer,
    induce_relation_banks,
    induce_typed_relation_banks,
    multiplex_snapshot_from_features,
    pool_control,
    pool_control_sequence,
    pool_features_with_steps,
    pool_multiplex_features_with_steps,
    pool_typed_features_with_steps,
    resolve_snapshot_inputs,
    snapshot_from_features,
    typed_snapshot_from_features,
    union_relation_edge_index,
    within_type_scoring_edge_index,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.training import FitHistory

SnapshotLike = Data | HeteroData
ReferenceSequence = GraphSnapshotSequence | HeteroGraphSnapshotSequence

ResolutionArg = Literal["fine", "coarse"] | int

_MANIFEST_NAME = "hierarchical_manifest.json"
_MODEL_NAME = "model.pt"
_WRAPPER_NAME = "wrapper.pt"


def _encoder_in_channels(model: GraphKoopmanModel) -> int:
    """Infer scoring feature width from a homogeneous / multiplex encoder.

    Parameters
    ----------
    model : GraphKoopmanModel
        Composed model with integer ``encoder.in_channels``.

    Returns
    -------
    int
        Positive encoder input width.

    Raises
    ------
    ValueError
        If ``encoder.in_channels`` is missing or not a positive int.
    """
    encoder = model.encoder
    in_channels = getattr(encoder, "in_channels", None)
    if isinstance(in_channels, int) and in_channels > 0:
        return in_channels
    msg = (
        "could not infer encoder.in_channels for hierarchical pooling; "
        "pass in_channels= explicitly"
    )
    raise ValueError(msg)


def _encoder_out_channels(model: GraphKoopmanModel) -> int:
    """Return decoded feature width for homogeneous / multiplex unpooling.

    Parameters
    ----------
    model : GraphKoopmanModel
        Composed model with integer ``decoder.out_channels``.

    Returns
    -------
    int
        Positive decoder output width.

    Raises
    ------
    ValueError
        If ``decoder.out_channels`` is missing or not a positive int.
    """
    decoder = model.decoder
    out_channels = getattr(decoder, "out_channels", None)
    if isinstance(out_channels, int) and out_channels > 0:
        return out_channels
    msg = "could not infer decoder.out_channels for hierarchical unpooling"
    raise ValueError(msg)


def _channels_by_type(mapping: object, *, role: str) -> dict[str, int]:
    """Normalize a typed channel mapping to ``{type: positive int}``.

    Parameters
    ----------
    mapping : object
        Encoder / decoder channel mapping.
    role : str
        Label for error messages (``"encoder.in_channels"`` / …).

    Returns
    -------
    dict[str, int]
        Per-type positive widths.

    Raises
    ------
    ValueError
        If ``mapping`` is not a non-empty dict of positive ints.
    """
    if not isinstance(mapping, dict) or not mapping:
        msg = f"could not infer typed {role} for hierarchical pooling"
        raise ValueError(msg)
    out: dict[str, int] = {}
    for key, value in mapping.items():
        width = int(value)
        if width <= 0:
            msg = f"typed {role}[{key!r}] must be positive, got {width}"
            raise ValueError(msg)
        out[str(key)] = width
    return out


class HierarchicalGraphKoopmanModel(nn.Module):
    """Pool a fine graph, advance with a composed GraphKoopman model, unpool.

    Power-user type under :mod:`koopman_graph.hierarchical` (not on the root
    façade). Spectrum and graph-operator regularization use the **pooled**
    topology. Global controls pass through; per-node controls are indexed by
    the pooling ``perm`` chain so additive and bilinear semantics on the
    coarse latent match the composed operator.

    Parameters
    ----------
    model : GraphKoopmanModel
        Composed fine→latent→decode model used on the coarsest graph.
    pool_ratios : sequence of float, optional
        Per-level retention ratios in ``(0, 1]``. Default ``(0.5,)``.
        ``(1.0,)`` keeps all nodes (no-op size; may reorder).
    pooling : {"topk", "sag"}, optional
        PyG pooling operator. Default ``"topk"``.
    in_channels : int or None, optional
        Scoring feature width. Defaults to ``model.encoder.in_channels``.
    refine_unpool : bool, optional
        Identity-initialized linear refine after scatter-unpool. Default
        ``True``.
    pool_schedule : {"per_snapshot", "hold_perm"}, optional
        How :meth:`fit` pools the training sequence. ``"per_snapshot"``
        (default) re-runs TopK/SAG on every timestep so feature-dependent
        scores stay patch-compatible with 0.7.0. ``"hold_perm"`` pools
        ``sequence[0]`` once and reuses that perm / coarse topology for all
        timesteps (faster on static graphs; scores are not recomputed).

    Notes
    -----
    Multiplex hetero models (``koopman="hetero_graph"``, single node type)
    score TopK/SAG on the union of relation banks and induce each coarse
    relation with the same ``perm``. Typed multi-node models pool **per
    node type** on within-type banks and induce all relation banks
    (including cross-type) with paired source/destination perms.

    Hetero ``fit`` (multiplex or typed) requires ``pool_schedule="hold_perm"``
    because feature-dependent ``per_snapshot`` coarse banks are dynamic
    topology, which hetero ``GraphKoopmanModel.fit`` does not support yet.
    ``pool_down`` / ``predict`` accept either schedule.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        pool_ratios: Sequence[float] = (0.5,),
        pooling: PoolingKind = "topk",
        in_channels: int | None = None,
        refine_unpool: bool = True,
        pool_schedule: PoolSchedule = "per_snapshot",
    ) -> None:
        """Store the composed model and build pool / unpool modules.

        Parameters
        ----------

        model : GraphKoopmanModel
            See the function signature / summary for ``model``.
        pool_ratios : Sequence[float]
            See the function signature / summary for ``pool_ratios``.
        pooling : PoolingKind
            See the function signature / summary for ``pooling``.
        in_channels : int | None
            See the function signature / summary for ``in_channels``.
        refine_unpool : bool
            See the function signature / summary for ``refine_unpool``.
        pool_schedule : PoolSchedule
            See the function signature / summary for ``pool_schedule``.

        Notes
        -----

        Constructor parameters are documented on the class."""
        super().__init__()
        uses_hetero = bool(getattr(model, "uses_hetero_koopman", False))
        uses_typed = uses_hetero and bool(getattr(model, "uses_typed_hetero", False))
        if len(pool_ratios) == 0:
            msg = "pool_ratios must contain at least one ratio"
            raise ValueError(msg)
        if pool_schedule not in ("per_snapshot", "hold_perm"):
            msg = (
                "pool_schedule must be 'per_snapshot' or 'hold_perm', "
                f"got {pool_schedule!r}"
            )
            raise ValueError(msg)
        self.model = model
        self._uses_hetero = uses_hetero
        self._uses_typed = uses_typed
        self._node_types: tuple[str, ...] = ()
        if uses_hetero:
            koopman = model.koopman
            node_types = tuple(str(t) for t in koopman.node_types)
            self._edge_types = tuple(
                (str(src), str(rel), str(dst)) for src, rel, dst in koopman.edge_types
            )
            if uses_typed:
                if len(node_types) < 2:
                    msg = (
                        "typed hierarchical pooling requires at least two node "
                        f"types; got {node_types!r}"
                    )
                    raise TypeError(msg)
                self._node_types = node_types
                self._node_type = ""
            else:
                if len(node_types) != 1:
                    msg = (
                        "multiplex hierarchical pooling requires exactly one "
                        f"node type; got {node_types!r}"
                    )
                    raise TypeError(msg)
                self._node_type = node_types[0]
                self._node_types = node_types
        else:
            self._node_type = ""
            self._edge_types = ()
        self.pool_ratios = tuple(float(r) for r in pool_ratios)
        self.pooling: PoolingKind = pooling
        self.pool_schedule: PoolSchedule = pool_schedule

        if uses_typed:
            if in_channels is not None:
                msg = (
                    "typed hierarchical pooling infers per-type channels from "
                    "the encoder; do not pass in_channels="
                )
                raise ValueError(msg)
            in_by_type = _channels_by_type(
                getattr(model.encoder, "in_channels", None),
                role="encoder.in_channels",
            )
            out_by_type = _channels_by_type(
                getattr(model.decoder, "out_channels", None),
                role="decoder.out_channels",
            )
            missing_in = set(self._node_types) - set(in_by_type)
            missing_out = set(self._node_types) - set(out_by_type)
            if missing_in or missing_out:
                msg = (
                    "typed encoder/decoder channels must cover all node types; "
                    f"missing in={sorted(missing_in)} out={sorted(missing_out)}"
                )
                raise ValueError(msg)
            self.in_channels_by_type = {
                node_type: in_by_type[node_type] for node_type in self._node_types
            }
            self.out_channels_by_type = {
                node_type: out_by_type[node_type] for node_type in self._node_types
            }
            # Scalar fields unused for typed; kept for wrapper payload shape.
            self.in_channels = -1
            self.out_channels = -1
            self.pool_layers = nn.ModuleDict(
                {
                    node_type: nn.ModuleList(
                        [
                            build_pool_layer(
                                self.in_channels_by_type[node_type],
                                ratio,
                                pooling=pooling,
                            )
                            for ratio in self.pool_ratios
                        ]
                    )
                    for node_type in self._node_types
                }
            )
            self.unpool_layers = nn.ModuleDict(
                {
                    node_type: nn.ModuleList(
                        [
                            ScatterUnpool(
                                self.out_channels_by_type[node_type],
                                refine=refine_unpool,
                            )
                            for _ in self.pool_ratios
                        ]
                    )
                    for node_type in self._node_types
                }
            )
        else:
            self.in_channels_by_type = {}
            self.out_channels_by_type = {}
            channels = (
                _encoder_in_channels(model) if in_channels is None else in_channels
            )
            self.in_channels = channels
            out_channels = _encoder_out_channels(model)
            self.out_channels = out_channels
            self.pool_layers = nn.ModuleList(
                [
                    build_pool_layer(channels, ratio, pooling=pooling)
                    for ratio in self.pool_ratios
                ]
            )
            self.unpool_layers = nn.ModuleList(
                [
                    ScatterUnpool(out_channels, refine=refine_unpool)
                    for _ in self.pool_ratios
                ]
            )

    @property
    def n_levels(self) -> int:
        """Return the number of pooling levels.

        Returns
        -------
        int
            Number of configured pooling ratios.
        """
        return len(self.pool_ratios)

    @property
    def control_dim(self) -> int:
        """Return the exogenous control dimension of the composed model.

        Returns
        -------
        int
            Control feature width.
        """
        return self.model.control_dim

    def pool_down(
        self,
        graph: Tensor | SnapshotLike,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> tuple[SnapshotLike, list[PoolStep]]:
        """Apply all pooling levels fine → coarse.

        Parameters
        ----------
        graph : Tensor, Data, or HeteroData
            Fine snapshot or node features. ``HeteroData`` is required when
            the composed model is hetero (multiplex or typed).
        edge_index, edge_weight
            Required when ``graph`` is a tensor (homogeneous only).

        Returns
        -------
        tuple
            Coarse ``Data`` or multiplex ``HeteroData`` and per-level
            :class:`PoolStep` metadata (fine→coarse order).
        """
        if self._uses_hetero:
            if not isinstance(graph, HeteroData):
                msg = (
                    "hetero hierarchical pool_down requires HeteroData; "
                    f"got {type(graph).__name__}"
                )
                raise TypeError(msg)
            if self._uses_typed:
                return self._pool_down_typed(graph)
            return self._pool_down_multiplex(graph)
        if isinstance(graph, HeteroData):
            msg = (
                "homogeneous HierarchicalGraphKoopmanModel cannot pool "
                "HeteroData snapshots"
            )
            raise TypeError(msg)
        assert isinstance(self.pool_layers, nn.ModuleList)
        x, edge_index, edge_weight = resolve_snapshot_inputs(
            graph, edge_index, edge_weight
        )
        steps: list[PoolStep] = []
        for layer in self.pool_layers:
            num_fine = x.size(0)
            x, edge_index, edge_weight, perm = apply_pool_layer(
                layer, x, edge_index, edge_weight
            )
            steps.append(
                PoolStep(
                    perm=perm,
                    num_fine=num_fine,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            )
        return snapshot_from_features(x, edge_index, edge_weight), steps

    def _pool_down_multiplex(
        self,
        graph: HeteroData,
    ) -> tuple[HeteroData, list[PoolStep]]:
        """Pool a multiplex hetero snapshot on the union of relation banks.

        Parameters
        ----------
        graph : HeteroData
            Multiplex fine snapshot.

        Returns
        -------
        tuple
            Coarse multiplex ``HeteroData`` and :class:`PoolStep` metadata.
        """
        if len(tuple(graph.node_types)) != 1:
            msg = (
                "multiplex hierarchical pooling requires exactly one node "
                f"type; got {tuple(graph.node_types)!r}"
            )
            raise ValueError(msg)
        present = {tuple(edge_type) for edge_type in graph.edge_types}
        edge_indices: list[Tensor] = []
        edge_weights: list[Tensor | None] = []
        for triple in self._edge_types:
            if triple not in present:
                msg = (
                    f"HeteroData snapshot is missing edge type {triple!r}; "
                    f"present edge types are {sorted(present)!r}"
                )
                raise ValueError(msg)
            edge_indices.append(graph[triple].edge_index)
            edge_weights.append(graph[triple].get("edge_weight", None))

        x = graph[self._node_type].x
        if x is None:
            msg = f"HeteroData node type {self._node_type!r} is missing features x"
            raise ValueError(msg)
        union_index = union_relation_edge_index(edge_indices)
        union_weight: Tensor | None = None
        steps: list[PoolStep] = []
        for layer in self.pool_layers:
            num_fine = x.size(0)
            x, union_index, union_weight, perm = apply_pool_layer(
                layer, x, union_index, union_weight
            )
            rel_indices, rel_weights = induce_relation_banks(
                perm,
                num_fine,
                edge_indices,
                edge_weights,
            )
            edge_indices = list(rel_indices)
            edge_weights = list(rel_weights)
            steps.append(
                PoolStep(
                    perm=perm,
                    num_fine=num_fine,
                    edge_index=union_index,
                    edge_weight=union_weight,
                    relation_edge_indices=rel_indices,
                    relation_edge_weights=rel_weights,
                )
            )
        coarse = multiplex_snapshot_from_features(
            x,
            node_type=self._node_type,
            edge_types=self._edge_types,
            relation_edge_indices=edge_indices,
            relation_edge_weights=edge_weights,
        )
        return coarse, steps

    def _pool_down_typed(
        self,
        graph: HeteroData,
    ) -> tuple[HeteroData, list[PoolStep]]:
        """Pool a typed hetero snapshot with per-type TopK/SAG permutations.

        Parameters
        ----------
        graph : HeteroData
            Typed fine snapshot.

        Returns
        -------
        tuple
            Coarse typed ``HeteroData`` and :class:`PoolStep` metadata.
        """
        present_nodes = {str(t) for t in graph.node_types}
        expected_nodes = set(self._node_types)
        if present_nodes != expected_nodes:
            msg = (
                "typed hierarchical pooling requires node types "
                f"{self._node_types!r}; snapshot has {sorted(present_nodes)!r}"
            )
            raise ValueError(msg)
        present_edges = {tuple(edge_type) for edge_type in graph.edge_types}
        edge_indices: list[Tensor] = []
        edge_weights: list[Tensor | None] = []
        for triple in self._edge_types:
            if triple not in present_edges:
                msg = (
                    f"HeteroData snapshot is missing edge type {triple!r}; "
                    f"present edge types are {sorted(present_edges)!r}"
                )
                raise ValueError(msg)
            edge_indices.append(graph[triple].edge_index)
            edge_weights.append(graph[triple].get("edge_weight", None))

        features: dict[str, Tensor] = {}
        for node_type in self._node_types:
            x = graph[node_type].x
            if x is None:
                msg = f"HeteroData node type {node_type!r} is missing features x"
                raise ValueError(msg)
            features[node_type] = x

        assert isinstance(self.pool_layers, nn.ModuleDict)
        steps: list[PoolStep] = []
        for level in range(self.n_levels):
            num_fine_by_type = {
                node_type: int(features[node_type].size(0))
                for node_type in self._node_types
            }
            perms_by_type: dict[str, Tensor] = {}
            scoring_edges_first: Tensor | None = None
            for node_type in self._node_types:
                layer = self.pool_layers[node_type][level]
                scoring = within_type_scoring_edge_index(
                    self._edge_types,
                    edge_indices,
                    node_type=node_type,
                )
                if scoring_edges_first is None:
                    scoring_edges_first = scoring
                x_c, _edge_c, _w_c, perm = apply_pool_layer(
                    layer,
                    features[node_type],
                    scoring,
                    None,
                )
                features[node_type] = x_c
                perms_by_type[node_type] = perm
            rel_indices, rel_weights = induce_typed_relation_banks(
                self._edge_types,
                edge_indices,
                edge_weights,
                perms_by_type=perms_by_type,
                num_fine_by_type=num_fine_by_type,
            )
            edge_indices = list(rel_indices)
            edge_weights = list(rel_weights)
            first_type = self._node_types[0]
            steps.append(
                PoolStep(
                    perm=perms_by_type[first_type],
                    num_fine=num_fine_by_type[first_type],
                    edge_index=(
                        scoring_edges_first
                        if scoring_edges_first is not None
                        else features[first_type].new_zeros(2, 0, dtype=torch.long)
                    ),
                    edge_weight=None,
                    relation_edge_indices=rel_indices,
                    relation_edge_weights=rel_weights,
                    typed_node_types=self._node_types,
                    typed_perms=tuple(
                        perms_by_type[node_type] for node_type in self._node_types
                    ),
                    typed_num_fine=tuple(
                        num_fine_by_type[node_type] for node_type in self._node_types
                    ),
                )
            )
        coarse = typed_snapshot_from_features(
            features,
            node_types=self._node_types,
            edge_types=self._edge_types,
            relation_edge_indices=edge_indices,
            relation_edge_weights=edge_weights,
        )
        return coarse, steps

    def unpool_up(
        self,
        coarse_x: Tensor | dict[str, Tensor],
        steps: Sequence[PoolStep],
        *,
        levels: int | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Unpool coarse features toward fine resolution.

        Parameters
        ----------
        coarse_x : Tensor or dict[str, Tensor]
            Features at the coarsest level (or an intermediate start).
            Typed hetero models require a ``{node_type: features}`` mapping.
        steps : sequence of PoolStep
            Pool metadata in fine→coarse order (same as :meth:`pool_down`).
        levels : int or None, optional
            Number of unpool steps from the coarse end. ``None`` fully restores
            the finest node count.

        Returns
        -------
        Tensor or dict[str, Tensor]
            Features after the requested unpool steps (dict when typed).
        """
        if len(steps) != self.n_levels:
            msg = f"expected {self.n_levels} pool steps, got {len(steps)}"
            raise ValueError(msg)
        n_unpool = self.n_levels if levels is None else levels
        if n_unpool < 0 or n_unpool > self.n_levels:
            msg = f"levels must be in [0, {self.n_levels}], got {n_unpool}"
            raise ValueError(msg)
        if self._uses_typed:
            if not isinstance(coarse_x, dict):
                msg = (
                    "typed hierarchical unpool_up requires a "
                    "{node_type: features} mapping"
                )
                raise TypeError(msg)
            return self._unpool_up_typed(coarse_x, steps, levels=n_unpool)
        if isinstance(coarse_x, dict):
            msg = (
                "homogeneous/multiplex hierarchical unpool_up expects a "
                "feature Tensor, not a dict"
            )
            raise TypeError(msg)
        assert isinstance(self.unpool_layers, nn.ModuleList)
        x = coarse_x
        # Unpool reverse: last pool step first.
        for offset in range(n_unpool):
            step = steps[-(offset + 1)]
            unpool = self.unpool_layers[-(offset + 1)]
            x = unpool(x, step.perm, step.num_fine)
        return x

    def _unpool_up_typed(
        self,
        coarse_by_type: dict[str, Tensor],
        steps: Sequence[PoolStep],
        *,
        levels: int,
    ) -> dict[str, Tensor]:
        """Unpool typed coarse features with per-type ScatterUnpool modules.

        Parameters
        ----------
        coarse_by_type : dict[str, Tensor]
            Coarse features keyed by node type.
        steps : sequence of PoolStep
            Fine→coarse metadata with typed perm fields.
        levels : int
            Number of unpool steps from the coarse end.

        Returns
        -------
        dict[str, Tensor]
            Features after ``levels`` unpool steps.
        """
        assert isinstance(self.unpool_layers, nn.ModuleDict)
        features = {
            node_type: coarse_by_type[node_type] for node_type in self._node_types
        }
        for offset in range(levels):
            step = steps[-(offset + 1)]
            if (
                step.typed_node_types is None
                or step.typed_perms is None
                or step.typed_num_fine is None
            ):
                msg = "typed unpool requires typed perm fields on PoolStep"
                raise ValueError(msg)
            for node_type, perm, num_fine in zip(
                step.typed_node_types,
                step.typed_perms,
                step.typed_num_fine,
                strict=True,
            ):
                unpool = self.unpool_layers[node_type][-(offset + 1)]
                features[node_type] = unpool(features[node_type], perm, num_fine)
        return features

    def _perms(self, steps: Sequence[PoolStep]) -> list[Tensor]:
        """Extract fine-to-coarse node permutations.

        Parameters
        ----------

        steps : Sequence[PoolStep]
            See the function signature / summary for ``steps``.

        Returns
        -------

        list of Tensor
            One permutation per pooling level."""
        return [step.perm for step in steps]

    def _pool_controls(
        self,
        controls: Sequence[Tensor] | None,
        steps: Sequence[PoolStep],
    ) -> list[Tensor] | None:
        """Pool global or per-node controls through the permutation chain.

        Parameters
        ----------

        controls : Sequence[Tensor] | None
            See the function signature / summary for ``controls``.
        steps : Sequence[PoolStep]
            See the function signature / summary for ``steps``.

        Returns
        -------

        list of Tensor or None
            Coarse controls, or ``None`` when controls are absent."""
        if controls is None:
            return None
        perms = self._perms(steps)
        return [pool_control(control, perms) for control in controls]

    def _resolve_resolution(self, resolution: ResolutionArg) -> int:
        """Return how many unpool steps to apply from the coarse end.

        Parameters
        ----------

        resolution : ResolutionArg
            See the function signature / summary for ``resolution``.

        Returns
        -------

        int
            Number of unpooling levels."""
        if resolution == "coarse":
            return 0
        if resolution == "fine":
            return self.n_levels
        if isinstance(resolution, int):
            if resolution < 0 or resolution > self.n_levels:
                msg = (
                    f"resolution int must be in [0, {self.n_levels}] "
                    f"(0=coarse, {self.n_levels}=fine), got {resolution}"
                )
                raise ValueError(msg)
            return resolution
        msg = f"resolution must be 'fine', 'coarse', or int, got {resolution!r}"
        raise ValueError(msg)

    def spectrum(
        self,
        reference_graph: Tensor | Data,
        *,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        delta_t: float | None = None,
    ) -> Any:
        """Spectrum of the composed operator on the **pooled** topology.

        For ``koopman="graph"``, requires pooling a reference graph so the
        effective matrix uses the coarse ``edge_index`` / node count — never
        report ``K_self`` alone as the networked spectrum.

        Parameters
        ----------

        reference_graph : Tensor | Data
            See the function signature / summary for ``reference_graph``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.
        delta_t : float | None
            See the function signature / summary for ``delta_t``.

        Returns
        -------

        Any
            Spectrum result returned by the composed model."""
        coarse, _steps = self.pool_down(
            reference_graph, edge_index=edge_index, edge_weight=edge_weight
        )
        return self.model.spectrum(
            delta_t=delta_t,
            edge_index=coarse.edge_index,
            num_nodes=int(coarse.x.size(0)),
            edge_weight=getattr(coarse, "edge_weight", None),
        )

    def predict(
        self,
        initial_graph: Tensor | SnapshotLike,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[SnapshotLike] | None = None,
        future_presence: Tensor | None = None,
        history: Sequence[SnapshotLike] | None = None,
        *,
        resolution: ResolutionArg = "fine",
    ) -> list[SnapshotLike]:
        """Pool once, forecast on the coarse graph, optionally unpool.

        Parameters
        ----------
        initial_graph : Tensor, Data, or HeteroData
            Fine initial snapshot.
        steps : int
            Forecast horizon.
        edge_index, edge_weight
            Topology when ``initial_graph`` is a tensor (homogeneous only).
        controls : sequence of Tensor or None, optional
            Fine-level controls (global or per-node). Per-node rows are pooled
            with the initial pooling ``perm`` chain.
        future_topologies : sequence of Data or HeteroData or None, optional
            Fine future topologies; each is pooled with the **same** pool
            layers (scores recomputed) before being forwarded to the composed
            model.
        future_presence : Tensor or None, optional
            Accepted for call-site parity with
            :class:`~koopman_graph.model.GraphKoopmanModel` /
            :func:`~koopman_graph.metrics.evaluate_forecast`. Hierarchical
            pooling does not implement presence-mask hold; the argument must
            be ``None``.
        history : sequence of Data or HeteroData or None, optional
            Delay history; each snapshot is pooled independently.
        resolution : {"fine", "coarse"} or int, optional
            ``"fine"`` / ``n_levels`` fully unpools; ``"coarse"`` / ``0`` returns
            coarse forecasts; intermediate ints unpool that many levels.

        Returns
        -------
        list of Data or HeteroData
            Forecasts at the requested resolution. Fine outputs carry the
            initial fine topology (hold-last at the fine level).

        Raises
        ------
        ValueError
            If ``future_presence`` is not ``None``.
        """
        if future_presence is not None:
            msg = (
                "HierarchicalGraphKoopmanModel does not support future_presence; "
                "presence-mask hold is not defined across pool/unpool"
            )
            raise ValueError(msg)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                if self._uses_hetero:
                    if self._uses_typed:
                        return self._predict_typed(
                            initial_graph,
                            steps,
                            controls=controls,
                            future_topologies=future_topologies,
                            history=history,
                            resolution=resolution,
                        )
                    return self._predict_multiplex(
                        initial_graph,
                        steps,
                        controls=controls,
                        future_topologies=future_topologies,
                        history=history,
                        resolution=resolution,
                    )
                fine_x, fine_edge, fine_weight = resolve_snapshot_inputs(
                    initial_graph,
                    edge_index,
                    edge_weight,  # type: ignore[arg-type]
                )
                fine_template = snapshot_from_features(fine_x, fine_edge, fine_weight)
                coarse, pool_steps = self.pool_down(fine_template)
                coarse_controls = self._pool_controls(controls, pool_steps)

                coarse_future: list[Data] | None = None
                if future_topologies is not None:
                    coarse_future = []
                    for topo in future_topologies:
                        pooled_topo, _ = self.pool_down(topo)
                        assert isinstance(pooled_topo, Data)
                        coarse_future.append(
                            snapshot_from_features(
                                pooled_topo.x.new_zeros(pooled_topo.x.shape),
                                pooled_topo.edge_index,
                                getattr(pooled_topo, "edge_weight", None),
                            )
                        )

                coarse_history: list[Data] | None = None
                if history is not None:
                    coarse_history = []
                    for snap in history:
                        pooled, _ = self.pool_down(snap)
                        assert isinstance(pooled, Data)
                        coarse_history.append(pooled)

                assert isinstance(coarse, Data)
                coarse_preds = self.model.predict(
                    coarse,
                    steps,
                    controls=coarse_controls,
                    future_topologies=coarse_future,
                    history=coarse_history,
                )

                n_unpool = self._resolve_resolution(resolution)
                if n_unpool == 0:
                    return list(coarse_preds)

                output: list[SnapshotLike] = []
                for pred in coarse_preds:
                    fine_feat = self.unpool_up(pred.x, pool_steps, levels=n_unpool)
                    if n_unpool == self.n_levels:
                        output.append(
                            snapshot_from_features(fine_feat, fine_edge, fine_weight)
                        )
                    else:
                        # Intermediate resolution: use topology after
                        # (n_levels - n_unpool) pool steps.
                        stop = self.n_levels - n_unpool
                        mid = pool_steps[stop - 1]
                        output.append(
                            snapshot_from_features(
                                fine_feat, mid.edge_index, mid.edge_weight
                            )
                        )
                return output
        finally:
            self.train(was_training)

    def _predict_multiplex(
        self,
        initial_graph: Tensor | SnapshotLike,
        steps: int,
        *,
        controls: Sequence[Tensor] | None,
        future_topologies: Sequence[SnapshotLike] | None,
        history: Sequence[SnapshotLike] | None,
        resolution: ResolutionArg,
    ) -> list[SnapshotLike]:
        """Pool / forecast / unpool path for multiplex hetero models.

        Parameters
        ----------
        initial_graph
            Value for ``initial_graph``.
        steps
            Value for ``steps``.
        controls
            Value for ``controls``.
        future_topologies
            Value for ``future_topologies``.
        history
            Value for ``history``.
        resolution
            Value for ``resolution``.

        Returns
        -------
        object
            Function result.
        """
        if not isinstance(initial_graph, HeteroData):
            msg = (
                "multiplex hierarchical predict requires a HeteroData origin; "
                f"got {type(initial_graph).__name__}"
            )
            raise TypeError(msg)
        fine_template = initial_graph
        coarse, pool_steps = self.pool_down(fine_template)
        coarse_controls = self._pool_controls(controls, pool_steps)

        coarse_future: list[HeteroData] | None = None
        if future_topologies is not None:
            coarse_future = []
            for topo in future_topologies:
                if not isinstance(topo, HeteroData):
                    msg = "future_topologies must be HeteroData for multiplex hetero"
                    raise TypeError(msg)
                pooled_topo, _ = self.pool_down(topo)
                assert isinstance(pooled_topo, HeteroData)
                zero = pooled_topo.clone()
                zero[self._node_type].x = pooled_topo[self._node_type].x.new_zeros(
                    pooled_topo[self._node_type].x.shape
                )
                coarse_future.append(zero)

        coarse_history: list[HeteroData] | None = None
        if history is not None:
            coarse_history = []
            for snap in history:
                if not isinstance(snap, HeteroData):
                    msg = "history must be HeteroData for multiplex hetero"
                    raise TypeError(msg)
                pooled, _ = self.pool_down(snap)
                assert isinstance(pooled, HeteroData)
                coarse_history.append(pooled)

        assert isinstance(coarse, HeteroData)
        coarse_preds = self.model.predict(
            coarse,
            steps,
            controls=coarse_controls,
            future_topologies=coarse_future,
            history=coarse_history,
        )

        n_unpool = self._resolve_resolution(resolution)
        if n_unpool == 0:
            return list(coarse_preds)

        output: list[SnapshotLike] = []
        for pred in coarse_preds:
            assert isinstance(pred, HeteroData)
            fine_feat = self.unpool_up(
                pred[self._node_type].x,
                pool_steps,
                levels=n_unpool,
            )
            if n_unpool == self.n_levels:
                out = fine_template.clone()
                out[self._node_type].x = fine_feat
                output.append(out)
            else:
                stop = self.n_levels - n_unpool
                mid = pool_steps[stop - 1]
                if mid.relation_edge_indices is None:
                    msg = "intermediate multiplex unpool requires relation banks"
                    raise RuntimeError(msg)
                output.append(
                    multiplex_snapshot_from_features(
                        fine_feat,
                        node_type=self._node_type,
                        edge_types=self._edge_types,
                        relation_edge_indices=mid.relation_edge_indices,
                        relation_edge_weights=mid.relation_edge_weights,
                    )
                )
        return output

    def _predict_typed(
        self,
        initial_graph: Tensor | SnapshotLike,
        steps: int,
        *,
        controls: Sequence[Tensor] | None,
        future_topologies: Sequence[SnapshotLike] | None,
        history: Sequence[SnapshotLike] | None,
        resolution: ResolutionArg,
    ) -> list[SnapshotLike]:
        """Pool / forecast / unpool path for typed hetero models.

        Parameters
        ----------
        initial_graph
            Value for ``initial_graph``.
        steps
            Value for ``steps``.
        controls
            Value for ``controls``.
        future_topologies
            Value for ``future_topologies``.
        history
            Value for ``history``.
        resolution
            Value for ``resolution``.

        Returns
        -------
        list of HeteroData
            Forecasts at the requested resolution.
        """
        if not isinstance(initial_graph, HeteroData):
            msg = (
                "typed hierarchical predict requires a HeteroData origin; "
                f"got {type(initial_graph).__name__}"
            )
            raise TypeError(msg)
        fine_template = initial_graph
        coarse, pool_steps = self.pool_down(fine_template)
        coarse_controls = self._pool_controls(controls, pool_steps)

        coarse_future: list[HeteroData] | None = None
        if future_topologies is not None:
            coarse_future = []
            for topo in future_topologies:
                if not isinstance(topo, HeteroData):
                    msg = "future_topologies must be HeteroData for typed hetero"
                    raise TypeError(msg)
                pooled_topo, _ = self.pool_down(topo)
                assert isinstance(pooled_topo, HeteroData)
                zero = pooled_topo.clone()
                for node_type in self._node_types:
                    zero[node_type].x = pooled_topo[node_type].x.new_zeros(
                        pooled_topo[node_type].x.shape
                    )
                coarse_future.append(zero)

        coarse_history: list[HeteroData] | None = None
        if history is not None:
            coarse_history = []
            for snap in history:
                if not isinstance(snap, HeteroData):
                    msg = "history must be HeteroData for typed hetero"
                    raise TypeError(msg)
                pooled, _ = self.pool_down(snap)
                assert isinstance(pooled, HeteroData)
                coarse_history.append(pooled)

        assert isinstance(coarse, HeteroData)
        coarse_preds = self.model.predict(
            coarse,
            steps,
            controls=coarse_controls,
            future_topologies=coarse_future,
            history=coarse_history,
        )

        n_unpool = self._resolve_resolution(resolution)
        if n_unpool == 0:
            return list(coarse_preds)

        output: list[SnapshotLike] = []
        for pred in coarse_preds:
            assert isinstance(pred, HeteroData)
            coarse_feats = {
                node_type: pred[node_type].x for node_type in self._node_types
            }
            fine_feats = self.unpool_up(coarse_feats, pool_steps, levels=n_unpool)
            assert isinstance(fine_feats, dict)
            if n_unpool == self.n_levels:
                out = fine_template.clone()
                for node_type in self._node_types:
                    out[node_type].x = fine_feats[node_type]
                output.append(out)
            else:
                stop = self.n_levels - n_unpool
                mid = pool_steps[stop - 1]
                if mid.relation_edge_indices is None:
                    msg = "intermediate typed unpool requires relation banks"
                    raise RuntimeError(msg)
                output.append(
                    typed_snapshot_from_features(
                        fine_feats,
                        node_types=self._node_types,
                        edge_types=self._edge_types,
                        relation_edge_indices=mid.relation_edge_indices,
                        relation_edge_weights=mid.relation_edge_weights,
                    )
                )
        return output

    def _snapshot_features(self, snap: SnapshotLike) -> Tensor:
        """Return homogeneous ``x`` or multiplex node features.

        Parameters
        ----------
        snap
            Value for ``snap``.

        Returns
        -------
        Tensor
            Node feature matrix.
        """
        if isinstance(snap, HeteroData):
            features = snap[self._node_type].x
            if features is None:
                msg = (
                    f"HeteroData node type {self._node_type!r} is missing "
                    "feature matrix x"
                )
                raise ValueError(msg)
            return features
        if snap.x is None:
            msg = "hold_perm pooling requires snapshot.x"
            raise ValueError(msg)
        return snap.x

    def _typed_snapshot_features(self, snap: SnapshotLike) -> dict[str, Tensor]:
        """Return per-type feature matrices from a typed hetero snapshot.

        Parameters
        ----------
        snap : Data or HeteroData
            Fine typed snapshot.

        Returns
        -------
        dict[str, Tensor]
            Features keyed by node type.
        """
        if not isinstance(snap, HeteroData):
            msg = "typed hierarchical fit requires HeteroData snapshots"
            raise TypeError(msg)
        features: dict[str, Tensor] = {}
        for node_type in self._node_types:
            x = snap[node_type].x
            if x is None:
                msg = f"HeteroData node type {node_type!r} is missing features x"
                raise ValueError(msg)
            features[node_type] = x
        return features

    def _pool_sequence(
        self,
        sequence: ReferenceSequence,
    ) -> tuple[ReferenceSequence, list[list[PoolStep]]]:
        """Pool every snapshot and retain per-step metadata.

        Under ``pool_schedule="per_snapshot"``, runs :meth:`pool_down` once per
        timestep (feature-dependent TopK/SAG scores). Under ``"hold_perm"``,
        pools ``sequence[0]`` once and applies those perms / coarse edges to
        every timestep via :func:`pool_features_with_steps` (or the multiplex /
        typed peer).

        Parameters
        ----------
        sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
            Fine-resolution training sequence.

        Returns
        -------
        tuple
            Coarse sequence and pooling metadata for every snapshot.
        """
        if self._uses_hetero and not isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = "hetero hierarchical fit requires HeteroGraphSnapshotSequence"
            raise TypeError(msg)
        if not self._uses_hetero and isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = "homogeneous hierarchical fit cannot use HeteroGraphSnapshotSequence"
            raise TypeError(msg)
        if sequence.num_timesteps < 1:
            msg = "sequence must contain at least one snapshot"
            raise ValueError(msg)

        coarse_snaps: list[SnapshotLike] = []
        all_steps: list[list[PoolStep]] = []
        if self.pool_schedule == "hold_perm":
            _, held_steps = self.pool_down(sequence[0])
            for snap in sequence:
                if self._uses_typed:
                    coarse_snaps.append(
                        pool_typed_features_with_steps(
                            self._typed_snapshot_features(snap),
                            held_steps,
                            node_types=self._node_types,
                            edge_types=self._edge_types,
                        )
                    )
                elif self._uses_hetero:
                    coarse_snaps.append(
                        pool_multiplex_features_with_steps(
                            self._snapshot_features(snap),
                            held_steps,
                            node_type=self._node_type,
                            edge_types=self._edge_types,
                        )
                    )
                else:
                    coarse_snaps.append(
                        pool_features_with_steps(
                            self._snapshot_features(snap), held_steps
                        )
                    )
                all_steps.append(held_steps)
            allow_dynamic_topology = False
        else:
            for snap in sequence:
                coarse, steps = self.pool_down(snap)
                coarse_snaps.append(coarse)
                all_steps.append(steps)
            # Feature-dependent TopK/SAG perms can change coarse edges over time.
            allow_dynamic_topology = True

        control_inputs = None
        if sequence.has_controls:
            assert sequence.control_inputs is not None
            perms = [self._perms(steps) for steps in all_steps]
            control_inputs = pool_control_sequence(sequence.control_inputs, perms)

        kwargs: dict[str, Any] = {
            "allow_dynamic_topology": allow_dynamic_topology,
        }
        if sequence.timestamps is not None:
            kwargs["timestamps"] = sequence.timestamps
        # Observation masks are fine-node specific; drop on coarse (documented).
        if self._uses_hetero:
            return (
                HeteroGraphSnapshotSequence(
                    coarse_snaps,  # type: ignore[arg-type]
                    control_inputs=control_inputs,
                    **kwargs,
                ),
                all_steps,
            )
        return (
            GraphSnapshotSequence(
                coarse_snaps,  # type: ignore[arg-type]
                control_inputs=control_inputs,
                **kwargs,
            ),
            all_steps,
        )

    def _fit_unpool(
        self,
        sequence: ReferenceSequence,
        all_steps: Sequence[Sequence[PoolStep]],
        *,
        epochs: int,
        lr: float,
    ) -> None:
        """Train scatter-unpool refine layers to reconstruct fine features.

        Parameters
        ----------

        sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
            See the function signature / summary for ``sequence``.
        all_steps : Sequence[Sequence[PoolStep]]
            See the function signature / summary for ``all_steps``.
        epochs : int
            See the function signature / summary for ``epochs``.
        lr : float
            See the function signature / summary for ``lr``.

        Notes
        -----

        A non-positive epoch count leaves the refine layers unchanged."""
        if epochs <= 0 or len(self.unpool_layers) == 0:
            return
        params = list(self.unpool_layers.parameters())
        if not params:
            return
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            if self._uses_typed:
                feats0 = self._typed_snapshot_features(sequence[0])
                loss = next(iter(feats0.values())).new_zeros(())
                for snap, steps in zip(sequence, all_steps, strict=True):
                    features = self._typed_snapshot_features(snap)
                    coarse = dict(features)
                    for step in steps:
                        if step.typed_node_types is None or step.typed_perms is None:
                            msg = (
                                "typed unpool fit requires typed perm fields "
                                "on PoolStep"
                            )
                            raise ValueError(msg)
                        for node_type, perm in zip(
                            step.typed_node_types, step.typed_perms, strict=True
                        ):
                            coarse[node_type] = coarse[node_type][perm]
                    recon = self.unpool_up(coarse, steps)
                    assert isinstance(recon, dict)
                    for node_type in self._node_types:
                        loss = loss + torch.mean(
                            (recon[node_type] - features[node_type]) ** 2
                        )
            else:
                loss = self._snapshot_features(sequence[0]).new_zeros(())
                for snap, steps in zip(sequence, all_steps, strict=True):
                    # Teacher: pool features without refine, then unpool back.
                    x = self._snapshot_features(snap)
                    # Reconstruct from the last coarse features obtained by
                    # indexing fine features with the perm chain (no score net).
                    coarse_x = x
                    for step in steps:
                        coarse_x = coarse_x[step.perm]
                    recon = self.unpool_up(coarse_x, steps)
                    assert isinstance(recon, Tensor)
                    loss = loss + torch.mean((recon - x) ** 2)
            loss = loss / len(sequence)
            loss.backward()
            opt.step()

    def fit(
        self,
        sequence: ReferenceSequence,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        unpool_epochs: int | None = None,
        unpool_lr: float | None = None,
        **kwargs: Any,
    ) -> FitHistory:
        """Pool the sequence, fit the composed model, then train unpool.

        Pooling scores are held fixed (eval) during the composed ``fit`` so the
        coarse topology stays a consistent reduction for the inner training
        loop. How often :meth:`pool_down` runs is controlled by
        :attr:`pool_schedule` (default ``"per_snapshot"``; see class docs).
        Unpool refine layers are trained afterward to map coarse features
        back toward fine node features.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
            Fine-resolution training sequence (container type must match the
            composed model).
        epochs : int, optional
            Epochs for the composed :meth:`GraphKoopmanModel.fit`.
        lr : float, optional
            Learning rate for the composed model.
        unpool_epochs : int or None, optional
            Epochs for unpool refine training. Defaults to ``max(5, epochs // 5)``.
        unpool_lr : float or None, optional
            Unpool learning rate. Defaults to ``lr``.
        **kwargs
            Forwarded to :meth:`GraphKoopmanModel.fit`.

        Returns
        -------
        FitHistory
            History from the composed model ``fit``.
        """
        if (
            self._uses_hetero
            and self.pool_schedule == "per_snapshot"
            and isinstance(sequence, HeteroGraphSnapshotSequence)
        ):
            msg = (
                "hetero hierarchical fit with pool_schedule='per_snapshot' "
                "produces dynamic coarse relation banks, which hetero "
                "GraphKoopmanModel.fit does not support; use "
                "pool_schedule='hold_perm' (or pool_down/predict under "
                "per_snapshot)"
            )
            raise ValueError(msg)

        was_training = self.training
        self.pool_layers.eval()
        try:
            with torch.no_grad():
                coarse_sequence, all_steps = self._pool_sequence(sequence)
            history = self.model.fit(coarse_sequence, epochs=epochs, lr=lr, **kwargs)
        finally:
            self.train(was_training)

        u_epochs = max(5, epochs // 5) if unpool_epochs is None else unpool_epochs
        u_lr = lr if unpool_lr is None else unpool_lr
        self._fit_unpool(sequence, all_steps, epochs=u_epochs, lr=u_lr)
        return history

    def save(self, directory: str | Path) -> None:
        """Persist composed model (format-1) plus wrapper weights.

        Parameters
        ----------
        directory : str or Path
            Destination directory (created when missing).
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.model.save(root / _MODEL_NAME)
        torch.save(
            {
                "pool_ratios": list(self.pool_ratios),
                "pooling": self.pooling,
                "pool_schedule": self.pool_schedule,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "pool_state_dict": self.pool_layers.state_dict(),
                "unpool_state_dict": self.unpool_layers.state_dict(),
            },
            root / _WRAPPER_NAME,
        )
        manifest: dict[str, Any] = {
            "kind": "HierarchicalGraphKoopmanModel",
            "model_file": _MODEL_NAME,
            "wrapper_file": _WRAPPER_NAME,
            "member_format": "GraphKoopmanModel.save",
            "pool_ratios": list(self.pool_ratios),
            "pooling": self.pooling,
            "pool_schedule": self.pool_schedule,
        }
        if self._uses_hetero:
            # Coarse relation schema preserved across pool levels.
            manifest["node_types"] = list(self._node_types)
            manifest["edge_types"] = [list(triple) for triple in self._edge_types]
            manifest["hetero_mode"] = "typed" if self._uses_typed else "multiplex"
        (root / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> HierarchicalGraphKoopmanModel:
        """Load a hierarchical wrapper saved by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Directory with manifest, model checkpoint, and wrapper weights.
        map_location : str, device, or None, optional
            Forwarded to model / wrapper loaders.

        Returns
        -------
        HierarchicalGraphKoopmanModel
            Reconstructed wrapper.
        """
        root = Path(directory)
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.is_file():
            msg = f"hierarchical manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = GraphKoopmanModel.load(
            root / manifest.get("model_file", _MODEL_NAME),
            map_location=map_location,
        )
        wrapper_path = root / manifest.get("wrapper_file", _WRAPPER_NAME)
        payload = torch.load(
            wrapper_path,
            map_location=map_location,
            weights_only=False,
        )
        in_channels = payload["in_channels"]
        # Typed wrappers store sentinel -1; channels are inferred from the model.
        if in_channels is not None and int(in_channels) < 0:
            in_channels = None
        inst = cls(
            model,
            pool_ratios=payload["pool_ratios"],
            pooling=payload["pooling"],
            in_channels=in_channels,
            pool_schedule=payload.get("pool_schedule", "per_snapshot"),
        )
        inst.pool_layers.load_state_dict(payload["pool_state_dict"])
        inst.unpool_layers.load_state_dict(payload["unpool_state_dict"])
        return inst
