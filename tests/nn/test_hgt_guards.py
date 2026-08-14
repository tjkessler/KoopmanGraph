"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.nn.heterogeneous import (
    HGTDecoder,
    _lookup_typed_edge_index,
    _resolve_hgt_activation,
    _stack_typed_latents,
    resolve_typed_relation_inputs,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    """Build a one-type, two-relation multiplex snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot with one cross relation."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def test_hgt_decoder_and_container_error_paths() -> None:
    """HGT decoder and hetero sequence accessors reject invalid inputs."""
    decoder = HGTDecoder(
        latent_dim=2,
        hidden_channels=4,
        out_channels={"a": 2, "b": 2},
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
        heads=1,
    )
    snap = HeteroData()
    snap["a"].x = torch.randn(2, 2)
    snap["b"].x = torch.randn(2, 2)
    snap["a", "r0", "b"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="does not match HeteroData counts"):
        decoder(z, snap, num_nodes_dict={"a": 9, "b": 2})
    edge_map = {("a", "r0", "b"): snap["a", "r0", "b"].edge_index}
    with pytest.raises(ValueError, match="num_nodes_dict is required"):
        decoder(z, edge_map, num_nodes_dict=None)
    with pytest.raises(ValueError, match="z must have shape"):
        decoder(torch.randn(4, 3), snap)
    with pytest.raises(ValueError, match="num_nodes_dict sums"):
        decoder(torch.randn(3, 2), snap)

    multi = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    with pytest.raises(KeyError, match="unknown node type"):
        multi.num_nodes_of("ghost")
    typed_seq = HeteroGraphSnapshotSequence([_typed_snapshot() for _ in range(2)])
    with pytest.raises(ValueError, match="num_nodes is defined only"):
        _ = typed_seq.num_nodes
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        multi.observation_mask_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        multi.pair_observation_mask(0)
    masked = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        observation_masks={"node": torch.ones(2, 4, dtype=torch.bool)},
    )
    with pytest.raises(IndexError, match="0 <= index"):
        masked.observation_mask_at(5)
    with pytest.raises(IndexError, match="0 <= index"):
        masked.pair_observation_mask(5)
    # Per-type control dict path on control_dim.
    ctrl = HeteroGraphSnapshotSequence(
        [_typed_snapshot() for _ in range(2)],
        control_inputs={
            "gen": torch.randn(2, 1),
            "load": torch.randn(2, 1),
        },
    )
    assert ctrl.control_dim == 1
    assert ctrl.has_controls
    assert len(ctrl.snapshots) == 2


def test_hgt_activation_lookup_and_typed_edge_helpers() -> None:
    """Hit HGT activation branches and typed edge-index lookup fallbacks."""
    assert isinstance(_resolve_hgt_activation("sigmoid"), torch.nn.Sigmoid)
    assert isinstance(_resolve_hgt_activation("tanh"), torch.nn.Tanh)

    edge = torch.tensor([[0], [0]], dtype=torch.long)
    assert torch.equal(
        _lookup_typed_edge_index(
            {("a", "r0", "b"): edge},
            ("a", "r0", "b"),
            expected_keys=(("a", "r0", "b"),),
        ),
        edge,
    )

    # Iterable non-tuple keys coerce via the fallback loop.
    class _Triple:
        def __iter__(self):
            return iter(("a", "r0", "b"))

        def __hash__(self) -> int:
            return hash(("a", "r0", "b"))

        def __eq__(self, other: object) -> bool:
            return False

    assert torch.equal(
        _lookup_typed_edge_index(
            {_Triple(): edge},
            ("a", "r0", "b"),
            expected_keys=(("a", "r0", "b"),),
        ),
        edge,
    )
    with pytest.raises(ValueError, match="missing edge type"):
        _lookup_typed_edge_index(
            {("a", "other", "b"): edge, 42: edge},
            ("a", "r0", "b"),
            expected_keys=(("a", "r0", "b"),),
        )
    with pytest.raises(ValueError, match="produced no embedding"):
        _stack_typed_latents({"a": torch.randn(2, 2), "b": None}, ("a", "b"))

    # Mapping topology path on HGTDecoder (num_nodes_dict present).
    decoder = HGTDecoder(
        latent_dim=2,
        hidden_channels=4,
        out_channels={"a": 2, "b": 2},
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
        heads=1,
    )
    edge_map = {
        ("a", "r0", "b"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    }
    # Mapping + num_nodes_dict reaches the typed HGT bank path; HGTConv may
    # omit source-type embeddings when only one directed relation is present.
    with pytest.raises(ValueError, match="produced no embedding"):
        decoder(
            torch.randn(4, 2),
            edge_map,
            num_nodes_dict={"a": 2, "b": 2},
        )

    # Typed resolve with mapping + None edge_weight bank ordering.
    feats = {"gen": torch.randn(2, 3), "load": torch.randn(3, 2)}
    banks = {
        ("gen", "feeds", "load"): torch.tensor([[0], [2]], dtype=torch.long),
    }
    _features, edges, weights, counts = resolve_typed_relation_inputs(
        feats,
        banks,
        None,
        num_relations=1,
        node_types=("gen", "load"),
        edge_types=(("gen", "feeds", "load"),),
    )
    assert counts == {"gen": 2, "load": 3}
    assert weights == [None]
    with pytest.raises(ValueError, match="Expected 2 edge types"):
        resolve_typed_relation_inputs(
            _typed_snapshot(),
            num_relations=2,
            node_types=("gen", "load"),
            edge_types=None,
        )
    with pytest.raises(TypeError, match="expect HeteroData or a mapping"):
        resolve_typed_relation_inputs(
            torch.randn(2, 3),
            num_relations=1,
            node_types=("gen", "load"),
            edge_types=(("gen", "feeds", "load"),),
        )
    with pytest.raises(ValueError, match="missing node type"):
        resolve_typed_relation_inputs(
            {"gen": torch.randn(2, 3)},
            banks,
            num_relations=1,
            node_types=("gen", "load"),
            edge_types=(("gen", "feeds", "load"),),
        )
    with pytest.raises(ValueError, match="must have shape"):
        resolve_typed_relation_inputs(
            {"gen": torch.randn(2), "load": torch.randn(3, 2)},
            banks,
            num_relations=1,
            node_types=("gen", "load"),
            edge_types=(("gen", "feeds", "load"),),
        )
