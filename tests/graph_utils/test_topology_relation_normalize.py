"""Tests for per-relation R-GCN / random-walk normalization helpers."""

from __future__ import annotations

import ast

import pytest
import torch
from tests.helpers import REPO_ROOT

from koopman_graph.graph_utils import (
    RELATION_NORMALIZATION_MODES,
    dense_random_walk_normalized_adjacency,
    dense_relation_normalized_adjacency,
    random_walk_normalized_adjacency_edge_weights,
    random_walk_normalized_adjacency_matvec,
    relation_degree_normalize,
    relation_normalized_adjacency_matvec,
)


def test_rgcn_in_degree_matches_hand_oracle() -> None:
    """Verify R-GCN in-degree normalize matches hand-built ``D_in^{-1} A``.

    Edges ``0->1``, ``2->1``, ``1->3`` (unit weights). Node 1 has in-degree 2;
    nodes 0 and 2 are isolates (in-degree 0).
    """
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 3]], dtype=torch.long)
    num_nodes = 4
    weights = relation_degree_normalize(
        edge_index,
        num_nodes=num_nodes,
        dtype=torch.float32,
        normalization="rgcn_in_degree",
    )
    # Edges into node 1: 1/2; edge into node 3: 1/1.
    assert weights == pytest.approx([0.5, 0.5, 1.0], abs=1e-6)

    dense = dense_relation_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float32,
        normalization="rgcn_in_degree",
    )
    expected = torch.zeros(4, 4)
    expected[1, 0] = 0.5
    expected[1, 2] = 0.5
    expected[3, 1] = 1.0
    assert torch.allclose(dense, expected, atol=1e-6)

    x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = relation_normalized_adjacency_matvec(
        edge_index,
        x,
        normalization="rgcn_in_degree",
    )
    assert torch.allclose(out, expected @ x, atol=1e-6)


def test_isolated_node_produces_zero_row_without_nan() -> None:
    """Verify isolates stay zero rows (floor does not invent self-loops)."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    dense = dense_relation_normalized_adjacency(
        edge_index,
        num_nodes=3,
        dtype=torch.float32,
        normalization="rgcn_in_degree",
    )
    assert torch.count_nonzero(dense[0]) == 0
    assert torch.count_nonzero(dense[2]) == 0
    assert not torch.isnan(dense).any()
    x = torch.randn(3, 2)
    out = relation_normalized_adjacency_matvec(
        edge_index,
        x,
        normalization="rgcn_in_degree",
    )
    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[2], torch.zeros(2))
    assert not torch.isnan(out).any()


def test_empty_relation_bank() -> None:
    """Verify empty edge banks return empty weights and zero matvec."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    weights = relation_degree_normalize(
        edge_index,
        num_nodes=3,
        dtype=torch.float32,
    )
    assert weights.shape == (0,)
    dense = dense_relation_normalized_adjacency(
        edge_index,
        3,
        dtype=torch.float32,
    )
    assert torch.equal(dense, torch.zeros(3, 3))
    x = torch.randn(3, 4)
    out = relation_normalized_adjacency_matvec(edge_index, x)
    assert torch.equal(out, torch.zeros_like(x))


def test_random_walk_mode_matches_forward_helper() -> None:
    """Verify ``normalization='random_walk'`` matches forward RW helpers."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 2.0, 3.0])
    got = relation_degree_normalize(
        edge_index,
        num_nodes=3,
        edge_weight=edge_weight,
        dtype=torch.float32,
        normalization="random_walk",
    )
    expected = random_walk_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=3,
        edge_weight=edge_weight,
        dtype=torch.float32,
        direction="forward",
    )
    assert torch.allclose(got, expected)

    x = torch.randn(3, 2)
    assert torch.allclose(
        relation_normalized_adjacency_matvec(
            edge_index,
            x,
            edge_weight=edge_weight,
            normalization="random_walk",
        ),
        random_walk_normalized_adjacency_matvec(
            edge_index,
            x,
            edge_weight=edge_weight,
            direction="forward",
        ),
    )
    assert torch.allclose(
        dense_relation_normalized_adjacency(
            edge_index,
            3,
            edge_weight=edge_weight,
            dtype=torch.float32,
            normalization="random_walk",
        ),
        dense_random_walk_normalized_adjacency(
            edge_index,
            3,
            edge_weight=edge_weight,
            dtype=torch.float32,
            direction="forward",
        ),
    )


def test_rgcn_matches_backward_random_walk_helpers() -> None:
    """Verify default R-GCN mode matches library backward RW helpers."""
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 3]], dtype=torch.long)
    got = relation_degree_normalize(
        edge_index,
        num_nodes=4,
        dtype=torch.float32,
        normalization="rgcn_in_degree",
    )
    expected = random_walk_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=4,
        dtype=torch.float32,
        direction="backward",
    )
    assert torch.allclose(got, expected)


def test_invalid_normalization_raises() -> None:
    """Verify unsupported normalization strings raise ``ValueError``."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(ValueError, match="normalization must be one of"):
        relation_degree_normalize(
            edge_index,
            num_nodes=2,
            dtype=torch.float32,
            normalization="symmetric",  # type: ignore[arg-type]
        )


def test_relation_normalization_modes_documented() -> None:
    """Verify the serialization-facing enum set is stable."""
    expected = frozenset({"rgcn_in_degree", "random_walk"})
    assert expected == RELATION_NORMALIZATION_MODES


def test_helpers_live_in_topology_not_propagation() -> None:
    """Verify relation helpers are defined in ``topology.py`` only."""
    root = REPO_ROOT / "src/koopman_graph/graph_utils"
    topo = root / "topology.py"
    prop = root / "propagation.py"
    topo_tree = ast.parse(topo.read_text())
    prop_tree = ast.parse(prop.read_text())
    topo_names = {
        node.name
        for node in topo_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prop_names = {
        node.name
        for node in prop_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "relation_degree_normalize" in topo_names
    assert "relation_normalized_adjacency_matvec" in topo_names
    assert "dense_relation_normalized_adjacency" in topo_names
    assert "relation_degree_normalize" not in prop_names
