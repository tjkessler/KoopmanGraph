"""Shared type aliases for networked (graph / continuous-graph) operators.

Kept outside :mod:`koopman_graph.operators.graph` so inverse helpers can type
adjacency modes without creating an import cycle with the operator module.
"""

from __future__ import annotations

from typing import Literal

GraphSparsity = Literal["dense", "block_diagonal", "distributed"]
GraphAdjacency = Literal["symmetric", "random_walk", "dual_random_walk"]
GRAPH_ADJACENCY_MODES: frozenset[str] = frozenset(
    {"symmetric", "random_walk", "dual_random_walk"}
)

__all__ = [
    "GRAPH_ADJACENCY_MODES",
    "GraphAdjacency",
    "GraphSparsity",
]
