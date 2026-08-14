"""Tests for ``synthesize_reverse_edge_types`` schema expansion."""

from __future__ import annotations

import pytest

from koopman_graph.graph_utils import synthesize_reverse_edge_types
from koopman_graph.graph_utils.topology import synthesize_reverse_edge_types as topo_fn


def test_export_matches_topology_symbol() -> None:
    """Capability export re-exports the topology helper."""
    assert synthesize_reverse_edge_types is topo_fn


def test_multiplex_doubles_cardinality() -> None:
    """Multiplex schema without reverses expands to 2|R|."""
    edge_types = (
        ("node", "r1", "node"),
        ("node", "r2", "node"),
    )
    expanded = synthesize_reverse_edge_types(edge_types)
    assert expanded == (
        ("node", "r1", "node"),
        ("node", "r2", "node"),
        ("node", "rev_r1", "node"),
        ("node", "rev_r2", "node"),
    )
    assert len(expanded) == 2 * len(edge_types)


def test_typed_swaps_endpoints() -> None:
    """Typed reverse swaps src/dst and prefixes the relation name."""
    edge_types = (
        ("a", "r0", "b"),
        ("b", "r1", "a"),
        ("a", "r2", "a"),
    )
    expanded = synthesize_reverse_edge_types(edge_types)
    assert ("b", "rev_r0", "a") in expanded
    assert ("a", "rev_r1", "b") in expanded
    assert ("a", "rev_r2", "a") in expanded
    assert expanded[:3] == edge_types
    assert len(expanded) == 6


def test_skip_when_reverse_already_present() -> None:
    """Existing reverse triples are not duplicated."""
    edge_types = (
        ("a", "r0", "b"),
        ("b", "rev_r0", "a"),
    )
    expanded = synthesize_reverse_edge_types(edge_types)
    assert expanded == edge_types
    assert expanded.count(("b", "rev_r0", "a")) == 1


def test_idempotent_on_expanded_schema() -> None:
    """Calling again on an expanded schema does not add reverse-of-reverse."""
    edge_types = (("node", "r1", "node"),)
    once = synthesize_reverse_edge_types(edge_types)
    twice = synthesize_reverse_edge_types(once)
    assert twice == once
    assert not any(rel.startswith("rev_rev_") for _, rel, _ in twice)


def test_skips_reverse_of_reverse_prefix() -> None:
    """Relations already prefixed with rev_ are not reverse-synthesized."""
    edge_types = (("a", "rev_hand", "b"),)
    expanded = synthesize_reverse_edge_types(edge_types)
    assert expanded == edge_types


def test_collision_on_rev_name_raises() -> None:
    """Occupied rev_ relation name that is not the geometric reverse raises."""
    edge_types = (
        ("a", "r0", "b"),
        ("a", "rev_r0", "a"),  # same rev name, wrong endpoints
    )
    with pytest.raises(ValueError, match="cannot synthesize reverse"):
        synthesize_reverse_edge_types(edge_types)


def test_rejects_duplicate_input_triples() -> None:
    """Duplicate input triples raise a clear error."""
    with pytest.raises(ValueError, match="unique"):
        synthesize_reverse_edge_types((("node", "r1", "node"), ("node", "r1", "node")))


def test_rejects_malformed_triple() -> None:
    """Non-triple entries raise ValueError."""
    with pytest.raises(ValueError, match="triples"):
        synthesize_reverse_edge_types((("a", "r0"),))  # type: ignore[arg-type]


def test_custom_reverse_prefix() -> None:
    """Custom reverse_prefix is honored."""
    expanded = synthesize_reverse_edge_types(
        (("a", "r0", "b"),),
        reverse_prefix="inv_",
    )
    assert expanded == (("a", "r0", "b"), ("b", "inv_r0", "a"))
