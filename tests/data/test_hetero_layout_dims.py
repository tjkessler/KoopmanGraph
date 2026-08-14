"""Opt-in ``latent_dims`` layout helpers (TASK-1814)."""

from __future__ import annotations

import pytest

from koopman_graph.data import (
    latent_type_slices,
    latent_type_slices_from_dims,
    node_type_slices,
    stacked_latent_numel,
    validate_latent_dims,
)

NODE_TYPES = ("a", "b")
NUM_NODES = {"a": 4, "b": 3}
LATENT_DIMS = {"a": 2, "b": 5}
SHARED_D = 4


def test_validate_latent_dims_none_is_shared_path() -> None:
    """Absent ``latent_dims`` selects the shared-d path (returns None)."""
    assert validate_latent_dims(NODE_TYPES, None, shared_latent_dim=SHARED_D) is None


def test_validate_latent_dims_rejects_bad_shared_width() -> None:
    """Shared path still validates ``shared_latent_dim`` when provided."""
    with pytest.raises(ValueError, match="shared_latent_dim must be positive"):
        validate_latent_dims(NODE_TYPES, None, shared_latent_dim=0)


def test_validate_latent_dims_accepts_unequal_widths() -> None:
    """Opt-in mapping is returned in node_type_names order."""
    dims = validate_latent_dims(NODE_TYPES, LATENT_DIMS)
    assert dims == {"a": 2, "b": 5}
    assert list(dims) == list(NODE_TYPES)


def test_validate_latent_dims_rejects_missing_key() -> None:
    """Missing type keys raise clearly (no silent ignore)."""
    with pytest.raises(ValueError, match="missing node type"):
        validate_latent_dims(NODE_TYPES, {"a": 2})


def test_validate_latent_dims_rejects_extra_key() -> None:
    """Extra type keys raise clearly."""
    with pytest.raises(ValueError, match="outside node_type_names"):
        validate_latent_dims(NODE_TYPES, {"a": 2, "b": 5, "c": 1})


def test_validate_latent_dims_rejects_non_positive_width() -> None:
    """Per-type widths must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_latent_dims(NODE_TYPES, {"a": 2, "b": 0})


def test_latent_type_slices_from_dims_unequal() -> None:
    """Unequal d_τ flat slices tile [0, Σ N_τ·d_τ) without gaps."""
    slices = latent_type_slices_from_dims(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    assert slices["a"] == slice(0, 4 * 2)
    assert slices["b"] == slice(4 * 2, 4 * 2 + 3 * 5)
    covered: list[int] = []
    for name in NODE_TYPES:
        type_slice = slices[name]
        covered.extend(range(type_slice.start, type_slice.stop))
    total = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    assert covered == list(range(total))
    assert total == 4 * 2 + 3 * 5


def test_latent_type_slices_from_dims_follows_declared_order() -> None:
    """Reversing node_type_names reverses the flat block order."""
    slices = latent_type_slices_from_dims(
        ("b", "a"),
        NUM_NODES,
        {"b": 5, "a": 2},
    )
    assert slices["b"] == slice(0, 3 * 5)
    assert slices["a"] == slice(3 * 5, 3 * 5 + 4 * 2)


def test_shared_d_latent_type_slices_unchanged() -> None:
    """Shared-d latent_type_slices still expands node rows by scalar d."""
    node_slices = node_type_slices(NODE_TYPES, NUM_NODES)
    flat = latent_type_slices(node_slices, latent_dim=SHARED_D)
    assert flat["a"] == slice(0, 4 * SHARED_D)
    assert flat["b"] == slice(4 * SHARED_D, (4 + 3) * SHARED_D)


def test_equal_latent_dims_matches_shared_flat_length() -> None:
    """When every d_τ equals d, flat length matches shared N·d."""
    equal = {"a": SHARED_D, "b": SHARED_D}
    numel = stacked_latent_numel(NODE_TYPES, NUM_NODES, equal)
    assert numel == (4 + 3) * SHARED_D
    from_dims = latent_type_slices_from_dims(NODE_TYPES, NUM_NODES, equal)
    shared = latent_type_slices(node_type_slices(NODE_TYPES, NUM_NODES), SHARED_D)
    assert from_dims == shared
