"""Tests for simplicial-1 / Hodge observable lifts (TASK-1848)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.observables import (
    boundary_incidence_b1,
    coerce_face_index,
    hodge_gradient_features,
    resolve_physics_lifting_fn,
    simplicial_one_laplacian_features,
    simplicial_one_laplacian_matvec,
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "observables.py"
)


def _triangle_plus_isolate(*, channels: int = 2) -> Data:
    """Three-node triangle with one isolated node; oriented unique edges."""
    # Oriented undirected edges: 0→1, 1→2, 0→2 (not doubled).
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    x = torch.tensor(
        [
            [1.0, 0.5],
            [0.0, 1.0],
            [-1.0, 0.25],
            [3.0, 3.0],
        ],
        dtype=torch.float64,
    )[:, :channels]
    return Data(x=x, edge_index=edge_index, face_index=face_index)


def test_coerce_face_index_accepts_triangle() -> None:
    """``coerce_face_index`` accepts a single triangle column."""
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    coerced = coerce_face_index(face_index, num_nodes=4)
    assert coerced.shape == (3, 1)
    assert torch.equal(coerced, face_index)


def test_coerce_face_index_rejects_bad_rank_and_oob() -> None:
    """Invalid face_index rank or node ids raise ``ValueError``."""
    with pytest.raises(ValueError, match="shape \\(3, num_faces\\)"):
        coerce_face_index(torch.tensor([[0, 1]], dtype=torch.long), num_nodes=3)
    with pytest.raises(ValueError, match="lie in"):
        coerce_face_index(
            torch.tensor([[0], [1], [9]], dtype=torch.long),
            num_nodes=3,
        )


def test_boundary_incidence_b1_shape_and_column_sums() -> None:
    """Signed ``B1`` has shape ``(N, E)`` and columns sum to zero."""
    data = _triangle_plus_isolate()
    incidence = boundary_incidence_b1(data.edge_index, num_nodes=4)
    assert incidence.shape == (4, 3)
    assert torch.allclose(incidence.sum(dim=0), torch.zeros(3), atol=1e-8)
    assert incidence.dtype == torch.float32


def test_simplicial_one_laplacian_matches_dense_b1() -> None:
    """``L1 @ x`` matches dense ``B1 @ B1.T @ x``; isolate row is zero."""
    data = _triangle_plus_isolate()
    lifted = simplicial_one_laplacian_features(data)
    assert lifted.shape == data.x.shape
    incidence = boundary_incidence_b1(data.edge_index, num_nodes=4).to(
        dtype=data.x.dtype
    )
    expected = (incidence @ incidence.T) @ data.x
    assert torch.allclose(lifted, expected, atol=1e-10)
    assert torch.allclose(lifted[3], torch.zeros_like(lifted[3]), atol=1e-10)
    matvec = simplicial_one_laplacian_matvec(
        data.edge_index,
        data.x,
        num_nodes=4,
    )
    assert torch.allclose(matvec, expected, atol=1e-10)


def test_hodge_gradient_features_shape_and_isolate() -> None:
    """Hodge gradient lift is non-negative with a zero isolate row."""
    data = _triangle_plus_isolate()
    lifted = hodge_gradient_features(data)
    assert lifted.shape == data.x.shape
    assert bool(torch.all(lifted >= 0))
    assert torch.allclose(lifted[3], torch.zeros_like(lifted[3]), atol=1e-10)


def test_hodge_presets_resolve() -> None:
    """New presets resolve through ``resolve_physics_lifting_fn``."""
    data = _triangle_plus_isolate()
    for name in ("hodge_gradient", "simplicial_one_laplacian"):
        fn = resolve_physics_lifting_fn(physics_preset=name)
        assert fn is not None
        out = fn(data)
        assert out.shape == data.x.shape


def test_invalid_face_index_rejected_by_lifts() -> None:
    """Lifts validate optional ``face_index`` when present."""
    data = _triangle_plus_isolate()
    data.face_index = torch.tensor([[0], [1], [99]], dtype=torch.long)
    with pytest.raises(ValueError, match="lie in"):
        simplicial_one_laplacian_features(data)
    with pytest.raises(ValueError, match="lie in"):
        hodge_gradient_features(data)


def test_hodge_honesty_docs() -> None:
    """Module docs distinguish simplicial-1 / L1 from sheaf and L_sym."""
    source = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "simplicial-1" in source or "simplicial 1" in source
    assert "sheaf" in source
    assert "not" in source
    assert "l_1" in source or "b_1" in source


def test_face_index_validation_edge_cases() -> None:
    """Face validation handles invalid node counts, empty faces, and float ids."""
    with pytest.raises(ValueError, match="num_nodes must be"):
        coerce_face_index(torch.empty((3, 0), dtype=torch.long), num_nodes=0)
    empty = torch.empty((3, 0), dtype=torch.int32)
    coerced = coerce_face_index(empty, num_nodes=2)
    assert coerced.dtype == torch.long
    assert coerced.shape == (3, 0)
    with pytest.raises(ValueError, match="integer tensor"):
        coerce_face_index(torch.zeros((3, 1)), num_nodes=2)


def test_boundary_incidence_validation_and_degenerate_edges() -> None:
    """Incidence validates its domain and handles empty edges and self-loops."""
    with pytest.raises(ValueError, match="num_nodes must be"):
        boundary_incidence_b1(torch.empty((2, 0), dtype=torch.long), num_nodes=0)
    with pytest.raises(ValueError, match="shape \\(2, num_edges\\)"):
        boundary_incidence_b1(torch.empty((3, 0), dtype=torch.long), num_nodes=2)
    with pytest.raises(ValueError, match="lie in"):
        boundary_incidence_b1(torch.tensor([[0], [2]]), num_nodes=2)

    empty = boundary_incidence_b1(
        torch.empty((2, 0), dtype=torch.long),
        num_nodes=2,
    )
    self_loop = boundary_incidence_b1(torch.tensor([[1], [1]]), num_nodes=2)
    assert empty.shape == (2, 0)
    assert torch.count_nonzero(self_loop) == 0


def test_simplicial_matvec_rejects_invalid_shapes() -> None:
    """The simplicial matvec requires a matching two-dimensional node signal."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="x must be 2D"):
        simplicial_one_laplacian_matvec(edge_index, torch.zeros(2))
    with pytest.raises(ValueError, match="does not match"):
        simplicial_one_laplacian_matvec(
            edge_index,
            torch.zeros(2, 1),
            num_nodes=3,
        )


@pytest.mark.parametrize(
    ("lift", "data", "message"),
    [
        (
            simplicial_one_laplacian_features,
            Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
            "data.x is required",
        ),
        (
            simplicial_one_laplacian_features,
            Data(x=torch.zeros(2), edge_index=torch.empty((2, 0), dtype=torch.long)),
            "data.x must be 2D",
        ),
        (
            simplicial_one_laplacian_features,
            Data(x=torch.zeros(2, 1)),
            "data.edge_index is required",
        ),
        (
            hodge_gradient_features,
            Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
            "data.x is required",
        ),
        (
            hodge_gradient_features,
            Data(x=torch.zeros(2), edge_index=torch.empty((2, 0), dtype=torch.long)),
            "data.x must be 2D",
        ),
        (
            hodge_gradient_features,
            Data(x=torch.zeros(2, 1)),
            "data.edge_index is required",
        ),
    ],
)
def test_hodge_lifts_reject_missing_or_malformed_inputs(
    lift: object,
    data: Data,
    message: str,
) -> None:
    """Hodge lifts report missing features, rank errors, and missing topology."""
    with pytest.raises(ValueError, match=message):
        lift(data)  # type: ignore[operator]
