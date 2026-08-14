"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.nn.cell_complex import (
    MAX_CELL_COMPLEX_DEGREE,
    CellComplex,
    hodge_laplacian,
)


def test_cell_complex_tetra_boundary() -> None:
    """Tetra incidence covers ``B_3``, bad ``tetra_index`` shapes, and missing faces."""
    from koopman_graph.nn.cell_complex import CellComplex, boundary_operator

    edge_index = torch.tensor(
        [[0, 1, 0, 0, 1, 2], [1, 2, 2, 3, 3, 3]],
        dtype=torch.long,
    )
    face_index = torch.tensor(
        [[0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="tetra_index must have shape"):
        CellComplex(
            num_nodes=4,
            edge_index=edge_index,
            face_index=face_index,
            tetra_index=torch.zeros(3, 1, dtype=torch.long),
        )
    tet = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    complex_ = CellComplex(
        num_nodes=4,
        edge_index=edge_index,
        face_index=face_index,
        tetra_index=tet,
    )
    b3 = boundary_operator(complex_, 3)
    assert b3.shape == (4, 1)
    incomplete = CellComplex(
        num_nodes=4,
        edge_index=edge_index,
        face_index=face_index[:, :3],
        tetra_index=tet,
    )
    with pytest.raises(ValueError, match="missing from face_index"):
        boundary_operator(incomplete, 3)
    object.__setattr__(complex_, "tetra_index", None)
    assert complex_.num_tets == 0


def test_cell_complex_degree_three_empty_tetra() -> None:
    """Empty tetrahedra keep ``L_2`` well-defined at degree cap 3."""
    assert MAX_CELL_COMPLEX_DEGREE == 3
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    complex_ = CellComplex(num_nodes=3, edge_index=edge_index, face_index=face_index)
    lap2 = hodge_laplacian(complex_, 2)
    assert lap2.shape == (1, 1)
    lap3 = hodge_laplacian(complex_, 3)
    assert lap3.shape == (0, 0)
