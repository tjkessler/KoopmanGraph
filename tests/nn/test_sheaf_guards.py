"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data


def test_coerce_face_index_and_sheaf_matvec_guards() -> None:
    """Face-index and sheaf Laplacian MVP validation branches."""
    from koopman_graph.observables import (
        coerce_face_index,
        diagonal_sheaf_laplacian_matvec,
        general_sheaf_laplacian_matvec,
    )

    with pytest.raises(ValueError, match="num_nodes must be >= 1"):
        coerce_face_index(torch.empty((3, 0), dtype=torch.long), num_nodes=0)
    with pytest.raises(ValueError, match=r"face_index must have shape \(3"):
        coerce_face_index(torch.zeros(2, 1, dtype=torch.long), num_nodes=2)
    with pytest.raises(ValueError, match="integer tensor"):
        coerce_face_index(torch.zeros(3, 1), num_nodes=2)
    with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
        coerce_face_index(
            torch.tensor([[0], [1], [9]], dtype=torch.long),
            num_nodes=2,
        )
    empty = coerce_face_index(torch.empty((3, 0), dtype=torch.long), num_nodes=2)
    assert empty.shape == (3, 0) and empty.dtype == torch.long

    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 3)
    with pytest.raises(ValueError, match="x must be 2D"):
        diagonal_sheaf_laplacian_matvec(
            edge, torch.randn(2), torch.ones(3), torch.ones(3)
        )
    with pytest.raises(ValueError, match="source_diag/target_diag"):
        diagonal_sheaf_laplacian_matvec(edge, x, torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="num_nodes="):
        diagonal_sheaf_laplacian_matvec(
            edge, x, torch.ones(3), torch.ones(3), num_nodes=5
        )
    with pytest.raises(ValueError, match=r"edge_index must have shape \(2"):
        diagonal_sheaf_laplacian_matvec(
            torch.zeros(3, 2), x, torch.ones(3), torch.ones(3)
        )

    eye = torch.eye(3)
    with pytest.raises(ValueError, match="x must be 2D"):
        general_sheaf_laplacian_matvec(edge, torch.randn(2), eye, eye)
    with pytest.raises(ValueError, match="source_map/target_map"):
        general_sheaf_laplacian_matvec(edge, x, torch.ones(3), eye)
    with pytest.raises(ValueError, match="num_nodes="):
        general_sheaf_laplacian_matvec(edge, x, eye, eye, num_nodes=4)
    with pytest.raises(ValueError, match=r"edge_index must have shape \(2"):
        general_sheaf_laplacian_matvec(torch.zeros(1, 2), x, eye, eye)


def test_sheaf_construction_and_forward_guards() -> None:
    """Restriction-map / Data / feature-dim / stack-type guards."""
    from koopman_graph.nn.sheaf import (
        SheafConv,
        SheafGNNEncoder,
        build_sheaf_convs,
    )

    with pytest.raises(ValueError, match="restriction_maps must be"):
        SheafConv(4, 4, restriction_maps="laplacian")  # type: ignore[arg-type]
    layers = build_sheaf_convs(2, 4, 3, num_layers=1)
    assert len(layers) == 1

    encoder = SheafGNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(Data(x=torch.randn(3, 2)))
    with pytest.raises(ValueError, match="edge_index is required when"):
        encoder(torch.randn(3, 2))
    with pytest.raises(ValueError, match="in_channels=2"):
        encoder(torch.randn(3, 5), torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    with pytest.raises(ValueError, match="Expected x with shape"):
        encoder(torch.randn(3), torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

    encoder.convs[0] = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="expected SheafConv"):
        encoder(torch.randn(2, 2), torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
