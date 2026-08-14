"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.nn.cell_complex import (
    CellComplex,
    CellComplexConv,
    build_cell_complex_convs,
    hodge_laplacian_matvec,
)


def test_cell_complex_validation_and_residual() -> None:
    """Construction guards, matvec shapes, residual skip, single-layer stack."""
    with pytest.raises(ValueError, match="num_nodes must be >= 1"):
        CellComplex(
            num_nodes=0,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            face_index=torch.empty((3, 0), dtype=torch.long),
        )
    with pytest.raises(ValueError, match=r"edge_index must have shape \(2"):
        CellComplex(
            num_nodes=3,
            edge_index=torch.zeros(3, 3),
            face_index=torch.empty((3, 0), dtype=torch.long),
        )
    with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
        CellComplex(
            num_nodes=2,
            edge_index=torch.tensor([[0], [5]], dtype=torch.long),
            face_index=torch.empty((3, 0), dtype=torch.long),
        )

    complex_ = CellComplex(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long),
        face_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="1D or 2D"):
        hodge_laplacian_matvec(complex_, 0, torch.zeros(3, 3, 1))
    with pytest.raises(ValueError, match="leading dim"):
        hodge_laplacian_matvec(complex_, 0, torch.zeros(4, 1))

    conv = CellComplexConv(4, 4, residual=True)
    x = torch.randn(3, 4)
    out = conv(x, complex_.edge_index)
    assert out.shape == (3, 4)

    layers = build_cell_complex_convs(2, 4, 3, num_layers=1)
    assert len(layers) == 1


def test_cell_complex_encoder_decoder_forward_guards() -> None:
    """Encoder/decoder refuse missing topology, bad shapes, and bad stacks."""
    from koopman_graph.nn.cell_complex import (
        CellComplexGNNDecoder,
        CellComplexGNNEncoder,
        bind_cell_complex_decoder,
        build_cell_complex_convs,
        hodge_laplacian_matvec,
    )

    encoder = CellComplexGNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    decoder = CellComplexGNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2)
    edge = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    faces = torch.tensor([[0], [1], [2]], dtype=torch.long)

    with pytest.raises(ValueError, match="data.x is required"):
        encoder(Data(edge_index=edge, face_index=faces))
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(Data(x=torch.randn(3, 2), face_index=faces))
    with pytest.raises(ValueError, match="edge_index is required when"):
        encoder(torch.randn(3, 2))
    with pytest.raises(ValueError, match="Expected x with shape"):
        encoder(torch.randn(3), edge)
    with pytest.raises(ValueError, match="in_channels=2"):
        encoder(torch.randn(3, 5), edge)

    encoder.convs[0] = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="expected CellComplexConv"):
        encoder(torch.randn(3, 2), edge)

    with pytest.raises(ValueError, match=r"edge_index must have shape \(2"):
        bind_cell_complex_decoder(decoder, torch.zeros(3, 2), faces)

    layers = build_cell_complex_convs(2, 4, 3, num_layers=3)
    assert len(layers) == 3

    complex_ = CellComplex(num_nodes=3, edge_index=edge, face_index=faces)
    assert complex_.num_cells(3) == 0
    with pytest.raises(ValueError, match=r"degree k must be in"):
        complex_.num_cells(4)
    vec = hodge_laplacian_matvec(complex_, 0, torch.randn(3))
    assert vec.shape == (3,)
