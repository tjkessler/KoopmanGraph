"""Tests for cell-complex operators and encoder peers (TASK-1933 / 1934)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.factory import build_encoder_peers
from koopman_graph.nn import (
    CellComplexGNNDecoder,
    CellComplexGNNEncoder,
    GNNDecoder,
    bind_cell_complex_decoder,
)
from koopman_graph.nn.cell_complex import (
    MAX_CELL_COMPLEX_DEGREE,
    CellComplex,
    boundary_incidence_b2,
    boundary_operator,
    hodge_laplacian,
    hodge_laplacian_matvec,
)
from koopman_graph.observables import (
    boundary_incidence_b1,
    simplicial_one_laplacian_matvec,
)
from koopman_graph.serialization import (
    FORMAT_VERSION,
    _build_encoder,
    build_model_config,
)

# float32 dense products on tiny fixtures; looser than float64 machine eps.
_ATOL = 1e-6


def _filled_triangle() -> CellComplex:
    """One oriented triangle (nodes 0-1-2) with matching 1-skeleton."""
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    return CellComplex(num_nodes=3, edge_index=edge_index, face_index=face_index)


def test_cell_counts_and_shapes() -> None:
    """``B_k`` / ``L_k`` shapes match documented ``n_k`` counts."""
    complex_ = _filled_triangle()
    assert complex_.num_cells(0) == 3
    assert complex_.num_cells(1) == 3
    assert complex_.num_cells(2) == 1
    b0 = boundary_operator(complex_, 0)
    b1 = boundary_operator(complex_, 1)
    b2 = boundary_operator(complex_, 2)
    assert b0.shape == (0, 3)
    assert b1.shape == (3, 3)
    assert b2.shape == (3, 1)
    assert hodge_laplacian(complex_, 0).shape == (3, 3)
    assert hodge_laplacian(complex_, 1).shape == (3, 3)
    assert hodge_laplacian(complex_, 2).shape == (1, 1)


def test_b1_reuses_simplicial_helper() -> None:
    """``B_1`` is exactly ``boundary_incidence_b1`` (no duplicated signs)."""
    complex_ = _filled_triangle()
    expected = boundary_incidence_b1(complex_.edge_index, num_nodes=3)
    assert torch.equal(boundary_operator(complex_, 1), expected)


def test_boundary_nilpotency_b1_b2() -> None:
    """``B_1 B_2 = 0`` on a filled triangle (orientation guard)."""
    complex_ = _filled_triangle()
    b1 = boundary_operator(complex_, 1)
    b2 = boundary_operator(complex_, 2)
    product = b1 @ b2
    assert product.shape == (3, 1)
    assert torch.allclose(product, torch.zeros_like(product), atol=_ATOL, rtol=0.0)


def test_l0_matches_simplicial_one_laplacian() -> None:
    """Hodge ``L_0`` matches the 0.10 simplicial node Laplacian helper."""
    complex_ = _filled_triangle()
    x = torch.randn(3, 4)
    dense = hodge_laplacian(complex_, 0) @ x
    matvec = hodge_laplacian_matvec(complex_, 0, x)
    simplicial = simplicial_one_laplacian_matvec(complex_.edge_index, x, num_nodes=3)
    assert torch.allclose(dense, simplicial, atol=_ATOL, rtol=1e-5)
    assert torch.allclose(matvec, simplicial, atol=_ATOL, rtol=1e-5)


def test_l1_includes_up_and_down() -> None:
    """``L_1 = B_1^T B_1 + B_2 B_2^T`` on the filled triangle."""
    complex_ = _filled_triangle()
    b1 = boundary_operator(complex_, 1)
    b2 = boundary_operator(complex_, 2)
    expected = b1.transpose(0, 1) @ b1 + b2 @ b2.transpose(0, 1)
    assert torch.allclose(hodge_laplacian(complex_, 1), expected, atol=_ATOL)


def test_skeleton_only_has_empty_b2() -> None:
    """1-skeleton complexes expose ``B_2`` with zero face columns."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    complex_ = CellComplex(num_nodes=3, edge_index=edge_index)
    b2 = boundary_operator(complex_, 2)
    assert b2.shape == (2, 0)
    # L_0 still matches simplicial helper on the path graph.
    x = torch.randn(3, 2)
    assert torch.allclose(
        hodge_laplacian_matvec(complex_, 0, x),
        simplicial_one_laplacian_matvec(edge_index, x, num_nodes=3),
        atol=_ATOL,
    )


def test_boundary_incidence_b2_public_helper() -> None:
    """Public ``boundary_incidence_b2`` matches ``boundary_operator(..., 2)``."""
    complex_ = _filled_triangle()
    assert torch.equal(
        boundary_incidence_b2(
            complex_.edge_index,
            complex_.face_index,
            num_nodes=3,
        ),
        boundary_operator(complex_, 2),
    )


def test_missing_face_edge_raises() -> None:
    """Faces whose 1-skeleton is incomplete raise clearly at ``B_2`` build."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # missing 0–2
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    # Construction only validates ids; incidence build enforces the skeleton.
    complex_ = CellComplex(
        num_nodes=3,
        edge_index=edge_index,
        face_index=face_index,
    )
    with pytest.raises(ValueError, match="missing from edge_index"):
        boundary_operator(complex_, 2)
    with pytest.raises(ValueError, match="missing from edge_index"):
        boundary_incidence_b2(edge_index, face_index, num_nodes=3)


def test_duplicate_undirected_edge_raises() -> None:
    """Duplicate undirected edges are rejected at construction."""
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="duplicate undirected edge"):
        CellComplex(num_nodes=3, edge_index=edge_index)


def test_degree_ceiling() -> None:
    """Degrees above the teaching MVP raise."""
    complex_ = _filled_triangle()
    with pytest.raises(ValueError, match="teaching MVP"):
        boundary_operator(complex_, MAX_CELL_COMPLEX_DEGREE + 1)
    with pytest.raises(ValueError, match="teaching MVP"):
        hodge_laplacian(complex_, -1)


def test_hodge_matvec_k1_gradients() -> None:
    """Dense ``L_1`` matvec admits gradient flow on a tiny fixture."""
    complex_ = _filled_triangle()
    x = torch.randn(3, 2, requires_grad=True)
    out = hodge_laplacian_matvec(complex_, 1, x)
    assert out.shape == (3, 2)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_exports_from_nn() -> None:
    """Cell-complex helpers are exported from ``koopman_graph.nn``."""
    import koopman_graph.nn as nn_pkg

    assert "CellComplex" in nn_pkg.__all__
    assert "boundary_operator" in nn_pkg.__all__
    assert "hodge_laplacian" in nn_pkg.__all__
    assert nn_pkg.CellComplex is CellComplex


def _triangle_data(*, channels: int = 2) -> Data:
    """Filled triangle ``Data`` with required ``face_index``."""
    complex_ = _filled_triangle()
    return Data(
        x=torch.randn(3, channels),
        edge_index=complex_.edge_index,
        face_index=complex_.face_index,
    )


def _cell_sequence(
    *,
    num_timesteps: int = 4,
    channels: int = 2,
) -> GraphSnapshotSequence:
    """Sequence of filled-triangle snapshots carrying ``face_index``."""
    template = _triangle_data(channels=channels)
    snapshots = [
        Data(
            x=torch.randn(3, channels),
            edge_index=template.edge_index.clone(),
            face_index=template.face_index.clone(),
        )
        for _ in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_encode_decode_shapes_and_gradients() -> None:
    """Encoder/decoder preserve shapes and admit gradient flow."""
    data = _triangle_data(channels=3)
    encoder = CellComplexGNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    decoder = CellComplexGNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
    )
    z = encoder(data)
    assert z.shape == (3, 4)
    recon = decoder(z, data.edge_index)
    assert recon.shape == (3, 3)
    encoder.zero_grad(set_to_none=True)
    z.sum().backward(retain_graph=True)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
    )
    decoder.zero_grad(set_to_none=True)
    recon.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
    )


def test_snapshot_to_device_preserves_face_index() -> None:
    """Device moves keep triangular ``face_index`` for cell-complex peers."""
    from koopman_graph.graph_utils import snapshot_to_device

    data = _triangle_data(channels=2)
    moved = snapshot_to_device(data, torch.device("cpu"))
    assert moved.face_index is not None
    assert torch.equal(moved.face_index, data.face_index)


def test_missing_face_index_raises_clearly() -> None:
    """``Data`` without faces refuses silent GNN-style fallback."""
    encoder = CellComplexGNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    data = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="face_index is required"):
        encoder(data)
    with pytest.raises(ValueError, match="at least one 2-cell"):
        data.face_index = torch.empty((3, 0), dtype=torch.long)
        encoder(data)


def test_factory_encoder_cell_complex_builds_matched_peers() -> None:
    """``build_encoder_peers(encoder=\"cell_complex\")`` returns matched peers."""
    encoder, decoder = build_encoder_peers(
        "cell_complex",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
    )
    assert isinstance(encoder, CellComplexGNNEncoder)
    assert isinstance(decoder, CellComplexGNNDecoder)


def test_exports_peers_from_nn_and_root() -> None:
    """Cell-complex peers are on ``nn`` and root ``__all__``."""
    assert "CellComplexGNNEncoder" in koopman_graph.nn.__all__
    assert "CellComplexGNNDecoder" in koopman_graph.nn.__all__
    assert "bind_cell_complex_decoder" in koopman_graph.nn.__all__
    assert "CellComplexGNNEncoder" in koopman_graph.__all__
    assert "CellComplexGNNDecoder" in koopman_graph.__all__
    assert koopman_graph.CellComplexGNNEncoder is CellComplexGNNEncoder


def test_model_fit_predict_cell_complex() -> None:
    """``GraphKoopmanModel`` fit/predict smoke keeps linear ``K``."""
    sequence = _cell_sequence(num_timesteps=4, channels=2)
    encoder, decoder = build_encoder_peers(
        "cell_complex",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    assert type(model.koopman).__name__ == "GraphKoopmanOperator"
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=1)
    assert len(preds) == 1
    assert preds[0].x.shape == (3, 2)


def test_checkpoint_round_trip_cell_enc(tmp_path) -> None:
    """Format-1 checkpoints round-trip ``cell_enc`` / ``cell_dec`` peers."""
    sequence = _cell_sequence(num_timesteps=3, channels=2)
    encoder, decoder = build_encoder_peers(
        "cell_complex",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["encoder"]["type"] == "cell_enc"
    assert config["decoder"]["type"] == "cell_dec"
    assert FORMAT_VERSION == 1
    path = tmp_path / "cell_peers.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, CellComplexGNNEncoder)
    assert isinstance(loaded.decoder, CellComplexGNNDecoder)


def test_sheaf_checkpoint_still_round_trips(tmp_path) -> None:
    """Sheaf peers still round-trip alongside the cell-complex path."""
    from koopman_graph.nn import SheafGNNDecoder, SheafGNNEncoder

    xs = torch.randn(3, 3, 2)
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    sequence = GraphSnapshotSequence.from_arrays(xs, edge_index)
    encoder, decoder = build_encoder_peers(
        "sheaf",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    model.fit(sequence, epochs=1)
    path = tmp_path / "sheaf_peers.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, SheafGNNEncoder)
    assert isinstance(loaded.decoder, SheafGNNDecoder)


def test_unknown_checkpoint_encoder_does_not_fall_back_to_gnn() -> None:
    """Unsupported encoder types raise; they never become ``GNNEncoder``."""
    with pytest.raises(ValueError, match="Unsupported encoder type"):
        _build_encoder(
            {
                "type": "not_a_cell_or_gnn",
                "in_channels": 2,
                "hidden_channels": 4,
                "latent_dim": 2,
                "num_layers": 2,
                "activation": "relu",
            }
        )


def test_mismatched_cell_complex_peers_raise() -> None:
    """Mixed cell-complex / GNN peers are rejected at fit."""
    sequence = _cell_sequence(num_timesteps=3, channels=2)
    model = GraphKoopmanModel(
        encoder=CellComplexGNNEncoder(
            in_channels=2,
            hidden_channels=8,
            latent_dim=4,
        ),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="must be used together"):
        model.fit(sequence, epochs=1)


def test_bind_cell_complex_decoder_closure() -> None:
    """Bound decoder validates faces at bind time and ignores call edges."""
    data = _triangle_data(channels=2)
    decoder = CellComplexGNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2)
    bound = bind_cell_complex_decoder(decoder, data.edge_index, data.face_index)
    z = torch.randn(3, 2)
    out = bound(z, torch.zeros(2, 0, dtype=torch.long), None)
    assert out.shape == (3, 2)
    with pytest.raises(ValueError, match="face_index is required"):
        bind_cell_complex_decoder(
            decoder,
            data.edge_index,
            None,  # type: ignore[arg-type]
        )
