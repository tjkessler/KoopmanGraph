"""Tests for simplicial-1 encoder / decoder peers (Phase 60 suite)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    GNNDecoder,
    SimplicialDecoder,
    SimplicialEncoder,
    bind_simplicial_decoder,
)
from koopman_graph.nn import simplicial as simplicial_mod
from koopman_graph.nn.simplicial import SimplicialConv
from koopman_graph.serialization import build_model_config

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "koopman_graph"
    / "nn"
    / "simplicial.py"
)


def _triangle_plus_isolate(*, channels: int = 2) -> Data:
    """Three-node triangle with one isolated node; oriented unique edges."""
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    x = torch.randn(4, channels)
    return Data(x=x, edge_index=edge_index, face_index=face_index)


def _simplicial_sequence(
    *,
    num_timesteps: int = 4,
    channels: int = 2,
) -> GraphSnapshotSequence:
    """Static triangle+isolate sequence for fit/predict smokes."""
    template = _triangle_plus_isolate(channels=channels)
    xs = torch.randn(num_timesteps, 4, channels)
    return GraphSnapshotSequence.from_arrays(
        xs,
        template.edge_index,
    )


def test_encode_decode_shapes_and_gradients() -> None:
    """Encoder/decoder preserve node count and admit gradient flow."""
    data = _triangle_plus_isolate(channels=3)
    encoder = SimplicialEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    decoder = SimplicialDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
    )
    z = encoder(data)
    assert z.shape == (4, 4)
    recon = decoder(z, data.edge_index)
    assert recon.shape == (4, 3)

    # Exercise each stack independently: combinatorial L1 can put random
    # latents near its nullspace, so a chained recon loss is not a reliable
    # encoder-gradient probe.
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


def test_exports_from_nn_and_root() -> None:
    """Simplicial peers are exported from ``nn`` and root ``__all__``."""
    assert "SimplicialEncoder" in koopman_graph.nn.__all__
    assert "SimplicialDecoder" in koopman_graph.nn.__all__
    assert "bind_simplicial_decoder" in koopman_graph.nn.__all__
    assert "SimplicialEncoder" in koopman_graph.__all__
    assert "SimplicialDecoder" in koopman_graph.__all__
    assert koopman_graph.SimplicialEncoder is SimplicialEncoder
    assert koopman_graph.SimplicialDecoder is SimplicialDecoder


def test_model_fit_predict_with_graph_koopman() -> None:
    """``GraphKoopmanModel`` fit/predict smoke with standard graph ``K``."""
    sequence = _simplicial_sequence(num_timesteps=4, channels=2)
    model = GraphKoopmanModel(
        encoder=SimplicialEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=SimplicialDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=1)
    assert len(preds) == 1
    assert preds[0].x.shape == (4, 2)


def test_checkpoint_round_trip_sim_enc(tmp_path) -> None:
    """Format-1 checkpoints round-trip ``sim_enc`` / ``sim_dec`` peers."""
    sequence = _simplicial_sequence(num_timesteps=3, channels=2)
    model = GraphKoopmanModel(
        encoder=SimplicialEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=SimplicialDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["encoder"]["type"] == "sim_enc"
    assert config["decoder"]["type"] == "sim_dec"
    path = tmp_path / "sim_peers.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, SimplicialEncoder)
    assert isinstance(loaded.decoder, SimplicialDecoder)


def test_invalid_face_index_raises() -> None:
    """Invalid ``face_index`` on ``Data`` input raises ``ValueError``."""
    data = _triangle_plus_isolate(channels=2)
    data.face_index = torch.tensor([[0, 1]], dtype=torch.long)
    encoder = SimplicialEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    with pytest.raises(ValueError, match="shape \\(3, num_faces\\)"):
        encoder(data)


def test_mismatched_simplicial_peers_raise() -> None:
    """Mixed simplicial/GNN peers are rejected at fit."""
    sequence = _simplicial_sequence(num_timesteps=3, channels=2)
    model = GraphKoopmanModel(
        encoder=SimplicialEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="must be used together"):
        model.fit(sequence, epochs=1)


def test_bind_simplicial_decoder_closure() -> None:
    """Bound decoder ignores positional topology and uses static edges."""
    data = _triangle_plus_isolate(channels=2)
    decoder = SimplicialDecoder(latent_dim=2, hidden_channels=4, out_channels=2)
    bound = bind_simplicial_decoder(decoder, data.edge_index, data.face_index)
    z = torch.randn(4, 2)
    out = bound(z, torch.zeros(2, 0, dtype=torch.long), None)
    assert out.shape == (4, 2)


def test_simplicial_conv_residual_skip() -> None:
    """Residual ``SimplicialConv`` adds the input when widths match."""
    data = _triangle_plus_isolate(channels=4)
    conv = SimplicialConv(4, 4, residual=True)
    out = conv(data.x, data.edge_index)
    assert out.shape == data.x.shape
    assert conv.residual is True


def test_build_simplicial_convs_layer_counts() -> None:
    """``build_simplicial_convs`` builds 1- and 3-layer stacks."""
    one = simplicial_mod.build_simplicial_convs(2, 8, 4, num_layers=1)
    assert len(one) == 1
    assert one[0].in_channels == 2 and one[0].out_channels == 4

    three = simplicial_mod.build_simplicial_convs(2, 8, 4, num_layers=3)
    assert len(three) == 3
    assert three[0].in_channels == 2 and three[0].out_channels == 8
    assert three[1].in_channels == 8 and three[1].out_channels == 8
    assert three[2].in_channels == 8 and three[2].out_channels == 4
    assert three[2].residual is False


def test_encoder_rejects_missing_data_fields_and_tensor_without_edges() -> None:
    """Forward validation covers missing ``x``/``edge_index`` and tensor API."""
    encoder = SimplicialEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    data = _triangle_plus_isolate(channels=2)
    data.x = None
    with pytest.raises(ValueError, match="data.x is required"):
        encoder(data)
    data = _triangle_plus_isolate(channels=2)
    data.edge_index = None
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(data)
    with pytest.raises(ValueError, match="edge_index is required when x_or_data"):
        encoder(torch.randn(4, 2))


def test_encoder_rejects_bad_feature_shape() -> None:
    """Wrong ``x`` rank or width raises clearly."""
    data = _triangle_plus_isolate(channels=2)
    encoder = SimplicialEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    with pytest.raises(ValueError, match="Expected x with shape"):
        encoder(torch.randn(4), data.edge_index)
    with pytest.raises(ValueError, match="Expected in_channels=2"):
        encoder(torch.randn(4, 3), data.edge_index)


def test_encoder_rejects_non_simplicial_conv_in_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-``SimplicialConv`` layer in the stack raises ``TypeError``."""
    encoder = SimplicialEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    encoder.convs[0] = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    data = _triangle_plus_isolate(channels=2)
    with pytest.raises(TypeError, match="expected SimplicialConv"):
        encoder(data)


def test_module_doc_honesty_keywords() -> None:
    """Module docstring states simplicial-1 scope and excludes sheaf/cell."""
    text = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "simplicial-1" in text or "simplicial 1" in text or "l_1" in text
    assert "sheaf" in text
    assert "cell" in text
    assert "simplicial" in simplicial_mod.__doc__.lower()
