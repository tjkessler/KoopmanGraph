"""Tests for sheaf encoder / decoder peers (TASK-1931 / TASK-1932)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.factory import build_encoder_peers
from koopman_graph.nn import (
    GNNDecoder,
    SheafGNNDecoder,
    SheafGNNEncoder,
    bind_sheaf_decoder,
)
from koopman_graph.nn.sheaf import MAX_GENERAL_SHEAF_CHANNELS, SheafConv
from koopman_graph.observables import (
    diagonal_sheaf_laplacian_matvec,
    general_sheaf_laplacian_matvec,
    simplicial_one_laplacian_matvec,
)
from koopman_graph.serialization import build_model_config

_SHEAF_MODULE = (
    Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "nn" / "sheaf.py"
)


def _triangle_plus_isolate(*, channels: int = 2) -> Data:
    """Three-node triangle with one isolated node; oriented unique edges."""
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    x = torch.randn(4, channels)
    return Data(x=x, edge_index=edge_index)


def _sheaf_sequence(
    *,
    num_timesteps: int = 4,
    channels: int = 2,
) -> GraphSnapshotSequence:
    """Static triangle+isolate sequence for fit/predict smokes."""
    template = _triangle_plus_isolate(channels=channels)
    xs = torch.randn(num_timesteps, 4, channels)
    return GraphSnapshotSequence.from_arrays(xs, template.edge_index)


def test_identity_diagonals_match_simplicial_l1() -> None:
    """Identity restriction maps recover combinatorial ``L_1 = B_1 B_1^T``."""
    data = _triangle_plus_isolate(channels=3)
    ones = torch.ones(3)
    sheaf = diagonal_sheaf_laplacian_matvec(
        data.edge_index,
        data.x,
        ones,
        ones,
    )
    simplicial = simplicial_one_laplacian_matvec(data.edge_index, data.x)
    assert torch.allclose(sheaf, simplicial, atol=1e-6, rtol=1e-6)


def test_encode_decode_shapes_and_gradients() -> None:
    """Encoder/decoder preserve node count and admit gradient flow."""
    data = _triangle_plus_isolate(channels=3)
    encoder = SheafGNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    decoder = SheafGNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
    )
    z = encoder(data)
    assert z.shape == (4, 4)
    recon = decoder(z, data.edge_index)
    assert recon.shape == (4, 3)

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


def test_factory_encoder_sheaf_builds_matched_peers() -> None:
    """``build_encoder_peers(encoder=\"sheaf\")`` returns matched sheaf peers."""
    encoder, decoder = build_encoder_peers(
        "sheaf",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
        num_layers=2,
    )
    assert isinstance(encoder, SheafGNNEncoder)
    assert isinstance(decoder, SheafGNNDecoder)
    assert encoder.latent_dim == 4
    assert decoder.out_channels == 2
    with pytest.raises(ValueError, match="Unknown encoder"):
        build_encoder_peers(
            "not_a_kind",  # type: ignore[arg-type]
            in_channels=2,
            hidden_channels=4,
            latent_dim=2,
            out_channels=2,
        )


def test_exports_from_nn_and_root() -> None:
    """Sheaf peers are exported from ``nn`` and root ``__all__``."""
    assert "SheafGNNEncoder" in koopman_graph.nn.__all__
    assert "SheafGNNDecoder" in koopman_graph.nn.__all__
    assert "bind_sheaf_decoder" in koopman_graph.nn.__all__
    assert "SheafGNNEncoder" in koopman_graph.__all__
    assert "SheafGNNDecoder" in koopman_graph.__all__
    assert koopman_graph.SheafGNNEncoder is SheafGNNEncoder
    assert koopman_graph.SheafGNNDecoder is SheafGNNDecoder


def test_model_fit_predict_with_graph_koopman() -> None:
    """``GraphKoopmanModel`` fit/predict smoke keeps linear ``K``."""
    sequence = _sheaf_sequence(num_timesteps=4, channels=2)
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
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    # Latent operator remains a linear graph Koopman module.
    assert type(model.koopman).__name__ == "GraphKoopmanOperator"
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=1)
    assert len(preds) == 1
    assert preds[0].x.shape == (4, 2)


def test_checkpoint_round_trip_sheaf_enc(tmp_path) -> None:
    """Format-1 checkpoints round-trip ``sheaf_enc`` / ``sheaf_dec`` peers."""
    sequence = _sheaf_sequence(num_timesteps=3, channels=2)
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
    config = build_model_config(model)
    assert config["encoder"]["type"] == "sheaf_enc"
    assert config["decoder"]["type"] == "sheaf_dec"
    path = tmp_path / "sheaf_peers.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, SheafGNNEncoder)
    assert isinstance(loaded.decoder, SheafGNNDecoder)


def test_mismatched_sheaf_peers_raise() -> None:
    """Mixed sheaf/GNN peers are rejected at fit."""
    sequence = _sheaf_sequence(num_timesteps=3, channels=2)
    model = GraphKoopmanModel(
        encoder=SheafGNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="must be used together"):
        model.fit(sequence, epochs=1)


def test_bind_sheaf_decoder_closure() -> None:
    """Bound decoder ignores positional topology and uses static edges."""
    data = _triangle_plus_isolate(channels=2)
    decoder = SheafGNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2)
    bound = bind_sheaf_decoder(decoder, data.edge_index)
    z = torch.randn(4, 2)
    out = bound(z, torch.zeros(2, 0, dtype=torch.long), None)
    assert out.shape == (4, 2)


def test_sheaf_conv_residual_skip() -> None:
    """Residual ``SheafConv`` adds the input when widths match."""
    data = _triangle_plus_isolate(channels=4)
    conv = SheafConv(4, 4, residual=True)
    out = conv(data.x, data.edge_index)
    assert out.shape == data.x.shape
    assert conv.residual is True


def test_encoder_rejects_missing_edges_and_bad_features() -> None:
    """Forward validation rejects missing edges and bad feature shapes."""
    encoder = SheafGNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    with pytest.raises(ValueError, match="edge_index is required"):
        encoder(torch.randn(3, 2))
    with pytest.raises(ValueError, match="data.x is required"):
        encoder(Data(edge_index=torch.tensor([[0], [1]], dtype=torch.long)))
    data = _triangle_plus_isolate(channels=3)
    with pytest.raises(ValueError, match="in_channels=2"):
        encoder(data)


def test_module_doc_honesty_keywords() -> None:
    """Module docs name diagonal default, general cost, and linear ``K``."""
    text = _SHEAF_MODULE.read_text(encoding="utf-8").lower()
    assert "diagonal" in text
    assert "sheaf" in text
    assert "general" in text
    assert "linear" in text  # latent K stays linear
    assert "ceiling" in text or str(MAX_GENERAL_SHEAF_CHANNELS) in text


def test_identity_general_maps_match_simplicial_l1() -> None:
    """Identity dense restriction maps recover combinatorial ``L_1``."""
    data = _triangle_plus_isolate(channels=3)
    eye = torch.eye(3)
    sheaf = general_sheaf_laplacian_matvec(
        data.edge_index,
        data.x,
        eye,
        eye,
    )
    simplicial = simplicial_one_laplacian_matvec(data.edge_index, data.x)
    assert torch.allclose(sheaf, simplicial, atol=1e-6, rtol=1e-6)


def test_diag_embedded_general_matches_diagonal_helper() -> None:
    """Diagonal matrices through the general helper match the diagonal path."""
    data = _triangle_plus_isolate(channels=3)
    diag = torch.tensor([0.5, 1.25, -0.75])
    source = torch.diag(diag)
    target = torch.diag(diag.flip(0))
    general = general_sheaf_laplacian_matvec(
        data.edge_index,
        data.x,
        source,
        target,
    )
    diagonal = diagonal_sheaf_laplacian_matvec(
        data.edge_index,
        data.x,
        diag,
        diag.flip(0),
    )
    assert torch.allclose(general, diagonal, atol=1e-6, rtol=1e-6)


def test_general_restriction_shapes_and_gradients() -> None:
    """Opt-in general maps preserve shapes and admit gradient flow."""
    data = _triangle_plus_isolate(channels=3)
    encoder = SheafGNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
        restriction_maps="general",
    )
    decoder = SheafGNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=2,
        restriction_maps="general",
    )
    assert encoder.restriction_maps == "general"
    assert decoder.restriction_maps == "general"
    z = encoder(data)
    assert z.shape == (4, 4)
    recon = decoder(z, data.edge_index)
    assert recon.shape == (4, 3)
    encoder.zero_grad(set_to_none=True)
    z.sum().backward(retain_graph=True)
    map_grads = [
        p.grad
        for name, p in encoder.named_parameters()
        if "source_map" in name or "target_map" in name
    ]
    assert map_grads
    assert any(g is not None and g.abs().sum() > 0 for g in map_grads)


def test_general_restriction_channel_ceiling() -> None:
    """General maps refuse widths above ``MAX_GENERAL_SHEAF_CHANNELS``."""
    too_wide = MAX_GENERAL_SHEAF_CHANNELS + 1
    with pytest.raises(ValueError, match="channels ≤"):
        SheafConv(too_wide, too_wide, restriction_maps="general")
    with pytest.raises(ValueError, match="channels ≤"):
        SheafGNNEncoder(
            in_channels=too_wide,
            hidden_channels=8,
            latent_dim=4,
            restriction_maps="general",
        )


def test_default_restriction_maps_remain_diagonal() -> None:
    """Factory / encoder default stays diagonal (1931 bit-compat path)."""
    encoder, decoder = build_encoder_peers(
        "sheaf",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
    )
    assert encoder.restriction_maps == "diagonal"
    assert decoder.restriction_maps == "diagonal"
    assert isinstance(encoder.convs[0], SheafConv)
    assert encoder.convs[0].source_diag is not None
    assert encoder.convs[0].source_map is None


def test_factory_and_checkpoint_general_restriction(tmp_path) -> None:
    """Factory opt-in + checkpoint round-trip for ``restriction_maps='general'``."""
    sequence = _sheaf_sequence(num_timesteps=3, channels=2)
    encoder, decoder = build_encoder_peers(
        "sheaf",
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        out_channels=2,
        restriction_maps="general",
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
    assert config["encoder"]["restriction_maps"] == "general"
    assert config["decoder"]["restriction_maps"] == "general"
    path = tmp_path / "sheaf_general.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.encoder.restriction_maps == "general"
    assert loaded.decoder.restriction_maps == "general"


def test_invalid_restriction_maps_string() -> None:
    """Unknown restriction_maps values raise clearly."""
    with pytest.raises(ValueError, match="restriction_maps must be"):
        SheafGNNEncoder(
            in_channels=2,
            hidden_channels=4,
            latent_dim=2,
            restriction_maps="block",  # type: ignore[arg-type]
        )
