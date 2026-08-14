"""Tests for optional typed HGTEncoder / HGTDecoder peers."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.nn import HGTDecoder, HGTEncoder
from koopman_graph.nn.heterogeneous import HGTEncoder as ModuleHGTEncoder

NODE_TYPES = ("a", "b")
EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"), ("a", "r2", "a"))
FEATURE_DIMS = {"a": 2, "b": 3}
NUM_NODES = {"a": 4, "b": 3}
LATENT_DIM = 4


def _typed_snapshot(*, seed: int = 0) -> HeteroData:
    """Build a two-type snapshot with edges into every node type."""
    generator = torch.Generator().manual_seed(seed)
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(
        NUM_NODES["a"],
        FEATURE_DIMS["a"],
        generator=generator,
    )
    snapshot["b"].x = torch.randn(
        NUM_NODES["b"],
        FEATURE_DIMS["b"],
        generator=generator,
    )
    snapshot["a", "r0", "b"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]],
        dtype=torch.long,
    )
    snapshot["b", "r1", "a"].edge_index = torch.tensor(
        [[0, 1], [1, 3]],
        dtype=torch.long,
    )
    snapshot["a", "r2", "a"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    return snapshot


def _encoder(**kwargs: object) -> HGTEncoder:
    """Build a typed HGT encoder over the shared test schema."""
    return HGTEncoder(
        FEATURE_DIMS,
        hidden_channels=8,
        latent_dim=LATENT_DIM,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        num_layers=1,
        **kwargs,  # type: ignore[arg-type]
    )


def _decoder(**kwargs: object) -> HGTDecoder:
    """Build a typed HGT decoder over the shared test schema."""
    return HGTDecoder(
        latent_dim=LATENT_DIM,
        hidden_channels=8,
        out_channels=FEATURE_DIMS,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        num_layers=1,
        **kwargs,  # type: ignore[arg-type]
    )


def test_hgt_exported_from_nn_not_root() -> None:
    """HGT peers are nn power-user exports, not root ``__all__`` members."""
    import koopman_graph

    assert HGTEncoder is ModuleHGTEncoder
    assert "HGTEncoder" in koopman_graph.nn.__all__
    assert "HGTDecoder" in koopman_graph.nn.__all__
    assert "HGTEncoder" not in koopman_graph.__all__
    assert "HGTDecoder" not in koopman_graph.__all__


def test_hgt_encoder_decoder_shapes() -> None:
    """Encode→decode preserves typed shapes and stacked latent contract."""
    torch.manual_seed(7)
    snapshot = _typed_snapshot(seed=7)
    encoder = _encoder()
    decoder = _decoder()

    z = encoder(snapshot)
    assert z.shape == (sum(NUM_NODES.values()), LATENT_DIM)
    reconstructed = decoder(z, snapshot)
    assert set(reconstructed) == set(NODE_TYPES)
    assert reconstructed["a"].shape == (NUM_NODES["a"], FEATURE_DIMS["a"])
    assert reconstructed["b"].shape == (NUM_NODES["b"], FEATURE_DIMS["b"])


def test_hgt_gradient_flows() -> None:
    """Backward pass yields nonzero grads on HGT parameters."""
    torch.manual_seed(11)
    snapshot = _typed_snapshot(seed=11)
    encoder = _encoder()
    decoder = _decoder()
    z = encoder(snapshot)
    reconstructed = decoder(z, snapshot)
    loss = reconstructed["a"].square().mean() + reconstructed["b"].square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in decoder.parameters()
    )


def test_hgt_mapping_input_matches_heterodata() -> None:
    """Feature-dict + type-local edge mapping matches HeteroData encode."""
    torch.manual_seed(3)
    snapshot = _typed_snapshot(seed=3)
    encoder = _encoder()
    encoder.eval()
    features = {name: snapshot[name].x for name in NODE_TYPES}
    edge_index = {edge_type: snapshot[edge_type].edge_index for edge_type in EDGE_TYPES}
    with torch.no_grad():
        z_data = encoder(snapshot)
        z_map = encoder(features, edge_index)
    assert torch.allclose(z_data, z_map)


def test_hgt_rejects_heads_not_dividing_width() -> None:
    """Constructor rejects latent widths not divisible by heads."""
    with pytest.raises(ValueError, match="divisible by heads"):
        HGTEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=5,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            heads=2,
            num_layers=1,
        )


def test_hgt_docstring_states_optional_not_required() -> None:
    """Honesty: HGT is optional and not required for hetero support."""
    doc = (HGTEncoder.__doc__ or "") + (HGTDecoder.__doc__ or "")
    assert "optional" in doc.lower()
    assert "not" in doc.lower() and "required" in doc.lower()
