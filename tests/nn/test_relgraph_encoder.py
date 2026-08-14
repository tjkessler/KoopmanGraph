"""Tests for RelGraph encode / decode peers (multiplex R-GCN-lite)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph import RelGraphDecoder, RelGraphEncoder
from koopman_graph.nn import RelGraphConv
from koopman_graph.nn import RelGraphDecoder as NNRelGraphDecoder
from koopman_graph.nn.heterogeneous import RelGraphEncoder as NNRelGraphEncoder
from koopman_graph.serialization import (
    _build_decoder,
    _build_encoder,
    _decoder_type,
    _encoder_type,
)


def _multiplex_snapshot(
    *,
    num_nodes: int = 4,
    in_channels: int = 3,
    seed: int = 0,
) -> HeteroData:
    """Build a deterministic one-type, two-relation multiplex snapshot."""
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2, 3], [2, 0, 1]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot (rejected by RelGraph multiplex peers)."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_relgraph_encoder_decoder_shapes(activation: str) -> None:
    """Seeded encode→decode round-trip preserves multiplex shapes."""
    torch.manual_seed(7)
    in_channels = 3
    latent_dim = 5
    num_relations = 2
    snapshot = _multiplex_snapshot(in_channels=in_channels, seed=7)
    encoder = RelGraphEncoder(
        in_channels=in_channels,
        hidden_channels=8,
        latent_dim=latent_dim,
        num_relations=num_relations,
        activation=activation,  # type: ignore[arg-type]
    )
    decoder = RelGraphDecoder(
        latent_dim=latent_dim,
        hidden_channels=8,
        out_channels=in_channels,
        num_relations=num_relations,
        activation=activation,  # type: ignore[arg-type]
    )

    z = encoder(snapshot)
    assert z.shape == (snapshot["node"].num_nodes, latent_dim)
    edge_indices = [
        snapshot["node", "r1", "node"].edge_index,
        snapshot["node", "r2", "node"].edge_index,
    ]
    # HeteroData sorts edge types by repr; r1 before r2 for these keys.
    reconstructed = decoder(z, edge_indices)
    assert reconstructed.shape == (snapshot["node"].num_nodes, in_channels)

    z_tensor = encoder(snapshot["node"].x, edge_indices)
    assert z_tensor.shape == z.shape
    assert torch.allclose(z_tensor, z)


def test_relgraph_gradient_flow_both_relation_banks() -> None:
    """Gradients w.r.t. both relation weight banks are nonzero."""
    torch.manual_seed(11)
    snapshot = _multiplex_snapshot(seed=11)
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
    )
    decoder = RelGraphDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_relations=2,
    )
    z = encoder(snapshot)
    edge_indices = [
        snapshot["node", "r1", "node"].edge_index,
        snapshot["node", "r2", "node"].edge_index,
    ]
    out = decoder(z, edge_indices)
    out.sum().backward()

    for module in (encoder, decoder):
        for layer in module.convs:
            assert isinstance(layer, RelGraphConv)
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all()
            # Both relation banks must receive gradient signal.
            assert layer.weight.grad[0].abs().sum() > 0
            assert layer.weight.grad[1].abs().sum() > 0


def test_relgraph_rejects_typed_hetero() -> None:
    """Typed multi-node HeteroData is rejected (multiplex-only)."""
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=1,
    )
    with pytest.raises(ValueError, match="exactly one node type"):
        encoder(_typed_snapshot())


def test_relgraph_rejects_relation_count_mismatch() -> None:
    """Wrong number of relation banks raises a clear error."""
    snapshot = _multiplex_snapshot()
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=3,
    )
    with pytest.raises(ValueError, match="Expected 3 edge types"):
        encoder(snapshot)


def test_relgraph_activation_validation() -> None:
    """Invalid activation identifiers raise like GNN peers."""
    with pytest.raises(ValueError, match="activation"):
        RelGraphEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            activation="swish",  # type: ignore[arg-type]
        )


def test_relgraph_normalization_validation() -> None:
    """Unsupported normalization modes raise at construction."""
    with pytest.raises(ValueError, match="normalization"):
        RelGraphEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            normalization="symmetric",  # type: ignore[arg-type]
        )


def test_relgraph_checkpoint_type_hooks() -> None:
    """Serialization registries recognize RelGraph peers."""
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
        normalization="random_walk",
        root_weight=False,
    )
    decoder = RelGraphDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_relations=2,
        normalization="random_walk",
        root_weight=False,
    )
    assert _encoder_type(encoder) == "relgraph_enc"
    assert _decoder_type(decoder) == "relgraph_dec"

    rebuilt_enc = _build_encoder(
        {
            "type": "relgraph_enc",
            "in_channels": 3,
            "hidden_channels": 8,
            "latent_dim": 4,
            "num_layers": 2,
            "activation": "relu",
            "num_relations": 2,
            "normalization": "random_walk",
            "root_weight": False,
        }
    )
    rebuilt_dec = _build_decoder(
        {
            "type": "relgraph_dec",
            "latent_dim": 4,
            "hidden_channels": 8,
            "out_channels": 3,
            "num_layers": 2,
            "activation": "relu",
            "num_relations": 2,
            "normalization": "random_walk",
            "root_weight": False,
        }
    )
    assert isinstance(rebuilt_enc, RelGraphEncoder)
    assert isinstance(rebuilt_dec, RelGraphDecoder)
    assert rebuilt_enc.num_relations == 2
    assert rebuilt_enc.normalization == "random_walk"
    assert rebuilt_enc.root_weight is False
    assert rebuilt_dec.root_weight is False


def test_relgraph_exported_from_nn_and_root() -> None:
    """RelGraph peers are on the root façade and ``koopman_graph.nn``."""
    assert RelGraphEncoder is NNRelGraphEncoder
    assert RelGraphDecoder is NNRelGraphDecoder


def test_relgraph_tensor_mapping_banks_and_no_root() -> None:
    """Tensor path accepts mapping banks; ``root_weight=False`` still runs."""
    torch.manual_seed(3)
    snapshot = _multiplex_snapshot(seed=3)
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
        root_weight=False,
    )
    edge_index = {
        "r1": snapshot["node", "r1", "node"].edge_index,
        "r2": snapshot["node", "r2", "node"].edge_index,
    }
    z = encoder(snapshot["node"].x, edge_index)
    assert z.shape == (4, 4)
    assert torch.isfinite(z).all()


def test_relgraph_rejects_non_hetero_non_tensor() -> None:
    """Unsupported input types raise ``TypeError``."""
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
    )
    with pytest.raises(TypeError, match="Tensor or HeteroData"):
        encoder(object())  # type: ignore[arg-type]


def test_relgraph_rejects_missing_tensor_edge_banks() -> None:
    """Tensor input without relation banks raises ``ValueError``."""
    encoder = RelGraphEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
    )
    with pytest.raises(ValueError, match="edge_index relation banks"):
        encoder(torch.randn(4, 3))
