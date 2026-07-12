"""Unit tests for GNNDecoder."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.decoder import GNNDecoder


def test_forward_with_data_object(synthetic_graph: Data) -> None:
    """Verify forward accepts a PyG ``Data`` object."""
    z = torch.randn(5, 4)
    graph = Data(x=z, edge_index=synthetic_graph.edge_index)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    out = decoder(graph)
    assert out.shape == (5, 3)


def test_forward_with_tensor_inputs(synthetic_graph: Data) -> None:
    """Verify forward accepts separate tensor inputs."""
    z = torch.randn(5, 4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    out = decoder(z, synthetic_graph.edge_index)
    assert out.shape == (5, 3)


def test_single_layer_output_shape() -> None:
    """Verify output shape with a single GNN layer."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    z = torch.randn(2, 6)
    decoder = GNNDecoder(
        latent_dim=6,
        hidden_channels=8,
        out_channels=3,
        num_layers=1,
    )
    out = decoder(z, edge_index)
    assert out.shape == (2, 3)


def test_multi_layer_output_shape() -> None:
    """Verify output shape with multiple GNN layers."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    z = torch.randn(3, 8)
    decoder = GNNDecoder(
        latent_dim=8,
        hidden_channels=16,
        out_channels=2,
        num_layers=3,
    )
    out = decoder(z, edge_index)
    assert out.shape == (3, 2)


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_activation_options(activation: str, synthetic_graph: Data) -> None:
    """Verify each supported activation produces finite outputs."""
    z = torch.randn(5, 4)
    decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        activation=activation,  # type: ignore[arg-type]
    )
    out = decoder(z, synthetic_graph.edge_index)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_permutation_equivariance(synthetic_graph: Data) -> None:
    """Verify outputs are equivariant to node permutations."""
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    decoder.eval()

    z = torch.randn(5, 4)
    perm = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel())

    permuted_graph = Data(
        x=z[perm],
        edge_index=inv_perm[synthetic_graph.edge_index],
    )
    original_graph = Data(x=z, edge_index=synthetic_graph.edge_index)

    out_original = decoder(original_graph)
    out_permuted = decoder(permuted_graph)
    assert torch.allclose(out_original, out_permuted[inv_perm], atol=1e-5)


def test_gradient_flow(synthetic_graph: Data) -> None:
    """Verify gradients flow through the operator forward pass."""
    z = torch.randn(5, 4, requires_grad=True)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    out = decoder(z, synthetic_graph.edge_index)
    loss = out.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    for param in decoder.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_missing_edge_index_raises() -> None:
    """Verify missing ``edge_index`` raises ``ValueError``."""
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    z = torch.randn(5, 4)
    with pytest.raises(ValueError, match="edge_index is required"):
        decoder(z)


def test_invalid_num_layers_raises() -> None:
    """Verify non-positive ``num_layers`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="num_layers must be positive"):
        GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=0)


def test_invalid_latent_dim_raises() -> None:
    """Verify non-positive ``latent_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        GNNDecoder(latent_dim=0, hidden_channels=8, out_channels=3)


def test_invalid_hidden_channels_raises() -> None:
    """Verify non-positive ``hidden_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        GNNDecoder(latent_dim=4, hidden_channels=0, out_channels=3)


def test_invalid_out_channels_raises() -> None:
    """Verify non-positive ``out_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="out_channels must be positive"):
        GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=0)


def test_invalid_input_rank_raises(synthetic_graph: Data) -> None:
    """Verify non-matrix latent input raises ``ValueError``."""
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    bad_z = torch.randn(5, 4, 1)
    with pytest.raises(ValueError, match="Expected x with shape"):
        decoder(bad_z, synthetic_graph.edge_index)


def test_invalid_feature_dim_raises(synthetic_graph: Data) -> None:
    """Verify invalid feature dimension raises ``ValueError``."""
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    bad_z = torch.randn(5, 2)
    with pytest.raises(ValueError, match="Expected latent_dim=4"):
        decoder(bad_z, synthetic_graph.edge_index)


def test_exported_from_package() -> None:
    """Verify the symbol is exported from the package root."""
    from koopman_graph import GNNDecoder as ExportedDecoder

    assert ExportedDecoder is GNNDecoder


def test_weighted_vs_unweighted_outputs_differ() -> None:
    """Verify scalar edge weights change GCN decoder outputs."""
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_weight = torch.tensor([2.0, 0.5, 0.5, 2.0])
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    decoder.eval()
    out_unweighted = decoder(z, edge_index)
    out_weighted = decoder(z, edge_index, edge_weight)
    assert not torch.allclose(out_unweighted, out_weighted)
