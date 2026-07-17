"""Unit tests for GNNEncoder."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.nn import GATEncoder, GNNEncoder


def test_forward_with_data_object(synthetic_graph: Data) -> None:
    """Verify forward accepts a PyG ``Data`` object."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph)
    assert out.shape == (5, 4)


def test_forward_with_tensor_inputs(synthetic_graph: Data) -> None:
    """Verify forward accepts separate tensor inputs."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph.x, synthetic_graph.edge_index)
    assert out.shape == (5, 4)


def test_single_layer_output_shape() -> None:
    """Verify output shape with a single GNN layer."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 3)
    encoder = GNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=6,
        num_layers=1,
    )
    out = encoder(x, edge_index)
    assert out.shape == (2, 6)


def test_multi_layer_output_shape() -> None:
    """Verify output shape with multiple GNN layers."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 2)
    encoder = GNNEncoder(
        in_channels=2,
        hidden_channels=16,
        latent_dim=8,
        num_layers=3,
    )
    out = encoder(x, edge_index)
    assert out.shape == (3, 8)


@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_activation_options(activation: str, synthetic_graph: Data) -> None:
    """Verify each supported activation produces finite outputs."""
    encoder = GNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        activation=activation,  # type: ignore[arg-type]
    )
    out = encoder(synthetic_graph)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()


def test_permutation_equivariance(synthetic_graph: Data) -> None:
    """Verify outputs are equivariant to node permutations."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    encoder.eval()

    perm = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel())

    permuted_graph = Data(
        x=synthetic_graph.x[perm],
        edge_index=inv_perm[synthetic_graph.edge_index],
    )

    out_original = encoder(synthetic_graph)
    out_permuted = encoder(permuted_graph)
    assert torch.allclose(out_original, out_permuted[inv_perm], atol=1e-5)


def test_gradient_flow(synthetic_graph: Data) -> None:
    """Verify gradients flow through the operator forward pass."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph)
    loss = out.sum()
    loss.backward()
    for param in encoder.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_missing_edge_index_raises() -> None:
    """Verify missing ``edge_index`` raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    x = torch.randn(5, 3)
    with pytest.raises(ValueError, match="edge_index is required"):
        encoder(x)


def test_invalid_num_layers_raises() -> None:
    """Verify non-positive ``num_layers`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="num_layers must be positive"):
        GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=0)


def test_invalid_in_channels_raises() -> None:
    """Verify non-positive ``in_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="in_channels must be positive"):
        GNNEncoder(in_channels=0, hidden_channels=8, latent_dim=4)


def test_invalid_hidden_channels_raises() -> None:
    """Verify non-positive ``hidden_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        GNNEncoder(in_channels=3, hidden_channels=0, latent_dim=4)


def test_invalid_latent_dim_raises() -> None:
    """Verify non-positive ``latent_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=0)


def test_invalid_input_rank_raises(synthetic_graph: Data) -> None:
    """Verify non-matrix node input raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    bad_x = torch.randn(5, 3, 1)
    with pytest.raises(ValueError, match="Expected x with shape"):
        encoder(bad_x, synthetic_graph.edge_index)


def test_resolve_activation_unknown_raises() -> None:
    """Verify unknown activation names raise ``ValueError``."""
    from koopman_graph.nn.gnn import _resolve_activation

    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_activation("leaky_relu")  # type: ignore[arg-type]


def test_invalid_feature_dim_raises(synthetic_graph: Data) -> None:
    """Verify invalid feature dimension raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    bad_x = torch.randn(5, 2)
    with pytest.raises(ValueError, match="Expected in_channels=3"):
        encoder(bad_x, synthetic_graph.edge_index)


def test_exported_from_package() -> None:
    """Verify the symbol is exported from the package root."""
    from koopman_graph import GNNEncoder as ExportedEncoder

    assert ExportedEncoder is GNNEncoder


def test_gat_forward_with_data_object(synthetic_graph: Data) -> None:
    """Verify GAT forward accepts a PyG ``Data`` object."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph)
    assert out.shape == (5, 4)


def test_gat_forward_with_tensor_inputs(synthetic_graph: Data) -> None:
    """Verify GAT forward accepts separate tensor inputs."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph.x, synthetic_graph.edge_index)
    assert out.shape == (5, 4)


def test_gat_multi_layer_output_shape() -> None:
    """Verify GAT output shape with multiple layers."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 2)
    encoder = GATEncoder(
        in_channels=2,
        hidden_channels=16,
        latent_dim=8,
        num_layers=3,
    )
    out = encoder(x, edge_index)
    assert out.shape == (3, 8)


def test_gat_single_layer_output_shape() -> None:
    """Verify GAT output shape with a single layer."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 3)
    encoder = GATEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=6,
        num_layers=1,
    )
    out = encoder(x, edge_index)
    assert out.shape == (2, 6)


def test_gat_invalid_in_channels_raises() -> None:
    """Verify non-positive GAT ``in_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="in_channels must be positive"):
        GATEncoder(in_channels=0, hidden_channels=8, latent_dim=4)


def test_gat_invalid_hidden_channels_raises() -> None:
    """Verify non-positive GAT ``hidden_channels`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        GATEncoder(in_channels=3, hidden_channels=0, latent_dim=4)


def test_gat_invalid_latent_dim_raises() -> None:
    """Verify non-positive GAT ``latent_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        GATEncoder(in_channels=3, hidden_channels=8, latent_dim=0)


def test_gat_invalid_num_layers_raises() -> None:
    """Verify non-positive GAT ``num_layers`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="num_layers must be positive"):
        GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=0)


def test_gat_invalid_dropout_raises() -> None:
    """Verify out-of-range GAT ``dropout`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="dropout must be in"):
        GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4, dropout=1.5)


def test_gat_missing_edge_index_raises() -> None:
    """Verify missing GAT ``edge_index`` raises ``ValueError``."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    x = torch.randn(5, 3)
    with pytest.raises(ValueError, match="edge_index is required"):
        encoder(x)


def test_gat_invalid_input_rank_raises(synthetic_graph: Data) -> None:
    """Verify non-matrix GAT input raises ``ValueError``."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    bad_x = torch.randn(5, 3, 1)
    with pytest.raises(ValueError, match="Expected x with shape"):
        encoder(bad_x, synthetic_graph.edge_index)


def test_gat_invalid_feature_dim_raises(synthetic_graph: Data) -> None:
    """Verify invalid GAT feature dimension raises ``ValueError``."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    bad_x = torch.randn(5, 2)
    with pytest.raises(ValueError, match="Expected in_channels=3"):
        encoder(bad_x, synthetic_graph.edge_index)


def test_gat_gradient_flow(synthetic_graph: Data) -> None:
    """Verify gradients flow through the GAT encoder."""
    encoder = GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    out = encoder(synthetic_graph)
    loss = out.sum()
    loss.backward()
    for param in encoder.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_gat_invalid_heads_raises() -> None:
    """Verify non-positive ``heads`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="heads must be positive"):
        GATEncoder(in_channels=3, hidden_channels=8, latent_dim=4, heads=0)


def test_gat_exported_from_package() -> None:
    """Verify ``GATEncoder`` is exported from the package root."""
    from koopman_graph import GATEncoder as ExportedGATEncoder

    assert ExportedGATEncoder is GATEncoder


def test_weighted_vs_unweighted_outputs_differ() -> None:
    """Verify scalar edge weights change GCN encoder outputs."""
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_weight = torch.tensor([2.0, 0.5, 0.5, 2.0])
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    encoder.eval()
    out_unweighted = encoder(x, edge_index)
    out_weighted = encoder(x, edge_index, edge_weight)
    assert not torch.allclose(out_unweighted, out_weighted)
