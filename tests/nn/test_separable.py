"""Node-wise separable dictionary encoder / decoder."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.nn import (
    GNNEncoder,
    SeparableDictionaryDecoder,
    SeparableDictionaryEncoder,
    is_separable_dictionary,
)
from koopman_graph.nn.delay import DelayEmbeddingEncoder


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_separable_encoder_ignores_edges_and_permutes_with_nodes() -> None:
    """Node permutation permutes latents; swapping edges does not."""
    torch.manual_seed(0)
    encoder = SeparableDictionaryEncoder(3, 8, 4, num_layers=2)
    x = torch.randn(5, 3)
    path = _path_edge_index(5)
    cycle_src = list(range(5)) + list(range(1, 5)) + [0]
    cycle_dst = list(range(1, 5)) + [0] + list(range(5))
    cycle = torch.tensor([cycle_src, cycle_dst], dtype=torch.long)
    z_path = encoder(x, path)
    z_cycle = encoder(x, cycle)
    torch.testing.assert_close(z_path, z_cycle, atol=1e-6, rtol=0.0)
    perm = torch.tensor([2, 0, 4, 1, 3])
    z_perm = encoder(x[perm], path)
    torch.testing.assert_close(z_perm, z_path[perm], atol=1e-5, rtol=0.0)
    data = Data(x=x, edge_index=path)
    torch.testing.assert_close(encoder(data), z_path, atol=1e-6, rtol=0.0)


def test_gnn_encoder_is_not_separable() -> None:
    """Default GCN lifts are not a separable dictionary."""
    gnn = GNNEncoder(3, 8, 4, num_layers=1)
    assert not is_separable_dictionary(gnn)
    separable = SeparableDictionaryEncoder(3, 8, 4, num_layers=1)
    assert is_separable_dictionary(separable)
    wrapped = DelayEmbeddingEncoder(separable, n_delays=1)
    assert is_separable_dictionary(wrapped)
    assert separable.receptive_field_hops() == 0


def test_separable_decoder_round_trip_shape() -> None:
    """Decoder maps ``(N, d)`` to ``(N, F)`` without using edges."""
    torch.manual_seed(1)
    decoder = SeparableDictionaryDecoder(4, 8, 3, num_layers=1)
    z = torch.randn(6, 4)
    edges = _path_edge_index(6)
    out = decoder(z, edges)
    assert out.shape == (6, 3)
    torch.testing.assert_close(out, decoder(z, None), atol=0.0, rtol=0.0)


def test_separable_rejects_bad_feature_width() -> None:
    """Wrong channel width raises."""
    encoder = SeparableDictionaryEncoder(3, 8, 4, num_layers=1)
    with pytest.raises(ValueError, match="expects features"):
        encoder(torch.randn(4, 2))
    decoder = SeparableDictionaryDecoder(4, 8, 3, num_layers=1)
    with pytest.raises(ValueError, match="expects latents"):
        decoder(torch.randn(4, 2))


def test_separable_types_stay_off_root_all() -> None:
    """Separable dictionary types are ``nn`` exports only (Q6)."""
    assert "SeparableDictionaryEncoder" in koopman_graph.nn.__all__
    assert "SeparableDictionaryDecoder" in koopman_graph.nn.__all__
    assert "is_separable_dictionary" in koopman_graph.nn.__all__
    exported = set(koopman_graph.__all__)
    assert "SeparableDictionaryEncoder" not in exported
    assert "SeparableDictionaryDecoder" not in exported
    assert "is_separable_dictionary" not in exported
