"""Tests for Koopman spectral clustering (TASK-1312)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from koopman_graph.analysis import ClusteringResult, koopman_spectral_clustering
from koopman_graph.analysis.clustering import _canonicalize_columns
from koopman_graph.graph_utils import dense_symmetric_normalized_adjacency
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.spectrum_types import compute_spectrum


def _adjusted_rand_index(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
    """Adjusted Rand index in ``[-1, 1]`` (1 = perfect agreement)."""
    y = labels_true.to(torch.long).reshape(-1)
    pred = labels_pred.to(torch.long).reshape(-1)
    if y.numel() != pred.numel():
        msg = "label tensors must have the same length"
        raise ValueError(msg)
    n = y.numel()
    if n < 2:
        return 1.0

    def _contingency(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n_a = int(a.max().item()) + 1
        n_b = int(b.max().item()) + 1
        table = torch.zeros(n_a, n_b, dtype=torch.float64)
        for i in range(n):
            table[int(a[i]), int(b[i])] += 1.0
        return table

    table = _contingency(y, pred)
    sum_comb_c = torch.sum(table * (table - 1) / 2).item()
    a_rows = table.sum(dim=1)
    b_cols = table.sum(dim=0)
    sum_comb_a = torch.sum(a_rows * (a_rows - 1) / 2).item()
    sum_comb_b = torch.sum(b_cols * (b_cols - 1) / 2).item()
    comb_n = n * (n - 1) / 2.0
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0
    return float((sum_comb_c - expected) / denom)


def _two_community_edge_index(n_per: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Two cliques with a single bridge edge; return edge_index and planted labels."""
    edges: list[list[int]] = []
    for offset in (0, n_per):
        nodes = list(range(offset, offset + n_per))
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                edges.extend([[src, dst], [dst, src]])
    # Weak bridge between communities.
    edges.extend([[n_per - 1, n_per], [n_per, n_per - 1]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    labels = torch.tensor([0] * n_per + [1] * n_per, dtype=torch.long)
    return edge_index, labels


def test_canonicalize_sign_is_stable() -> None:
    """Flipping a mode column then canonicalizing recovers the same embedding."""
    matrix = torch.tensor([[1.0, -2.0], [0.5, 0.1], [-3.0, 0.2]])
    flipped = matrix.clone()
    flipped[:, 0] = -flipped[:, 0]
    assert torch.allclose(_canonicalize_columns(matrix), _canonicalize_columns(flipped))


def test_planted_two_community_diffusion_ari() -> None:
    """Spectrum+modes path recovers planted communities with high ARI."""
    edge_index, planted = _two_community_edge_index(6)
    num_nodes = int(planted.numel())
    adj = dense_symmetric_normalized_adjacency(
        edge_index, num_nodes, dtype=torch.float32
    )
    # Diffusion-like discrete map on scalar node states.
    spectrum = compute_spectrum(adj, time_step=1.0)
    # Leading two magnitude-sorted modes as node coordinates (d=1).
    modes = spectrum.eigenvectors[:, :2].real
    result = koopman_spectral_clustering(
        spectrum,
        n_clusters=2,
        n_modes=2,
        modes=modes,
        seed=0,
    )
    assert isinstance(result, ClusteringResult)
    assert result.labels.shape == (num_nodes,)
    assert int(result.labels.unique().numel()) == 2
    assert result.eigen_indices == (0, 1)
    ari = _adjusted_rand_index(planted, result.labels)
    assert ari >= 0.8


def test_determinism_under_fixed_seed() -> None:
    """Same seed yields identical labels."""
    edge_index, planted = _two_community_edge_index(5)
    num_nodes = int(planted.numel())
    adj = dense_symmetric_normalized_adjacency(
        edge_index, num_nodes, dtype=torch.float32
    )
    spectrum = compute_spectrum(adj, time_step=1.0)
    modes = spectrum.eigenvectors[:, :2].real
    a = koopman_spectral_clustering(spectrum, 2, n_modes=2, modes=modes, seed=7)
    b = koopman_spectral_clustering(spectrum, 2, n_modes=2, modes=modes, seed=7)
    assert torch.equal(a.labels, b.labels)
    assert a.eigen_indices == (0, 1)


def test_graph_model_path_with_topology() -> None:
    """Networked model path uses resolve_spectrum + effective eigenvectors."""
    edge_index, planted = _two_community_edge_index(5)
    num_nodes = int(planted.numel())
    latent_dim = 1
    encoder = GNNEncoder(
        in_channels=1,
        hidden_channels=4,
        latent_dim=latent_dim,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=latent_dim,
        hidden_channels=4,
        out_channels=1,
        num_layers=1,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(
        torch.tensor([[0.55]]),
        torch.tensor([[0.4]]),
    )
    result = koopman_spectral_clustering(
        model,
        n_clusters=2,
        n_modes=2,
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=0,
    )
    assert result.labels.shape == (num_nodes,)
    assert int(result.labels.unique().numel()) == 2
    ari = _adjusted_rand_index(planted, result.labels)
    assert ari >= 0.8


def test_spectrum_without_modes_raises() -> None:
    """Bare KoopmanSpectrum requires an explicit modes matrix."""
    eye = torch.eye(3)
    spectrum = compute_spectrum(eye, time_step=1.0)
    with pytest.raises(ValueError, match="modes is required"):
        koopman_spectral_clustering(spectrum, n_clusters=2)


def test_clustering_result_is_frozen() -> None:
    """ClusteringResult rejects attribute assignment."""
    modes = torch.tensor([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.95, 0.05]])
    eye = torch.eye(4)
    spectrum = compute_spectrum(0.5 * eye + 0.5 * torch.ones(4, 4) / 4, time_step=1.0)
    result = koopman_spectral_clustering(spectrum, 2, n_modes=2, modes=modes, seed=0)
    with pytest.raises(FrozenInstanceError):
        result.inertia = 0.0  # type: ignore[misc]


def test_modes_orientation_transpose_accepted() -> None:
    """``(n_modes, N)`` mode matrices are accepted and transposed."""
    edge_index, planted = _two_community_edge_index(4)
    num_nodes = int(planted.numel())
    adj = dense_symmetric_normalized_adjacency(
        edge_index, num_nodes, dtype=torch.float32
    )
    spectrum = compute_spectrum(adj, time_step=1.0)
    modes_tn = spectrum.eigenvectors[:, :2].real.T  # (2, N)
    result = koopman_spectral_clustering(
        spectrum,
        n_clusters=2,
        n_modes=2,
        modes=modes_tn,
        seed=0,
    )
    assert result.embedding.shape == (num_nodes, 2)
    assert int(result.labels.unique().numel()) == 2
