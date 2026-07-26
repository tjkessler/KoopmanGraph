"""Tests for Koopman spectral clustering (TASK-1312)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.analysis import ClusteringResult, koopman_spectral_clustering
from koopman_graph.analysis.clustering import (
    _canonicalize_columns,
    _embedding_from_effective_eigenvectors,
    _embedding_from_mode_shapes,
    _kmeans,
    _normalize_modes_matrix,
    _relabel_by_embedding_mean,
)
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


def test_canonicalize_rejects_non_2d_and_handles_complex() -> None:
    """Canonicalization validates rank and flips complex columns by real pivot."""
    with pytest.raises(ValueError, match="matrix must be 2D"):
        _canonicalize_columns(torch.tensor([1.0, 2.0]))
    empty_cols = _canonicalize_columns(torch.zeros(0, 2))
    assert empty_cols.shape == (0, 2)
    complex_matrix = torch.tensor([[1.0 + 0.0j, -2.0 + 0.0j]], dtype=torch.complex64)
    out = _canonicalize_columns(complex_matrix)
    assert out.dtype == torch.float32
    assert out[0, 1].item() == pytest.approx(2.0)


def test_normalize_modes_matrix_validation() -> None:
    """Mode matrices must be 2D with a matching ``n_modes`` axis."""
    with pytest.raises(ValueError, match="modes must be 2D"):
        _normalize_modes_matrix(torch.tensor([1.0, 2.0]), 2)
    with pytest.raises(ValueError, match="one axis equal to n_modes"):
        _normalize_modes_matrix(torch.ones(3, 3), 2)


def test_embedding_from_effective_eigenvectors_validation() -> None:
    """Effective eigenvector embedding checks divisibility and mode count."""
    eigenvectors = torch.randn(7, 4)
    with pytest.raises(ValueError, match="not divisible by num_nodes"):
        _embedding_from_effective_eigenvectors(eigenvectors, num_nodes=3, n_modes=2)
    with pytest.raises(ValueError, match="exceeds spectrum size"):
        _embedding_from_effective_eigenvectors(
            torch.randn(6, 2), num_nodes=3, n_modes=3
        )


def test_embedding_from_mode_shapes_validation_and_reduction() -> None:
    """Mode-shape reduction validates rank and averages signed channels."""
    with pytest.raises(ValueError, match="mode_shapes must have shape"):
        _embedding_from_mode_shapes(torch.ones(2, 3), 2)
    with pytest.raises(ValueError, match="provides 1 modes"):
        _embedding_from_mode_shapes(torch.ones(1, 4, 2), n_modes=2)
    shapes = torch.tensor(
        [
            [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]],
            [[-0.5, 0.5], [1.0, 1.0], [0.0, 2.0]],
        ]
    )
    embedding = _embedding_from_mode_shapes(shapes, n_modes=2)
    assert embedding.shape == (3, 2)
    assert embedding[0, 0].item() == pytest.approx(0.0)


def test_relabel_by_embedding_mean_skips_empty_clusters() -> None:
    """Relabeling tolerates missing cluster ids by sorting finite means."""
    labels = torch.tensor([0, 0, 2, 2])
    embedding = torch.tensor([[0.0], [1.0], [5.0], [6.0]])
    remapped = _relabel_by_embedding_mean(labels, embedding)
    assert remapped.tolist() == [0, 0, 1, 1]


def test_kmeans_validation_and_empty_cluster_reseed(monkeypatch) -> None:
    """K-means validates arguments and re-seeds empty clusters."""
    embedding = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [10.0, 10.0]])
    with pytest.raises(ValueError, match="embedding must be 2D"):
        _kmeans(torch.ones(3), 2, seed=0, max_iter=1)
    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        _kmeans(embedding, 0, seed=0, max_iter=1)
    with pytest.raises(ValueError, match="cannot exceed num_nodes"):
        _kmeans(embedding, 5, seed=0, max_iter=1)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        _kmeans(embedding, 2, seed=0, max_iter=0)

    seen_reseed = {"called": False}
    original_randint = torch.randint
    original_any = torch.any
    force_empty_once = {"done": False}

    def _randint(*args, **kwargs):
        seen_reseed["called"] = True
        return original_randint(*args, **kwargs)

    def _any(tensor, *args, **kwargs):
        result = original_any(tensor, *args, **kwargs)
        if (
            not force_empty_once["done"]
            and tensor.dtype == torch.bool
            and tensor.numel() == embedding.shape[0]
            and bool(result)
        ):
            force_empty_once["done"] = True
            return torch.tensor(False)
        return result

    monkeypatch.setattr(torch, "randint", _randint)
    monkeypatch.setattr(torch, "any", _any)
    labels, inertia = _kmeans(embedding, 3, seed=0, max_iter=10)
    assert labels.shape == (4,)
    assert inertia >= 0.0
    assert seen_reseed["called"]


def test_public_api_rejects_invalid_cluster_counts() -> None:
    """``n_clusters`` and ``n_modes`` must be positive."""
    spectrum = compute_spectrum(torch.eye(2), time_step=1.0)
    modes = torch.ones(2, 1)
    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        koopman_spectral_clustering(spectrum, 0, modes=modes)
    with pytest.raises(ValueError, match="n_modes must be >= 1"):
        koopman_spectral_clustering(spectrum, 1, n_modes=0, modes=modes)


def test_modes_exceeds_spectrum_size_with_spectrum_and_provider() -> None:
    """Explicit modes cannot request more columns than the resolved spectrum."""
    spectrum = compute_spectrum(torch.eye(2), time_step=1.0)
    modes = torch.ones(2, 3)
    with pytest.raises(ValueError, match="exceeds spectrum size"):
        koopman_spectral_clustering(spectrum, 2, n_modes=3, modes=modes)

    edge_index, _ = _two_community_edge_index(4)
    num_nodes = 8
    model = GraphKoopmanModel(
        encoder=GNNEncoder(1, 4, 1, num_layers=1),
        decoder=GNNDecoder(1, 4, 1, num_layers=1),
        latent_dim=1,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(torch.tensor([[0.5]]), torch.tensor([[0.4]]))
    with pytest.raises(ValueError, match="exceeds spectrum size"):
        koopman_spectral_clustering(
            model,
            2,
            n_modes=32,
            modes=torch.ones(num_nodes, 32),
            edge_index=edge_index,
            num_nodes=num_nodes,
        )


def _pernode_clustering_model(**kwargs: object) -> GraphKoopmanModel:
    """Return a matrix Koopman model with a ``decode`` alias for clustering."""
    params: dict[str, object] = {
        "encoder": GNNEncoder(3, 4, 2, num_layers=1),
        "decoder": GNNDecoder(2, 4, 3, num_layers=1),
        "latent_dim": 2,
        "time_step": 0.1,
    }
    params.update(kwargs)
    model = GraphKoopmanModel(**params)
    model.decode = model.decoder  # type: ignore[attr-defined]
    return model


def test_graph_model_requires_num_nodes() -> None:
    """Networked models need ``num_nodes`` for spectral clustering."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(1, 4, 1, num_layers=1),
        decoder=GNNDecoder(1, 4, 1, num_layers=1),
        latent_dim=1,
        time_step=0.1,
        koopman="graph",
    )
    with pytest.raises(ValueError, match="num_nodes is required"):
        koopman_spectral_clustering(model, n_clusters=2)


def test_pernode_decode_mode_shapes_path(synthetic_graph: Data) -> None:
    """Per-node models cluster via ``decode_mode_shapes`` when given a snapshot."""
    model = _pernode_clustering_model()
    result = koopman_spectral_clustering(
        model,
        n_clusters=2,
        n_modes=2,
        x_or_data=synthetic_graph,
        seed=0,
    )
    assert result.labels.shape == (synthetic_graph.num_nodes,)
    assert result.embedding.shape == (synthetic_graph.num_nodes, 2)


def test_pernode_path_requires_x_or_data() -> None:
    """Per-node clustering rejects missing reference snapshots."""
    model = _pernode_clustering_model(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
    )
    with pytest.raises(ValueError, match="x_or_data is required"):
        koopman_spectral_clustering(model, n_clusters=2)


def test_generic_provider_without_embedding_path_raises() -> None:
    """Bare spectrum providers must pass explicit modes or a full model."""

    class _SpectrumOnly:
        def spectrum(self) -> object:
            return compute_spectrum(torch.eye(2), time_step=1.0)

    with pytest.raises(ValueError, match="Unable to build a node embedding"):
        koopman_spectral_clustering(_SpectrumOnly(), n_clusters=2)
