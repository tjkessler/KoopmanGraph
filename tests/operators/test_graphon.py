"""Coverage and oracle tests for :mod:`koopman_graph.operators.graphon`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.operators import (
    MAX_GRAPHON_NODES,
    estimate_graphon,
    sample_graphon_adjacency,
)

# Constant-oracle abs tolerance: N=40, 12 graphs, p=0.4 gives ~9e3 Bernoulli
# trials; SE of the mean is about 0.005, so 0.05 is a conservative floor.
_CONSTANT_ABS = 0.05
# Product Pearson floor: degree scores on 16 mean-aggregated graphs at N=40.
_PRODUCT_PEARSON = 0.85


def _generator(seed: int) -> torch.Generator:
    """Return a CPU generator with ``seed``.

    Parameters
    ----------
    seed : int
        RNG seed.

    Returns
    -------
    torch.Generator
        Seeded generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _as_data(num_nodes: int, edge_index: Tensor) -> Data:
    """Wrap an adjacency as homogeneous ``Data``.

    Parameters
    ----------
    num_nodes : int
        Node count.
    edge_index : Tensor
        COO index.

    Returns
    -------
    Data
        Snapshot with a dummy feature column.
    """
    return Data(x=torch.zeros(num_nodes, 1), edge_index=edge_index)


def _pearson(left: Tensor, right: Tensor) -> float:
    """Return Pearson correlation of two 1-D tensors.

    Parameters
    ----------
    left, right : Tensor
        Equal-length vectors.

    Returns
    -------
    float
        Sample Pearson correlation.
    """
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denom = float(centered_left.norm() * centered_right.norm())
    if denom == 0.0:
        return 0.0
    return float((centered_left * centered_right).sum() / denom)


def test_graphon_product_kernel_and_small_n() -> None:
    """Product graphon samples; ``N < 2`` is rejected."""
    with pytest.raises(ValueError, match=">= 2"):
        sample_graphon_adjacency(1)
    edges = sample_graphon_adjacency(5, kernel="product")
    assert edges.shape[0] == 2


def test_sample_graphon_rejects_bad_density_and_positions_shape() -> None:
    """Constant density and latent shape are validated."""
    with pytest.raises(ValueError, match="density must lie"):
        sample_graphon_adjacency(4, density=1.5)
    with pytest.raises(ValueError, match="positions must have shape"):
        sample_graphon_adjacency(3, kernel="product", positions=torch.zeros(2))
    with pytest.raises(ValueError, match="unknown graphon"):
        sample_graphon_adjacency(4, kernel="bogus")


def test_estimate_graphon_rejects_unusable_snapshots() -> None:
    """Missing node count and out-of-range endpoints raise."""
    with pytest.raises(ValueError, match="cannot infer"):
        estimate_graphon([Data()], kernel_family="constant")
    graph = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 5], [1, 0]]),
    )
    with pytest.raises(ValueError, match="edge_index endpoints"):
        estimate_graphon([graph], kernel_family="constant")


def test_sample_graphon_rejects_positions_outside_unit_interval() -> None:
    """Latent coordinates must lie in ``[0, 1]``."""
    with pytest.raises(ValueError, match="positions must lie"):
        sample_graphon_adjacency(
            3,
            kernel="product",
            positions=torch.tensor([0.0, 0.5, 1.2]),
        )


def test_estimate_graphon_recovers_constant_density() -> None:
    """Mean off-diagonal density tracks a seeded constant graphon."""
    num_nodes = 40
    density = 0.4
    graphs = [
        _as_data(
            num_nodes,
            sample_graphon_adjacency(
                num_nodes,
                kernel="constant",
                density=density,
                generator=_generator(seed),
            ),
        )
        for seed in range(12)
    ]
    estimate = estimate_graphon(graphs, kernel_family="constant")
    assert estimate.kernel_family == "constant"
    assert estimate.num_nodes == num_nodes
    assert estimate.n_graphs == 12
    assert estimate.latent_scores is None
    assert estimate.density == pytest.approx(density, abs=_CONSTANT_ABS)
    upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
    assert float(
        estimate.probability_matrix[upper[0], upper[1]].mean()
    ) == pytest.approx(estimate.density, abs=1e-12)
    assert torch.diag(estimate.probability_matrix).abs().max() == pytest.approx(
        0.0, abs=1e-12
    )


def test_estimate_graphon_recovers_product_latents() -> None:
    """Degree scores correlate with fixed product-kernel positions."""
    num_nodes = 40
    positions = torch.linspace(0.2, 0.9, num_nodes)
    graphs = [
        _as_data(
            num_nodes,
            sample_graphon_adjacency(
                num_nodes,
                kernel="product",
                positions=positions,
                generator=_generator(seed),
            ),
        )
        for seed in range(16)
    ]
    estimate = estimate_graphon(graphs, kernel_family="product")
    assert estimate.latent_scores is not None
    corr = _pearson(estimate.latent_scores, positions.to(dtype=torch.float64))
    assert corr == pytest.approx(1.0, abs=1.0 - _PRODUCT_PEARSON)
    assert torch.diag(estimate.probability_matrix).abs().max() == pytest.approx(
        0.0, abs=1e-12
    )


def test_estimate_graphon_accepts_explicit_num_nodes() -> None:
    """``num_nodes`` on ``Data`` is used when ``x`` is absent."""
    graph = Data(
        num_nodes=4,
        edge_index=sample_graphon_adjacency(4, generator=_generator(3)),
    )
    estimate = estimate_graphon([graph], kernel_family="constant")
    assert estimate.num_nodes == 4


def test_estimate_graphon_low_rank_is_clipped_sketch() -> None:
    """Low-rank SVD is a teaching sketch, not an oracle claim."""
    num_nodes = 12
    graphs = [
        _as_data(
            num_nodes,
            sample_graphon_adjacency(
                num_nodes,
                kernel="constant",
                density=0.35,
                generator=_generator(0),
            ),
        )
    ]
    estimate = estimate_graphon(graphs, kernel_family="low_rank", rank=3)
    assert estimate.rank == 3
    assert estimate.factors is not None
    assert tuple(estimate.factors.shape) == (num_nodes, 3)
    assert float(estimate.probability_matrix.min()) >= 0.0
    assert float(estimate.probability_matrix.max()) <= 1.0
    assert torch.diag(estimate.probability_matrix).abs().max() == pytest.approx(
        0.0, abs=1e-12
    )


def test_estimate_graphon_refuses_mixed_and_unbounded_n() -> None:
    """Mixed ``N``, tiny ``N``, and the dense ceiling raise."""
    small = _as_data(4, sample_graphon_adjacency(4, generator=_generator(1)))
    other = _as_data(6, sample_graphon_adjacency(6, generator=_generator(2)))
    with pytest.raises(ValueError, match="shared N"):
        estimate_graphon([small, other], kernel_family="constant")
    singleton = Data(
        x=torch.zeros(1, 1),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    with pytest.raises(ValueError, match=">= 2"):
        estimate_graphon([singleton], kernel_family="constant")
    huge = Data(
        x=torch.zeros(MAX_GRAPHON_NODES + 1, 1),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="teaching ceiling"):
        estimate_graphon([huge], kernel_family="constant")
    with pytest.raises(ValueError, match="at least one graph"):
        estimate_graphon([], kernel_family="constant")
    with pytest.raises(ValueError, match="kernel_family"):
        estimate_graphon([small], kernel_family="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must be"):
        estimate_graphon([small], kernel_family="low_rank", rank=0)
    hetero = HeteroData()
    hetero["n"].x = torch.zeros(3, 1)
    with pytest.raises(TypeError, match="homogeneous Data"):
        estimate_graphon([hetero], kernel_family="constant")


def test_graphon_module_does_not_import_data_package() -> None:
    """Operators graphon helpers take PyG ``Data`` without importing ``data``."""
    import inspect

    import koopman_graph.operators.graphon as graphon_mod

    source = inspect.getsource(graphon_mod)
    assert "koopman_graph.data" not in source
