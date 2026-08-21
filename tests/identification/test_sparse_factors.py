"""Sparse graph-factor identification versus dense joint LS and SINDy."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT

import koopman_graph
import koopman_graph.identification as identification
from koopman_graph.analysis import identify_sparse_dynamics
from koopman_graph.identification import (
    SparseFactorReport,
    identify_sparse_graph_factors,
)
from koopman_graph.losses import KoopmanSparsityLoss

_SOURCE = REPO_ROOT / "src" / "koopman_graph" / "identification" / "sparse_factors.py"


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected path edges on ``num_nodes`` nodes.

    Parameters
    ----------
    num_nodes : int
        Path length.

    Returns
    -------
    Tensor
        COO index with shape ``(2, 2*(num_nodes-1))``.
    """
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def _rollout_pairs(
    k_self: torch.Tensor,
    k_nbr: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    num_times: int,
    seed: int,
) -> torch.Tensor:
    """Build consecutive encodings from the one-tap graph map.

    Parameters
    ----------
    k_self, k_nbr : Tensor
        True factors ``(d, d)``.
    adjacency : Tensor
        Dense ``Â``.
    num_times : int
        Length ``T``.
    seed : int
        RNG seed for the initial snapshot.

    Returns
    -------
    Tensor
        Encodings with shape ``(T, N, d)``.
    """
    generator = torch.Generator(device=k_self.device)
    generator.manual_seed(seed)
    num_nodes = int(adjacency.shape[0])
    latent_dim = int(k_self.shape[0])
    state = torch.randn(
        num_nodes,
        latent_dim,
        dtype=k_self.dtype,
        generator=generator,
    )
    trajectory = [state]
    for _ in range(num_times - 1):
        neighbor = adjacency @ state
        state = state @ k_self.T + neighbor @ k_nbr.T
        trajectory.append(state)
    return torch.stack(trajectory, dim=0)


def _dense_adjacency(num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """Symmetric-normalized path adjacency used by the identifier.

    Parameters
    ----------
    num_nodes : int
        Path length.
    dtype : torch.dtype
        Floating dtype.

    Returns
    -------
    Tensor
        Dense ``Â``.
    """
    from koopman_graph.graph_utils.topology import (
        dense_symmetric_normalized_adjacency,
    )

    return dense_symmetric_normalized_adjacency(
        _path_edge_index(num_nodes),
        num_nodes,
        dtype=dtype,
    )


def _dense_random_walk_adjacency(num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """Forward random-walk normalized path adjacency.

    Parameters
    ----------
    num_nodes : int
        Path length.
    dtype : torch.dtype
        Floating dtype.

    Returns
    -------
    Tensor
        Dense ``Â``.
    """
    from koopman_graph.graph_utils.topology import (
        dense_random_walk_normalized_adjacency,
    )

    return dense_random_walk_normalized_adjacency(
        _path_edge_index(num_nodes),
        num_nodes,
        dtype=dtype,
        direction="forward",
    )


def test_sparse_factors_exported_and_not_on_root_all() -> None:
    """Sparse-factor helpers are identification exports, not root symbols."""
    for name in ("SparseFactorReport", "identify_sparse_graph_factors"):
        assert name in identification.__all__
        assert name not in set(koopman_graph.__all__)
        assert not hasattr(koopman_graph, name)
    text = _SOURCE.read_text(encoding="utf-8")
    assert "10.1073/pnas.1517384113" in text
    assert "Brunton2016SINDy" in text
    assert "10.1017/jfm.2021.271" in text
    assert "Pan2021SparseSubspace" in text
    assert "identify_sparse_dynamics" in text
    assert "KoopmanSparsityLoss" in text
    assert "multi-task" in text


def test_stlsq_recovers_sparse_factors_sparser_than_dense_ls() -> None:
    """Elementwise STLSQ matches a sparse oracle and zeros more than dense LS.

    Independent construction of ``Z_{t+1} = Z_t K_self^T + (Â Z_t) K_nbr^T``
    on a path; float64 residual of that linear map, so ``rtol`` ``1e-8`` /
    ``atol`` ``1e-10`` on recovered factors.
    """
    dtype = torch.float64
    num_nodes = 6
    k_self = torch.tensor(
        [[0.50, 0.20, 0.00], [0.00, 0.40, 0.00], [0.00, 0.00, 0.30]],
        dtype=dtype,
    )
    k_nbr = torch.tensor(
        [[0.00, 0.15, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]],
        dtype=dtype,
    )
    adjacency = _dense_adjacency(num_nodes, dtype)
    z_pairs = _rollout_pairs(k_self, k_nbr, adjacency, num_times=24, seed=0)
    edges = _path_edge_index(num_nodes)
    dense = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="none",
        method="stlsq",
        threshold=0.0,
    )
    sparse = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="none",
        method="stlsq",
        threshold=0.05,
    )
    assert sparse.nnz < dense.nnz
    assert sparse.nnz == 5
    torch.testing.assert_close(sparse.K_self, k_self, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(sparse.K_nbr, k_nbr, rtol=1e-8, atol=1e-10)
    assert sparse.residual == pytest.approx(0.0, abs=1e-12)
    assert sparse.method == "stlsq"
    assert sparse.group == "none"


def test_self_nbr_group_zeros_true_zero_neighbor_factor() -> None:
    """``group='self_nbr'`` drops a whole factor whose Frobenius norm is small.

    Oracle ``K_nbr = 0``; threshold sits between the two factor norms of
    unconstrained least squares. Construction residual, so recovered
    ``K_self`` uses ``rtol`` ``1e-8`` / ``atol`` ``1e-10``.
    """
    dtype = torch.float64
    k_self = torch.diag(torch.tensor([0.8, 0.6, 0.5], dtype=dtype))
    k_nbr = torch.zeros(3, 3, dtype=dtype)
    adjacency = _dense_adjacency(5, dtype)
    z_pairs = _rollout_pairs(k_self, k_nbr, adjacency, num_times=20, seed=1)
    edges = _path_edge_index(5)
    dense = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="self_nbr",
        method="stlsq",
        threshold=0.0,
    )
    neighbor_norm = float(dense.K_nbr.norm().item())
    self_norm = float(dense.K_self.norm().item())
    assert neighbor_norm < 0.5 * self_norm
    cutoff = 0.5 * (neighbor_norm + self_norm)
    report = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="self_nbr",
        method="stlsq",
        threshold=cutoff,
    )
    assert torch.equal(report.K_nbr, torch.zeros_like(k_nbr))
    torch.testing.assert_close(report.K_self, k_self, rtol=1e-8, atol=1e-10)


def test_orbit_group_and_group_lasso_smoke() -> None:
    """Latent-row grouping and ISTA group-lasso return finite refit factors."""
    dtype = torch.float64
    k_self = torch.diag(torch.tensor([0.7, 0.4, 0.0], dtype=dtype))
    k_nbr = torch.zeros(3, 3, dtype=dtype)
    adjacency = _dense_adjacency(4, dtype)
    z_pairs = _rollout_pairs(k_self, k_nbr, adjacency, num_times=16, seed=2)
    edges = _path_edge_index(4)
    orbit = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="orbit",
        method="stlsq",
        threshold=0.2,
    )
    assert orbit.K_self.shape == (3, 3)
    assert torch.isfinite(orbit.K_self).all()
    lasso = identify_sparse_graph_factors(
        z_pairs,
        edges,
        group="self_nbr",
        method="group_lasso",
        threshold=0.05,
        max_iter=80,
    )
    assert lasso.method == "group_lasso"
    assert torch.isfinite(lasso.K_self).all()
    assert lasso.nnz <= 2 * 9
    walk_adj = _dense_random_walk_adjacency(4, dtype)
    walk_pairs = _rollout_pairs(k_self, k_nbr, walk_adj, num_times=16, seed=3)
    walk = identify_sparse_graph_factors(
        walk_pairs,
        edges,
        group="none",
        method="stlsq",
        threshold=0.05,
        adjacency="random_walk",
    )
    assert walk.n_samples == 15 * 4
    torch.testing.assert_close(walk.K_self, k_self, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(walk.K_nbr, k_nbr, rtol=1e-8, atol=1e-10)


def test_sparse_factors_distinct_from_sindy_and_l1_loss() -> None:
    """Graph-factor ID is not latent SINDy and not a training L1 penalty."""
    assert identify_sparse_graph_factors is not identify_sparse_dynamics
    assert SparseFactorReport is not type(KoopmanSparsityLoss)
    assert not hasattr(identify_sparse_graph_factors, "library")


def test_sparse_factor_guards() -> None:
    """Invalid layouts, methods, and dual adjacency raise with the constraint."""
    z_pairs = torch.randn(4, 3, 2, dtype=torch.float64)
    edges = _path_edge_index(3)
    with pytest.raises(ValueError, match="T >= 2"):
        identify_sparse_graph_factors(
            z_pairs[:1],
            edges,
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="group must be one of"):
        identify_sparse_graph_factors(
            z_pairs,
            edges,
            group="nodes",  # type: ignore[arg-type]
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="method must be one of"):
        identify_sparse_graph_factors(
            z_pairs,
            edges,
            method="omp",  # type: ignore[arg-type]
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="dual random-walk is out of scope"):
        identify_sparse_graph_factors(
            z_pairs,
            edges,
            adjacency="dual_random_walk",  # type: ignore[arg-type]
            threshold=0.1,
        )
    with pytest.raises(ValueError, match="threshold must be a finite"):
        identify_sparse_graph_factors(z_pairs, edges, threshold=-0.1)
    with pytest.raises(ValueError, match="z_pairs must have shape"):
        identify_sparse_graph_factors(torch.randn(5, 2), edges, threshold=0.1)
