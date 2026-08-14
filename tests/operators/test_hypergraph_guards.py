"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator
from koopman_graph.training.pair_objectives import (
    _dense_networked_inverse_for_snapshot,
    topologies_equal,
)


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_hypergraph_distributed_inverse_paths() -> None:
    """Distributed hypergraph inversion rejects overrides and assembles its inverse."""
    operator = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        sparsity="distributed",
    )
    z = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    hyperedges = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        operator.inverse_advance(
            z,
            hyperedge_index=hyperedges,
            inverse_matrix=torch.eye(6),
        )
    recovered = operator.inverse_advance(z, hyperedge_index=hyperedges)
    torch.testing.assert_close(recovered, z)


def test_hypergraph_dual_and_effective_matrix_guards() -> None:
    """K_bwd / effective_matrix conflict paths on Zhou vs dual modes."""
    zhou = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="zhou_symmetric",
    )
    with pytest.raises(RuntimeError, match="K_bwd is only available"):
        _ = zhou.K_bwd
    with pytest.raises(ValueError, match="k_bwd is only valid"):
        zhou.set_dense_matrices(
            torch.eye(2),
            torch.zeros(2, 2),
            k_bwd=torch.eye(2),
        )

    dual = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="dual_random_walk",
    )
    # Single hyperarc 0 → {1, 2} (bipartite incidence; see directed tests).
    tail = torch.tensor([[0], [0]], dtype=torch.long)
    head = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="at most one of k_self"):
        dual.effective_matrix(
            None,
            3,
            tail_index=tail,
            head_index=head,
            k_self=torch.eye(2),
            k_self_blocks=torch.eye(2).unsqueeze(0).repeat(3, 1, 1),
        )
    matrix = dual.effective_matrix(None, 3, tail_index=tail, head_index=head)
    assert matrix.shape == (6, 6)
    spec = dual.spectrum(None, 3, tail_index=tail, head_index=head)
    assert spec.eigenvalues.numel() == 6


def test_hypergraph_forward_rw_inverse_and_dense_paths() -> None:
    """forward_random_walk spectrum/forward, inverse guards, dual bound_metric."""
    tail = torch.tensor([[0], [0]], dtype=torch.long)
    head = torch.tensor([[1], [0]], dtype=torch.long)
    z = torch.randn(2, 2)

    fwd = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="forward_random_walk",
    )
    with pytest.raises(ValueError, match="requires tail_index and head_index"):
        fwd.forward(z)
    out = fwd.forward(z, tail_index=tail, head_index=head)
    assert out.shape == z.shape
    matrix = fwd.effective_matrix(None, 2, tail_index=tail, head_index=head)
    assert matrix.shape == (4, 4)
    spec = fwd.spectrum(None, 2, tail_index=tail, head_index=head)
    assert spec.eigenvalues.numel() == 4

    # Empty directed incidence → empty 2-section for orbit binding.
    empty_t = torch.zeros(2, 0, dtype=torch.long)
    empty_h = torch.zeros(2, 0, dtype=torch.long)
    orbit_op = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="forward_random_walk",
        auto_orbits=True,
    )
    with patch(
        "koopman_graph.operators.orbit_ties.node_orbit_partition",
        return_value=((0,), (1,)),
    ):
        _ = orbit_op.forward(
            z,
            tail_index=empty_t,
            head_index=empty_h,
        )
    assert orbit_op._orbit_nbrs is not None
    assert orbit_op._hedge is orbit_op._orbit_nbrs[0]

    dual = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="dual_random_walk",
    )
    dual.set_dense_matrices(
        0.8 * torch.eye(2), 0.1 * torch.eye(2), k_bwd=0.05 * torch.eye(2)
    )
    dual.reset_parameters()
    assert dual.bound_metric().ndim == 0

    block = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="forward_random_walk",
        sparsity="block_diagonal",
    )
    with pytest.raises(
        ValueError, match="supports only incidence_mode='zhou_symmetric'"
    ):
        block.inverse_advance(z, tail_index=tail, head_index=head)
    zhou_block = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="zhou_symmetric",
        sparsity="block_diagonal",
    )
    with pytest.raises(ValueError, match="hyperedge_index is required"):
        zhou_block.inverse_advance(z)

    controlled = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        incidence_mode="zhou_symmetric",
        control_dim=1,
        control_mode="bilinear",
    )
    hyp = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled.inverse_advance(
            z,
            hyperedge_index=hyp,
            control=torch.ones(1, 1, 1),
        )


def test_topologies_equal_hyperedge_branches() -> None:
    """Cover weight/hyperedge presence and content mismatches."""
    edges = _path_edges(3)
    weights = torch.ones(edges.shape[1])
    hyp_a = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyp_b = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    hyp_w = torch.ones(1)

    assert not topologies_equal(edges, weights, edges, None)
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_index_b=None,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_index_b=hyp_b,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a,
        hyperedge_weight_b=None,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a,
        hyperedge_weight_b=2.0 * hyp_w,
    )
    assert topologies_equal(
        edges,
        None,
        edges.clone(),
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a.clone(),
        hyperedge_weight_b=hyp_w.clone(),
    )


def test_dense_networked_inverse_missing_hyperedge_returns_none() -> None:
    """Hypergraph inverse helper returns ``None`` without incidence."""
    op = HypergraphKoopmanOperator(latent_dim=2, sparsity="dense")
    snap = Data(x=torch.randn(3, 2), edge_index=_path_edges(3))
    assert _dense_networked_inverse_for_snapshot(op, snap) is None


def test_hypergraph_dense_effective_inverse_rejects_block_diagonal() -> None:
    """Hypergraph ``dense_effective_inverse`` requires ``sparsity='dense'``."""
    op = HypergraphKoopmanOperator(latent_dim=2, sparsity="block_diagonal")
    hyp = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        op.dense_effective_inverse(hyp, num_nodes=3)
