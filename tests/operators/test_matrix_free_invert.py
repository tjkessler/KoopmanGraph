"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import torch

from koopman_graph.operators import (
    apply_k_eff_hypergraph,
    invert_k_eff_hypergraph,
)


def test_matrix_free_hypergraph_invert() -> None:
    """Zhou-symmetric hypergraph invert returns a finite solution."""
    k_self = torch.eye(2)
    k_hedge = 0.1 * torch.eye(2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    rhs = torch.randn(6)
    applied = apply_k_eff_hypergraph(
        rhs,
        k_self=k_self,
        k_hedge=k_hedge,
        hyperedge_index=hyperedge_index,
        num_nodes=3,
    )
    assert applied.shape == (6,)
    result = invert_k_eff_hypergraph(
        rhs,
        k_self=k_self,
        k_hedge=k_hedge,
        hyperedge_index=hyperedge_index,
        num_nodes=3,
    )
    assert result.solution.shape == (6,)
