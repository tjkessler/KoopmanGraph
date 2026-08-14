"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import (
    GraphKoopmanOperator,
)
from koopman_graph.operators.graph_inverse import block_diagonal_graph_inverse_advance


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_graph_inverse_and_operator_adjacency_guards() -> None:
    """Inverse helper / dual reset / bound_metric / set_dense k_bwd guards."""
    edge_index = _path_edges(3)
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="k_bwd is required"):
        block_diagonal_graph_inverse_advance(
            z,
            k_self=torch.eye(2),
            k_nbr=0.1 * torch.eye(2),
            edge_index=edge_index,
            adjacency="dual_random_walk",
        )
    with pytest.raises(ValueError, match="adjacency must be"):
        block_diagonal_graph_inverse_advance(
            z,
            k_self=torch.eye(2),
            k_nbr=0.1 * torch.eye(2),
            edge_index=edge_index,
            adjacency="bogus",  # type: ignore[arg-type]
        )

    dual = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    dual.reset_parameters()
    assert torch.isfinite(dual.bound_metric())

    plain = GraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="k_bwd is only valid"):
        plain.set_dense_matrices(
            0.8 * torch.eye(2), 0.1 * torch.eye(2), k_bwd=0.05 * torch.eye(2)
        )
