"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.operators import (
    GraphKoopmanOperator,
)
from koopman_graph.operators.graph import _koopman_spectrum_from_eigenvalues


def test_graph_spectrum_certificate_and_distributed_inverse_guards() -> None:
    """Graph helpers validate time, assemble certificates, and reject tied inverse."""
    with pytest.raises(ValueError, match="time_step must be positive"):
        _koopman_spectrum_from_eigenvalues(torch.ones(1, dtype=torch.complex64), 0.0)

    graph = GraphKoopmanOperator(2, init_mode="identity")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    certificate = graph.stability_certificate(edge_index, 2, kind="schur")
    assert torch.isfinite(certificate.bound)

    distributed = GraphKoopmanOperator(
        2,
        init_mode="identity",
        sparsity="distributed",
        orbit_partition=((0, 1),),
    )
    z = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="requires a shared K_self"):
        distributed.inverse_advance(z, edge_index=edge_index)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        GraphKoopmanOperator(2, sparsity="distributed").inverse_advance(
            z,
            edge_index=edge_index,
            inverse_matrix=torch.eye(4),
        )
