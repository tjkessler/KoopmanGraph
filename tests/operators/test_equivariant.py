"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.operators import (
    EquivariantKoopmanOperator,
)


def test_equivariant_operator_blocks_inverse_and_forward() -> None:
    """Scalar/vector operator covers validation, vector-only, inverse, forward."""
    with pytest.raises(ValueError, match="non-negative"):
        EquivariantKoopmanOperator(n_scalars=-1, n_vectors=1)
    with pytest.raises(ValueError, match="at least one"):
        EquivariantKoopmanOperator(n_scalars=0, n_vectors=0)
    vectors = EquivariantKoopmanOperator(n_scalars=0, n_vectors=1)
    z = torch.randn(2, 3)
    assert vectors.matrix.shape == (3, 3)
    assert vectors.bound_metric().ndim == 0
    assert vectors.inverse_advance(z).shape == z.shape
    assert vectors(z).shape == z.shape


def test_equivariant_block_operator() -> None:
    """Vector blocks are multiples of ``I_3``."""
    op = EquivariantKoopmanOperator(n_scalars=2, n_vectors=1)
    assert op.latent_dim == 5
    z = torch.randn(3, 5)
    out = op.advance(z)
    assert out.shape == z.shape
    matrix = op.matrix
    assert matrix.shape == (5, 5)
    vec_block = matrix[2:, 2:]
    assert torch.allclose(vec_block, op.vector_scales[0] * torch.eye(3), atol=1e-6)
