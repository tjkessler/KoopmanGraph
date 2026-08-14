"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import (
    KoopmanOperator,
)
from koopman_graph.operators.discrete_propagation import (
    advance_step as disc_advance_step,
)
from koopman_graph.operators.discrete_propagation import (
    inverse_bilinear as disc_inverse_bilinear,
)
from koopman_graph.operators.discrete_propagation import (
    inverse_step as disc_inverse_step,
)


def test_discrete_set_dense_matrix_and_inverse_gaps() -> None:
    structured = KoopmanOperator(2, parameterization="schur")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        structured.set_dense_matrix(torch.eye(2))

    uncontrolled = KoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="Expected matrix shape"):
        uncontrolled.set_dense_matrix(torch.ones(2, 3))
    with pytest.raises(ValueError, match="control_matrix provided"):
        uncontrolled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(1, 2))
    uncontrolled._parameters.pop("K")
    with pytest.raises(AttributeError, match="K"):
        uncontrolled.set_dense_matrix(torch.eye(2))

    controlled = KoopmanOperator(2, control_dim=1, init_mode="identity")
    with pytest.raises(ValueError, match="control_matrix is required"):
        controlled.set_dense_matrix(torch.eye(2))
    with pytest.raises(ValueError, match="Expected control_matrix shape"):
        controlled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(2, 2))

    with pytest.raises(ValueError, match="Unknown init_mode"):
        KoopmanOperator(2, parameterization="schur", init_mode="nope")  # type: ignore[arg-type]

    op = KoopmanOperator(2, init_mode="identity")
    op.parameterization = "mystery"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown parameterization"):
        op._inverse_matrix()


def test_discrete_propagation_bilinear_inverse_branches() -> None:
    latent_dim = 4
    control_dim = 2
    matrix = 0.5 * torch.eye(latent_dim)
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    z = torch.randn(latent_dim)
    z_nodes = torch.randn(3, latent_dim)
    u = torch.randn(control_dim)
    u_nodes = torch.randn(3, control_dim)

    advanced = disc_advance_step(
        z,
        u,
        matrix=matrix,
        control_matrix=control_matrix,
        control_mode="bilinear",
        control_dim=control_dim,
        latent_dim=latent_dim,
        coupling=coupling,
    )
    assert advanced.shape == z.shape
    with pytest.raises(ValueError, match="bilinear coupling"):
        disc_advance_step(
            z,
            u,
            matrix=matrix,
            control_matrix=control_matrix,
            control_mode="bilinear",
            control_dim=control_dim,
            latent_dim=latent_dim,
            coupling=None,
        )

    assert (
        disc_inverse_bilinear(z, u, matrix=matrix, coupling=coupling).shape == z.shape
    )
    assert (
        disc_inverse_bilinear(z_nodes, u_nodes, matrix=matrix, coupling=coupling).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching"):
        disc_inverse_bilinear(z, u_nodes, matrix=matrix, coupling=coupling)
    with pytest.raises(ValueError, match="control input must have shape"):
        disc_inverse_bilinear(
            z,
            torch.randn(2, 2, control_dim),
            matrix=matrix,
            coupling=coupling,
        )
    singular_k = torch.zeros(latent_dim, latent_dim)
    singular_coupling = torch.zeros(control_dim, latent_dim, latent_dim)
    _ = disc_inverse_bilinear(
        z,
        torch.zeros(control_dim),
        matrix=singular_k,
        coupling=singular_coupling,
    )
    _ = disc_inverse_bilinear(
        z_nodes,
        torch.zeros(3, control_dim),
        matrix=singular_k,
        coupling=singular_coupling,
    )
    inverse_matrix = torch.linalg.inv(matrix)
    assert (
        disc_inverse_step(
            advanced,
            control=u,
            matrix=matrix,
            inverse_matrix=inverse_matrix,
            control_matrix=control_matrix,
            control_mode="bilinear",
            control_dim=control_dim,
            latent_dim=latent_dim,
            coupling=coupling,
        ).shape
        == z.shape
    )
