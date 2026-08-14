"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import (
    ContinuousKoopmanOperator,
)
from koopman_graph.operators import (
    matrix_log,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.continuous_parameterizations import (
    max_real_part_of_generator,
    reset_dissipative_generator,
    reset_lyapunov_generator,
    reset_odo_generator,
    reset_schur_generator,
)


def test_matrix_log_preserves_complex_dtype() -> None:
    matrix = torch.eye(2, dtype=torch.complex64) * (0.5 + 0.1j)
    result = matrix_log(matrix)
    assert result.is_complex()


def test_van_loan_generator_rejects_nonpositive_delta_t() -> None:
    with pytest.raises(ValueError, match="delta_t must be positive"):
        van_loan_generator_from_discrete(torch.eye(2), torch.zeros(1, 2), 0.0)


def test_continuous_operator_constructor_validation() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ContinuousKoopmanOperator(0)
    with pytest.raises(ValueError, match="init_scale must be non-negative"):
        ContinuousKoopmanOperator(2, init_scale=-0.1)
    with pytest.raises(ValueError, match="max_real_eigenvalue must be positive"):
        ContinuousKoopmanOperator(2, max_real_eigenvalue=0.0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        ContinuousKoopmanOperator(2, control_dim=-1)
    with pytest.raises(ValueError, match="Unknown parameterization"):
        ContinuousKoopmanOperator(2, parameterization="not-a-mode")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "parameterization",
    ["dense", "odo", "schur", "dissipative", "lyapunov"],
)
def test_continuous_xavier_init_modes(parameterization: str) -> None:
    op = ContinuousKoopmanOperator(
        3,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="xavier",
    )
    assert op.L.shape == (3, 3)


@pytest.mark.parametrize(
    "parameterization",
    ["dense", "odo", "schur", "lyapunov"],
)
def test_continuous_unknown_init_mode(parameterization: str) -> None:
    with pytest.raises(ValueError, match="Unknown init_mode"):
        ContinuousKoopmanOperator(
            2,
            parameterization=parameterization,  # type: ignore[arg-type]
            init_mode="not-real",  # type: ignore[arg-type]
        )


def test_continuous_reset_control_parameters_noop_without_control() -> None:
    op = ContinuousKoopmanOperator(2, control_dim=0)
    op.reset_control_parameters()


def test_continuous_control_term_and_advance_validation() -> None:
    uncontrolled = ContinuousKoopmanOperator(3, init_mode="identity")
    with pytest.raises(ValueError, match="control_term requires control_dim > 0"):
        uncontrolled.control_term(torch.ones(1))

    controlled = ContinuousKoopmanOperator(
        3, control_dim=2, init_mode="identity", parameterization="dense"
    )
    with pytest.raises(ValueError, match="Expected global control shape"):
        controlled.control_term(torch.ones(3))
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        controlled.control_term(torch.ones(2, 3), num_nodes=2)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled.control_term(torch.ones(2, 2, 2), num_nodes=2)

    z = torch.randn(2, 3)
    with pytest.raises(ValueError, match="delta_t is required"):
        controlled.advance(z, None, control=torch.ones(2))
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        controlled.advance(torch.randn(2, 4), 0.1, control=torch.ones(2))
    with pytest.raises(ValueError, match="ignored when delta_t is zero"):
        controlled.advance(z, 0.0, control=torch.ones(2))
    with pytest.raises(ValueError, match="uncontrolled operator"):
        uncontrolled.advance(z, 0.1, control=torch.ones(1))
    with pytest.raises(ValueError, match="control input is required"):
        controlled.advance(z, 0.1, control=None)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled.advance(z, 0.1, control=torch.ones(2, 2, 2))

    offset = controlled.control_term(torch.ones(2))
    assert offset.shape == (3,)
    per_node = controlled.control_term(torch.ones(2, 2), num_nodes=2)
    assert per_node.shape == (2, 3)
    advanced = controlled.advance(z, 0.2, control=torch.ones(2))
    assert advanced.shape == z.shape
    advanced_rows = controlled.advance(z, 0.2, control=torch.ones(2, 2))
    assert advanced_rows.shape == z.shape
    assert torch.allclose(controlled(z, control=torch.ones(2), delta_t=0.1), advanced)


def test_continuous_set_dense_matrix_and_inverse_paths() -> None:
    structured = ContinuousKoopmanOperator(2, parameterization="schur")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        structured.set_dense_matrix(torch.eye(2))

    uncontrolled = ContinuousKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="Expected generator shape"):
        uncontrolled.set_dense_matrix(torch.ones(2, 3))
    with pytest.raises(ValueError, match="control_matrix provided"):
        uncontrolled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(1, 2))
    uncontrolled._parameters.pop("L")
    with pytest.raises(AttributeError, match="L"):
        uncontrolled.set_dense_matrix(torch.eye(2))

    controlled = ContinuousKoopmanOperator(2, control_dim=1, init_mode="identity")
    with pytest.raises(ValueError, match="control_matrix is required"):
        controlled.set_dense_matrix(torch.eye(2))
    with pytest.raises(ValueError, match="Expected control_matrix shape"):
        controlled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(2, 2))
    controlled.set_dense_matrix(
        -torch.eye(2),
        control_matrix=torch.zeros(1, 2),
    )

    z = torch.randn(2, 2)
    with pytest.raises(ValueError, match="delta_t is required"):
        controlled.inverse_advance(z, None, control=torch.ones(1))
    with pytest.raises(ValueError, match="control input is required"):
        controlled.inverse_advance(z, 0.1, control=None)
    recovered = controlled.inverse_advance(z, 0.1, control=torch.ones(1))
    assert recovered.shape == z.shape
    recovered_rows = controlled.inverse_advance(z, 0.1, control=torch.ones(2, 1))
    assert recovered_rows.shape == z.shape


def test_continuous_parameterization_reset_and_certificate_guards() -> None:
    d = 4
    for init_mode in ("identity", "identity_noise", "xavier"):
        reset_odo_generator(
            torch.empty(d, d),
            torch.empty(d, d),
            torch.empty(d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
        reset_schur_generator(
            torch.empty(d, d),
            torch.empty(d),
            torch.empty(d, d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
        reset_dissipative_generator(
            torch.empty(d, d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
        )
        reset_lyapunov_generator(
            torch.empty(d, d),
            torch.empty(d),
            torch.empty(d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_odo_generator(
            torch.empty(d, d),
            torch.empty(d, d),
            torch.empty(d),
            init_mode="nope",  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_dissipative_generator(
            torch.empty(d, d),
            init_mode="nope",  # type: ignore[arg-type]
            init_scale=0.01,
        )
    assert float(max_real_part_of_generator(-torch.eye(d))) <= 0.0
    assert ContinuousKoopmanOperator(
        latent_dim=d,
        parameterization="lyapunov",
        max_real_eigenvalue=0.05,
    ).L.shape == (d, d)
    assert ContinuousKoopmanOperator(
        latent_dim=d,
        parameterization="dissipative",
    ).L.shape == (d, d)
