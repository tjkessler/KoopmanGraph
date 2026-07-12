"""Unit tests for KoopmanOperator."""

import pytest
import torch

from koopman_graph.operator import KoopmanOperator


@pytest.mark.parametrize("init_mode", ["identity", "identity_noise", "xavier"])
def test_init_modes_construct(init_mode: str) -> None:
    """Verify KoopmanOperator constructs for each init mode."""
    op = KoopmanOperator(4, init_mode=init_mode)  # type: ignore[arg-type]
    assert op.K.shape == (4, 4)
    assert torch.isfinite(op.K).all()


def test_identity_init() -> None:
    """Verify identity initialization sets ``K`` to the identity matrix."""
    op = KoopmanOperator(3, init_mode="identity")
    assert torch.allclose(op.K, torch.eye(3))


def test_forward_single_graph_shape() -> None:
    """Verify forward output shape for a single graph."""
    op = KoopmanOperator(8)
    z = torch.randn(5, 8)
    out = op(z)
    assert out.shape == (5, 8)


def test_forward_batched_shape() -> None:
    """Verify forward output shape for batched latent states."""
    op = KoopmanOperator(8)
    z = torch.randn(2, 5, 8)
    out = op(z)
    assert out.shape == (2, 5, 8)


def test_forward_matches_manual_matmul() -> None:
    """Verify forward matches ``z @ K.T``."""
    op = KoopmanOperator(4, init_mode="xavier")
    z = torch.randn(3, 4)
    expected = z @ op.K.T
    assert torch.allclose(op(z), expected)


def test_gradient_flow() -> None:
    """Verify gradients flow through the operator forward pass."""
    op = KoopmanOperator(6, init_mode="identity_noise")
    z = torch.randn(4, 6, requires_grad=True)
    loss = op(z).sum()
    loss.backward()
    assert op.K.grad is not None
    assert torch.isfinite(op.K.grad).all()
    assert z.grad is not None


def test_invalid_latent_dim_raises() -> None:
    """Verify non-positive ``latent_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        KoopmanOperator(0)


def test_shape_mismatch_raises() -> None:
    """Verify trailing dimension mismatch raises ``ValueError``."""
    op = KoopmanOperator(4)
    z = torch.randn(3, 5)
    with pytest.raises(ValueError, match="Expected trailing dimension 4"):
        op(z)


def test_exported_from_package() -> None:
    """Verify the symbol is exported from the package root."""
    from koopman_graph import KoopmanOperator as ExportedOperator

    assert ExportedOperator is KoopmanOperator


def test_invalid_init_scale_raises() -> None:
    """Verify negative ``init_scale`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="init_scale must be non-negative"):
        KoopmanOperator(4, init_scale=-0.1)


def test_reset_parameters_unknown_init_mode_raises() -> None:
    """Verify unknown ``init_mode`` in ``reset_parameters`` raises ``ValueError``."""
    op = KoopmanOperator(3)
    op.init_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown init_mode"):
        op.reset_parameters()


def test_odo_parameterization_constructs() -> None:
    """Verify ODO parameterization builds a finite assembled matrix."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="identity")
    assert op.K.shape == (4, 4)
    assert torch.isfinite(op.K).all()


def test_odo_spectral_radius_bounded() -> None:
    """Verify ODO spectral radius stays within the configured bound."""
    op = KoopmanOperator(6, parameterization="odo", max_spectral_radius=0.9)
    op.diag_raw.data.fill_(3.0)
    assert op.spectral_radius().item() <= 0.9 + 1e-6
    eigenvalues = torch.linalg.eigvals(op.K)
    assert eigenvalues.abs().max().item() <= 0.9 + 1e-5


def test_odo_inverse_recovers_forward_step() -> None:
    """Verify ODO inverse step recovers the previous latent state."""
    op = KoopmanOperator(5, parameterization="odo", init_mode="identity_noise")
    z = torch.randn(7, 5)
    z_next = op(z)
    recovered = op.inverse_step(z_next)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_odo_gradient_flow() -> None:
    """Verify gradients reach ODO factor parameters."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="identity_noise")
    z = torch.randn(3, 4, requires_grad=True)
    loss = op(z).sum()
    loss.backward()
    assert op.cayley_O1.grad is not None
    assert op.cayley_O2.grad is not None
    assert op.diag_raw.grad is not None


def test_invalid_max_spectral_radius_raises() -> None:
    """Verify non-positive ``max_spectral_radius`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="max_spectral_radius must be positive"):
        KoopmanOperator(4, parameterization="odo", max_spectral_radius=0.0)


def test_dense_inverse_matrix_reused() -> None:
    """Verify dense inverse matrix matches a single inverse step."""
    op = KoopmanOperator(4, init_mode="identity")
    z = torch.randn(5, 4)
    inverse = op.dense_inverse_matrix()
    assert torch.allclose(op.inverse_step(z, inverse_matrix=inverse), z, atol=1e-5)


def test_negative_control_dim_raises() -> None:
    """Verify negative ``control_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        KoopmanOperator(4, control_dim=-1)


def test_unknown_parameterization_raises() -> None:
    """Verify unknown parameterization raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown parameterization"):
        KoopmanOperator(4, parameterization="bogus")  # type: ignore[arg-type]


def test_reset_control_parameters_noop_without_control() -> None:
    """Verify control reset is a no-op for uncontrolled operators."""
    op = KoopmanOperator(4)
    op.reset_control_parameters()
    assert not hasattr(op, "B")


def test_control_term_requires_positive_control_dim() -> None:
    """Verify ``control_term`` rejects uncontrolled operators."""
    op = KoopmanOperator(4)
    with pytest.raises(ValueError, match="requires control_dim > 0"):
        op.control_term(torch.zeros(1))


def test_control_term_validates_shapes() -> None:
    """Verify global, per-node, and rank validation in ``control_term``."""
    op = KoopmanOperator(4, control_dim=2)
    with pytest.raises(ValueError, match="Expected global control shape"):
        op.control_term(torch.zeros(3))
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        op.control_term(torch.zeros(5, 3))
    with pytest.raises(ValueError, match="Per-node control has 4 rows"):
        op.control_term(torch.zeros(4, 2), num_nodes=5)
    with pytest.raises(ValueError, match="control input must have shape"):
        op.control_term(torch.zeros(2, 2, 2))


def test_forward_rejects_control_for_uncontrolled_operator() -> None:
    """Verify controls passed to an uncontrolled operator raise."""
    op = KoopmanOperator(4)
    with pytest.raises(ValueError, match="uncontrolled operator"):
        op(torch.randn(3, 4), control=torch.zeros(1))


def test_forward_requires_control_when_controlled() -> None:
    """Verify a controlled operator requires a control input."""
    op = KoopmanOperator(4, control_dim=1)
    with pytest.raises(ValueError, match="control input is required"):
        op(torch.randn(3, 4))


def test_inverse_step_requires_control_when_controlled() -> None:
    """Verify inverse step requires control for controlled operators."""
    op = KoopmanOperator(4, control_dim=1)
    with pytest.raises(ValueError, match="control input is required"):
        op.inverse_step(torch.randn(3, 4))


def test_inverse_step_with_per_node_control() -> None:
    """Verify controlled inverse step recovers state for per-node controls."""
    op = KoopmanOperator(3, control_dim=2, init_mode="identity")
    with torch.no_grad():
        op.B.copy_(torch.randn(2, 3) * 0.1)
    z = torch.randn(5, 3)
    control = torch.randn(5, 2)
    z_next = op(z, control=control)
    recovered = op.inverse_step(z_next, control=control)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_odo_xavier_init_constructs() -> None:
    """Verify xavier initialization for the ODO parameterization."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="xavier")
    assert torch.isfinite(op.K).all()


def test_odo_reset_unknown_init_mode_raises() -> None:
    """Verify unknown ``init_mode`` in ODO reset raises ``ValueError``."""
    op = KoopmanOperator(3, parameterization="odo")
    op.init_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown init_mode"):
        op.reset_parameters()


def test_dense_spectral_radius_matches_eigvals() -> None:
    """Verify dense spectral radius equals the max eigenvalue magnitude."""
    op = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        op._parameters["K"].copy_(torch.diag(torch.tensor([0.5, -1.5, 0.9])))
    assert torch.isclose(op.spectral_radius(), torch.tensor(1.5))


def test_dense_inverse_matrix_rejects_odo() -> None:
    """Verify ``dense_inverse_matrix`` is dense-only."""
    op = KoopmanOperator(3, parameterization="odo")
    with pytest.raises(ValueError, match="only available for dense"):
        op.dense_inverse_matrix()


def test_dense_inverse_matrix_falls_back_to_pinv() -> None:
    """Verify singular dense matrices fall back to the pseudo-inverse."""
    op = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        op._parameters["K"].zero_()
    inverse = op.dense_inverse_matrix()
    assert torch.allclose(inverse, torch.zeros(3, 3))
