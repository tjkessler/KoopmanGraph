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
