"""Block-equivariant Koopman operator, including the l=2 tensor irrep."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
from koopman_graph.model.factory import parse_koopman_arg
from koopman_graph.operators import EquivariantKoopmanOperator
from koopman_graph.operators.equivariant import (
    TENSOR_IRREP_DIM,
    TENSOR_IRREP_L,
    VECTOR_IRREP_DIM,
)

# Independent construction: scale * I_l commutes with Wigner D^{(l)}.
_ORACLE_ABS = 1e-6
_SEED = 0


def test_equivariant_operator_off_root() -> None:
    """Block operator lives on ``operators.__all__``, not the root façade."""
    assert "EquivariantKoopmanOperator" in koopman_graph.operators.__all__
    assert "EquivariantKoopmanOperator" not in koopman_graph.__all__


def test_equivariant_module_does_not_import_e3nn() -> None:
    """Generator leaf stays extra-free; rotation tests import e3nn."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/operators/equivariant.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    offenders = [
        name for name in imported if name == "e3nn" or name.startswith("e3nn.")
    ]
    assert not offenders


def test_equivariant_operator_blocks_inverse_and_forward() -> None:
    """Scalar/vector operator covers validation, vector-only, inverse, forward."""
    with pytest.raises(ValueError, match="non-negative"):
        EquivariantKoopmanOperator(n_scalars=-1, n_vectors=1)
    with pytest.raises(ValueError, match="at least one"):
        EquivariantKoopmanOperator(n_scalars=0, n_vectors=0, n_tensors=0)
    vectors = EquivariantKoopmanOperator(n_scalars=0, n_vectors=1)
    z = torch.randn(2, VECTOR_IRREP_DIM)
    assert vectors.matrix.shape == (VECTOR_IRREP_DIM, VECTOR_IRREP_DIM)
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
    assert torch.allclose(
        vec_block,
        op.vector_scales[0] * torch.eye(VECTOR_IRREP_DIM),
        atol=_ORACLE_ABS,
    )


def test_scalar_only_bound_metric_does_not_require_vector_scales() -> None:
    """Empty vector / tensor scale banks are skipped in ``bound_metric``."""
    op = EquivariantKoopmanOperator(n_scalars=2)
    assert op.n_tensors == 0
    assert op.latent_dim == 2
    assert op.bound_metric().ndim == 0


def test_l2_tensor_block_is_scale_times_i5() -> None:
    """The additional irrep is ``scale * I_5``, not a mixed Clebsch–Gordan map."""
    op = EquivariantKoopmanOperator(n_scalars=0, n_vectors=0, n_tensors=1)
    assert TENSOR_IRREP_L == 2
    assert op.latent_dim == TENSOR_IRREP_DIM
    with torch.no_grad():
        op.tensor_scales.copy_(torch.tensor([-0.4]))
    block = op.matrix
    assert torch.allclose(
        block,
        op.tensor_scales[0] * torch.eye(TENSOR_IRREP_DIM),
        atol=_ORACLE_ABS,
    )
    z = torch.randn(3, TENSOR_IRREP_DIM)
    assert op.advance(z).shape == z.shape
    assert op.bound_metric().item() == pytest.approx(0.4, abs=_ORACLE_ABS)


def test_l2_tensor_block_commutes_with_rotation() -> None:
    """``K`` commutes with the SO(3) representation on ``0e ⊕ 1o ⊕ 2e``.

    Independent oracle: Schur says each irrep block is a scalar times
    identity, so it commutes with the Wigner matrix of that irrep.
    Requires ``e3nn`` (``[equivariance]``).
    """
    o3 = pytest.importorskip("e3nn.o3")
    op = EquivariantKoopmanOperator(
        n_scalars=1,
        n_vectors=1,
        n_tensors=1,
    ).double()
    with torch.no_grad():
        op.vector_scales.copy_(torch.tensor([0.7], dtype=torch.float64))
        op.tensor_scales.copy_(torch.tensor([-0.4], dtype=torch.float64))
    torch.manual_seed(_SEED)
    latents = torch.randn(4, op.latent_dim, dtype=torch.float64)
    rotation = o3.rand_matrix()
    irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
    if hasattr(irreps, "D_from_matrix"):
        representation = irreps.D_from_matrix(rotation)
    else:
        representation = irreps.D_from_angles(*o3.matrix_to_angles(rotation))
    representation = representation.to(dtype=torch.float64)
    rotated = latents @ representation.transpose(-1, -2)
    left = op.advance(rotated)
    right = op.advance(latents) @ representation.transpose(-1, -2)
    assert torch.allclose(left, right, atol=_ORACLE_ABS, rtol=0.0)


def test_equivariant_is_not_a_factory_kind() -> None:
    """``koopman=None`` stays per-node; ``equivariant`` is not a kind."""
    kind, injected = parse_koopman_arg(None)
    assert kind == "pernode"
    assert injected is None
    with pytest.raises(ValueError, match="hodge"):
        parse_koopman_arg("equivariant")
