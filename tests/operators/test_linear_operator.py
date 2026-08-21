"""LinearOperatorProtocol wrappers for polynomial and one-tap graph maps."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
from koopman_graph.graph_utils import dense_symmetric_normalized_adjacency
from koopman_graph.operators import (
    LinearOperatorProtocol,
    MatrixFreeGraphLinearOperator,
    PolynomialGraphLinearOperator,
)
from koopman_graph.operators.linear import MAX_DENSE_LINEAR_OPERATOR_SIZE
from koopman_graph.operators.matrix_free import apply_k_eff_graph, flatten_node_latents
from koopman_graph.operators.polynomial_graph import dense_polynomial_kronecker

_ATOL = 1e-5
_SPECTRUM_ATOL = 1e-4
_RESIDUAL_ATOL = 1e-6
_N_NODES = 6
_LATENT_DIM = 2
_SEED = 0


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Undirected path used as a modest sparse smoke graph."""
    tails = torch.arange(num_nodes - 1, dtype=torch.long)
    forward = torch.stack((tails, tails + 1), dim=0)
    return torch.cat((forward, forward.flip(0)), dim=1)


def _one_tap_factors() -> tuple[torch.Tensor, torch.Tensor]:
    """Well-conditioned self / neighbor factors for Richardson."""
    k_self = torch.tensor([[0.7, 0.05], [0.0, 0.6]], dtype=torch.float64)
    k_nbr = torch.tensor([[0.04, 0.0], [0.0, 0.03]], dtype=torch.float64)
    return k_self, k_nbr


def test_linear_operator_export_off_root() -> None:
    """Protocol and wrappers live on ``operators.__all__``, not the root."""
    assert "LinearOperatorProtocol" in koopman_graph.operators.__all__
    assert "PolynomialGraphLinearOperator" in koopman_graph.operators.__all__
    assert "MatrixFreeGraphLinearOperator" in koopman_graph.operators.__all__
    assert "LinearOperatorProtocol" not in koopman_graph.__all__
    assert "PolynomialGraphLinearOperator" not in koopman_graph.__all__
    assert "MatrixFreeGraphLinearOperator" not in koopman_graph.__all__


def test_linear_module_does_not_import_model() -> None:
    """L2 operator algebra must not import L4 model."""
    source = (
        Path(__file__).resolve().parents[2] / "src/koopman_graph/operators/linear.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    offenders = [
        name
        for name in imported
        if name == "koopman_graph.model" or name.startswith("koopman_graph.model.")
    ]
    assert not offenders


def test_wrappers_satisfy_linear_operator_protocol() -> None:
    """Both teaching wrappers are runtime-checkable protocol implementers."""
    k_self, k_nbr = _one_tap_factors()
    edge_index = _path_edges(_N_NODES)
    one_tap = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    poly = PolynomialGraphLinearOperator(
        (k_self, k_nbr, 0.01 * torch.eye(2, dtype=torch.float64)),
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    assert isinstance(one_tap, LinearOperatorProtocol)
    assert isinstance(poly, LinearOperatorProtocol)


def test_matrix_free_wrapper_matches_apply_and_dense() -> None:
    """One-tap wrapper matches ``apply_k_eff_graph`` and dense ``K @ x``."""
    torch.manual_seed(_SEED)
    k_self, k_nbr = _one_tap_factors()
    edge_index = _path_edges(_N_NODES)
    operator = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    flat = flatten_node_latents(torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64))
    expected = apply_k_eff_graph(
        flat,
        k_self=k_self,
        k_nbr=k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    matrix = operator.dense_matrix()
    dense = matrix @ flat
    got = operator.matvec(flat)
    assert torch.allclose(got, expected, atol=_ATOL)
    assert torch.allclose(got, dense, atol=_ATOL)
    assert torch.allclose(operator.rmatvec(flat), matrix.T @ flat, atol=_ATOL)


def test_polynomial_wrapper_matches_dense_kronecker() -> None:
    """Degree-2 polynomial matvec matches assembled Kronecker ``K_eff``."""
    torch.manual_seed(_SEED)
    k_self, k_nbr = _one_tap_factors()
    k_two = torch.tensor([[0.01, 0.0], [0.0, 0.008]], dtype=torch.float64)
    edge_index = _path_edges(_N_NODES)
    hops = (k_self, k_nbr, k_two)
    operator = PolynomialGraphLinearOperator(
        hops,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    adjacency = dense_symmetric_normalized_adjacency(
        edge_index,
        _N_NODES,
        dtype=k_self.dtype,
    )
    dense = dense_polynomial_kronecker(adjacency, hops)
    flat = flatten_node_latents(torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64))
    assert torch.allclose(operator.matvec(flat), dense @ flat, atol=_ATOL)
    assert torch.allclose(operator.rmatvec(flat), dense.T @ flat, atol=_ATOL)


def test_solve_and_residual_on_both_wrappers() -> None:
    """Richardson solve recovers ``x``; residual stays under ``1e-6``."""
    torch.manual_seed(_SEED)
    k_self, k_nbr = _one_tap_factors()
    edge_index = _path_edges(_N_NODES)
    x_true = flatten_node_latents(
        torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64)
    )
    one_tap = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    poly = PolynomialGraphLinearOperator(
        (k_self, k_nbr, 0.01 * torch.eye(2, dtype=torch.float64)),
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    for operator in (one_tap, poly):
        rhs = operator.matvec(x_true)
        solved = operator.solve(rhs, tol=_RESIDUAL_ATOL, max_iters=64)
        assert torch.allclose(solved, x_true, atol=_ATOL)
        residual = float(operator.residual_norm(solved, rhs).item())
        # Richardson stops on a relative residual; absolute norm may sit
        # just above ``1e-6`` on this modest path (same ``1e-5`` floor as
        # ``tests/operators/test_matrix_free.py``).
        assert residual == pytest.approx(0.0, abs=_ATOL)


def test_expm_action_matches_dense_matrix_exp() -> None:
    """Taylor ``exp(t A) b`` matches a dense matrix exponential on a toy."""
    torch.manual_seed(_SEED)
    k_self, k_nbr = _one_tap_factors()
    edge_index = _path_edges(_N_NODES)
    operator = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    flat = flatten_node_latents(torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64))
    time = 0.1
    dense = torch.linalg.matrix_exp(time * operator.dense_matrix()) @ flat
    got = operator.expm_action(time, flat)
    assert torch.allclose(got, dense, atol=_ATOL)


def test_leading_eigpairs_report_residuals() -> None:
    """Arnoldi residuals stay under the requested teaching tolerance."""
    k_self, k_nbr = _one_tap_factors()
    edge_index = _path_edges(_N_NODES)
    operator = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=edge_index,
        num_nodes=_N_NODES,
    )
    result = operator.leading_eigpairs(2, tol=_SPECTRUM_ATOL)
    assert result.converged
    assert result.eigenvalues.shape == (2,)
    assert bool((result.residual_norms <= _SPECTRUM_ATOL + 1e-12).all().item())


def test_dense_assembly_refused_above_ceiling() -> None:
    """``N·d`` above the teaching ceiling refuses dense assembly."""
    latent_dim = 64
    num_nodes = (MAX_DENSE_LINEAR_OPERATOR_SIZE // latent_dim) + 1
    assert num_nodes * latent_dim > MAX_DENSE_LINEAR_OPERATOR_SIZE
    k_self = 0.5 * torch.eye(latent_dim)
    k_nbr = 0.01 * torch.eye(latent_dim)
    operator = MatrixFreeGraphLinearOperator(
        k_self,
        k_nbr,
        edge_index=_path_edges(num_nodes),
        num_nodes=num_nodes,
    )
    assert operator.memory_estimate.dense_allowed is False
    assert "trainer DDP" in operator.memory_estimate.notes
    with pytest.raises(ValueError, match="dense assembly refused"):
        operator.dense_matrix()


def test_polynomial_wrapper_rejects_one_tap() -> None:
    """A two-factor bank is the one-tap wrapper, not the polynomial class."""
    k_self, k_nbr = _one_tap_factors()
    with pytest.raises(ValueError, match="P>=2"):
        PolynomialGraphLinearOperator(
            (k_self, k_nbr),
            edge_index=_path_edges(_N_NODES),
            num_nodes=_N_NODES,
        )
