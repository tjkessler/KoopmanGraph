"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from koopman_graph.operators import continuous_graph as continuous_graph_mod
from koopman_graph.operators import graph as graph_mod
from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator
from koopman_graph.operators.graph import GraphKoopmanOperator
from koopman_graph.operators.kronecker_spectrum import (
    _adjacency_eigenpairs,
    _complex_dtype,
    _factors_finite,
    _k_eff_eigenpairs_kronecker_sum,
    _random_walk_eigendecomposition,
    eigenvalues_k_eff_kronecker_sum,
    kronecker_sum_spectrum_eligible,
    spectrum_k_eff_kronecker_sum,
    spectrum_l_eff_kronecker_sum,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def test_eligibility_rejects_unknown_adjacency_and_sparsity() -> None:
    """Unknown adjacency / sparsity tokens short-circuit to False."""
    assert not kronecker_sum_spectrum_eligible(
        adjacency="not_an_adjacency",
        sparsity="dense",
        shared_self=True,
    )
    assert not kronecker_sum_spectrum_eligible(
        adjacency="symmetric",
        sparsity="not_a_sparsity",
        shared_self=True,
    )


def test_factors_finite_and_complex_dtype_helpers() -> None:
    """Private finite-check and dtype helpers cover float32 / float64 paths."""
    assert _factors_finite(torch.ones(2), torch.zeros(3, 3))
    assert not _factors_finite(torch.tensor([1.0, float("nan")]))
    assert _complex_dtype(torch.float32) == torch.complex64
    assert _complex_dtype(torch.float64) == torch.complex128


def test_random_walk_eigendecomposition_nonfinite_residual() -> None:
    """Non-finite residual ratio refuses the random-walk factorization."""
    adj = torch.eye(2)
    real_norm = torch.linalg.vector_norm
    calls = {"n": 0}

    def _nan_norm(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        calls["n"] += 1
        # Second call is ||ÂV - VΛ||; return NaN so the residual is non-finite.
        if calls["n"] >= 2:
            return torch.tensor(float("nan"), dtype=torch.float32)
        return real_norm(tensor, *args, **kwargs)

    with patch("torch.linalg.vector_norm", side_effect=_nan_norm):
        assert _random_walk_eigendecomposition(adj, residual_tol=1e-6) is None


def test_adjacency_eigenpairs_symmetric_nonfinite_adj() -> None:
    """Non-finite symmetric adjacency assembly yields None."""
    edge_index = _path_edge_index(3)
    with patch(
        "koopman_graph.operators.kronecker_spectrum.dense_symmetric_normalized_adjacency",
        return_value=torch.full((3, 3), float("nan")),
    ):
        assert (
            _adjacency_eigenpairs(
                edge_index,
                3,
                adjacency="symmetric",
                edge_weight=None,
                dtype=torch.float32,
                residual_tol=1e-6,
            )
            is None
        )


def test_adjacency_eigenpairs_symmetric_nonfinite_eigh() -> None:
    """Non-finite symmetric eigenpairs yield None."""
    edge_index = _path_edge_index(3)
    with patch(
        "torch.linalg.eigh",
        return_value=(
            torch.tensor([float("nan"), 1.0, 1.0]),
            torch.eye(3),
        ),
    ):
        assert (
            _adjacency_eigenpairs(
                edge_index,
                3,
                adjacency="symmetric",
                edge_weight=None,
                dtype=torch.float32,
                residual_tol=1e-6,
            )
            is None
        )


def test_adjacency_eigenpairs_random_walk_nonfinite_adj() -> None:
    """Non-finite random-walk adjacency assembly yields None."""
    edge_index = _path_edge_index(3)
    with patch(
        "koopman_graph.operators.kronecker_spectrum."
        "dense_random_walk_normalized_adjacency",
        return_value=torch.full((3, 3), float("nan")),
    ):
        assert (
            _adjacency_eigenpairs(
                edge_index,
                3,
                adjacency="random_walk",
                edge_weight=None,
                dtype=torch.float32,
                residual_tol=1e-6,
            )
            is None
        )


def test_adjacency_eigenpairs_unknown_mode_returns_none() -> None:
    """Unsupported adjacency tokens fall through to None."""
    edge_index = _path_edge_index(2)
    assert (
        _adjacency_eigenpairs(
            edge_index,
            2,
            adjacency="dual_random_walk",
            edge_weight=None,
            dtype=torch.float32,
            residual_tol=1e-6,
        )
        is None
    )


@pytest.mark.parametrize(
    ("k_self", "k_nbr", "num_nodes"),
    [
        (torch.ones(2), torch.eye(2), 2),  # ndim != 2
        (torch.eye(2), torch.eye(3), 2),  # shape mismatch
        (torch.ones(2, 3), torch.ones(2, 3), 2),  # non-square
        (torch.eye(2), torch.eye(2), 0),  # num_nodes < 1
    ],
)
def test_k_eff_eigenpairs_rejects_bad_shapes(
    k_self: torch.Tensor,
    k_nbr: torch.Tensor,
    num_nodes: int,
) -> None:
    """Shape / node-count guards refuse the Kronecker eigenpair core."""
    assert (
        _k_eff_eigenpairs_kronecker_sum(
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=_path_edge_index(max(num_nodes, 2)),
            num_nodes=num_nodes,
            adjacency="symmetric",
            edge_weight=None,
            residual_tol=1e-6,
        )
        is None
    )


def test_k_eff_eigenpairs_block_eig_nonfinite_returns_none() -> None:
    """Non-finite per-λ block eigendecomposition aborts the reduction."""
    edge_index = _path_edge_index(3)
    k_self = torch.eye(2)
    k_nbr = 0.1 * torch.eye(2)
    real_eig = torch.linalg.eig
    calls = {"n": 0}

    def _spoof(matrix: torch.Tensor):
        calls["n"] += 1
        # First call is adjacency (random_walk path unused); for symmetric,
        # adjacency uses eigh. Block eig uses eig — first eig call is a block.
        values, vectors = real_eig(matrix)
        if matrix.shape == (2, 2):
            values = values.clone()
            values[0] = complex(float("nan"), 0.0)
        return values, vectors

    with patch("torch.linalg.eig", side_effect=_spoof):
        assert (
            _k_eff_eigenpairs_kronecker_sum(
                k_self=k_self,
                k_nbr=k_nbr,
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
                edge_weight=None,
                residual_tol=1e-6,
            )
            is None
        )
    assert calls["n"] >= 1


def test_k_eff_eigenpairs_nonfinite_ambient_returns_none() -> None:
    """Non-finite ambient Kronecker product columns abort the reduction."""
    edge_index = _path_edge_index(3)
    with patch(
        "torch.kron",
        return_value=torch.full((6,), float("nan"), dtype=torch.complex64),
    ):
        assert (
            _k_eff_eigenpairs_kronecker_sum(
                k_self=torch.eye(2),
                k_nbr=torch.zeros(2, 2),
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
                edge_weight=None,
                residual_tol=1e-6,
            )
            is None
        )


def test_k_eff_eigenpairs_shape_mismatch_returns_none() -> None:
    """Mismatched stacked ambient dimension refuses the assembled spectrum."""
    edge_index = _path_edge_index(3)
    real_stack = torch.stack

    def _short_stack(tensors: list[torch.Tensor], *args: object, **kwargs: object):
        if args or kwargs.get("dim") == 1:
            # Eigenvector stack path uses dim=1; shorten eigenvalues stack only.
            pass
        stacked = real_stack(tensors, *args, **kwargs)
        if stacked.ndim == 1 and stacked.numel() > 1:
            return stacked[:1]
        return stacked

    with patch("torch.stack", side_effect=_short_stack):
        assert (
            _k_eff_eigenpairs_kronecker_sum(
                k_self=torch.eye(2),
                k_nbr=torch.zeros(2, 2),
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
                edge_weight=None,
                residual_tol=1e-6,
            )
            is None
        )


def test_spectrum_helpers_return_none_when_pairs_fail() -> None:
    """Public spectrum wrappers propagate None from the eigenpair core."""
    edge_index = _path_edge_index(3)
    with patch(
        "koopman_graph.operators.kronecker_spectrum._k_eff_eigenpairs_kronecker_sum",
        return_value=None,
    ):
        assert (
            eigenvalues_k_eff_kronecker_sum(
                k_self=torch.eye(2),
                k_nbr=torch.zeros(2, 2),
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
            )
            is None
        )
        assert (
            spectrum_k_eff_kronecker_sum(
                k_self=torch.eye(2),
                k_nbr=torch.zeros(2, 2),
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
                time_step=1.0,
            )
            is None
        )
        assert (
            spectrum_l_eff_kronecker_sum(
                l_self=torch.eye(2),
                l_nbr=torch.zeros(2, 2),
                edge_index=edge_index,
                num_nodes=3,
                adjacency="symmetric",
            )
            is None
        )


def test_operator_spectrum_falls_back_when_kronecker_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible discrete spectrum dense-routes when the helper returns None."""

    monkeypatch.setattr(graph_mod, "spectrum_k_eff_kronecker_sum", lambda **_: None)
    edge_index = _path_edge_index(3)
    op = GraphKoopmanOperator(2, init_mode="identity")
    spectrum = op.spectrum(edge_index, 3, time_step=0.2)
    assert spectrum.eigenvalues.shape == (6,)
    assert spectrum.eigenvectors.shape == (6, 6)


def test_continuous_spectrum_falls_back_when_kronecker_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible continuous spectrum dense-routes when the helper returns None."""

    monkeypatch.setattr(
        continuous_graph_mod, "spectrum_l_eff_kronecker_sum", lambda **_: None
    )
    edge_index = _path_edge_index(3)
    op = ContinuousGraphKoopmanOperator(2, init_mode="identity")
    spectrum = op.spectrum(edge_index, 3)
    assert spectrum.eigenvalues.shape == (6,)
    assert spectrum.eigenvectors.shape == (6, 6)
