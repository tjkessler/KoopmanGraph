"""Finite-sample subspace invariance leakage :math:`\\eta`."""

from __future__ import annotations

import pytest
import torch
from tests.identification.test_fit_identification import _identity_model, _IdentityCodec
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.identification import (
    SubspaceInvarianceReport,
    subspace_invariance_report,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import GraphKoopmanOperator


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Undirected path ``edge_index``.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    Tensor
        COO edge index.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _line_encodings(*, n_times: int = 8, n_nodes: int = 4) -> torch.Tensor:
    """Time-major encodings on :math:`\\mathrm{span}\\{e_1\\}\\subset\\mathbb{R}^{2}`.

    Parameters
    ----------
    n_times, n_nodes : int
        Layout.

    Returns
    -------
    Tensor
        ``(T, N, 2)`` float64 encodings.
    """
    encodings = torch.zeros(n_times, n_nodes, 2, dtype=torch.float64)
    encodings[:, :, 0] = torch.linspace(
        0.2, 1.6, n_times, dtype=torch.float64
    ).unsqueeze(1)
    return encodings


def _line_sequence(*, n_times: int = 8, n_nodes: int = 4) -> GraphSnapshotSequence:
    """Identity-codec snapshots on the first coordinate axis.

    Parameters
    ----------
    n_times, n_nodes : int
        Sequence length and graph size.

    Returns
    -------
    GraphSnapshotSequence
        Homogeneous snapshots.
    """
    encodings = _line_encodings(n_times=n_times, n_nodes=n_nodes)
    edge_index = _path_edges(n_nodes)
    snapshots = [
        Data(x=encodings[index].clone(), edge_index=edge_index)
        for index in range(n_times)
    ]
    return GraphSnapshotSequence(snapshots)


def test_leakage_vanishes_on_invariant_line() -> None:
    """Diagonal ``K`` leaves :math:`\\mathrm{span}\\{e_1\\}` invariant, so ``η→0``.

    Noiseless float64 construction; ``atol=1e-8``.
    """
    encodings = _line_encodings()
    matrix = torch.diag(torch.tensor([0.9, 0.5], dtype=torch.float64))
    report = subspace_invariance_report(encodings, matrix, held_out=True)
    assert isinstance(report, SubspaceInvarianceReport)
    assert report.leakage == pytest.approx(0.0, abs=1e-8)
    assert report.rank == 1
    assert report.held_out is True
    assert report.n_samples == 4 * 4


def test_leakage_positive_when_map_leaves_the_span() -> None:
    """A 90° map sends :math:`e_1` to :math:`e_2`, so leakage is order-one."""
    encodings = _line_encodings()
    matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    report = subspace_invariance_report(encodings, matrix, held_out=True)
    assert report.leakage > 0.5
    assert report.rank == 1


def test_held_out_uses_last_half_only() -> None:
    """Full-span early snapshots do not enlarge ``P`` when ``held_out=True``."""
    n_times, n_nodes = 8, 4
    encodings = torch.zeros(n_times, n_nodes, 2, dtype=torch.float64)
    encodings[:4, 0, 0] = 1.0
    encodings[:4, 1, 1] = 1.0
    encodings[4:, :, 0] = torch.linspace(0.2, 0.8, 4, dtype=torch.float64).unsqueeze(1)
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    held = subspace_invariance_report(encodings, rotation, held_out=True)
    full = subspace_invariance_report(encodings, rotation, held_out=False)
    assert held.rank == 1
    assert held.leakage > 0.5
    assert full.rank == 2
    assert full.leakage == pytest.approx(0.0, abs=1e-8)


def test_held_out_requires_four_snapshots() -> None:
    """``held_out=True`` refuses short trajectories."""
    encodings = torch.zeros(3, 2, 2, dtype=torch.float64)
    encodings[:, :, 0] = 1.0
    matrix = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 4 snapshots"):
        subspace_invariance_report(encodings, matrix, held_out=True)


def test_zero_mapped_norm_raises() -> None:
    """A zero map makes the leakage denominator undefined."""
    encodings = _line_encodings()
    matrix = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"E\|\|K"):
        subspace_invariance_report(encodings, matrix, held_out=True)


def test_evaluate_default_leaves_invariance_unset() -> None:
    """Default ``evaluate`` metrics match an explicit ``include_invariance=False``."""
    sequence = _line_sequence()
    model = _identity_model(2, dtype=torch.float64)
    with torch.no_grad():
        model.koopman.set_dense_matrix(
            torch.diag(torch.tensor([0.9, 0.5], dtype=torch.float64))
        )
    default = model.evaluate(sequence, horizons=(1,))
    explicit = model.evaluate(sequence, horizons=(1,), include_invariance=False)
    assert default.invariance is None
    assert explicit.invariance is None
    assert default.aggregate_rmse == pytest.approx(explicit.aggregate_rmse)


def test_evaluate_include_invariance_preserves_rmse() -> None:
    """Opt-in leakage attaches without changing aggregate RMSE."""
    sequence = _line_sequence()
    model = _identity_model(2, dtype=torch.float64)
    with torch.no_grad():
        model.koopman.set_dense_matrix(
            torch.diag(torch.tensor([0.9, 0.5], dtype=torch.float64))
        )
    baseline = model.evaluate(sequence, horizons=(1,))
    result = model.evaluate(sequence, horizons=(1,), include_invariance=True)
    assert isinstance(result.invariance, SubspaceInvarianceReport)
    assert result.invariance.leakage == pytest.approx(0.0, abs=1e-8)
    assert result.aggregate_rmse == pytest.approx(baseline.aggregate_rmse)
    assert result.aggregate_mae == pytest.approx(baseline.aggregate_mae)


def test_model_subspace_invariance_report_matches_tensor_path() -> None:
    """The model helper encodes then matches the tensor API."""
    sequence = _line_sequence()
    model = _identity_model(2, dtype=torch.float64)
    matrix = torch.diag(torch.tensor([0.9, 0.5], dtype=torch.float64))
    with torch.no_grad():
        model.koopman.set_dense_matrix(matrix)
    report = model.subspace_invariance_report(sequence, held_out=True)
    direct = subspace_invariance_report(_line_encodings(), matrix, held_out=True)
    assert report.leakage == pytest.approx(direct.leakage, abs=1e-8)
    assert report.rank == direct.rank


def test_graph_operator_refuses_invariance() -> None:
    """Networked operators are out of scope for this increment."""
    sequence = _line_sequence()
    model = GraphKoopmanModel(
        encoder=_IdentityCodec(2),
        decoder=_IdentityCodec(2),
        latent_dim=2,
        time_step=1.0,
        koopman=GraphKoopmanOperator(latent_dim=2),
    )
    with pytest.raises(ValueError, match="per-node"):
        model.subspace_invariance_report(sequence)
    with pytest.raises(ValueError, match="per-node"):
        model.evaluate(sequence, horizons=(1,), include_invariance=True)


def test_hetero_evaluate_refuses_include_invariance() -> None:
    """Hetero evaluate raises before silently dropping the flag."""
    data = HeteroData()
    data["node"].x = torch.randn(4, 3)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]], dtype=torch.long
    )
    sequence = HeteroGraphSnapshotSequence([data.clone() for _ in range(5)])
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, 8, 4, 2, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 3, 2, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )
    with pytest.raises(ValueError, match="hetero"):
        model.evaluate(sequence, horizons=(1,), include_invariance=True)
    with pytest.raises(ValueError, match="Hetero"):
        model.subspace_invariance_report(sequence)
