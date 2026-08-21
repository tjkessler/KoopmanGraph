"""Generator EDMD baseline versus discrete EDMD and derivative-mode SINDy."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
import koopman_graph.baselines as baselines_pkg
from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines import GEDMDBaseline
from koopman_graph.protocols import ForecastModel, UncontrolledForecastModel
from koopman_graph.spectrum_types import compute_generator_spectrum


def _sequence_from_states(
    states: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    in_channels: int,
    derivatives: list[torch.Tensor] | None = None,
    timestamps: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a graph snapshot sequence from flattened states.

    Parameters
    ----------
    states : list of Tensor
        Flattened states.
    edge_index : Tensor
        Shared COO edges.
    num_nodes, in_channels : int
        Snapshot layout.
    derivatives : list of Tensor or None, optional
        Flattened ``dx/dt`` stored as ``Data.dx_dt``.
    timestamps : Tensor or None, optional
        Optional strictly increasing times.

    Returns
    -------
    GraphSnapshotSequence
        Homogeneous snapshots.
    """
    snapshots = []
    for index, state in enumerate(states):
        payload: dict[str, torch.Tensor] = {
            "x": state.reshape(num_nodes, in_channels),
            "edge_index": edge_index,
        }
        if derivatives is not None:
            payload["dx_dt"] = derivatives[index].reshape(num_nodes, in_channels)
        snapshots.append(Data(**payload))
    return GraphSnapshotSequence(snapshots, timestamps=timestamps)


def test_gedmd_exported_and_not_on_root_all() -> None:
    """``GEDMDBaseline`` is a baselines export, not a root façade symbol."""
    assert "GEDMDBaseline" in baselines_pkg.__all__
    assert "GEDMDBaseline" not in set(koopman_graph.__all__)
    assert not hasattr(koopman_graph, "GEDMDBaseline")
    source = REPO_ROOT / "src" / "koopman_graph" / "baselines" / "gedmd.py"
    text = source.read_text(encoding="utf-8")
    assert "10.1016/j.physd.2020.132416" in text
    assert "Klus2020gEDMD" in text
    assert "identify_sparse_dynamics" in text


def test_gedmd_satisfies_forecast_protocols() -> None:
    """gEDMD is an uncontrolled Data-only ``predict`` peer."""
    baseline = GEDMDBaseline(polynomial_degree=1)
    assert isinstance(baseline, ForecastModel)
    assert isinstance(baseline, UncontrolledForecastModel)


def test_gedmd_recovers_linear_generator(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Identity-dictionary gEDMD recovers a supplied linear generator.

    Independent ``(x, \\dot x)`` samples of ``\\dot x = x @ A^\\top``;
    construction residual, so ``rtol`` ``1e-8`` / ``atol`` ``1e-10`` on
    float64.
    """
    generator = torch.tensor(
        [[-0.2, 1.0], [-1.0, -0.3]],
        dtype=torch.float64,
    )
    torch.manual_seed(0)
    states = [torch.randn(2, dtype=torch.float64) for _ in range(12)]
    derivatives = [state @ generator.T for state in states]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        derivatives=derivatives,
    )
    baseline = GEDMDBaseline(polynomial_degree=1, time_step=0.25).fit(sequence)
    assert baseline.K is not None
    torch.testing.assert_close(baseline.K, generator, rtol=1e-8, atol=1e-10)
    expected_spectrum = compute_generator_spectrum(generator)
    fitted_spectrum = baseline.spectrum()
    torch.testing.assert_close(
        fitted_spectrum.eigenvalues,
        expected_spectrum.eigenvalues,
        rtol=1e-8,
        atol=1e-10,
    )
    torch.testing.assert_close(
        fitted_spectrum.growth_rates,
        expected_spectrum.growth_rates,
        rtol=1e-8,
        atol=1e-10,
    )
    initial = sequence[0]
    predicted = baseline.predict(initial, steps=3)[-1]
    step = torch.linalg.matrix_exp(generator * 0.25)
    expected_state = states[0]
    for _ in range(3):
        expected_state = expected_state @ step.T
    torch.testing.assert_close(
        predicted.x.reshape(-1),
        expected_state,
        rtol=1e-8,
        atol=1e-10,
    )


def test_gedmd_fit_derivatives_kwarg_and_irregular_timestamps(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """``derivatives=`` recovers ``A`` even when timestamps are irregular.

    Timestamps are unused at fit; they do not create ``L``. Construction
    residual, so ``rtol`` ``1e-8`` / ``atol`` ``1e-10`` on float64.
    """
    generator = torch.diag(torch.tensor([-0.5, -1.2], dtype=torch.float64))
    torch.manual_seed(1)
    states = torch.randn(10, 2, dtype=torch.float64)
    derivatives = states @ generator.T
    timestamps = torch.tensor(
        [0.0, 0.05, 0.2, 0.21, 0.5, 0.9, 1.0, 1.7, 1.75, 3.0],
        dtype=torch.float64,
    )
    sequence = _sequence_from_states(
        list(states),
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        timestamps=timestamps,
    )
    baseline = GEDMDBaseline(polynomial_degree=1).fit(
        sequence,
        derivatives=derivatives.reshape(10, 2, 1),
    )
    assert baseline.K is not None
    torch.testing.assert_close(baseline.K, generator, rtol=1e-8, atol=1e-10)


def test_gedmd_refuses_timestamps_without_derivatives(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Irregular timestamps alone do not identify the generator."""
    states = [
        torch.tensor([1.0, 0.3], dtype=torch.float64),
        torch.tensor([0.8, 0.1], dtype=torch.float64),
        torch.tensor([0.5, -0.2], dtype=torch.float64),
    ]
    timestamps = torch.tensor([0.0, 0.1, 0.4], dtype=torch.float64)
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        timestamps=timestamps,
    )
    with pytest.raises(ValueError, match="generator-action data"):
        GEDMDBaseline(polynomial_degree=1).fit(sequence)


def test_gedmd_refuses_rbf_dictionary(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """RBF / kernel dictionaries are out of scope for this generator path."""
    state = torch.tensor([1.0, -0.5], dtype=torch.float64)
    derivative = torch.tensor([-0.2, 0.1], dtype=torch.float64)
    sequence = _sequence_from_states(
        [state, state],
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        derivatives=[derivative, derivative],
    )
    with pytest.raises(ValueError, match="dictionary='polynomial' only"):
        GEDMDBaseline(dictionary="rbf", polynomial_degree=1).fit(sequence)
