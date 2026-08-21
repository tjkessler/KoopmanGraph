"""Measure-preserving EDMD baseline versus unconstrained EDMD."""

from __future__ import annotations

import math

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
import koopman_graph.baselines as baselines_pkg
from koopman_graph import GraphSnapshotSequence
from koopman_graph.analysis._galerkin import assemble_galerkin_grams
from koopman_graph.baselines import EDMDBaseline, MpEDMDBaseline
from koopman_graph.baselines.mpedmd import fit_mpedmd_row_operator
from koopman_graph.protocols import ForecastModel, UncontrolledForecastModel


def _sequence_from_states(
    states: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    in_channels: int,
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

    Returns
    -------
    GraphSnapshotSequence
        Homogeneous snapshots.
    """
    snapshots = [
        Data(x=state.reshape(num_nodes, in_channels), edge_index=edge_index)
        for state in states
    ]
    return GraphSnapshotSequence(snapshots)


def _orbit(
    operator: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    n_times: int,
) -> list[torch.Tensor]:
    """Advance ``x_{t+1} = x_t @ K^\\top``.

    Parameters
    ----------
    operator : Tensor
        Row-convention map.
    initial_state : Tensor
        Flattened initial state.
    n_times : int
        Trajectory length.

    Returns
    -------
    list of Tensor
        Flattened states.
    """
    states = [initial_state]
    for _ in range(n_times - 1):
        states.append(states[-1] @ operator.T)
    return states


def test_mpedmd_exported_and_not_on_root_all() -> None:
    """``MpEDMDBaseline`` is a baselines export, not a root façade symbol."""
    assert "MpEDMDBaseline" in baselines_pkg.__all__
    assert "MpEDMDBaseline" not in set(koopman_graph.__all__)
    assert not hasattr(koopman_graph, "MpEDMDBaseline")
    source = REPO_ROOT / "src" / "koopman_graph" / "baselines" / "mpedmd.py"
    text = source.read_text(encoding="utf-8")
    assert "10.1137/22M1521407" in text
    assert "Colbrook2023mpEDMD" in text


def test_mpedmd_satisfies_forecast_protocols() -> None:
    """mpEDMD is an uncontrolled Data-only ``predict`` peer."""
    baseline = MpEDMDBaseline(polynomial_degree=1)
    assert isinstance(baseline, ForecastModel)
    assert isinstance(baseline, UncontrolledForecastModel)


def test_mpedmd_recovers_rotation_like_edmd(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Identity-dictionary mpEDMD matches EDMD on a regular polygonal rotation.

    A closed regular polygon makes the empirical dictionary Gram a multiple
    of the identity, so the Euclidean rotation is G-unitary and the polar
    factor agrees with unconstrained EDMD. Synthetic construction; ``rtol``
    ``1e-8`` / ``atol`` ``1e-10`` is float64 residual from that orbit.
    """
    n_pairs = 16
    angle = 2.0 * math.pi / n_pairs
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]],
        dtype=torch.float64,
    )
    states = _orbit(
        rotation,
        torch.tensor([1.0, 0.3], dtype=torch.float64),
        n_times=n_pairs + 1,
    )
    sequence = _sequence_from_states(
        states, synthetic_edge_index, num_nodes=2, in_channels=1
    )
    edmd = EDMDBaseline(polynomial_degree=1, time_step=1.0).fit(sequence)
    mpedmd = MpEDMDBaseline(polynomial_degree=1, time_step=1.0).fit(sequence)
    assert edmd.K is not None
    assert mpedmd.K is not None
    torch.testing.assert_close(edmd.K, rotation, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(mpedmd.K, rotation, rtol=1e-8, atol=1e-10)
    radii = torch.linalg.eigvals(mpedmd.K).abs()
    torch.testing.assert_close(
        radii,
        torch.ones_like(radii),
        rtol=1e-8,
        atol=1e-10,
    )
    prediction = mpedmd.predict(sequence[0], steps=3)[-1]
    torch.testing.assert_close(
        prediction.x.reshape(-1),
        states[3],
        rtol=1e-8,
        atol=1e-10,
    )


def test_mpedmd_is_unitary_in_the_gram_inner_product() -> None:
    """On a non-isotropic orbit, ``K`` stays G-unitary with unit-circle eigenvalues.

    The Euclidean rotation is not G-unitary when the sample Gram is
    anisotropic, so ``K`` need not equal the rotation. Colbrook Algorithm 4.1
    still yields ``K^\\top G K = G``. Synthetic construction; ``rtol`` ``1e-8``
    / ``atol`` ``1e-10`` is float64 residual of that identity.
    """
    angle = 0.4
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]],
        dtype=torch.float64,
    )
    states = torch.stack(
        _orbit(
            rotation,
            torch.tensor([1.0, 0.3], dtype=torch.float64),
            n_times=16,
        )
    )
    left, right = states[:-1], states[1:]
    fitted = fit_mpedmd_row_operator(left, right, rank=None)
    assert not torch.allclose(fitted, rotation, rtol=1e-3, atol=1e-3)
    gram = assemble_galerkin_grams(left, right).g00
    column_map = fitted.T
    torch.testing.assert_close(
        column_map.T @ gram @ column_map,
        gram,
        rtol=1e-8,
        atol=1e-10,
    )
    radii = torch.linalg.eigvals(fitted).abs()
    torch.testing.assert_close(
        radii,
        torch.ones_like(radii),
        rtol=1e-8,
        atol=1e-10,
    )


def test_mpedmd_does_not_recover_dissipative_map(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """On a contraction, EDMD recovers ``K`` while mpEDMD stays on the circle."""
    dissipative = torch.diag(torch.tensor([0.7, 0.4], dtype=torch.float64))
    states = _orbit(
        dissipative,
        torch.tensor([1.2, -0.8], dtype=torch.float64),
        n_times=12,
    )
    sequence = _sequence_from_states(
        states, synthetic_edge_index, num_nodes=2, in_channels=1
    )
    edmd = EDMDBaseline(polynomial_degree=1).fit(sequence)
    mpedmd = MpEDMDBaseline(polynomial_degree=1).fit(sequence)
    assert edmd.K is not None
    assert mpedmd.K is not None
    torch.testing.assert_close(edmd.K, dissipative, rtol=1e-8, atol=1e-10)
    assert not torch.allclose(mpedmd.K, dissipative, rtol=1e-2, atol=1e-2)
    radii = torch.linalg.eigvals(mpedmd.K).abs()
    torch.testing.assert_close(
        radii,
        torch.ones_like(radii),
        rtol=1e-6,
        atol=1e-8,
    )


def test_fit_mpedmd_row_operator_empty_gram_raises() -> None:
    """A zero dictionary Gram is refused."""
    left = torch.zeros(4, 2, dtype=torch.float64)
    right = torch.zeros(4, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Gram square-root is empty"):
        fit_mpedmd_row_operator(left, right, rank=None)


def test_mpedmd_requires_two_snapshots(synthetic_edge_index: torch.Tensor) -> None:
    """Fewer than two snapshots raise like EDMD."""
    sequence = _sequence_from_states(
        [torch.tensor([1.0, 0.0], dtype=torch.float64)],
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )
    with pytest.raises(ValueError, match="at least two snapshots"):
        MpEDMDBaseline(polynomial_degree=1).fit(sequence)
