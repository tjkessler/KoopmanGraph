"""Hankel-DMD and HAVOK teaching baselines versus delay-encoder stacking."""

from __future__ import annotations

import math

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
import koopman_graph.baselines as baselines_pkg
from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines import HankelDMDBaseline, HAVOKBaseline
from koopman_graph.baselines.base import ClassicalBaseline
from koopman_graph.baselines.hankel_dmd import delay_embed_rows
from koopman_graph.nn import DelayEmbeddingEncoder, GNNEncoder
from koopman_graph.protocols import ForecastModel, UncontrolledForecastModel

_HANKEL_SRC = REPO_ROOT / "src" / "koopman_graph" / "baselines" / "hankel_dmd.py"
_HAVOK_SRC = REPO_ROOT / "src" / "koopman_graph" / "baselines" / "havok.py"
_SCALAR_EDGE_INDEX = torch.tensor([[0], [0]], dtype=torch.long)


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


def _scalar_rotation_orbit(*, n_times: int, dtype: torch.dtype) -> list[torch.Tensor]:
    """Return the first coordinate of a planar rotation as 1-D states.

    Parameters
    ----------
    n_times : int
        Number of samples.
    dtype : torch.dtype
        Floating type.

    Returns
    -------
    list of Tensor
        Scalar states with shape ``(1,)``.
    """
    angle = 2.0 * math.pi / 16.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=dtype)
    state = torch.tensor([1.0, 0.3], dtype=dtype)
    orbit: list[torch.Tensor] = []
    for _ in range(n_times):
        orbit.append(state[0:1].clone())
        state = state @ rotation.T
    return orbit


def test_hankel_havok_exported_and_not_on_root_all() -> None:
    """Hankel-DMD and HAVOK are baselines exports, not root façade symbols."""
    for name in ("HankelDMDBaseline", "HAVOKBaseline"):
        assert name in baselines_pkg.__all__
        assert name not in set(koopman_graph.__all__)
        assert not hasattr(koopman_graph, name)
    assert "delay_embed_rows" not in baselines_pkg.__all__
    assert "assemble_delay_state" not in baselines_pkg.__all__
    hankel_text = _HANKEL_SRC.read_text(encoding="utf-8")
    havok_text = _HAVOK_SRC.read_text(encoding="utf-8")
    assert "10.1137/17M1125236" in hankel_text
    assert "Arbabi2017HankelDMD" in hankel_text
    assert "10.1038/s41467-017-00030-8" in havok_text
    assert "Brunton2017HAVOK" in havok_text


def test_hankel_havok_satisfy_forecast_protocols() -> None:
    """Both delay-row solvers are uncontrolled Data-only ``predict`` peers."""
    for baseline in (
        HankelDMDBaseline(n_delays=2),
        HAVOKBaseline(n_delays=4, havok_rank=3),
    ):
        assert isinstance(baseline, ForecastModel)
        assert isinstance(baseline, UncontrolledForecastModel)
        assert isinstance(baseline, ClassicalBaseline)


def test_delay_embed_rows_oldest_to_newest() -> None:
    """Window ``i`` concatenates ``states[i:i+n_delays]`` oldest → newest.

    Exact integer-valued construction, so equality is bitwise.
    """
    states = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=torch.float64,
    )
    rows = delay_embed_rows(states, n_delays=2)
    expected = torch.tensor(
        [
            [1.0, 10.0, 2.0, 20.0],
            [2.0, 20.0, 3.0, 30.0],
            [3.0, 30.0, 4.0, 40.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rows, expected, rtol=0.0, atol=0.0)


def test_hankel_dmd_predicts_scalar_rotation_with_history() -> None:
    """Hankel-DMD one-step forecast matches a held-out scalar rotation sample.

    The measurement is the first coordinate of a planar 16-gon rotation.
    ``n_delays=4`` with supplied history is an exact delay IC; float64
    residual of the linear delay map, so ``rtol`` ``1e-8`` / ``atol``
    ``1e-10``. Dominant eigenvalues lie on the unit circle to the same
    construction residual.
    """
    states = _scalar_rotation_orbit(n_times=33, dtype=torch.float64)
    sequence = _sequence_from_states(
        states[:-1],
        _SCALAR_EDGE_INDEX,
        num_nodes=1,
        in_channels=1,
    )
    baseline = HankelDMDBaseline(n_delays=4, time_step=1.0).fit(sequence)
    assert baseline.K is not None
    radii = torch.linalg.eigvals(baseline.K).abs()
    dominant = torch.sort(radii, descending=True).values[:2]
    torch.testing.assert_close(
        dominant,
        torch.ones_like(dominant),
        rtol=1e-8,
        atol=1e-10,
    )
    predicted = baseline.predict(
        sequence[-1],
        steps=1,
        history=list(sequence[-4:-1]),
    )[-1]
    torch.testing.assert_close(
        predicted.x.reshape(-1),
        states[-1],
        rtol=1e-8,
        atol=1e-10,
    )


def test_havok_energy_concentrates_on_leading_modes() -> None:
    """A sampled sinusoid Hankel is numerically rank-2; ``A`` stays finite.

    Exact 2-D oscillator coordinate; ``σ_3 / σ_1`` is a float64 SVD
    residual (``1e-8`` relative). Autonomous ``predict`` uses ``u=0``.
    """
    states = _scalar_rotation_orbit(n_times=40, dtype=torch.float64)
    sequence = _sequence_from_states(
        states,
        _SCALAR_EDGE_INDEX,
        num_nodes=1,
        in_channels=1,
    )
    baseline = HAVOKBaseline(n_delays=4, havok_rank=3, time_step=1.0).fit(sequence)
    assert baseline.K is not None
    assert baseline.B is not None
    assert baseline.singular_values is not None
    assert baseline.K.shape == (2, 2)
    assert baseline.B.shape == (2, 1)
    assert torch.isfinite(baseline.K).all()
    leading = baseline.singular_values[0]
    third = baseline.singular_values[2]
    assert float(third / leading) == pytest.approx(0.0, abs=1e-8)
    forecasts = baseline.predict(
        sequence[-1],
        steps=2,
        history=list(sequence[-4:-1]),
    )
    assert len(forecasts) == 2
    assert forecasts[0].x.shape == (1, 1)
    spectrum = baseline.spectrum()
    assert spectrum.eigenvalues.shape == (2,)


def test_hankel_havok_distinct_from_delay_encoder() -> None:
    """Delay-encoder stacking is a GNN wrapper, not a delay-row DMD/HAVOK fit."""
    encoder = DelayEmbeddingEncoder(
        GNNEncoder(in_channels=4, hidden_channels=4, latent_dim=2, num_layers=1),
        n_delays=2,
    )
    assert not isinstance(encoder, ClassicalBaseline)
    assert not hasattr(HankelDMDBaseline(), "base_encoder")
    assert not hasattr(HAVOKBaseline(n_delays=4), "base_encoder")
    assert not isinstance(HankelDMDBaseline(), DelayEmbeddingEncoder)


def test_delay_embed_rows_rejects_invalid_window() -> None:
    """``n_delays < 1`` and short sequences raise with the constraint."""
    states = torch.ones(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="n_delays must be >= 1"):
        delay_embed_rows(states, n_delays=0)
    with pytest.raises(ValueError, match="need at least n_delays=4"):
        delay_embed_rows(states, n_delays=4)


def test_hankel_dmd_rejects_short_sequence() -> None:
    """Fit requires ``T >= n_delays + 1`` consecutive snapshots."""
    states = [torch.ones(1, dtype=torch.float64) for _ in range(3)]
    sequence = _sequence_from_states(
        states,
        _SCALAR_EDGE_INDEX,
        num_nodes=1,
        in_channels=1,
    )
    with pytest.raises(ValueError, match="n_delays\\+1=4"):
        HankelDMDBaseline(n_delays=3).fit(sequence)
    with pytest.raises(ValueError, match="n_delays must be >= 1"):
        HankelDMDBaseline(n_delays=0)


def test_havok_rank_guards() -> None:
    """``havok_rank`` must be at least 2 and fit inside the Hankel bound."""
    with pytest.raises(ValueError, match="havok_rank must be >= 2"):
        HAVOKBaseline(havok_rank=1)
    states = [torch.tensor([float(index)], dtype=torch.float64) for index in range(8)]
    sequence = _sequence_from_states(
        states,
        _SCALAR_EDGE_INDEX,
        num_nodes=1,
        in_channels=1,
    )
    with pytest.raises(ValueError, match="havok_rank=3 exceeds Hankel rank bound"):
        HAVOKBaseline(n_delays=2, havok_rank=3).fit(sequence)
    with pytest.raises(ValueError, match="n_delays\\+1=5"):
        HAVOKBaseline(n_delays=4, havok_rank=3).fit(
            _sequence_from_states(
                states[:4],
                _SCALAR_EDGE_INDEX,
                num_nodes=1,
                in_channels=1,
            )
        )
