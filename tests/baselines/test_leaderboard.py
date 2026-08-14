"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest

from koopman_graph.baselines.gnn import (
    LeaderboardProtocol,
    metr_la_leaderboard_protocol,
    multi_seed_summary,
    pems_leaderboard_protocol,
)
from koopman_graph.baselines.gnn.protocol import (
    EmptyProtocolDeviationsError,
    ForecasterProtocol,
)


def test_leaderboard_protocol_validation_and_pems() -> None:
    """Leaderboard adapters reject bad splits and empty summaries."""
    with pytest.raises(ValueError, match="positive"):
        LeaderboardProtocol(
            name="bad",
            history_len=0,
            horizon=12,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            metric="mae",
        )
    with pytest.raises(ValueError, match="must equal 1"):
        LeaderboardProtocol(
            name="bad",
            history_len=12,
            horizon=12,
            train_ratio=0.5,
            val_ratio=0.1,
            test_ratio=0.2,
            metric="mae",
        )
    protocol = pems_leaderboard_protocol("pems-bay")
    assert protocol.name == "pems-bay"
    from koopman_graph.baselines.gnn.leaderboard import multi_seed_summary

    with pytest.raises(ValueError, match="non-empty"):
        multi_seed_summary([])


def test_leaderboard_protocol_allows_empty_deviations() -> None:
    """Leaderboard adapters may have empty deviations; teaching ports may not."""
    proto = metr_la_leaderboard_protocol()
    assert proto.history_len == 12
    assert proto.deviations == ()
    empty_ok = LeaderboardProtocol(
        name="toy",
        history_len=2,
        horizon=2,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        metric="mae",
        deviations=(),
    )
    assert empty_ok.deviations == ()
    with pytest.raises(EmptyProtocolDeviationsError):
        ForecasterProtocol(
            name="bad",
            history_len=2,
            horizon=2,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            metric="mae",
            deviations=(),
        )
    mean, std = multi_seed_summary([1.0, 2.0, 3.0])
    assert float(mean) == pytest.approx(2.0)
