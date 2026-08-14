"""Tests for hetero training-input coercion and widened ``MultiTrajectory``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
    as_multi_trajectory,
    resolve_hetero_sequence,
)
from koopman_graph.training.inputs import (
    resolve_training_sequences,
    resolve_validation_sequences,
)


def _multiplex_snapshot(*, num_nodes: int = 4) -> HeteroData:
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, 3)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


def _hetero_sequence(num_timesteps: int = 3) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(num_timesteps)]
    )


def _homo_sequence(num_timesteps: int = 3) -> GraphSnapshotSequence:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.randn(3, 2), edge_index=edge_index) for _ in range(num_timesteps)]
    )


def test_resolve_training_accepts_hetero_sequence() -> None:
    """Verify a hetero sequence coerces without casting to ``Data``."""
    sequence = _hetero_sequence()
    resolved = resolve_training_sequences(sequence)
    assert resolved == [sequence]
    assert isinstance(resolved[0], HeteroGraphSnapshotSequence)


def test_resolve_training_accepts_hetero_multi_trajectory() -> None:
    """Verify ``MultiTrajectory`` of hetero sequences is accepted."""
    first = _hetero_sequence(2)
    second = _hetero_sequence(4)
    resolved = resolve_training_sequences(MultiTrajectory((first, second)))
    assert resolved == [first, second]


def test_multi_trajectory_rejects_homo_hetero_mix() -> None:
    """Verify mixed families inside ``MultiTrajectory`` raise clearly."""
    with pytest.raises(ValueError, match="cannot mix"):
        MultiTrajectory((_homo_sequence(), _hetero_sequence()))


def test_as_multi_trajectory_accepts_hetero() -> None:
    """Verify the helper builds an all-hetero ``MultiTrajectory``."""
    first = _hetero_sequence(2)
    second = _hetero_sequence(3)
    bundle = as_multi_trajectory(first, second)
    assert bundle.sequences == (first, second)


def test_bare_list_of_hetero_sequences_rejected() -> None:
    """Verify bare hetero-sequence lists require ``MultiTrajectory``."""
    with pytest.raises(TypeError, match="MultiTrajectory"):
        resolve_training_sequences([_hetero_sequence(), _hetero_sequence()])


def test_bare_list_of_heterodata_wraps_one_trajectory() -> None:
    """Verify a bare ``HeteroData`` list becomes one hetero trajectory."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    resolved = resolve_training_sequences(snapshots)
    assert len(resolved) == 1
    assert isinstance(resolved[0], HeteroGraphSnapshotSequence)
    assert resolved[0].num_timesteps == 3


def test_resolve_hetero_sequence_identity_and_wrap() -> None:
    """Verify ``resolve_hetero_sequence`` mirrors ``resolve_sequence``."""
    sequence = _hetero_sequence()
    assert resolve_hetero_sequence(sequence) is sequence
    wrapped = resolve_hetero_sequence([_multiplex_snapshot() for _ in range(2)])
    assert isinstance(wrapped, HeteroGraphSnapshotSequence)
    assert wrapped.num_timesteps == 2


def test_mix_data_and_heterodata_rejected() -> None:
    """Verify mixing ``Data`` and ``HeteroData`` in one fit input raises."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="cannot mix Data and HeteroData"):
        resolve_training_sequences(
            [
                Data(x=torch.randn(2, 2), edge_index=edge_index),
                _multiplex_snapshot(),
            ]
        )


def test_mix_homo_and_hetero_sequences_in_bare_list_rejected() -> None:
    """Verify mixing sequence families in a bare list raises."""
    with pytest.raises(ValueError, match="cannot mix GraphSnapshotSequence"):
        resolve_training_sequences([_homo_sequence(), _hetero_sequence()])


def test_homo_coercion_regressions_unchanged() -> None:
    """Verify homogeneous list[Data] and MultiTrajectory paths still work."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graphs = [
        Data(x=torch.randn(2, 2), edge_index=edge_index),
        Data(x=torch.randn(2, 2), edge_index=edge_index),
    ]
    from_data = resolve_training_sequences(graphs)
    assert len(from_data) == 1
    assert isinstance(from_data[0], GraphSnapshotSequence)
    assert from_data[0].num_timesteps == 2

    seq = _homo_sequence()
    multi = resolve_training_sequences(MultiTrajectory((seq, seq)))
    assert multi == [seq, seq]

    with pytest.raises(TypeError, match="MultiTrajectory"):
        resolve_training_sequences([seq, seq])


def test_validation_accepts_hetero_multi_trajectory() -> None:
    """Verify validation MultiTrajectory length checks apply to hetero."""
    first = _hetero_sequence(2)
    second = _hetero_sequence(2)
    resolved = resolve_validation_sequences(
        MultiTrajectory((first, second)),
        num_training_sequences=2,
    )
    assert resolved == [first, second]
    with pytest.raises(ValueError, match="validation_sequence list length"):
        resolve_validation_sequences(
            MultiTrajectory((first,)),
            num_training_sequences=2,
        )
