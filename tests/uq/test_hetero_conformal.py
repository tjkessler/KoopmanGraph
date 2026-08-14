"""Hetero parity tests for ``ConformalKoopmanUQ``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.data.hetero_layout import stack_typed_features
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.uq import ConformalKoopmanUQ, hetero_snapshot_with_features


def _multiplex_snapshot(*, seed: int = 0, num_nodes: int = 4) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, 3, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _multiplex_sequence(
    *,
    num_sequences: int = 4,
    timesteps: int = 3,
    seed: int = 0,
) -> list[HeteroGraphSnapshotSequence]:
    sequences: list[HeteroGraphSnapshotSequence] = []
    for seq_id in range(num_sequences):
        snaps = [
            _multiplex_snapshot(seed=seed + 100 * seq_id + t) for t in range(timesteps)
        ]
        sequences.append(HeteroGraphSnapshotSequence(snaps))
    return sequences


def _hetero_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=(
            ("node", "r1", "node"),
            ("node", "r2", "node"),
        ),
    )


def _homo_model() -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )


def test_multiplex_calibrate_and_predict_interval_smoke() -> None:
    """Calibrate + predict_interval return HeteroData bands on multiplex data."""
    torch.manual_seed(0)
    model = _hetero_model()
    cal = _multiplex_sequence(num_sequences=4, timesteps=3, seed=0)
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(cal, steps=2, alpha=0.1)
    assert uq.is_calibrated
    assert uq.quantiles.shape == (2,)

    origin = cal[0][0]
    interval = uq.predict_interval(origin, steps=2, level=0.9)
    assert len(interval.mean) == 2
    assert isinstance(interval.mean[0], HeteroData)
    assert isinstance(interval.lower[0], HeteroData)
    assert isinstance(interval.upper[0], HeteroData)
    assert interval.lower[0]["node"].x.shape == (4, 3)
    assert torch.all(interval.lower[0]["node"].x <= interval.mean[0]["node"].x)
    assert torch.all(interval.mean[0]["node"].x <= interval.upper[0]["node"].x)


def test_node_wise_scores_on_stacked_n() -> None:
    """node_wise quantiles have shape (steps, stacked N)."""
    torch.manual_seed(1)
    model = _hetero_model(seed=1)
    cal = _multiplex_sequence(num_sequences=12, timesteps=2, seed=10)
    uq = ConformalKoopmanUQ(model, method="split", score="node_wise")
    uq.calibrate(cal, steps=1, alpha=0.1)
    # Multiplex fixture: one type with 4 nodes → stacked N = 4.
    assert uq.quantiles.shape == (1, 4)

    interval = uq.predict_interval(cal[0][0], steps=1, level=0.9)
    widths = interval.upper[0]["node"].x - interval.lower[0]["node"].x
    assert widths.shape == (4, 3)
    # Half-width constant across features for each node.
    assert torch.allclose(widths[:, 0], widths[:, 1])


def test_neighbor_smoothing_uses_relation_union() -> None:
    """neighbor_smoothing calibrates on multiplex without raising."""
    torch.manual_seed(2)
    model = _hetero_model(seed=2)
    cal = _multiplex_sequence(num_sequences=12, timesteps=2, seed=20)
    uq = ConformalKoopmanUQ(
        model,
        method="split",
        score="node_wise",
        neighbor_smoothing=0.5,
    )
    uq.calibrate(cal, steps=1, alpha=0.1)
    assert uq.quantiles.shape == (1, 4)
    interval = uq.predict_interval(cal[0][0], steps=1, level=0.9)
    assert isinstance(interval.lower[0], HeteroData)


def test_homo_path_unchanged_with_data_intervals() -> None:
    """Homogeneous conformal still returns Data bands."""
    torch.manual_seed(3)
    model = _homo_model()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    sequences = []
    for seq_id in range(4):
        snaps = []
        for t in range(3):
            generator = torch.Generator().manual_seed(300 + 10 * seq_id + t)
            snaps.append(
                Data(
                    x=torch.randn(3, 3, generator=generator),
                    edge_index=edge_index,
                )
            )
        sequences.append(GraphSnapshotSequence(snaps))
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(sequences, steps=2, alpha=0.1)
    interval = uq.predict_interval(sequences[0][0], steps=2, level=0.9)
    assert isinstance(interval.mean[0], Data)
    assert not isinstance(interval.mean[0], HeteroData)


def test_hetero_model_rejects_homo_calibration() -> None:
    """Hetero conformal rejects Data / GraphSnapshotSequence calibration."""
    model = _hetero_model()
    uq = ConformalKoopmanUQ(model)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    seq = GraphSnapshotSequence(
        [
            Data(x=torch.randn(3, 3), edge_index=edge_index),
            Data(x=torch.randn(3, 3), edge_index=edge_index),
        ]
    )
    with pytest.raises(TypeError, match="hetero ConformalKoopmanUQ"):
        uq.calibrate([seq], steps=1)


def test_homo_model_rejects_hetero_predict_interval() -> None:
    """Homogeneous predict_interval rejects HeteroData origins."""
    model = _homo_model()
    uq = ConformalKoopmanUQ(model)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    cal = [
        GraphSnapshotSequence(
            [
                Data(x=torch.randn(3, 3), edge_index=edge_index),
                Data(x=torch.randn(3, 3), edge_index=edge_index),
            ]
        )
        for _ in range(3)
    ]
    uq.calibrate(cal, steps=1, alpha=0.1)
    with pytest.raises(TypeError, match="homogeneous ConformalKoopmanUQ"):
        uq.predict_interval(_multiplex_snapshot(), steps=1, level=0.9)


def test_hetero_snapshot_with_features_round_trip() -> None:
    """Packing helper unstacks stacked features onto a cloned template."""
    snapshot = _multiplex_snapshot(seed=5)
    stacked = stack_typed_features({"node": snapshot["node"].x}, ("node",))
    packed = hetero_snapshot_with_features(snapshot, stacked + 1.0, ("node",))
    torch.testing.assert_close(packed["node"].x, snapshot["node"].x + 1.0)
    assert ("node", "r1", "node") in packed.edge_types


def test_hetero_calibration_round_trip(tmp_path: Path) -> None:
    """Calibration payload round-trips for a hetero wrapper."""
    model = _hetero_model(seed=7)
    cal = _multiplex_sequence(num_sequences=4, timesteps=2, seed=70)
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(cal, steps=1, alpha=0.1)
    path = tmp_path / "hetero_cal.pt"
    uq.save_calibration(path)
    other = ConformalKoopmanUQ(model, method="split", score="aggregate")
    other.load_calibration(path)
    torch.testing.assert_close(other.quantiles, uq.quantiles)
    interval = other.predict_interval(cal[0][0], steps=1, level=0.9)
    assert isinstance(interval.mean[0], HeteroData)
