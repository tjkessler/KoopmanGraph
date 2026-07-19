"""Tests for delay / Hankel embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    DelayEmbeddingEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.datasets import Lorenz96GraphBenchmark
from koopman_graph.nn.delay import flatten_delay_window, stack_delay_features


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    return torch.tensor([sources + targets, targets + sources], dtype=torch.long)


def _make_sequence(
    *,
    num_timesteps: int = 8,
    num_nodes: int = 4,
    in_channels: int = 2,
    observation_masks: torch.Tensor | None = None,
    allow_dynamic_topology: bool = False,
) -> GraphSnapshotSequence:
    edge_index = _path_edge_index(num_nodes)
    snapshots = []
    for t in range(num_timesteps):
        x = torch.full((num_nodes, in_channels), float(t + 1))
        snapshots.append(Data(x=x, edge_index=edge_index))
    return GraphSnapshotSequence(
        snapshots,
        allow_dynamic_topology=allow_dynamic_topology,
        observation_masks=observation_masks,
    )


def test_stack_delay_features_pads_and_masks_history() -> None:
    sequence = _make_sequence(num_timesteps=5, in_channels=2)
    x_window, _edge_index, _weight, history_mask = stack_delay_features(
        sequence,
        index=1,
        n_delays=3,
        pad=True,
    )
    assert x_window.shape == (3, 4, 2)
    assert history_mask.tolist() == [False, True, True]
    assert torch.all(x_window[0] == 0.0)
    assert torch.all(x_window[1] == 1.0)
    assert torch.all(x_window[2] == 2.0)


def test_stack_delay_features_rejects_topology_change() -> None:
    edge_a = _path_edge_index(3)
    edge_b = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.ones(3, 1), edge_index=edge_a),
        Data(x=torch.ones(3, 1), edge_index=edge_a),
    ]
    sequence = GraphSnapshotSequence(snapshots)
    # Borrowed Data mutation after construction (documented container behavior).
    sequence[1].edge_index = edge_b
    with pytest.raises(ValueError, match="topology changed"):
        stack_delay_features(sequence, index=1, n_delays=2, pad=False)


def test_windowed_shapes_and_stride() -> None:
    sequence = _make_sequence(num_timesteps=6, in_channels=3)
    windowed = sequence.windowed(n_delays=3, stride=2, pad=True)
    assert windowed.num_timesteps == 3
    assert windowed[0].x.shape == (4, 9)
    flat = flatten_delay_window(
        stack_delay_features(sequence, 0, 3, pad=True)[0],
    )
    assert torch.allclose(windowed[0].x, flat)


def test_delay_encoder_gradient_through_pads() -> None:
    base = GNNEncoder(in_channels=6, hidden_channels=8, latent_dim=4)
    encoder = DelayEmbeddingEncoder(base, n_delays=3)
    x_window = torch.zeros(3, 5, 2, requires_grad=True)
    edge_index = _path_edge_index(5)
    z = encoder(x_window, edge_index)
    z.sum().backward()
    assert x_window.grad is not None
    assert x_window.grad.shape == x_window.shape


def test_n_delays_one_matches_bare_encoder() -> None:
    sequence = _make_sequence(num_timesteps=4, in_channels=2)
    torch.manual_seed(0)
    bare = GraphKoopmanModel(
        encoder=GNNEncoder(2, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=1.0,
        n_delays=1,
    )
    torch.manual_seed(0)
    delayed = GraphKoopmanModel(
        encoder=GNNEncoder(2, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=1.0,
        n_delays=1,
    )
    z0 = bare.encode(sequence[2])
    z1 = delayed.encode_at(sequence, 2)
    assert torch.allclose(z0, z1)


def test_model_auto_wrap_requires_sized_channels() -> None:
    with pytest.raises(ValueError, match="divisible by n_delays"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 8, 4),
            decoder=GNNDecoder(4, 8, 2),
            latent_dim=4,
            time_step=1.0,
            n_delays=3,
        )


def test_serialization_round_trip_n_delays(tmp_path: Path) -> None:
    model = GraphKoopmanModel(
        encoder=GNNEncoder(6, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=1.0,
        n_delays=3,
    )
    assert isinstance(model.encoder, DelayEmbeddingEncoder)
    path = tmp_path / "delay.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.n_delays == 3
    assert isinstance(loaded.encoder, DelayEmbeddingEncoder)
    assert loaded.encoder.base_encoder.in_channels == 6


def test_partial_observability_delay_improves_or_matches() -> None:
    """Soft check: delay embedding should not worsen short-horizon RMSE."""
    full = Lorenz96GraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=40,
        seed=7,
        forcing=8.0,
        burn_in=50,
    )
    num_nodes = full.num_nodes
    # Observe half the nodes every step (fixed mask).
    mask = torch.zeros(full.num_timesteps, num_nodes, dtype=torch.bool)
    mask[:, : num_nodes // 2] = True
    sequence = GraphSnapshotSequence(
        list(full),
        observation_masks=mask,
    )

    def _fit(n_delays: int) -> float:
        torch.manual_seed(0)
        feature_dim = 1
        encoder = GNNEncoder(
            in_channels=n_delays * feature_dim,
            hidden_channels=16,
            latent_dim=8,
        )
        decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=1)
        model = GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            time_step=0.05,
            n_delays=n_delays,
        )
        model.fit(sequence, epochs=12, lr=1e-2)
        result = model.evaluate(sequence, horizons=(1, 2))
        return float(result.aggregate_rmse)

    rmse_1 = _fit(1)
    rmse_3 = _fit(3)
    # Soft threshold: allow small noise; delay should be no worse than +25%.
    assert rmse_3 <= rmse_1 * 1.25 + 1e-3


def test_delay_helpers_validation_and_masking() -> None:
    """Exercise delay helper validation, masking, and history builders."""
    from koopman_graph.nn.delay import (
        apply_observation_mask_to_features,
        flatten_delay_window,
        history_from_snapshots,
        resolve_delay_encoder,
    )

    x = torch.ones(3, 2)
    assert torch.equal(apply_observation_mask_to_features(x, None), x)
    masked = apply_observation_mask_to_features(x, torch.tensor([True, False, True]))
    assert torch.equal(masked[1], torch.zeros(2))
    with pytest.raises(ValueError, match="does not match"):
        apply_observation_mask_to_features(x, torch.ones(2, dtype=torch.bool))

    with pytest.raises(ValueError, match="n_delays"):
        flatten_delay_window(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="n_delays must be"):
        stack_delay_features(_make_sequence(), index=0, n_delays=0)
    with pytest.raises(IndexError, match="index must satisfy"):
        stack_delay_features(_make_sequence(), index=99, n_delays=2)
    with pytest.raises(ValueError, match="insufficient history"):
        stack_delay_features(
            _make_sequence(num_timesteps=2),
            index=0,
            n_delays=3,
            pad=False,
        )

    sequence = _make_sequence(num_timesteps=4, in_channels=2)
    masks = torch.ones(4, 4, dtype=torch.bool)
    masks[1, 0] = False
    masked_seq = GraphSnapshotSequence(list(sequence), observation_masks=masks)
    window, *_rest = stack_delay_features(masked_seq, index=2, n_delays=2, pad=False)
    assert torch.allclose(window[0, 0], torch.zeros(2))

    edge_a = _path_edge_index(3)
    weight_mismatch = GraphSnapshotSequence(
        [
            Data(
                x=torch.ones(3, 1),
                edge_index=edge_a,
                edge_weight=torch.ones(edge_a.shape[1]),
            ),
            Data(
                x=torch.ones(3, 1),
                edge_index=edge_a,
                edge_weight=torch.ones(edge_a.shape[1]),
            ),
        ]
    )
    weight_mismatch[1].edge_weight = torch.ones(edge_a.shape[1]) * 2.0
    with pytest.raises(ValueError, match="topology changed"):
        stack_delay_features(weight_mismatch, index=1, n_delays=2, pad=False)

    # Presence mismatch after construction (borrowed Data mutation).
    presence = GraphSnapshotSequence(
        [
            Data(
                x=torch.ones(3, 1),
                edge_index=edge_a,
                edge_weight=torch.ones(edge_a.shape[1]),
            ),
            Data(
                x=torch.ones(3, 1),
                edge_index=edge_a,
                edge_weight=torch.ones(edge_a.shape[1]),
            ),
        ]
    )
    del presence[1].edge_weight
    with pytest.raises(ValueError, match="topology changed"):
        stack_delay_features(presence, index=1, n_delays=2, pad=False)

    mismatched = GraphSnapshotSequence(
        [
            Data(x=torch.ones(3, 1), edge_index=edge_a),
            Data(x=torch.ones(3, 1), edge_index=edge_a),
        ],
        allow_dynamic_topology=True,
    )
    mismatched[1].x = torch.ones(4, 1)
    with pytest.raises(ValueError, match="share num_nodes"):
        stack_delay_features(mismatched, index=1, n_delays=2, pad=False)

    newest = Data(x=torch.ones(3, 2), edge_index=edge_a)
    x_win, edge_index, _ew, history = history_from_snapshots(
        [newest],
        n_delays=3,
        pad=True,
    )
    assert x_win.shape == (3, 3, 2)
    assert history.tolist() == [False, False, True]
    assert torch.equal(edge_index, edge_a)
    with pytest.raises(ValueError, match="n_delays"):
        history_from_snapshots([newest], n_delays=0)
    with pytest.raises(ValueError, match="at least one"):
        history_from_snapshots([], n_delays=2)
    with pytest.raises(ValueError, match="pad=False"):
        history_from_snapshots([newest], n_delays=2, pad=False)
    long_hist = history_from_snapshots(
        [Data(x=torch.ones(3, 2), edge_index=edge_a) for _ in range(5)],
        n_delays=2,
        pad=False,
    )
    assert long_hist[0].shape[0] == 2

    base = GNNEncoder(in_channels=6, hidden_channels=4, latent_dim=3)
    wrapped, n = resolve_delay_encoder(base, 3)
    assert isinstance(wrapped, DelayEmbeddingEncoder)
    assert n == 3
    same, n1 = resolve_delay_encoder(wrapped, 1)
    assert same is wrapped and n1 == 3
    with pytest.raises(ValueError, match="conflicts"):
        resolve_delay_encoder(wrapped, 2)
    with pytest.raises(ValueError, match="n_delays must be"):
        resolve_delay_encoder(base, 0)
    with pytest.raises(TypeError, match="GNNEncoder/GATEncoder"):
        resolve_delay_encoder(torch.nn.Linear(2, 2), 3)
    with pytest.raises(ValueError, match="in_channels"):
        DelayEmbeddingEncoder(torch.nn.Linear(4, 2), n_delays=2)  # type: ignore[arg-type]


def test_delay_encoder_forward_paths_and_errors() -> None:
    """DelayEmbeddingEncoder accepts Data/stacked tensors and validates shapes."""
    base = GNNEncoder(in_channels=6, hidden_channels=8, latent_dim=4)
    encoder = DelayEmbeddingEncoder(base, n_delays=3)
    edge_index = _path_edge_index(4)
    window = torch.randn(3, 4, 2)
    stacked = flatten_delay_window(window)
    z_window = encoder(window, edge_index)
    z_stack = encoder(stacked, edge_index)
    assert z_window.shape == (4, 4)
    assert torch.allclose(z_window, z_stack)
    data = Data(x=stacked, edge_index=edge_index)
    assert encoder(data).shape == (4, 4)

    with pytest.raises(ValueError, match="edge_index is required"):
        encoder(window)
    with pytest.raises(ValueError, match="leading dim"):
        encoder(torch.randn(2, 4, 2), edge_index)
    with pytest.raises(ValueError, match="feature_dim"):
        encoder(torch.randn(3, 4, 3), edge_index)
    with pytest.raises(ValueError, match="in_channels"):
        encoder(torch.randn(4, 5), edge_index)
    with pytest.raises(ValueError, match="expected delay window"):
        encoder(torch.randn(4), edge_index)
