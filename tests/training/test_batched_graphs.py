"""Tests for opt-in vectorized multi-graph training."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
)
from koopman_graph.training import (
    LossWeights,
    compute_batched_training_loss,
    compute_training_loss,
    mean_training_loss_breakdown,
)

# Float32 encode / K matvec accumulation on modest graphs (N≤8, T≤4, two
# sequences). Tighter than typical eigendecomposition checks.
_BATCHED_LOSS_RTOL = 1e-5
_BATCHED_LOSS_ATOL = 1e-6


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index with shape ``(2, 2 * (num_nodes - 1))``.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _sequence(
    *,
    num_nodes: int,
    num_timesteps: int = 4,
    in_channels: int = 3,
    seed: int = 0,
    scale: float = 1.0,
) -> GraphSnapshotSequence:
    """Build a static-topology homogeneous sequence.

    Parameters
    ----------
    num_nodes : int
        Nodes per snapshot.
    num_timesteps : int, optional
        Snapshot count. Default is ``4``.
    in_channels : int, optional
        Feature width. Default is ``3``.
    seed : int, optional
        RNG seed. Default is ``0``.
    scale : float, optional
        Feature scale so unequal-``N`` graphs have distinct MSE. Default is
        ``1.0``.

    Returns
    -------
    GraphSnapshotSequence
        Random snapshots on a path graph.
    """
    torch.manual_seed(seed)
    edge_index = _path_edge_index(num_nodes)
    snapshots = [
        Data(
            x=scale * torch.randn(num_nodes, in_channels),
            edge_index=edge_index,
        )
        for _ in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def _make_model(*, koopman: str = "pernode", seed: int = 0) -> GraphKoopmanModel:
    """Construct a small hop-matched model for batching tests.

    Parameters
    ----------
    koopman : str, optional
        Factory kind. Default is ``"pernode"``.
    seed : int, optional
        RNG seed. Default is ``0``.

    Returns
    -------
    GraphKoopmanModel
        One-layer GCN stack (no receptive-field warning vs graph ``P=1``).
    """
    torch.manual_seed(seed)
    encoder = GNNEncoder(
        in_channels=3,
        hidden_channels=8,
        latent_dim=4,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=1,
    )
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,  # type: ignore[arg-type]
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require batched vs per-sequence losses to match documented tols.

    Parameters
    ----------
    actual : Tensor
        Batched scalar.
    expected : Tensor
        Mean of per-sequence scalars.
    """
    assert torch.allclose(
        actual,
        expected,
        rtol=_BATCHED_LOSS_RTOL,
        atol=_BATCHED_LOSS_ATOL,
    )


@pytest.mark.parametrize("koopman", ["pernode", "graph"])
def test_batched_reconstruction_matches_per_sequence_mean(koopman: str) -> None:
    """Batched recon equals the trajectory-equal mean on unequal ``N``."""
    model = _make_model(koopman=koopman, seed=7)
    model.eval()
    seq_a = _sequence(num_nodes=3, seed=11, scale=0.2)
    seq_b = _sequence(num_nodes=5, seed=12, scale=4.0)
    weights = LossWeights()
    per_sequence = mean_training_loss_breakdown(
        [
            compute_training_loss(model, seq_a, weights),
            compute_training_loss(model, seq_b, weights),
        ]
    )
    batched = compute_batched_training_loss(model, (seq_a, seq_b), weights)
    _assert_close(batched.reconstruction, per_sequence.reconstruction)
    _assert_close(batched.total, per_sequence.total)


def test_batched_forward_matches_per_sequence_mean() -> None:
    """Forward consistency uses the same per-graph-then-mean reduction."""
    model = _make_model(koopman="graph", seed=8)
    model.eval()
    seq_a = _sequence(num_nodes=3, seed=21)
    seq_b = _sequence(num_nodes=5, seed=22)
    weights = LossWeights(reconstruction=1.0, forward=1.0)
    per_sequence = mean_training_loss_breakdown(
        [
            compute_training_loss(model, seq_a, weights),
            compute_training_loss(model, seq_b, weights),
        ]
    )
    batched = compute_batched_training_loss(model, (seq_a, seq_b), weights)
    _assert_close(batched.reconstruction, per_sequence.reconstruction)
    _assert_close(batched.forward, per_sequence.forward)
    _assert_close(batched.total, per_sequence.total)


def test_batched_eigenvalue_term_stays_per_sequence() -> None:
    """Eigenvalue hinge is the mean of per-graph spectra, not the union."""
    model = _make_model(koopman="graph", seed=9)
    model.eval()
    seq_a = _sequence(num_nodes=3, seed=31)
    seq_b = _sequence(num_nodes=5, seed=32)
    weights = LossWeights(reconstruction=1.0, eigenvalue=1.0)
    per_sequence = mean_training_loss_breakdown(
        [
            compute_training_loss(model, seq_a, weights),
            compute_training_loss(model, seq_b, weights),
        ]
    )
    batched = compute_batched_training_loss(model, (seq_a, seq_b), weights)
    _assert_close(batched.eigenvalue, per_sequence.eigenvalue)
    _assert_close(batched.total, per_sequence.total)


def test_default_fit_does_not_call_batched_loss() -> None:
    """Default ``MultiTrajectory`` fit keeps the per-sequence Python loop."""
    model = _make_model(seed=10)
    seq_a = _sequence(num_nodes=3, num_timesteps=3, seed=41)
    seq_b = _sequence(num_nodes=3, num_timesteps=3, seed=42)
    with patch(
        "koopman_graph.training.epochs.compute_batched_training_loss"
    ) as batched:
        model.fit(MultiTrajectory((seq_a, seq_b)), epochs=1, lr=1e-2)
        batched.assert_not_called()


def test_fit_batch_graphs_trains_and_allows_single_trajectory() -> None:
    """``batch_graphs=True`` trains a multi-graph batch and a trivial batch."""
    model = _make_model(koopman="graph", seed=13)
    seq_a = _sequence(num_nodes=3, num_timesteps=3, seed=51)
    seq_b = _sequence(num_nodes=5, num_timesteps=3, seed=52)
    history = model.fit(
        MultiTrajectory((seq_a, seq_b)),
        epochs=2,
        lr=1e-2,
        batch_graphs=True,
    )
    assert history.epochs == 2
    assert len(history.loss) == 2

    single = _make_model(seed=14)
    solo = _sequence(num_nodes=4, num_timesteps=3, seed=53)
    solo_history = single.fit(solo, epochs=1, lr=1e-2, batch_graphs=True)
    assert solo_history.epochs == 1


def test_batch_graphs_rejects_windowed_and_ddp() -> None:
    """Windowed sampling and DDP are mutually exclusive with graph batching."""
    model = _make_model(seed=15)
    sequence = _sequence(num_nodes=3, num_timesteps=4, seed=61)
    with pytest.raises(ValueError, match="window_length"):
        model.fit(
            sequence,
            epochs=1,
            batch_graphs=True,
            window_length=3,
        )
    with pytest.raises(ValueError, match="ddp"):
        model.fit(
            sequence,
            epochs=1,
            batch_graphs=True,
            strategy="ddp",
        )


def test_batch_graphs_rejects_hetero_delay_orbits_and_topology() -> None:
    """Refused families raise a clear ``ValueError`` / ``TypeError``."""
    model = _make_model(seed=16)
    hetero = HeteroData()
    hetero["node"].x = torch.randn(4, 3)
    hetero["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    hetero_seq = HeteroGraphSnapshotSequence([hetero, hetero.clone()])
    with pytest.raises(TypeError, match="GraphSnapshotSequence"):
        compute_batched_training_loss(model, (hetero_seq,), LossWeights())

    delay_encoder = GNNEncoder(
        in_channels=6,
        hidden_channels=8,
        latent_dim=4,
        num_layers=1,
    )
    delay_decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=3,
        num_layers=1,
    )
    delayed = GraphKoopmanModel(
        encoder=delay_encoder,
        decoder=delay_decoder,
        latent_dim=4,
        time_step=0.1,
        n_delays=2,
    )
    sequence = _sequence(num_nodes=3, seed=71)
    with pytest.raises(ValueError, match="n_delays"):
        compute_batched_training_loss(delayed, (sequence,), LossWeights())

    orbit = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
        koopman_orbit_partition=((0, 1), (2,)),
    )
    with pytest.raises(ValueError, match="orbit"):
        compute_batched_training_loss(orbit, (sequence,), LossWeights())

    learned = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
        learn_topology="self_adaptive",
    )
    with pytest.raises(ValueError, match="topology"):
        compute_batched_training_loss(learned, (sequence,), LossWeights())

    continuous = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman="continuous_graph",
        dynamics_mode="continuous",
    )
    with pytest.raises(ValueError, match="discrete per-node or graph"):
        compute_batched_training_loss(continuous, (sequence,), LossWeights())
