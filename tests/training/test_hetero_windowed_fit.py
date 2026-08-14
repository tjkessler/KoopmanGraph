"""Single-process windowed fit for multiplex and typed hetero sequences."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    NeighborWindowSampler,
    WindowSampler,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.training import run_fit_loop

# --- multiplex fixtures -----------------------------------------------------

_MULTIPLEX_EDGES_R1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
_MULTIPLEX_EDGES_R2 = torch.tensor([[0, 2], [2, 3]], dtype=torch.long)


def _multiplex_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(4, 3, generator=generator)
    data["node", "r1", "node"].edge_index = _MULTIPLEX_EDGES_R1
    data["node", "r2", "node"].edge_index = _MULTIPLEX_EDGES_R2
    return data


def _multiplex_sequence(
    *,
    num_timesteps: int = 5,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_multiplex_snapshot(seed=seed + t) for t in range(num_timesteps)]
    )


def _multiplex_model(*, seed: int = 0) -> GraphKoopmanModel:
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
    )


# --- typed fixtures ---------------------------------------------------------

NODE_TYPES = ("a", "b")
EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"), ("a", "r2", "a"))
FEATURE_DIMS = {"a": 2, "b": 3}
NUM_NODES = {"a": 4, "b": 3}
LATENT_DIM = 4

_EDGES_AB = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
_EDGES_BA = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
_EDGES_AA = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def _typed_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(
        NUM_NODES["a"],
        FEATURE_DIMS["a"],
        generator=generator,
    )
    snapshot["b"].x = torch.randn(
        NUM_NODES["b"],
        FEATURE_DIMS["b"],
        generator=generator,
    )
    snapshot["a", "r0", "b"].edge_index = _EDGES_AB
    snapshot["b", "r1", "a"].edge_index = _EDGES_BA
    snapshot["a", "r2", "a"].edge_index = _EDGES_AA
    return snapshot


def _typed_sequence(
    *,
    num_timesteps: int = 5,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_typed_snapshot(seed=seed + t) for t in range(num_timesteps)]
    )


def _typed_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=LATENT_DIM,
            hidden_channels=8,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=LATENT_DIM,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
    )


# --- smokes -----------------------------------------------------------------


def test_multiplex_windowed_fit_smoke() -> None:
    """Multiplex ``fit(..., window_length=...)`` trains with finite loss."""
    sequence = _multiplex_sequence(seed=1)
    model = _multiplex_model(seed=1)
    history = model.fit(
        sequence,
        epochs=2,
        window_length=3,
        batch_size=2,
        window_seed=0,
        lr=1e-2,
    )
    assert history.epochs == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history.loss)


def test_multiplex_windowed_run_fit_loop_with_window_sampler() -> None:
    """Pre-built ``WindowSampler`` works for multiplex hetero sequences."""
    sequence = _multiplex_sequence(seed=2)
    model = _multiplex_model(seed=2)
    sampler = WindowSampler(
        sequence,
        window_length=3,
        batch_size=2,
        windows_per_epoch=4,
        shuffle=True,
        seed=0,
    )
    history = run_fit_loop(
        model,
        [sequence],
        epochs=1,
        sampler=sampler,
        lr=1e-2,
        device="cpu",
    )
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_typed_windowed_fit_smoke() -> None:
    """Typed multi-node-type windowed ``fit`` trains with finite loss."""
    sequence = _typed_sequence(seed=3)
    model = _typed_model(seed=3)
    history = model.fit(
        sequence,
        epochs=2,
        window_length=3,
        batch_size=2,
        window_seed=0,
        lr=1e-2,
    )
    assert history.epochs == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history.loss)


def test_neighbor_window_sampler_rejects_hetero_at_construction() -> None:
    """NeighborWindowSampler construction names the homo-only restriction."""
    with pytest.raises(
        ValueError,
        match="does not support HeteroGraphSnapshotSequence",
    ):
        NeighborWindowSampler(
            _multiplex_sequence(),
            window_length=2,
            num_nodes=2,
            num_hops=1,
            batch_size=1,
            shuffle=False,
        )


def test_run_fit_loop_rejects_neighbor_sampler_with_hetero_train() -> None:
    """Fit path rejects NeighborWindowSampler when train sequences are hetero."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    homo = GraphSnapshotSequence(
        [
            Data(x=torch.randn(2, 3), edge_index=edge_index, num_nodes=2)
            for _ in range(4)
        ]
    )
    sampler = NeighborWindowSampler(
        homo,
        window_length=2,
        num_nodes=1,
        num_hops=1,
        batch_size=1,
        shuffle=False,
    )
    model = _multiplex_model(seed=0)
    with pytest.raises(ValueError, match="NeighborWindowSampler is homogeneous-only"):
        run_fit_loop(
            model,
            [_multiplex_sequence(seed=0)],
            epochs=1,
            sampler=sampler,
            device="cpu",
        )
