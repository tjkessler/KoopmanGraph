"""Coverage and error-path tests for :mod:`koopman_graph.hierarchical`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.hierarchical.pooling import (
    ScatterUnpool,
    pool_control,
    pool_control_sequence,
    pool_features_with_steps,
)


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def _tiny_model(
    *, n_delays: int = 1, learn_topology: str | None = None
) -> GraphKoopmanModel:
    in_channels = n_delays * 2
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels, 8, 3, num_layers=1),
        decoder=GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        n_delays=n_delays,
        learn_topology=learn_topology,
    )


def test_hold_perm_rejects_empty_sequence_and_missing_features() -> None:
    """``hold_perm`` pool guards for empty sequences / missing ``x``."""
    hier = HierarchicalGraphKoopmanModel(
        _tiny_model(), pool_ratios=(0.5,), pool_schedule="hold_perm"
    )
    empty = MagicMock()
    empty.num_timesteps = 0
    with pytest.raises(ValueError, match="at least one snapshot"):
        hier._pool_sequence(empty)

    edge_index = _path_edges(4)
    first = Data(x=torch.randn(4, 2), edge_index=edge_index)
    missing_x = Data(x=None, edge_index=edge_index, num_nodes=4)
    sequence = MagicMock()
    sequence.num_timesteps = 2
    sequence.__getitem__ = lambda _self, index: first if index == 0 else missing_x
    sequence.__iter__ = lambda _self: iter([first, missing_x])
    with pytest.raises(ValueError, match="requires snapshot.x"):
        hier._pool_sequence(sequence)


def test_pool_features_with_steps_rejects_empty_steps() -> None:
    """Empty pool-step lists are rejected."""
    with pytest.raises(ValueError, match="at least one PoolStep"):
        pool_features_with_steps(torch.randn(4, 2), [])


def test_hierarchical_pooling_and_resolution_gaps() -> None:
    base = GraphKoopmanModel(
        encoder=GNNEncoder(1, 4, 2),
        decoder=GNNDecoder(2, 4, 1),
        latent_dim=2,
        time_step=0.1,
    )
    model = HierarchicalGraphKoopmanModel(base, pool_ratios=(0.5,))
    from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

    seq = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=6,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    model.fit(seq, epochs=1, lr=1e-2, unpool_epochs=0)
    with pytest.raises(ValueError, match="resolution"):
        model.predict(seq[0], steps=1, resolution="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolution int"):
        model.predict(seq[0], steps=1, resolution=99)
    assert len(model.predict(seq[0], steps=1, resolution=0)) == 1
    assert len(model.predict(seq[0], steps=1, resolution="coarse")) == 1

    with pytest.raises(ValueError, match="control must have shape"):
        pool_control(torch.randn(2, 2, 2), [torch.tensor([0, 1])])
    with pytest.raises(ValueError, match="control_inputs must have shape"):
        pool_control_sequence(torch.randn(2), [[torch.tensor([0, 1])]])
    with pytest.raises(ValueError, match="perm chains"):
        pool_control_sequence(
            torch.randn(2, 8, 1),
            [[torch.arange(4)]],
        )
    pooled = pool_control(torch.randn(8, 1), [torch.arange(4)])
    assert pooled.shape == (4, 1)
    pooled_seq = pool_control_sequence(
        torch.randn(2, 8, 1),
        [[torch.arange(4)], [torch.arange(4)]],
    )
    assert pooled_seq.shape == (2, 4, 1)

    unpool = ScatterUnpool(channels=2, refine=True)
    unpool.train()
    out = unpool(torch.randn(1, 2), torch.tensor([0]), num_fine=2)
    assert out.shape[0] == 2
