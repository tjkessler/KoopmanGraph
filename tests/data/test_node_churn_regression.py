"""Release-blocking node-churn regression suite (TASK-1921 / DESIGN R4).

**Release blocker.** Failures here mean presence-mask churn broke default-off
bit-compatibility with 0.10, or window / world-size-1 DDP / neighbor shards
misaligned presence rows relative to features. Do not ship such a change
without an explicit, documented fix.

Coverage
--------
1. Seeded golden one-step loss for homogeneous and multiplex sequences
   **without** presence masks (sequence path; same goldens as TASK-1820).
2. ``WindowSampler`` temporal windows carry correctly aligned presence masks.
3. ``DistributedWindowSampler`` with ``world_size=1`` matches
   ``WindowSampler`` and preserves presence alignment (homo + multiplex).
4. ``NeighborWindowSampler`` / induced subgraphs subset presence columns and
   ``entity_ids`` with the sampled node set.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import k_hop_subgraph

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    NeighborWindowSampler,
    WindowSampler,
)
from koopman_graph.data.sampling import induce_neighbor_subgraph_sequence
from koopman_graph.distributed import DistributedWindowSampler
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.training.pair_objectives import _one_step_pair

# Locked with tests/model/test_hetero_shared_d_regression.py (TASK-1820 / 2026-07-31).
_LOSS_ABS = 1e-5
_GOLDEN_HOMO_ONE_STEP = 0.6180354356765747
_GOLDEN_MULTIPLEX_ONE_STEP = 1.1788241863250732
_LATENT_DIM = 4


def _homo_snapshot(*, seed: int) -> Data:
    generator = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(4, 3, generator=generator),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )


def _multiplex_snapshot(*, seed: int) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(4, 3, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _homo_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=3,
            hidden_channels=8,
            latent_dim=_LATENT_DIM,
            num_layers=1,
        ),
        decoder=GNNDecoder(
            latent_dim=_LATENT_DIM,
            hidden_channels=8,
            out_channels=3,
            num_layers=1,
        ),
        latent_dim=_LATENT_DIM,
        time_step=1.0,
    )


def _multiplex_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=_LATENT_DIM,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=_LATENT_DIM,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=_LATENT_DIM,
        time_step=1.0,
        koopman="hetero_graph",
    )


def _presence_pattern(num_timesteps: int, num_nodes: int) -> torch.Tensor:
    masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
    if num_timesteps >= 3 and num_nodes >= 2:
        masks[2:, -1] = False
    return masks


def _homo_sequence_with_presence(
    *,
    timesteps: int = 5,
    num_nodes: int = 4,
    edge_index: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    if edge_index is None:
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    # Encode original node id in feature channel 0 for neighbor remapping checks.
    features = torch.zeros(timesteps, num_nodes, 3)
    for node in range(num_nodes):
        features[:, node, 0] = float(node)
        features[:, node, 1] = torch.arange(timesteps, dtype=torch.float32)
    presence = _presence_pattern(timesteps, num_nodes)
    ids = tuple(f"n{i}" for i in range(num_nodes))
    return GraphSnapshotSequence.from_arrays(
        features,
        edge_index,
        presence_masks=presence,
        entity_ids=ids,
        allow_node_churn=True,
    )


def _multiplex_sequence_with_presence(
    *,
    timesteps: int = 5,
) -> HeteroGraphSnapshotSequence:
    snapshots = [_multiplex_snapshot(seed=10 + t) for t in range(timesteps)]
    # Overwrite features with identifiable rows while keeping topology.
    for t, snap in enumerate(snapshots):
        x = torch.zeros(4, 3)
        for node in range(4):
            x[node, 0] = float(node)
            x[node, 1] = float(t)
        snap["node"].x = x
    presence = {"node": _presence_pattern(timesteps, 4)}
    return HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )


def test_homo_sequence_without_presence_matches_shared_d_golden() -> None:
    """Homogeneous sequence path (no presence) locks the 0.10 golden loss."""
    sequence = GraphSnapshotSequence([_homo_snapshot(seed=1), _homo_snapshot(seed=2)])
    assert not sequence.has_presence_masks
    model = _homo_model(seed=0)
    loss = _one_step_pair(model, sequence, 0)
    assert float(loss.detach()) == pytest.approx(_GOLDEN_HOMO_ONE_STEP, abs=_LOSS_ABS)


def test_multiplex_sequence_without_presence_matches_shared_d_golden() -> None:
    """Multiplex sequence path (no presence) locks the shared-d golden loss."""
    sequence = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot(seed=1), _multiplex_snapshot(seed=2)]
    )
    assert not sequence.has_presence_masks
    model = _multiplex_model(seed=0)
    loss = _one_step_pair(model, sequence, 0)
    assert float(loss.detach()) == pytest.approx(
        _GOLDEN_MULTIPLEX_ONE_STEP,
        abs=_LOSS_ABS,
    )


def test_window_sampler_preserves_presence_alignment() -> None:
    """WindowSampler slices keep presence rows aligned with window features."""
    sequence = _homo_sequence_with_presence(timesteps=5)
    assert sequence.presence_masks is not None
    sampler = WindowSampler(
        sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
    )
    origins = [(0, start) for start in range(sequence.num_timesteps - 3 + 1)]
    origin_iter = iter(origins)
    for batch in sampler.iter_epoch(0):
        for window in batch:
            seq_idx, start = next(origin_iter)
            assert seq_idx == 0
            expected = sequence.slice(start, start + 3)
            assert window.has_presence_masks
            assert window.allow_node_churn
            assert window.entity_ids == sequence.entity_ids
            assert window.presence_masks is not None
            assert expected.presence_masks is not None
            assert torch.equal(window.presence_masks, expected.presence_masks)
            for t in range(window.num_timesteps):
                assert torch.equal(window[t].x, expected[t].x)
                assert torch.equal(
                    window.presence_mask_at(t),
                    sequence.presence_mask_at(start + t),
                )


def test_distributed_world_size_one_matches_window_sampler_with_presence() -> None:
    """World-size-1 DDP shards match WindowSampler and keep presence aligned."""
    sequence = _homo_sequence_with_presence(timesteps=5)
    baseline = WindowSampler(
        sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
    )
    distributed = DistributedWindowSampler(
        sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=1,
    )
    base_batches = list(baseline.iter_epoch(0))
    dist_batches = list(distributed.iter_epoch(0))
    assert len(base_batches) == len(dist_batches)
    for left_batch, right_batch in zip(base_batches, dist_batches, strict=True):
        assert len(left_batch) == len(right_batch)
        for left, right in zip(left_batch, right_batch, strict=True):
            assert left.has_presence_masks and right.has_presence_masks
            assert left.presence_masks is not None
            assert right.presence_masks is not None
            assert torch.equal(left.presence_masks, right.presence_masks)
            assert torch.equal(left[0].x, right[0].x)
            assert left.entity_ids == right.entity_ids == sequence.entity_ids


def test_distributed_world_size_one_multiplex_presence_alignment() -> None:
    """World-size-1 hetero shards keep per-type presence aligned with slices."""
    sequence = _multiplex_sequence_with_presence(timesteps=5)
    distributed = DistributedWindowSampler(
        sequence,
        window_length=3,
        batch_size=1,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=1,
    )
    starts = list(range(sequence.num_timesteps - 3 + 1))
    start_iter = iter(starts)
    for batch in distributed.iter_epoch(0):
        for window in batch:
            start = next(start_iter)
            expected = sequence.slice(start, start + 3)
            assert isinstance(window, HeteroGraphSnapshotSequence)
            assert window.has_presence_masks
            assert window.presence_masks is not None
            assert expected.presence_masks is not None
            assert torch.equal(
                window.presence_masks["node"],
                expected.presence_masks["node"],
            )
            assert torch.equal(window[0]["node"].x, expected[0]["node"].x)


def test_neighbor_subgraph_subsets_presence_and_entity_ids() -> None:
    """Induced neighbor windows subset presence columns with sampled nodes."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    sequence = _homo_sequence_with_presence(
        timesteps=4,
        num_nodes=5,
        edge_index=edge_index,
    )
    temporal = sequence.slice(1, 4)
    seed_nodes = torch.tensor([1, 3], dtype=torch.long)
    subset, _, _, _ = k_hop_subgraph(
        seed_nodes,
        1,
        temporal.edge_index,
        relabel_nodes=False,
        num_nodes=temporal.num_nodes,
    )
    subset = subset.sort().values

    window = induce_neighbor_subgraph_sequence(
        temporal,
        seed_nodes=seed_nodes,
        num_hops=1,
    )
    assert window.has_presence_masks
    assert window.allow_node_churn
    assert window.presence_masks is not None
    assert temporal.presence_masks is not None
    assert torch.equal(window.presence_masks, temporal.presence_masks[:, subset])
    assert window.entity_ids == tuple(
        temporal.entity_ids[int(i)]
        for i in subset.tolist()  # type: ignore[index]
    )
    for t in range(window.num_timesteps):
        for local, original in enumerate(subset.tolist()):
            assert float(window[t].x[local, 0].item()) == float(original)
            assert bool(window.presence_mask_at(t)[local].item()) == bool(
                temporal.presence_mask_at(t)[original].item()
            )


def test_neighbor_window_sampler_preserves_presence_row_alignment() -> None:
    """NeighborWindowSampler yields windows whose presence matches feature ids."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    sequence = _homo_sequence_with_presence(
        timesteps=4,
        num_nodes=5,
        edge_index=edge_index,
    )
    sampler = NeighborWindowSampler(
        sequence,
        window_length=2,
        num_nodes=2,
        num_hops=1,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    for batch in sampler.iter_epoch(0):
        for window in batch:
            assert window.has_presence_masks
            assert window.allow_node_churn
            assert window.presence_masks is not None
            assert window.entity_ids is not None
            assert window.num_nodes == window.presence_masks.shape[1]
            assert len(window.entity_ids) == window.num_nodes
            for t in range(window.num_timesteps):
                for local in range(window.num_nodes):
                    original = int(window[t].x[local, 0].item())
                    # Feature channel-0 encodes the parent universe index.
                    assert window.entity_ids[local] == f"n{original}"
                    # Recover parent timestep from channel-1 (written in builder).
                    parent_t = int(window[t].x[local, 1].item())
                    assert bool(window.presence_mask_at(t)[local].item()) == bool(
                        sequence.presence_mask_at(parent_t)[original].item()
                    )
