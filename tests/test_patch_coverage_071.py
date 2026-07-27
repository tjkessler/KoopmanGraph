"""Patch-coverage gaps for the 0.7.1 Codecov patch gate (target ≥ 90%)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.data import WindowSampler
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.hierarchical.pooling import pool_features_with_steps
from koopman_graph.losses.rollout import _encode_rollout_origin_latent
from koopman_graph.operators.continuous import ContinuousKoopmanOperator
from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator
from koopman_graph.training import LossWeights
from koopman_graph.training.epochs import prepare_training_amp, train_windowed_epoch
from koopman_graph.training.extra_objectives import (
    compute_worst_case_reconstruction_loss,
)
from koopman_graph.training.pair_objectives import (
    _dense_networked_inverse_for_snapshot,
    _reconstruction_from_predictions,
    compute_sequence_loss,
    one_step_prediction,
    topologies_equal,
)
from koopman_graph.training.timestep_encode import encode_at_timestep


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


def _sequence(num_timesteps: int = 3, *, num_nodes: int = 3) -> GraphSnapshotSequence:
    edge_index = _path_edges(num_nodes)
    return GraphSnapshotSequence(
        [
            Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
            for _ in range(num_timesteps)
        ]
    )


def test_encode_at_timestep_falls_back_to_encode() -> None:
    """Models without ``encode_at`` use ``encode(snapshot)``."""
    model = _tiny_model()
    model.encode_at = None  # type: ignore[method-assign, assignment]
    sequence = _sequence(2)
    with torch.no_grad():
        z = encode_at_timestep(model, sequence, 0)
    assert z.shape == (3, 3)


def test_one_step_prediction_model_forward_fallback() -> None:
    """n_delays==1 without cache uses ``model(source, …)``."""
    model = _tiny_model()
    sequence = _sequence(2)
    with torch.no_grad():
        pred = one_step_prediction(model, sequence, 0)
    assert pred.shape == sequence[1].x.shape


def test_reconstruction_and_sequence_loss_guard_short_sequences() -> None:
    """Prediction helpers reject sequences shorter than two snapshots."""
    model = _tiny_model()
    short = _sequence(1)
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        _reconstruction_from_predictions(model, short, [])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_sequence_loss(model, short)


def test_reconstruction_rejects_prediction_length_mismatch() -> None:
    """``predictions`` length must equal the number of consecutive pairs."""
    model = _tiny_model()
    sequence = _sequence(3)
    with pytest.raises(ValueError, match="predictions length"):
        _reconstruction_from_predictions(model, sequence, [torch.zeros(3, 2)])


def test_sequence_loss_without_cache() -> None:
    """``cache is None`` routes through ``mean_pair_sequence_loss``."""
    model = _tiny_model()
    sequence = _sequence(3)
    loss = compute_sequence_loss(model, sequence, cache=None)
    assert torch.isfinite(loss)


def test_topologies_equal_hyperedge_branches() -> None:
    """Cover weight/hyperedge presence and content mismatches."""
    edges = _path_edges(3)
    weights = torch.ones(edges.shape[1])
    hyp_a = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyp_b = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    hyp_w = torch.ones(1)

    assert not topologies_equal(edges, weights, edges, None)
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_index_b=None,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_index_b=hyp_b,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a,
        hyperedge_weight_b=None,
    )
    assert not topologies_equal(
        edges,
        None,
        edges,
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a,
        hyperedge_weight_b=2.0 * hyp_w,
    )
    assert topologies_equal(
        edges,
        None,
        edges.clone(),
        None,
        hyperedge_index_a=hyp_a,
        hyperedge_weight_a=hyp_w,
        hyperedge_index_b=hyp_a.clone(),
        hyperedge_weight_b=hyp_w.clone(),
    )


def test_dense_networked_inverse_missing_hyperedge_returns_none() -> None:
    """Hypergraph inverse helper returns ``None`` without incidence."""
    op = HypergraphKoopmanOperator(latent_dim=2, sparsity="dense")
    snap = Data(x=torch.randn(3, 2), edge_index=_path_edges(3))
    assert _dense_networked_inverse_for_snapshot(op, snap) is None


def test_hypergraph_dense_effective_inverse_rejects_block_diagonal() -> None:
    """Hypergraph ``dense_effective_inverse`` requires ``sparsity='dense'``."""
    op = HypergraphKoopmanOperator(latent_dim=2, sparsity="block_diagonal")
    hyp = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        op.dense_effective_inverse(hyp, num_nodes=3)


def test_prepare_training_amp_cuda_path_without_gpu() -> None:
    """CUDA device type enables AMP even when no GPU is present (mocked)."""
    fake_cuda = SimpleNamespace(type="cuda")
    stub_scaler = MagicMock(name="GradScaler")
    with patch("torch.amp.GradScaler", return_value=stub_scaler) as ctor:
        enabled, dtype, scaler = prepare_training_amp(
            True,
            fake_cuda,  # type: ignore[arg-type]
            amp_dtype=torch.bfloat16,
        )
    assert enabled is True
    assert dtype is torch.bfloat16
    assert scaler is stub_scaler
    ctor.assert_called_once()

    with patch("torch.amp.GradScaler") as ctor2:
        enabled2, dtype2, scaler2 = prepare_training_amp(
            True,
            fake_cuda,  # type: ignore[arg-type]
            grad_scaler=stub_scaler,
        )
    assert enabled2 is True
    assert dtype2 is torch.float16
    assert scaler2 is stub_scaler
    ctor2.assert_not_called()


def test_train_windowed_epoch_amp_autocast_branch() -> None:
    """Forced AMP path exercises the windowed autocast body."""
    model = _tiny_model()
    sequence = _sequence(4)
    sampler = WindowSampler(sequence, window_length=3, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    stub_scaler = MagicMock()
    stub_scaler.scale.side_effect = lambda loss: loss
    stub_scaler.step.side_effect = lambda opt: opt.step()

    with (
        patch(
            "koopman_graph.training.epochs.prepare_training_amp",
            return_value=(True, torch.float16, stub_scaler),
        ),
        patch(
            "torch.amp.autocast",
            side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
        ),
    ):
        breakdown = train_windowed_epoch(
            model,
            sampler,
            optimizer,
            LossWeights(),
            epoch=0,
        )
    assert torch.isfinite(breakdown.total)


def test_continuous_graph_topology_payload_equal_mismatch_branches() -> None:
    """``_topology_payload_equal`` false paths for index/weight mismatches."""
    op = ContinuousGraphKoopmanOperator(latent_dim=2)
    edges = _path_edges(3)
    other = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    weights = torch.ones(edges.shape[1])
    assert op._topology_payload_equal(edges, None, other, None) is False
    assert op._topology_payload_equal(edges, weights, edges, None) is False
    assert op._topology_payload_equal(edges, None, edges, None) is True
    assert op._topology_payload_equal(edges, weights, edges, 2.0 * weights) is False


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


def test_encode_rollout_origin_delay_learned_topology() -> None:
    """Delay + self-adaptive origin uses history_from_snapshots + encode."""
    model = _tiny_model(n_delays=2, learn_topology="self_adaptive")
    edge_index = _path_edges(3)
    history = [Data(x=torch.randn(3, 2), edge_index=edge_index)]
    origin = Data(x=torch.randn(3, 2), edge_index=edge_index)
    model.eval()
    with torch.no_grad():
        z, resolved_ei, resolved_ew = model.encode_rollout_origin(
            origin, history=history
        )
    assert z.shape[0] == 3
    assert resolved_ei.shape[0] == 2
    assert resolved_ew is not None


def test_rollout_encode_origin_falls_back_without_encode_at() -> None:
    """Rollout helper falls back to ``encode`` when ``encode_at`` is absent."""
    model = _tiny_model()
    model.encode_at = None  # type: ignore[method-assign, assignment]
    sequence = _sequence(2)
    with torch.no_grad():
        z = _encode_rollout_origin_latent(model, sequence, 0)
    assert z.shape == (3, 3)


def test_worst_case_predictions_guard_short_sequence() -> None:
    """Worst-case path with precomputed predictions rejects short sequences."""
    model = _tiny_model()
    short = _sequence(1)
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_worst_case_reconstruction_loss(model, short, weight=1.0, predictions=[])


def test_structural_assembly_factors_dense_returns_empty() -> None:
    """Dense discrete/continuous parameterization hits the empty-factor branch."""
    discrete = KoopmanOperator(latent_dim=2, parameterization="dense")
    continuous = ContinuousKoopmanOperator(latent_dim=2, parameterization="dense")
    assert discrete._structural_assembly_factors() == ()
    assert continuous._structural_assembly_factors() == ()
