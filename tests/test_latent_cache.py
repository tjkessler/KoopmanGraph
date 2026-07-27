"""Tests for shared sequence latent cache (TASK-1500 / TASK-1502 / TASK-1510)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.losses import rollout_sequence_loss
from koopman_graph.operators import GlobalLocalKoopmanOperator, GraphKoopmanOperator
from koopman_graph.training import (
    ExtraLosses,
    LossWeights,
    compute_backward_consistency_sequence_loss,
    compute_forward_consistency_sequence_loss,
    compute_sequence_loss,
    compute_training_loss,
)
from koopman_graph.training.extra_objectives import (
    compute_pde_residual_loss,
    compute_worst_case_reconstruction_loss,
)
from koopman_graph.training.history import TrainingLossBreakdown
from koopman_graph.training.latent_cache import (
    SequenceLatentCache,
    encode_sequence_latents,
    latent_window_from_cache,
)
from koopman_graph.training.pair_objectives import (
    encode_at_timestep,
    teacher_forced_latent_window,
)

# Float32 eval parity: model() vs encode→advance→decode can differ by a few
# ulps on the n_delays==1 recon path; encode-path-only cases use exact match.
_PARITY_ATOL = 1e-6
_PAIR_WEIGHTS = LossWeights(reconstruction=1.0, forward=1.0, backward=1.0)


def _make_sequence(
    *,
    num_timesteps: int = 4,
    num_nodes: int = 5,
    channels: int = 3,
    seed: int = 0,
    with_masks: bool = False,
) -> GraphSnapshotSequence:
    """Build a short static-topology sequence for latent-cache tests."""
    torch.manual_seed(seed)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
        dtype=torch.long,
    )
    snapshots = [
        Data(x=torch.randn(num_nodes, channels), edge_index=edge_index)
        for _ in range(num_timesteps)
    ]
    kwargs: dict[str, torch.Tensor] = {}
    if with_masks:
        masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
        masks[1, 0] = False
        kwargs["observation_masks"] = masks
    return GraphSnapshotSequence(snapshots, **kwargs)


def _make_model(
    *,
    koopman: str = "pernode",
    n_delays: int = 1,
    seed: int = 0,
    feature_dim: int = 3,
) -> GraphKoopmanModel:
    """Construct a small model for encode-count and window tests."""
    torch.manual_seed(seed)
    in_channels = n_delays * feature_dim
    encoder = GNNEncoder(
        in_channels=in_channels,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=feature_dim,
        num_layers=2,
    )
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,  # type: ignore[arg-type]
        n_delays=n_delays,
    )


def _make_dynamic_topology_sequence(
    *,
    num_timesteps: int = 4,
    num_nodes: int = 5,
    channels: int = 3,
    seed: int = 0,
) -> GraphSnapshotSequence:
    """Alternating edge sets so pair topology differs from encode topology."""
    torch.manual_seed(seed)
    # Path graph vs cycle — must differ so target-snapshot topology is observable.
    edges_a = torch.tensor(
        [[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]],
        dtype=torch.long,
    )
    edges_b = torch.tensor(
        [
            [i for i in range(num_nodes)],
            [(i + 1) % num_nodes for i in range(num_nodes)],
        ],
        dtype=torch.long,
    )
    snapshots = []
    for timestep in range(num_timesteps):
        edges = edges_a if timestep % 2 == 0 else edges_b
        snapshots.append(
            Data(x=torch.randn(num_nodes, channels), edge_index=edges.clone())
        )
    return GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)


def _uncached_pair_breakdown(
    model: GraphKoopmanModel,
    sequence: GraphSnapshotSequence,
    weights: LossWeights,
) -> TrainingLossBreakdown:
    """Compose pair terms without a shared cache (legacy public API path)."""
    device = next(model.parameters()).device
    reconstruction = (
        compute_sequence_loss(model, sequence)
        if weights.reconstruction != 0.0
        else torch.zeros((), device=device)
    )
    forward = (
        compute_forward_consistency_sequence_loss(model, sequence)
        if weights.forward != 0.0
        else torch.zeros((), device=device)
    )
    backward = (
        compute_backward_consistency_sequence_loss(model, sequence)
        if weights.backward != 0.0
        else torch.zeros((), device=device)
    )
    total = (
        weights.reconstruction * reconstruction
        + weights.forward * forward
        + weights.backward * backward
    )
    zero = torch.zeros((), device=device)
    return TrainingLossBreakdown(
        reconstruction=reconstruction,
        forward=forward,
        backward=backward,
        rollout=zero,
        eigenvalue=zero,
        lie=zero,
        pde=zero,
        sparsity=zero,
        worst_case=zero,
        total=total,
    )


def _assert_pair_breakdown_close(
    cached: TrainingLossBreakdown,
    uncached: TrainingLossBreakdown,
    *,
    atol: float,
) -> None:
    """Compare recon/forward/backward/total; justify atol at call site."""
    torch.testing.assert_close(
        cached.reconstruction, uncached.reconstruction, rtol=0.0, atol=atol
    )
    torch.testing.assert_close(cached.forward, uncached.forward, rtol=0.0, atol=atol)
    torch.testing.assert_close(cached.backward, uncached.backward, rtol=0.0, atol=atol)
    torch.testing.assert_close(cached.total, uncached.total, rtol=0.0, atol=atol)


def _encoder_forward_count(
    model: GraphKoopmanModel,
    sequence: GraphSnapshotSequence,
    weights: LossWeights = _PAIR_WEIGHTS,
) -> int:
    """Return encoder forward hooks fired during ``compute_training_loss``."""
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    handle = model.encoder.register_forward_hook(_count)
    try:
        breakdown = compute_training_loss(model, sequence, weights)
    finally:
        handle.remove()
    assert torch.isfinite(breakdown.total)
    return hits["count"]


def test_encode_sequence_latents_length_and_values() -> None:
    """Cache length is T and entries match encode_at_timestep."""
    model = _make_model()
    sequence = _make_sequence(num_timesteps=4)
    cache = encode_sequence_latents(model, sequence)
    assert isinstance(cache, SequenceLatentCache)
    assert cache.num_timesteps == 4
    assert len(cache.z) == 4
    # Regression: teacher-forced encodings must match direct encode_at
    # (float32 allclose; same graph and weights, no stochastic layers).
    for index, latent in enumerate(cache.z):
        expected = encode_at_timestep(model, sequence, index)
        torch.testing.assert_close(latent, expected, rtol=0.0, atol=0.0)


def test_encode_sequence_latents_encoder_fires_once_per_timestep() -> None:
    """encode_sequence_latents runs the encoder forward exactly T times."""
    model = _make_model()
    sequence = _make_sequence(num_timesteps=5)
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    handle = model.encoder.register_forward_hook(_count)
    try:
        cache = encode_sequence_latents(model, sequence)
    finally:
        handle.remove()
    assert cache.num_timesteps == 5
    assert hits["count"] == 5


def test_encode_sequence_latents_delay_embedding_once_per_index() -> None:
    """Delay-embedding encode_at still yields one encoder forward per index."""
    # Base in_channels = n_delays * feature_dim (DelayEmbeddingEncoder contract).
    torch.manual_seed(0)
    n_delays = 3
    feature_dim = 3
    encoder = GNNEncoder(
        in_channels=n_delays * feature_dim,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    decoder = GNNDecoder(
        latent_dim=4, hidden_channels=8, out_channels=feature_dim, num_layers=2
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        n_delays=n_delays,
    )
    sequence = _make_sequence(num_timesteps=4, channels=feature_dim)
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    # Hook the delay wrapper: one forward per encode_at index.
    handle = model.encoder.register_forward_hook(_count)
    try:
        cache = encode_sequence_latents(model, sequence)
    finally:
        handle.remove()
    assert cache.num_timesteps == 4
    assert hits["count"] == 4


def test_latent_window_from_cache_matches_teacher_forced() -> None:
    """Cached global/local windows match teacher_forced_latent_window."""
    model = _make_model(koopman="global_local", seed=1)
    assert isinstance(model.koopman, GlobalLocalKoopmanOperator)
    sequence = _make_sequence(num_timesteps=6, seed=1)
    cache = encode_sequence_latents(model, sequence)
    window = model.koopman.local_window
    for timestep in range(sequence.num_timesteps):
        from_cache = latent_window_from_cache(cache, timestep, window)
        from_encode = teacher_forced_latent_window(model, sequence, timestep)
        assert from_encode is not None
        # Same encodings and stacking; exact match on CPU float32.
        torch.testing.assert_close(from_cache, from_encode, rtol=0.0, atol=0.0)


def test_latent_window_from_cache_validates_args() -> None:
    """Window helper rejects bad timestep / window."""
    model = _make_model()
    sequence = _make_sequence(num_timesteps=3)
    cache = encode_sequence_latents(model, sequence)
    with pytest.raises(ValueError, match="local_window"):
        latent_window_from_cache(cache, 0, 0)
    with pytest.raises(ValueError, match="timestep"):
        latent_window_from_cache(cache, 3, 2)


def test_cache_does_not_store_topology_or_masks() -> None:
    """Cache holds latents only; masks/topology remain on the sequence."""
    model = _make_model()
    sequence = _make_sequence(num_timesteps=3, with_masks=True)
    cache = encode_sequence_latents(model, sequence)
    assert hasattr(cache, "z")
    assert not hasattr(cache, "edge_index")
    assert not hasattr(cache, "observation_masks")
    assert sequence.has_observation_masks
    # Encoding with masks still produces finite latents (mask applied in encode_at).
    assert all(torch.isfinite(latent).all() for latent in cache.z)


def test_compute_training_loss_encodes_once_for_pair_terms() -> None:
    """Shared cache: recon+forward+backward encode the sequence once (TASK-1501)."""
    model = _make_model(seed=2)
    sequence = _make_sequence(num_timesteps=4, seed=2)
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    handle = model.encoder.register_forward_hook(_count)
    try:
        breakdown = compute_training_loss(
            model,
            sequence,
            LossWeights(reconstruction=1.0, forward=1.0, backward=1.0),
        )
    finally:
        handle.remove()
    assert torch.isfinite(breakdown.total)
    # One encode per timestep for the shared cache (T=4), not ~6*(T-1).
    assert hits["count"] == sequence.num_timesteps


def test_public_sequence_loss_without_cache_still_works() -> None:
    """compute_sequence_loss without cache keeps the public no-cache API."""
    model = _make_model(seed=3)
    sequence = _make_sequence(num_timesteps=3, seed=3)
    loss = compute_sequence_loss(model, sequence)
    assert torch.isfinite(loss)


def test_rollout_with_cache_matches_uncached_origin() -> None:
    """Rollout origin from cache matches a fresh encode (AR path unchanged)."""
    model = _make_model(seed=4)
    model.eval()
    sequence = _make_sequence(num_timesteps=5, seed=4)
    with torch.no_grad():
        cache = encode_sequence_latents(model, sequence)
        loss_cached = rollout_sequence_loss(
            model, sequence, horizon=2, start=0, cache=cache
        )
        loss_plain = rollout_sequence_loss(model, sequence, horizon=2, start=0)
    # Same origin latent and deterministic eval advance/decode.
    torch.testing.assert_close(loss_cached, loss_plain, rtol=0.0, atol=0.0)


def test_training_loss_parity_vs_uncached_pair_path() -> None:
    """Cached compute_training_loss matches legacy per-term encodes (TASK-1502)."""
    model = _make_model(seed=10)
    model.eval()
    sequence = _make_sequence(num_timesteps=5, seed=10)
    with torch.no_grad():
        cached = compute_training_loss(model, sequence, _PAIR_WEIGHTS)
        uncached = _uncached_pair_breakdown(model, sequence, _PAIR_WEIGHTS)
    # n_delays==1 recon: cached encode→advance→decode vs uncached model();
    # float32 eval noise bound (not bit-identical across those two paths).
    _assert_pair_breakdown_close(cached, uncached, atol=_PARITY_ATOL)


def test_training_loss_parity_delay_embedding_exact() -> None:
    """Delay path uses encode→advance→decode with or without cache (exact)."""
    n_delays = 3
    model = _make_model(n_delays=n_delays, seed=11)
    model.eval()
    sequence = _make_sequence(num_timesteps=5, seed=11)
    with torch.no_grad():
        cached = compute_training_loss(model, sequence, _PAIR_WEIGHTS)
        uncached = _uncached_pair_breakdown(model, sequence, _PAIR_WEIGHTS)
    # Same numeric path; only encode reuse differs → exact float32 match.
    _assert_pair_breakdown_close(cached, uncached, atol=0.0)


def test_training_loss_parity_global_local_exact() -> None:
    """Global/local always uses the encode path; cache is exact (TASK-1502)."""
    model = _make_model(koopman="global_local", seed=12)
    assert isinstance(model.koopman, GlobalLocalKoopmanOperator)
    model.eval()
    sequence = _make_sequence(num_timesteps=6, seed=12)
    with torch.no_grad():
        cached = compute_training_loss(model, sequence, _PAIR_WEIGHTS)
        uncached = _uncached_pair_breakdown(model, sequence, _PAIR_WEIGHTS)
    _assert_pair_breakdown_close(cached, uncached, atol=0.0)


def test_encoder_forward_count_pair_terms_is_t() -> None:
    """With recon+forward+backward, encoder fires once per timestep."""
    model = _make_model(seed=13)
    sequence = _make_sequence(num_timesteps=6, seed=13)
    assert _encoder_forward_count(model, sequence) == sequence.num_timesteps


def test_encoder_forward_count_delay_embedding_is_t() -> None:
    """Delay embedding still budgets T encoder forwards for three pair terms."""
    n_delays = 3
    model = _make_model(n_delays=n_delays, seed=14)
    sequence = _make_sequence(num_timesteps=5, seed=14)
    # Delay stacking happens inside encode_at; hook on wrapper sees T calls.
    assert _encoder_forward_count(model, sequence) == sequence.num_timesteps


def test_encoder_forward_count_global_local_is_t() -> None:
    """Global/local windows stack from cache; no extra encoder forwards."""
    model = _make_model(koopman="global_local", seed=15)
    sequence = _make_sequence(num_timesteps=6, seed=15)
    # Without cache, teacher_forced_latent_window would re-encode windows.
    assert _encoder_forward_count(model, sequence) == sequence.num_timesteps


def test_cached_forward_uses_target_snapshot_topology() -> None:
    """Cache does not bake topology; pairs pass target-snapshot edge_index."""
    model = _make_model(koopman="graph", seed=16)
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.eval()
    sequence = _make_dynamic_topology_sequence(num_timesteps=4, seed=16)
    cache = encode_sequence_latents(model, sequence)
    seen: list[torch.Tensor] = []
    original_advance = model.koopman.advance

    def _spy_advance(
        z,
        delta_t=None,
        *,
        control=None,
        edge_index=None,
        edge_weight=None,
    ):
        # Match GraphKoopmanOperator.advance so propagate_latent's
        # signature filter does not forward hyperedge kwargs.
        assert edge_index is not None
        seen.append(edge_index.detach().clone())
        return original_advance(
            z,
            delta_t,
            control=control,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    model.koopman.advance = _spy_advance  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            loss = compute_forward_consistency_sequence_loss(
                model, sequence, cache=cache
            )
    finally:
        model.koopman.advance = original_advance  # type: ignore[method-assign]

    assert torch.isfinite(loss)
    num_pairs = sequence.num_timesteps - 1
    assert len(seen) == num_pairs
    for timestep, edge_index in enumerate(seen):
        expected = sequence[timestep + 1].edge_index
        torch.testing.assert_close(edge_index, expected, rtol=0.0, atol=0.0)
    # Alternating topologies actually differ across consecutive pairs.
    assert not torch.equal(seen[0], seen[1])


def _pde_extra_losses() -> ExtraLosses:
    """PDE residual = prediction − target features (finite, simple)."""
    return ExtraLosses(
        pde_residual_fn=lambda prediction, snapshot: prediction - snapshot.x
    )


def test_shared_predictions_decoder_count_recon_pde_worst_case() -> None:
    """Recon + PDE + worst-case decode once per pair (TASK-1510 / G6)."""
    model = _make_model(seed=20)
    sequence = _make_sequence(num_timesteps=5, seed=20)
    weights = LossWeights(reconstruction=1.0, pde=1.0, worst_case=1.0)
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    handle = model.decoder.register_forward_hook(_count)
    try:
        breakdown = compute_training_loss(
            model,
            sequence,
            weights,
            extra_losses=_pde_extra_losses(),
        )
    finally:
        handle.remove()

    num_pairs = sequence.num_timesteps - 1
    assert hits["count"] == num_pairs
    assert torch.isfinite(breakdown.total)
    assert torch.isfinite(breakdown.pde)
    assert torch.isfinite(breakdown.worst_case)


def test_shared_predictions_parity_vs_legacy_model_path() -> None:
    """Shared preds match legacy model()/uncached recon within float tolerance."""
    model = _make_model(seed=21)
    model.eval()
    sequence = _make_sequence(num_timesteps=5, seed=21)
    weights = LossWeights(reconstruction=1.0, pde=1.0, worst_case=1.0)
    extras = _pde_extra_losses()
    with torch.no_grad():
        shared = compute_training_loss(model, sequence, weights, extra_losses=extras)
        legacy_recon = compute_sequence_loss(model, sequence)
        legacy_pde = compute_pde_residual_loss(
            model, sequence, weight=1.0, extra_losses=extras
        )
        legacy_worst = compute_worst_case_reconstruction_loss(
            model, sequence, weight=1.0
        )
    # Static unmasked n_delays==1: encode→decode and model() agree tightly.
    torch.testing.assert_close(
        shared.reconstruction, legacy_recon, rtol=0.0, atol=_PARITY_ATOL
    )
    torch.testing.assert_close(shared.pde, legacy_pde, rtol=0.0, atol=_PARITY_ATOL)
    torch.testing.assert_close(
        shared.worst_case, legacy_worst, rtol=0.0, atol=_PARITY_ATOL
    )


def test_shared_predictions_respect_observation_masks() -> None:
    """Shared recon/PDE/worst-case still honor target observation masks."""
    model = _make_model(seed=24)
    model.eval()
    masked = _make_sequence(num_timesteps=4, seed=24, with_masks=True)
    full = _make_sequence(num_timesteps=4, seed=24, with_masks=False)
    weights = LossWeights(reconstruction=1.0, pde=1.0, worst_case=1.0)
    extras = _pde_extra_losses()
    with torch.no_grad():
        masked_bd = compute_training_loss(model, masked, weights, extra_losses=extras)
        full_bd = compute_training_loss(model, full, weights, extra_losses=extras)
    assert not torch.equal(masked_bd.reconstruction, full_bd.reconstruction)
    assert not torch.equal(masked_bd.pde, full_bd.pde)
    assert not torch.equal(masked_bd.worst_case, full_bd.worst_case)


def test_pde_only_uses_latent_cache_encode_budget() -> None:
    """PDE alone still builds the shared latent cache (recommended default)."""
    model = _make_model(seed=22)
    sequence = _make_sequence(num_timesteps=4, seed=22)
    weights = LossWeights(reconstruction=0.0, pde=1.0)
    hits = {"count": 0}

    def _count(_module, _inputs, _output) -> None:
        hits["count"] += 1

    handle = model.encoder.register_forward_hook(_count)
    try:
        breakdown = compute_training_loss(
            model,
            sequence,
            weights,
            extra_losses=_pde_extra_losses(),
        )
    finally:
        handle.remove()

    assert hits["count"] == sequence.num_timesteps
    assert torch.isfinite(breakdown.pde)


def test_lie_helper_has_no_shared_predictions_hook() -> None:
    """Lie consistency stays on its own path (TASK-1510 out of scope)."""
    import inspect

    from koopman_graph.training.extra_objectives import compute_lie_consistency_loss

    params = inspect.signature(compute_lie_consistency_loss).parameters
    assert "predictions" not in params


def test_shared_predictions_reject_wrong_length() -> None:
    """PDE / worst-case / recon helpers reject mismatched prediction lists."""
    model = _make_model(seed=25)
    sequence = _make_sequence(num_timesteps=4, seed=25)
    short = [sequence[0].x]
    with pytest.raises(ValueError, match="predictions length"):
        compute_sequence_loss(model, sequence, predictions=short)
    with pytest.raises(ValueError, match="predictions length"):
        compute_pde_residual_loss(
            model,
            sequence,
            weight=1.0,
            extra_losses=_pde_extra_losses(),
            predictions=short,
        )
    with pytest.raises(ValueError, match="predictions length"):
        compute_worst_case_reconstruction_loss(
            model, sequence, weight=1.0, predictions=short
        )
