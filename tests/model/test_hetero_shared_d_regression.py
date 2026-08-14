"""Shared-d / homogeneous bit-compat regression lock (TASK-1820).

**Release blocker (DESIGN R1 / §5.4.6).** Failures here mean a d_τ layout
change altered the default shared-d hetero path or homogeneous defaults.
Do not ship such a change without an explicit, documented tolerance update
and release-note callout.

Tolerance contract
------------------
Seeded one-step MSE via :func:`~koopman_graph.training.one_step_loss` on
fixed fixtures (model ``manual_seed(0)``; snapshot generators seeds 1→2).
Compare to golden floats with ``abs=1e-5`` (float32 encode / relation
message path; same band as hetero WS1 loss parity).
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.serialization import build_checkpoint
from koopman_graph.training import one_step_loss

_LOSS_ABS = 1e-5

# Golden one-step MSE values locked 2026-07-31 (TASK-1820) under the fixtures
# and seeds below. Update only with an explicit release-note if defaults change.
_GOLDEN_HOMO_ONE_STEP = 0.6180354356765747
_GOLDEN_MULTIPLEX_ONE_STEP = 1.1788241863250732
_GOLDEN_TYPED_ONE_STEP = 0.7983493208885193

_TYPED_NODE_TYPES = ("a", "b")
_TYPED_EDGE_TYPES = (
    ("a", "to_b", "b"),
    ("b", "to_a", "a"),
)
_TYPED_FEATURE_DIMS = {"a": 2, "b": 2}
_TYPED_NUM_NODES = {"a": 2, "b": 3}
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


def _typed_snapshot(*, seed: int) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(
        _TYPED_NUM_NODES["a"],
        _TYPED_FEATURE_DIMS["a"],
        generator=generator,
    )
    snapshot["b"].x = torch.randn(
        _TYPED_NUM_NODES["b"],
        _TYPED_FEATURE_DIMS["b"],
        generator=generator,
    )
    snapshot["a", "to_b", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    snapshot["b", "to_a", "a"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )
    return snapshot


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


def _typed_model(
    *,
    seed: int = 0,
    latent_dims: dict[str, int] | None = None,
) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            _TYPED_FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=_LATENT_DIM,
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
            latent_dims=latent_dims,
        ),
        decoder=RelGraphDecoder(
            latent_dim=_LATENT_DIM,
            hidden_channels=8,
            out_channels=_TYPED_FEATURE_DIMS,
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
            latent_dims=latent_dims,
        ),
        latent_dim=_LATENT_DIM,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=_TYPED_NODE_TYPES,
        koopman_edge_types=_TYPED_EDGE_TYPES,
        koopman_latent_dims=latent_dims,
    )


def test_homogeneous_seeded_one_step_loss_matches_golden() -> None:
    """Homogeneous default one-step MSE matches pre-d_τ golden (release blocker)."""
    model = _homo_model(seed=0)
    loss = one_step_loss(model, _homo_snapshot(seed=1), _homo_snapshot(seed=2))
    assert float(loss.detach()) == pytest.approx(_GOLDEN_HOMO_ONE_STEP, abs=_LOSS_ABS)


def test_multiplex_shared_d_path_shapes_and_golden_loss() -> None:
    """Multiplex hetero without latent_dims stays shared-d with locked loss."""
    model = _multiplex_model(seed=0)
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.latent_dims is None
    assert not model.koopman.is_rectangular
    assert model.koopman.relation_matrix(0).shape == (_LATENT_DIM, _LATENT_DIM)

    origin = _multiplex_snapshot(seed=1)
    z = model.encode(origin)
    assert z.shape == (4, _LATENT_DIM)

    loss = one_step_loss(model, origin, _multiplex_snapshot(seed=2))
    assert float(loss.detach()) == pytest.approx(
        _GOLDEN_MULTIPLEX_ONE_STEP,
        abs=_LOSS_ABS,
    )


def test_typed_shared_d_path_shapes_golden_loss_and_checkpoint() -> None:
    """Typed hetero without latent_dims stays shared-d; checkpoint omits key."""
    model = _typed_model(seed=0)
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.latent_dims is None
    assert not model.koopman.is_rectangular
    assert model.encoder.latent_dims is None
    assert model.koopman.relation_matrix(0).shape == (_LATENT_DIM, _LATENT_DIM)

    origin = _typed_snapshot(seed=1)
    num_nodes = sum(_TYPED_NUM_NODES.values())
    z = model.encode(origin)
    assert z.shape == (num_nodes, _LATENT_DIM)

    loss = one_step_loss(model, origin, _typed_snapshot(seed=2))
    assert float(loss.detach()) == pytest.approx(_GOLDEN_TYPED_ONE_STEP, abs=_LOSS_ABS)

    checkpoint = build_checkpoint(model)
    assert "latent_dims" not in checkpoint["config"]


def test_equal_latent_dims_stays_on_square_shared_width_path() -> None:
    """Opt-in latent_dims with all d_τ == d stays non-rectangular (N, d)."""
    equal = {name: _LATENT_DIM for name in _TYPED_NODE_TYPES}
    model = _typed_model(seed=0, latent_dims=equal)
    assert model.koopman.latent_dims == equal
    assert not model.koopman.is_rectangular
    assert model.koopman.relation_matrix(0).shape == (_LATENT_DIM, _LATENT_DIM)

    origin = _typed_snapshot(seed=1)
    z = model.encode(origin)
    assert z.shape == (sum(_TYPED_NUM_NODES.values()), _LATENT_DIM)
    # Square-path smoke: one-step loss remains finite (not a bit-gold lock —
    # equal-width opt-in may diverge from absent-key init in future).
    loss = one_step_loss(model, origin, _typed_snapshot(seed=2))
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0
