"""d_τ parity for continuous hetero, conformal, and env (TASK-1821).

Covers unequal ``latent_dims`` end-to-end while shared-d paths remain covered
by ``tests/operators/test_hetero_continuous.py``,
``tests/uq/test_hetero_conformal.py``, ``tests/env/test_hetero_env.py``, and
``tests/model/test_hetero_shared_d_regression.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence, stacked_latent_numel
from koopman_graph.env import GraphKoopmanEnv, flatten_latent
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import ContinuousHeteroGraphKoopmanOperator
from koopman_graph.uq import ConformalKoopmanUQ

NODE_TYPES = ("a", "b")
EDGE_TYPES = (
    ("a", "to_b", "b"),
    ("b", "to_a", "a"),
)
NUM_NODES = {"a": 2, "b": 3}
FEATURE_DIMS = {"a": 2, "b": 2}
LATENT_DIMS = {"a": 2, "b": 3}
SHARED_D = 4


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
    snapshot["a", "to_b", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    snapshot["b", "to_a", "a"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )
    return snapshot


def _typed_sequence(
    *,
    num_sequences: int = 4,
    timesteps: int = 3,
    seed: int = 0,
) -> list[HeteroGraphSnapshotSequence]:
    sequences: list[HeteroGraphSnapshotSequence] = []
    for seq_id in range(num_sequences):
        snaps = [
            _typed_snapshot(seed=seed + 100 * seq_id + t) for t in range(timesteps)
        ]
        sequences.append(HeteroGraphSnapshotSequence(snaps))
    return sequences


def _rectangular_model(
    *,
    dynamics_mode: str = "discrete",
    seed: int = 0,
) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=4,
            latent_dim=SHARED_D,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=SHARED_D,
            hidden_channels=4,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=SHARED_D,
        time_step=0.2 if dynamics_mode == "continuous" else 1.0,
        dynamics_mode=dynamics_mode,  # type: ignore[arg-type]
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        koopman_latent_dims=LATENT_DIMS,
    )


def _global_edges() -> list[torch.Tensor]:
    # Stacked nodes: a0,a1,b0,b1,b2
    return [
        torch.tensor([[0, 1], [2, 4]], dtype=torch.long),
        torch.tensor([[2, 3], [0, 1]], dtype=torch.long),
    ]


def test_factory_continuous_rectangular_wires_latent_dims() -> None:
    """Continuous hetero factory accepts koopman_latent_dims (TASK-1821)."""
    model = _rectangular_model(dynamics_mode="continuous", seed=0)
    assert isinstance(model.koopman, ContinuousHeteroGraphKoopmanOperator)
    assert model.koopman.is_rectangular
    assert model.koopman.latent_dims == LATENT_DIMS
    assert model.encoder.is_rectangular
    assert model.decoder.is_rectangular


def test_continuous_rectangular_advance_matches_expm() -> None:
    """Rectangular continuous advance matches Φ @ flat (Appendix B orientation)."""
    torch.manual_seed(0)
    op = ContinuousHeteroGraphKoopmanOperator(
        SHARED_D,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        latent_dims=LATENT_DIMS,
        init_mode="identity",
        parameterization="dense",
        sparsity="dense",
    )
    l_self = {
        "a": torch.tensor([[-0.5, 0.1], [0.0, -0.4]]),
        "b": torch.tensor(
            [
                [-0.3, 0.0, 0.1],
                [0.0, -0.2, 0.0],
                [0.05, 0.0, -0.25],
            ]
        ),
    }
    l_ab = torch.tensor([[0.05, 0.0, 0.02], [0.0, 0.04, 0.0]])
    l_ba = torch.tensor([[0.03, 0.0], [0.0, 0.02], [0.01, 0.02]])
    op.set_dense_matrices(l_self, [l_ab, l_ba])
    banks = _global_edges()
    z_flat = op.pack_typed_latents(
        {
            "a": torch.randn(NUM_NODES["a"], LATENT_DIMS["a"]),
            "b": torch.randn(NUM_NODES["b"], LATENT_DIMS["b"]),
        },
        NUM_NODES,
    )
    delta_t = 0.15
    advanced = op.advance(
        z_flat,
        delta_t,
        edge_indices=banks,
        num_nodes_dict=NUM_NODES,
    )
    num_nodes = sum(NUM_NODES.values())
    l_eff = op.effective_generator(banks, num_nodes, num_nodes_dict=NUM_NODES)
    expected = torch.linalg.matrix_exp(delta_t * l_eff) @ z_flat
    assert torch.allclose(advanced, expected, atol=1e-4)
    total = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    assert advanced.shape == (total,)
    assert op.relation_matrix(0).shape == (LATENT_DIMS["a"], LATENT_DIMS["b"])


def test_continuous_rectangular_model_encode_predict_smoke() -> None:
    """Factory continuous rectangular model encode / predict smoke."""
    model = _rectangular_model(dynamics_mode="continuous", seed=1)
    origin = _typed_snapshot(seed=1)
    expected = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    z = model.encode(origin)
    assert z.ndim == 1
    assert z.numel() == expected
    model.eval()
    with torch.no_grad():
        preds = model.predict(origin, steps=2)
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["a"].x.shape == (NUM_NODES["a"], FEATURE_DIMS["a"])
    assert preds[0]["b"].x.shape == (NUM_NODES["b"], FEATURE_DIMS["b"])


def test_conformal_intervals_pack_under_unequal_latent_dims() -> None:
    """Conformal calibrate/predict packs HeteroData bands with unequal d_τ."""
    torch.manual_seed(0)
    model = _rectangular_model(dynamics_mode="discrete", seed=0)
    cal = _typed_sequence(num_sequences=6, timesteps=3, seed=0)
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(cal, steps=2, alpha=0.1)
    assert uq.is_calibrated
    assert uq.quantiles.shape == (2,)

    interval = uq.predict_interval(cal[0][0], steps=2, level=0.9)
    assert len(interval.mean) == 2
    assert isinstance(interval.mean[0], HeteroData)
    assert interval.lower[0]["a"].x.shape == (NUM_NODES["a"], FEATURE_DIMS["a"])
    assert interval.upper[0]["b"].x.shape == (NUM_NODES["b"], FEATURE_DIMS["b"])
    assert torch.all(interval.lower[0]["a"].x <= interval.mean[0]["a"].x)
    assert torch.all(interval.mean[0]["b"].x <= interval.upper[0]["b"].x)


def test_env_obs_space_uses_stacked_latent_numel() -> None:
    """Rectangular hetero env observation length is Σ N_τ·d_τ."""
    pytest.importorskip("gymnasium")
    model = _rectangular_model(dynamics_mode="discrete", seed=2)
    sequence = HeteroGraphSnapshotSequence([_typed_snapshot(seed=t) for t in range(3)])
    expected = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    env = GraphKoopmanEnv(
        model,
        sequence,
        reward_fn=lambda snap, _t: float(snap["a"].x.sum().item()),
        start_index=0,
        random_start=False,
        max_episode_steps=2,
    )
    assert env.observation_space.shape == (expected,)
    assert env._is_rectangular
    obs, info = env.reset(seed=0)
    assert obs.shape == (expected,)
    assert set(info["decoded_x_by_type"]) == set(NODE_TYPES)
    reshaped = env.reshape_observation(obs)
    assert reshaped.shape == (expected,)
    next_obs, _reward, _term, _trunc, _step_info = env.step(
        np.zeros(0, dtype=np.float32)
    )
    assert next_obs.shape == (expected,)
    with torch.no_grad():
        z = model.encode(_typed_snapshot(seed=0))
    assert flatten_latent(z).shape == (expected,)


def test_shared_d_continuous_path_still_green() -> None:
    """Absent latent_dims keeps shared-d continuous hetero (N, d) encode."""
    torch.manual_seed(0)
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=4,
            latent_dim=SHARED_D,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=SHARED_D,
            hidden_channels=4,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=SHARED_D,
        time_step=0.2,
        dynamics_mode="continuous",
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
    )
    assert isinstance(model.koopman, ContinuousHeteroGraphKoopmanOperator)
    assert not model.koopman.is_rectangular
    z = model.encode(_typed_snapshot(seed=0))
    assert z.shape == (sum(NUM_NODES.values()), SHARED_D)
