"""Factory + encode/predict path for ``koopman='hetero_graph'``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.uq import ConformalKoopmanUQ


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _hetero_model(
    *,
    latent_dim: int = 4,
    in_channels: int = 3,
    num_relations: int = 2,
    control_dim: int = 0,
) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            in_channels,
            hidden_channels=8,
            latent_dim=latent_dim,
            num_relations=num_relations,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=8,
            out_channels=in_channels,
            num_relations=num_relations,
            num_layers=1,
        ),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="hetero_graph",
        control_dim=control_dim,
    )


def test_factory_hetero_graph_builds_relgraph_operator() -> None:
    """``koopman='hetero_graph'`` builds relational operator + RelGraph peers."""
    model = _hetero_model()
    assert model.koopman_kind == "hetero_graph"
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert isinstance(model.encoder, RelGraphEncoder)
    assert isinstance(model.decoder, RelGraphDecoder)
    assert model.koopman.num_relations == 2
    assert model.koopman.normalization == "rgcn_in_degree"


def test_factory_hetero_graph_requires_relgraph_peers() -> None:
    """Homogeneous GNN peers are rejected for ``koopman='hetero_graph'``."""
    with pytest.raises(ValueError, match="RelGraphEncoder"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
            decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
            latent_dim=4,
            time_step=1.0,
            koopman="hetero_graph",
        )


def test_relgraph_peers_require_hetero_graph_kind() -> None:
    """RelGraph peers without ``koopman='hetero_graph'`` raise."""
    with pytest.raises(ValueError, match="hetero_graph"):
        GraphKoopmanModel(
            encoder=RelGraphEncoder(3, 8, 4, 2, num_layers=1),
            decoder=RelGraphDecoder(4, 8, 3, 2, num_layers=1),
            latent_dim=4,
            time_step=1.0,
            koopman="graph",
        )


def test_encode_forward_predict_heterodata() -> None:
    """Encode / forward / predict accept multiplex ``HeteroData``."""
    model = _hetero_model()
    origin = _multiplex_snapshot()
    z = model.encode(origin)
    assert z.shape == (4, 4)

    model.eval()
    with torch.no_grad():
        nxt = model(origin)
        assert nxt.shape == (4, 3)
        preds = model.predict(origin, steps=2)
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["node"].x.shape == (4, 3)
    assert set(preds[0].edge_types) == set(origin.edge_types)


def test_predict_hold_last_future_hetero_topology() -> None:
    """Future ``HeteroData`` topologies update relation banks under hold-last."""
    model = _hetero_model()
    origin = _multiplex_snapshot()
    future = _multiplex_snapshot()
    future["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 3], [3, 1]],
        dtype=torch.long,
    )
    model.eval()
    with torch.no_grad():
        preds = model.predict(origin, steps=2, future_topologies=[future])
    assert torch.equal(
        preds[0]["node", "r1", "node"].edge_index,
        future["node", "r1", "node"].edge_index,
    )
    assert torch.equal(
        preds[1]["node", "r1", "node"].edge_index,
        future["node", "r1", "node"].edge_index,
    )


def test_homogeneous_factory_path_unchanged() -> None:
    """Default homogeneous factory path still builds a per-node operator."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )
    assert model.koopman_kind == "pernode"
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    model.eval()
    with torch.no_grad():
        out = model(data)
    assert out.shape == (3, 3)


def test_hierarchical_rejects_hetero_model() -> None:
    """Hierarchical façade rejects multiplex operators."""
    model = _hetero_model()
    with pytest.raises(TypeError, match="homogeneous-only"):
        HierarchicalGraphKoopmanModel(model)


def test_conformal_rejects_hetero_model() -> None:
    """Conformal UQ rejects multiplex operators at construction."""
    model = _hetero_model()
    with pytest.raises(TypeError, match="homogeneous-only"):
        ConformalKoopmanUQ(model)


def test_conformal_rejects_hetero_calibration_sequences() -> None:
    """Conformal calibrate rejects hetero sequences even for homo models."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )
    uq = ConformalKoopmanUQ(model)
    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    with pytest.raises(TypeError, match="homogeneous-only"):
        uq.calibrate([seq], steps=1)


def test_hetero_fit_predict_smoke() -> None:
    """Seeded multiplex fit yields finite loss and predict rolls out."""
    torch.manual_seed(0)
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(5)])
    model = _hetero_model()
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history.loss)
    preds = model.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["node"].x.shape == (4, 3)


def test_hetero_fit_rejects_backward_consistency() -> None:
    """Backward consistency raises clearly on multiplex sequences."""
    from koopman_graph.training import LossWeights

    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    model = _hetero_model()
    with pytest.raises(ValueError, match="backward consistency"):
        model.fit(
            sequence,
            epochs=1,
            loss_weights=LossWeights(reconstruction=1.0, backward=1.0),
        )


def test_hetero_fit_rejects_windowed_sampler() -> None:
    """Windowed fit is unsupported for multiplex sequences."""
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(5)])
    model = _hetero_model()
    with pytest.raises(ValueError, match="windowed / sampler fit"):
        model.fit(sequence, epochs=1, window_length=3)


def test_sequence_to_device_preserves_hetero_container() -> None:
    """Device transfer keeps HeteroGraphSnapshotSequence type."""
    from koopman_graph.training.device import sequence_to_device

    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    moved = sequence_to_device(sequence, torch.device("cpu"))
    assert isinstance(moved, HeteroGraphSnapshotSequence)
    assert moved.num_timesteps == 2
    assert isinstance(moved[0], HeteroData)


def test_relgraph_rejects_learn_topology() -> None:
    """Self-adaptive topology remains homogeneous-only with RelGraph peers."""
    with pytest.raises(ValueError, match="learn_topology is unsupported"):
        GraphKoopmanModel(
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
            learn_topology="self_adaptive",
        )


def test_encode_sequence_latents_hetero() -> None:
    """Latent cache encodes multiplex sequences without homo fallback."""
    from koopman_graph.training.latent_cache import encode_sequence_latents

    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    model = _hetero_model()
    cache = encode_sequence_latents(model, sequence)
    assert cache.num_timesteps == 3
    assert cache.z[0].shape == (4, 4)


def test_env_rejects_hetero_model() -> None:
    """GraphKoopmanEnv rejects multiplex models."""
    gymnasium = pytest.importorskip("gymnasium")
    del gymnasium
    from koopman_graph.env import GraphKoopmanEnv

    controlled = _hetero_model(control_dim=1)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ref = GraphSnapshotSequence(
        [Data(x=torch.randn(3, 3), edge_index=edge_index) for _ in range(2)]
    )
    with pytest.raises(TypeError, match="homogeneous-only"):
        GraphKoopmanEnv(
            controlled,
            ref,
            reward_fn=lambda _data, _t: 0.0,
            control_low=-1.0,
            control_high=1.0,
            max_episode_steps=2,
        )
