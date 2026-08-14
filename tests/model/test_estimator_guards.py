"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.graph_utils.propagation import (
    autoregressive_hetero_latent_rollout,
    pack_hetero_rollout_snapshots,
)
from koopman_graph.losses.rollout import (
    _bind_hetero_decoder,
    rollout_sequence_loss,
)
from koopman_graph.nn.heterogeneous import (
    _relgraph_message_passing,
)
from koopman_graph.operators.heterogeneous import (
    HeteroGraphKoopmanOperator,
)
from koopman_graph.training.objectives import (
    _hetero_eigenvalue_regularization_over_sequence,
    compute_eigenvalue_regularization_loss,
)
from koopman_graph.training.pair_objectives import (
    _hetero_num_relations,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    """Build a one-type, two-relation multiplex snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot with one cross relation."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def _multiplex_model(*, num_relations: int = 2) -> GraphKoopmanModel:
    """Factory-built multiplex RelGraph model."""
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(3, 8, 4, num_relations=num_relations, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 3, num_relations=num_relations, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )


def test_estimator_ddp_strategy_rejects_callbacks() -> None:
    """Distributed ``strategy='ddp'`` refuses ``callbacks=`` until supported."""
    from koopman_graph.model.estimator import GraphKoopmanModel
    from koopman_graph.nn import GNNDecoder, GNNEncoder
    from koopman_graph.training.callbacks import NoOpFitCallback

    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 3, num_layers=1),
        decoder=GNNDecoder(3, 4, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(2, 2), edge_index=edge_index) for _ in range(4)]
    )
    with pytest.raises(ValueError, match="callbacks"):
        model.fit(
            sequence,
            epochs=1,
            strategy="ddp",
            callbacks=[NoOpFitCallback()],
        )


def test_estimator_propagation_rollout_and_objectives_gaps() -> None:
    """Cover estimator hetero guards, pack/rollout helpers, and eig-reg gaps."""
    model = _multiplex_model()
    with pytest.raises(ValueError, match="steps must be >= 1"):
        model._rollout_hetero(_multiplex_snapshot(), steps=0)
    with pytest.raises(ValueError, match="step_deltas"):
        model._rollout_hetero(
            _multiplex_snapshot(),
            steps=1,
            step_deltas=[1.0, 2.0],
        )
    with pytest.raises(TypeError, match="_rollout is homogeneous-only"):
        model._rollout(_multiplex_snapshot(), steps=1)
    with pytest.raises(ValueError, match="history / delay embedding"):
        model.predict(_multiplex_snapshot(), steps=1, history=torch.randn(1, 4, 3))
    with pytest.raises(TypeError, match="requires a HeteroData origin"):
        model.predict(torch.randn(4, 3), steps=1)

    # Swap RelGraph peers for GNN modules after construction.
    broken = _multiplex_model(num_relations=1)
    broken.encoder = GNNEncoder(3, 8, 4)
    broken.decoder = GNNDecoder(4, 8, 3)
    with pytest.raises(TypeError, match="HeteroData forward requires"):
        broken(_multiplex_snapshot())
    with pytest.raises(TypeError, match="Hetero rollout requires RelGraphEncoder"):
        broken._rollout_hetero(_multiplex_snapshot(), steps=1)

    op = HeteroGraphKoopmanOperator(2, 1)
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    with pytest.raises(ValueError, match="steps must be >= 1"):
        autoregressive_hetero_latent_rollout(
            op,
            lambda z, *_a, **_k: z,
            torch.randn(2, 2),
            steps=0,
            topology_at=lambda _s: (edges, [None]),
        )
    with pytest.raises(ValueError, match="at least one node type"):
        pack_hetero_rollout_snapshots(
            [(torch.randn(2, 2), edges, [None])],
            template=_multiplex_snapshot(),
            node_types=(),
        )
    multi_edges = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0], [3]], dtype=torch.long),
    ]
    with pytest.raises(ValueError, match="Expected 2 relation banks"):
        pack_hetero_rollout_snapshots(
            [(torch.randn(4, 3), multi_edges[:1], [None])],
            template=_multiplex_snapshot(),
        )
    packed = pack_hetero_rollout_snapshots(
        [
            (
                torch.randn(4, 3),
                multi_edges,
                [torch.ones(2), None],
            )
        ],
        template=_multiplex_snapshot(),
    )
    assert packed[0]["node", "r1", "node"].edge_weight is not None
    with pytest.raises(ValueError, match="missing node types"):
        pack_hetero_rollout_snapshots(
            [({"gen": torch.randn(2, 3)}, edges, [None])],
            template=_typed_snapshot(),
        )

    short = HeteroGraphSnapshotSequence([_multiplex_snapshot()])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        _hetero_eigenvalue_regularization_over_sequence(model, short)

    # Hypergraph / continuous-graph eig-reg reject hetero sequences.
    from koopman_graph.operators import (
        ContinuousGraphKoopmanOperator,
        HypergraphKoopmanOperator,
    )

    class _Fake:
        dynamics_mode = "discrete"

        def __init__(self, koopman: object) -> None:
            self.koopman = koopman

    hetero_seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    with pytest.raises(ValueError, match="homogeneous GraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(
            _Fake(ContinuousGraphKoopmanOperator(2)),  # type: ignore[arg-type]
            hetero_seq,
        )
    with pytest.raises(ValueError, match="homogeneous GraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(
            _Fake(HypergraphKoopmanOperator(2)),  # type: ignore[arg-type]
            hetero_seq,
        )

    # Rollout loss hetero guards.
    controlled = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(3)],
        control_inputs=torch.randn(3, 1),
    )
    with pytest.raises(ValueError, match="uncontrolled"):
        rollout_sequence_loss(model, controlled, start=0, horizon=1)

    # Decoder-only num_relations / non-typed bind.
    peer = SimpleNamespace(
        koopman=SimpleNamespace(),
        decoder=RelGraphDecoder(2, 4, 3, num_relations=2, num_layers=1),
    )
    assert _hetero_num_relations(peer) == 2  # type: ignore[arg-type]
    bound = _bind_hetero_decoder(peer.decoder, None)
    assert callable(bound)

    # Masked multiplex prediction loss branch.
    from koopman_graph.training.pair_objectives import _hetero_prediction_loss

    pred = torch.randn(4, 3)
    loss = _hetero_prediction_loss(
        pred,
        _multiplex_snapshot(),
        node_types=None,
        target_masks={"node": torch.ones(4, dtype=torch.bool)},
    )
    assert loss.ndim == 0

    # container control_dim tensor / None branches on hetero sequence.
    plain = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    assert plain.control_dim == 0
    tensor_ctrl = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 1),
    )
    assert tensor_ctrl.control_dim == 1
    node_ctrl = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 4, 1),
    )
    assert node_ctrl.control_dim == 1

    # RelGraph stack TypeError for non-RelGraphConv layers.
    enc = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    enc.convs[0] = torch.nn.Linear(3, 4)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="expected RelGraphConv"):
        _relgraph_message_passing(
            enc,
            torch.randn(3, 3),
            [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)],
            [None],
        )
