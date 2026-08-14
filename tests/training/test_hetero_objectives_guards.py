"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.losses.rollout import (
    _hetero_rollout_step_loss,
    _multiplex_target_features,
    _typed_target_features,
)
from koopman_graph.nn.heterogeneous import (
    _relgraph_message_passing,
)
from koopman_graph.operators.heterogeneous import (
    HeteroGraphKoopmanOperator,
)
from koopman_graph.training import LossWeights
from koopman_graph.training.history import ExtraLosses
from koopman_graph.training.inputs import (
    resolve_training_sequences,
    resolve_validation_sequences,
)
from koopman_graph.training.objectives import (
    _hetero_eigenvalue_regularization_over_sequence,
    _hetero_relation_banks,
    _validate_hetero_fit_surface,
    compute_eigenvalue_regularization_loss,
)
from koopman_graph.training.pair_objectives import (
    _backward_consistency_pair,
    _hetero_num_relations,
    _one_step_pair,
    multiplex_node_features,
    one_step_loss,
    one_step_prediction,
    pair_control,
    stack_typed_masks,
    typed_node_features,
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


def test_pair_objective_hetero_guards() -> None:
    """Pair helpers reject unsupported hetero training surfaces."""
    typed = _typed_snapshot()
    with pytest.raises(ValueError, match="exactly one node type"):
        multiplex_node_features(typed)

    homo = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=1.0,
    )
    with pytest.raises(TypeError, match="HeteroGraphKoopmanOperator"):
        _hetero_num_relations(homo)

    with pytest.raises(ValueError, match="missing node type"):
        typed_node_features(_typed_snapshot(), ("gen", "bus"))

    class _StoreNone:
        x = None

    class _TypedMissing:
        node_types = ("gen",)

        def __getitem__(self, _key: str) -> _StoreNone:
            return _StoreNone()

    with pytest.raises(ValueError, match="missing feature matrix x"):
        typed_node_features(_TypedMissing(), ("gen",))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="missing node type"):
        stack_typed_masks(
            {"gen": torch.ones(2, dtype=torch.bool)},
            ("gen", "load"),
        )

    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    controlled = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 1),
    )
    with pytest.raises(ValueError, match="controlled HeteroGraphSnapshotSequence"):
        pair_control(controlled, 0)

    model = _multiplex_model()
    with pytest.raises(ValueError, match="backward consistency is unsupported"):
        _backward_consistency_pair(model, seq, 0)

    pred = one_step_prediction(model, seq, 0)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape[0] == 4


def test_objectives_hetero_fit_surface_guards() -> None:
    """Eigenvalue / fit-surface helpers reject unsupported hetero mixes."""
    multi = [_multiplex_snapshot() for _ in range(2)]
    multi[1]["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    dynamic = HeteroGraphSnapshotSequence(multi, allow_dynamic_topology=True)
    with pytest.raises(ValueError, match="dynamic-topology"):
        _hetero_relation_banks(dynamic)

    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    one_rel = _multiplex_model(num_relations=1)
    with pytest.raises(ValueError, match="num_relations"):
        _hetero_eigenvalue_regularization_over_sequence(one_rel, seq)

    graph = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    with pytest.raises(ValueError, match="not HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(graph, seq)

    model = _multiplex_model()
    homo_seq = GraphSnapshotSequence(
        [
            Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            )
            for _ in range(2)
        ]
    )
    with pytest.raises(ValueError, match="requires a HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(model, homo_seq)

    weights = LossWeights(reconstruction=1.0, forward=1.0)
    with pytest.raises(ValueError, match="dynamic-topology"):
        _validate_hetero_fit_surface(dynamic, weights, extra_losses=None)

    controlled = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 1),
    )
    with pytest.raises(ValueError, match="controlled"):
        _validate_hetero_fit_surface(controlled, weights, extra_losses=None)

    stamped = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        timestamps=torch.tensor([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="timestamped"):
        _validate_hetero_fit_surface(stamped, weights, extra_losses=None)

    with pytest.raises(ValueError, match="lie / pde / worst_case"):
        _validate_hetero_fit_surface(
            seq,
            LossWeights(reconstruction=1.0, lie=0.1),
            extra_losses=None,
        )
    with pytest.raises(ValueError, match="lie_dynamics_fn"):
        _validate_hetero_fit_surface(
            seq,
            weights,
            extra_losses=ExtraLosses(lie_dynamics_fn=lambda *_a, **_k: 0.0),
        )


def test_training_input_and_pair_objective_remaining_guards() -> None:
    """Fit-input classifiers and hetero pair helpers hit remaining miss lines."""
    with pytest.raises(TypeError, match="single HeteroData is not a trajectory"):
        resolve_training_sequences(_multiplex_snapshot())
    with pytest.raises(TypeError, match="single HeteroData is not a trajectory"):
        resolve_validation_sequences(_multiplex_snapshot(), num_training_sequences=1)

    class _Snap:
        node_types = ("node",)

        def __getitem__(self, _key: str) -> SimpleNamespace:
            return SimpleNamespace(x=None)

    with pytest.raises(ValueError, match="missing feature matrix x"):
        multiplex_node_features(_Snap())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing feature matrix x"):
        _multiplex_target_features(_Snap())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing node type"):
        _typed_target_features(_typed_snapshot(), ("gen", "ghost"))
    with pytest.raises(ValueError, match="missing feature matrix x"):
        _typed_target_features(_Snap(), ("node",))  # type: ignore[arg-type]

    model = _multiplex_model()
    snap_t = _multiplex_snapshot()
    snap_t1 = _multiplex_snapshot()
    with pytest.raises(ValueError, match="target_mask is unsupported"):
        one_step_loss(
            model,
            snap_t,
            snap_t1,
            target_mask=torch.ones(4, dtype=torch.bool),
        )
    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    loss = _one_step_pair(model, seq, 0)
    assert loss.ndim == 0

    pred = torch.randn(4, 3)
    step = _hetero_rollout_step_loss(pred, seq[1], node_types=None, masks=None)
    assert step.ndim == 0
    masked = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        observation_masks={"node": torch.ones(2, 4, dtype=torch.bool)},
    )
    step_m = _hetero_rollout_step_loss(
        pred,
        masked[1],
        node_types=None,
        masks=masked.observation_mask_at(1),
    )
    assert step_m.ndim == 0

    enc = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    with pytest.raises(ValueError, match="Expected x with shape"):
        _relgraph_message_passing(enc, torch.randn(3), edges, [None])
    with pytest.raises(ValueError, match="Expected in_channels"):
        _relgraph_message_passing(enc, torch.randn(3, 5), edges, [None])

    op = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        init_mode="identity_noise",
        init_scale=0.01,
        parameterization="odo",
    )
    op.reset_parameters()
    assert op.num_relations == 1
