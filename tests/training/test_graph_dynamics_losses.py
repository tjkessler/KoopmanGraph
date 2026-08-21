"""Structural and presence losses for graph-state closure."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphDynamicsConfig, GraphSnapshotSequence
from koopman_graph.training import LossWeights, compute_training_loss
from koopman_graph.training.latent_cache import encode_sequence_latents
from koopman_graph.training.objectives import compute_graph_dynamics_losses


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional cycle.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    Tensor
        COO index.
    """
    forward_src = list(range(num_nodes))
    forward_dst = list(range(1, num_nodes)) + [0]
    src = forward_src + forward_dst
    dst = forward_dst + forward_src
    return torch.tensor([src, dst], dtype=torch.long)


def _model(config: GraphDynamicsConfig | None) -> GraphKoopmanModel:
    """Return a tiny homogeneous model.

    Parameters
    ----------
    config : GraphDynamicsConfig or None
        Graph-state config.

    Returns
    -------
    GraphKoopmanModel
        Untrained model.
    """
    return GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        graph_dynamics=config,
    )


def _sequence(
    *,
    dynamic: bool = True,
    presence: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a short homogeneous sequence.

    Parameters
    ----------
    dynamic : bool, optional
        Alternate path / cycle topologies when ``True``.
    presence : Tensor or None, optional
        Optional ``(T, N)`` presence masks.

    Returns
    -------
    GraphSnapshotSequence
        Three-step sequence on 5 nodes.
    """
    torch.manual_seed(0)
    snapshots = []
    for timestep in range(3):
        edges = _path_edge_index(5) if timestep % 2 == 0 else _cycle_edge_index(5)
        if not dynamic:
            edges = _path_edge_index(5)
        snapshots.append(Data(x=torch.randn(5, 2), edge_index=edges))
    return GraphSnapshotSequence(
        snapshots,
        allow_dynamic_topology=dynamic,
        presence_masks=presence,
        allow_node_churn=presence is not None,
    )


def test_graph_dynamics_none_keeps_zero_topology_terms() -> None:
    """Default models record zero topology / presence breakdown terms."""
    model = _model(None)
    sequence = _sequence(dynamic=False)
    breakdown = compute_training_loss(model, sequence, LossWeights())
    assert float(breakdown.topology) == 0.0
    assert float(breakdown.presence) == 0.0


def test_structural_bce_is_positive_on_dynamic_graphs() -> None:
    """An untrained sparse head yields a positive structural term."""
    model = _model(GraphDynamicsConfig(candidate_k=4))
    sequence = _sequence(dynamic=True)
    breakdown = compute_training_loss(model, sequence, LossWeights())
    assert float(breakdown.topology) > 0.0
    assert float(breakdown.presence) == 0.0
    assert float(breakdown.total) > float(breakdown.reconstruction)


def test_presence_bce_runs_when_masks_exist() -> None:
    """Presence BCE is computed only when masks are attached."""
    presence = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, True],
            [True, True, False, True, True],
        ]
    )
    sequence = _sequence(dynamic=False, presence=presence)
    model = _model(GraphDynamicsConfig(topology_head="none"))
    cache = encode_sequence_latents(model, sequence)
    topology, presence_loss = compute_graph_dynamics_losses(
        model,
        sequence,
        cache=cache,
        device=next(model.parameters()).device,
    )
    assert float(topology) == 0.0
    assert float(presence_loss) > 0.0


def test_teacher_force_does_not_change_reconstruction_topology() -> None:
    """``recursive_training=False`` keeps teacher-forced one-step reconstruction."""
    torch.manual_seed(1)
    sequence = _sequence(dynamic=True)
    teacher = _model(
        GraphDynamicsConfig(recursive_training=False, topology_loss_weight=0.0)
    )
    recursive = _model(
        GraphDynamicsConfig(recursive_training=True, topology_loss_weight=0.0)
    )
    recursive.encoder.load_state_dict(teacher.encoder.state_dict())
    recursive.decoder.load_state_dict(teacher.decoder.state_dict())
    recursive.koopman.load_state_dict(teacher.koopman.state_dict())
    recursive.predicted_topology.load_state_dict(
        teacher.predicted_topology.state_dict()
    )
    weights = LossWeights(reconstruction=1.0)
    teacher_break = compute_training_loss(teacher, sequence, weights)
    recursive_break = compute_training_loss(recursive, sequence, weights)
    assert not torch.allclose(
        teacher_break.reconstruction,
        recursive_break.reconstruction,
        atol=1e-6,
    )


def test_fit_history_records_topology_term() -> None:
    """Fit history stores the unweighted topology series."""
    model = _model(GraphDynamicsConfig(candidate_k=3))
    sequence = _sequence(dynamic=True)
    history = model.fit(sequence, epochs=2, lr=1e-2)
    assert len(history.topology_loss) == 2
    assert history.topology_loss[0] > 0.0
    assert len(history.presence_loss) == 2
