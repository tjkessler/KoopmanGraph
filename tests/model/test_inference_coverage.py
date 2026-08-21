"""Coverage for spectrum, presence, and invariance inference helpers."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.model.inference import (
    compute_model_spectrum,
    latent_decode_rollout,
    resolve_future_presence_at,
    sequence_subspace_invariance,
)
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
    KoopmanOperator,
)


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


def _tiny_model() -> GraphKoopmanModel:
    """Return a small homogeneous per-node model.

    Returns
    -------
    GraphKoopmanModel
        One-layer GCN stack.
    """
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 8, 3, num_layers=1),
        decoder=GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman_init_mode="identity",
        koopman_init_scale=0.0,
    )


def test_compute_model_spectrum_hetero_success() -> None:
    """Discrete hetero spectrum uses the assembled multiplex operator."""
    operator = HeteroGraphKoopmanOperator(3, num_relations=2)
    edge_indices = (
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 2], [2, 3]], dtype=torch.long),
    )
    spectrum = compute_model_spectrum(
        operator,
        uses_graph_koopman=False,
        is_continuous=False,
        time_step=1.0,
        uses_hetero_koopman=True,
        edge_indices=edge_indices,
        num_nodes=4,
    )
    assert spectrum.eigenvalues.numel() == 12

    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, 8, 4, 2, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 3, 2, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )
    origin = HeteroData()
    origin["node"].x = torch.randn(4, 3)
    origin["node", "r1", "node"].edge_index = edge_indices[0]
    origin["node", "r2", "node"].edge_index = edge_indices[1]
    via_model = model.spectrum(edge_indices=edge_indices, num_nodes=4)
    assert via_model.eigenvalues.numel() == 16


def test_compute_model_spectrum_family_guards_and_success() -> None:
    """Networked, continuous, and missing-topology spectrum branches run."""
    edge = _path_edge_index(3)
    hyper = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    banks = (
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        compute_model_spectrum(
            ContinuousHeteroGraphKoopmanOperator(2, num_relations=2),
            uses_graph_koopman=False,
            is_continuous=True,
            time_step=1.0,
            uses_continuous_hetero_koopman=True,
        )
    continuous_hetero = compute_model_spectrum(
        ContinuousHeteroGraphKoopmanOperator(2, num_relations=2),
        uses_graph_koopman=False,
        is_continuous=True,
        time_step=1.0,
        uses_continuous_hetero_koopman=True,
        edge_indices=banks,
        num_nodes=3,
    )
    assert continuous_hetero.eigenvalues.numel() == 6
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        compute_model_spectrum(
            HeteroGraphKoopmanOperator(2, num_relations=2),
            uses_graph_koopman=False,
            is_continuous=False,
            time_step=1.0,
            uses_hetero_koopman=True,
        )
    with pytest.raises(ValueError, match="edge_index and num_nodes"):
        compute_model_spectrum(
            ContinuousGraphKoopmanOperator(2),
            uses_graph_koopman=False,
            is_continuous=True,
            time_step=1.0,
            uses_continuous_graph_koopman=True,
        )
    continuous_graph = compute_model_spectrum(
        ContinuousGraphKoopmanOperator(2),
        uses_graph_koopman=False,
        is_continuous=True,
        time_step=1.0,
        uses_continuous_graph_koopman=True,
        edge_index=edge,
        num_nodes=3,
    )
    assert continuous_graph.eigenvalues.numel() == 6
    with pytest.raises(ValueError, match="hyperedge_index and num_nodes"):
        compute_model_spectrum(
            HypergraphKoopmanOperator(2),
            uses_graph_koopman=False,
            is_continuous=False,
            time_step=1.0,
            uses_hypergraph_koopman=True,
        )
    hyper_spec = compute_model_spectrum(
        HypergraphKoopmanOperator(2),
        uses_graph_koopman=False,
        is_continuous=False,
        time_step=1.0,
        uses_hypergraph_koopman=True,
        hyperedge_index=hyper,
        num_nodes=3,
    )
    assert hyper_spec.eigenvalues.numel() == 6
    with pytest.raises(ValueError, match="edge_index and num_nodes"):
        compute_model_spectrum(
            GraphKoopmanOperator(2),
            uses_graph_koopman=True,
            is_continuous=False,
            time_step=1.0,
        )
    graph_spec = compute_model_spectrum(
        GraphKoopmanOperator(2),
        uses_graph_koopman=True,
        is_continuous=False,
        time_step=1.0,
        edge_index=edge,
        num_nodes=3,
    )
    assert graph_spec.eigenvalues.numel() == 6
    auxiliary = ContinuousKoopmanOperator(2, parameterization="auxiliary_spectral")
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        compute_model_spectrum(
            auxiliary,
            uses_graph_koopman=False,
            is_continuous=True,
            time_step=1.0,
        )
    generator = compute_model_spectrum(
        ContinuousKoopmanOperator(2),
        uses_graph_koopman=False,
        is_continuous=True,
        time_step=1.0,
    )
    discrete = compute_model_spectrum(
        ContinuousKoopmanOperator(2),
        uses_graph_koopman=False,
        is_continuous=True,
        time_step=1.0,
        delta_t=0.25,
    )
    pernode = compute_model_spectrum(
        KoopmanOperator(2),
        uses_graph_koopman=False,
        is_continuous=False,
        time_step=1.0,
    )
    assert generator.eigenvalues.numel() == 2
    assert discrete.eigenvalues.numel() == 2
    assert pernode.eigenvalues.numel() == 2


def test_resolve_future_presence_sequence_schedule() -> None:
    """A sequence of 1-d masks builds a callable presence schedule."""
    masks = [
        torch.ones(3, dtype=torch.bool),
        torch.tensor([True, False, True]),
    ]
    schedule = resolve_future_presence_at(masks, steps=2, num_nodes=3)
    assert schedule is not None
    torch.testing.assert_close(schedule(0), masks[0])
    torch.testing.assert_close(schedule(1), masks[1].bool())


def test_latent_decode_rollout_hold_last_and_parameter_length() -> None:
    """Hold-last topology runs; a short parameter schedule raises."""
    model = _tiny_model()
    origin = Data(x=torch.randn(3, 2), edge_index=_path_edge_index(3))
    presence = [torch.ones(3, dtype=torch.bool)]
    rolled = latent_decode_rollout(
        model.koopman,
        model.decoder,
        model.encode_rollout_origin,
        x_or_data=origin,
        steps=1,
        control_dim=0,
        default_delta_t=0.1,
        future_presence=presence,
        topology_model=None,
    )
    assert len(rolled) == 1
    with pytest.raises(ValueError, match="parameters for rollout"):
        latent_decode_rollout(
            model.koopman,
            model.decoder,
            model.encode_rollout_origin,
            x_or_data=origin,
            steps=2,
            control_dim=0,
            default_delta_t=0.1,
            parameters=[torch.zeros(1)],
        )
    features = origin.x
    from_tensor = latent_decode_rollout(
        model.koopman,
        model.decoder,
        model.encode_rollout_origin,
        x_or_data=features,
        steps=1,
        control_dim=0,
        default_delta_t=0.1,
        edge_index=origin.edge_index,
        topology_model=None,
    )
    assert len(from_tensor) == 1
    hyper_origin = Data(
        x=torch.randn(3, 2),
        edge_index=origin.edge_index,
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
    )
    from_hyper = latent_decode_rollout(
        model.koopman,
        model.decoder,
        model.encode_rollout_origin,
        x_or_data=hyper_origin,
        steps=1,
        control_dim=0,
        default_delta_t=0.1,
        topology_model=None,
    )
    assert len(from_hyper) == 1
    preds = model.predict_at(origin, query_times=[0.1, 0.2])
    assert len(preds) == 2


def test_subspace_invariance_rejects_heterodata_lists() -> None:
    """Bare HeteroData sequences are refused before ``resolve_sequence``."""
    model = _tiny_model()
    snap = HeteroData()
    snap["node"].x = torch.randn(3, 2)
    snap["node", "r", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    with pytest.raises(ValueError, match="HeteroData sequences"):
        sequence_subspace_invariance(model, [snap, snap.clone()])
    with pytest.raises(ValueError, match="HeteroData sequences"):
        model.subspace_invariance_report([snap])
    hetero_seq = HeteroGraphSnapshotSequence([snap, snap.clone()])
    with pytest.raises(ValueError, match="HeteroGraphSnapshotSequence"):
        sequence_subspace_invariance(model, hetero_seq)


def test_subspace_invariance_accepts_plain_data_lists() -> None:
    """A list of ``Data`` snapshots is resolved then encoded."""
    model = _tiny_model()
    edge = _path_edge_index(3)
    snapshots = [Data(x=torch.randn(3, 2), edge_index=edge) for _ in range(6)]
    report = sequence_subspace_invariance(model, snapshots, held_out=False)
    assert report.n_samples >= 1
    wrapped = GraphSnapshotSequence(snapshots)
    again = model.subspace_invariance_report(wrapped, held_out=False)
    assert again.n_samples == report.n_samples
