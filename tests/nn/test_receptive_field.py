"""Tests for encoder vs operator receptive-field checks."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    DiffConvEncoder,
    ReceptiveFieldMismatchWarning,
    check_encoder_operator_receptive_field,
)
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator
from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected path edge index."""
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def test_diffusion_steps_exceeding_filter_degree_warns() -> None:
    """Acceptance: ``diffusion_steps > filter_degree`` warns (one DiffConv layer)."""
    encoder = DiffConvEncoder(2, 4, 3, num_layers=1, diffusion_steps=2)
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=1)
    assert encoder.receptive_field_hops() == 2
    with pytest.warns(ReceptiveFieldMismatchWarning, match="does not compensate"):
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.warned is True
    assert report.encoder_hops == 2
    assert report.operator_hops == 1


def test_gcn_depth_exceeding_filter_degree_warns() -> None:
    """Default two-layer GCN vs one-hop graph ``K`` warns."""
    encoder = GNNEncoder(2, 4, 3, num_layers=2)
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=1)
    with pytest.warns(ReceptiveFieldMismatchWarning, match="does not compensate"):
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.encoder_hops == 2
    assert report.operator_hops == 1


def test_matched_hops_are_silent() -> None:
    """Equal hop counts do not warn."""
    encoder = GNNEncoder(2, 4, 3, num_layers=1)
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReceptiveFieldMismatchWarning)
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.warned is False
    assert report.encoder_hops == 1
    assert report.operator_hops == 1


def test_operator_longer_than_encoder_is_silent() -> None:
    """Operator hops may exceed encoder hops without a warning."""
    encoder = GNNEncoder(2, 4, 3, num_layers=2)
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReceptiveFieldMismatchWarning)
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.warned is False
    assert report.encoder_hops == 2
    assert report.operator_hops == 2


def test_pernode_operator_skips_check() -> None:
    """Per-node ``K`` has no graph hop radius; the helper stays silent."""
    encoder = GNNEncoder(2, 4, 3, num_layers=3)
    operator = KoopmanOperator(3, init_mode="identity")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReceptiveFieldMismatchWarning)
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.warned is False
    assert report.operator_hops is None
    assert report.encoder_hops == 3


def test_continuous_graph_operator_skips_check() -> None:
    """Continuous graph has no ``filter_degree`` hop method."""
    encoder = GNNEncoder(2, 4, 3, num_layers=2)
    operator = ContinuousGraphKoopmanOperator(3, init_mode="identity")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReceptiveFieldMismatchWarning)
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.warned is False
    assert report.operator_hops is None


def test_diffconv_hops_are_layers_times_diffusion_steps() -> None:
    """Stacked DiffConv radius is the product, not ``diffusion_steps`` alone."""
    encoder = DiffConvEncoder(2, 4, 3, num_layers=2, diffusion_steps=3)
    assert encoder.receptive_field_hops() == 6
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=1)
    with pytest.warns(ReceptiveFieldMismatchWarning, match="6 hop"):
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.encoder_hops == 6


def test_delay_wrapper_uses_base_encoder_hops() -> None:
    """Delay stacking does not add spatial hops."""
    base = GNNEncoder(6, 4, 3, num_layers=2)
    encoder = DelayEmbeddingEncoder(base, n_delays=2)
    assert encoder.receptive_field_hops() == 2
    operator = GraphKoopmanOperator(3, init_mode="identity", filter_degree=1)
    with pytest.warns(ReceptiveFieldMismatchWarning, match="does not compensate"):
        report = check_encoder_operator_receptive_field(encoder, operator)
    assert report.encoder_hops == 2


def test_fit_warns_when_encoder_hops_exceed_filter_degree() -> None:
    """``GraphKoopmanModel.fit`` emits the mismatch warning once."""
    encoder = GNNEncoder(2, 4, 3, num_layers=2)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_init_mode="identity",
    )
    edge_index = _path_edge_index(4)
    snapshots = [Data(x=torch.randn(4, 2), edge_index=edge_index) for _ in range(2)]
    with pytest.warns(ReceptiveFieldMismatchWarning, match="does not compensate"):
        history = model.fit(GraphSnapshotSequence(snapshots), epochs=1)
    assert history.epochs == 1


def test_fit_pernode_default_is_silent() -> None:
    """Default per-node ``K`` does not warn on a two-layer GCN."""
    encoder = GNNEncoder(2, 4, 3, num_layers=2)
    decoder = GNNDecoder(3, 4, 2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=3,
        time_step=0.1,
    )
    edge_index = _path_edge_index(4)
    snapshots = [Data(x=torch.randn(4, 2), edge_index=edge_index) for _ in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReceptiveFieldMismatchWarning)
        history = model.fit(GraphSnapshotSequence(snapshots), epochs=1)
    assert history.epochs == 1
