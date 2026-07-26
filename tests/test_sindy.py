"""Tests for SINDy latent sparse identification (TASK-1311)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.analysis import SINDyReport, identify_sparse_dynamics
from koopman_graph.analysis.sindy import _build_graph_library, _build_poly_library
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _identity_latent_model(latent_dim: int = 2) -> GraphKoopmanModel:
    """Build a model whose encode path is patched to return snapshot features."""
    encoder = GNNEncoder(
        in_channels=latent_dim,
        hidden_channels=4,
        latent_dim=latent_dim,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=latent_dim,
        hidden_channels=4,
        out_channels=latent_dim,
        num_layers=1,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
    )

    def _encode_features(x_or_data, edge_index=None, edge_weight=None):
        if isinstance(x_or_data, Data):
            assert x_or_data.x is not None
            return x_or_data.x
        return x_or_data

    model.encode = _encode_features  # type: ignore[method-assign]
    return model


def _planted_discrete_sequence(
    *,
    num_nodes: int = 4,
    num_timesteps: int = 40,
    seed: int = 0,
) -> GraphSnapshotSequence:
    """Noiseless trajectory from a sparse quadratic discrete map."""
    torch.manual_seed(seed)
    edge_index = _path_edge_index(num_nodes)
    # z0' = 0.8 z0 + 0.15 z0^2 ; z1' = 0.7 z1
    z = 0.3 * torch.randn(num_nodes, 2)
    snapshots: list[Data] = [Data(x=z.clone(), edge_index=edge_index)]
    for _ in range(num_timesteps - 1):
        z_next = torch.empty_like(z)
        z_next[:, 0] = 0.8 * z[:, 0] + 0.15 * z[:, 0] ** 2
        z_next[:, 1] = 0.7 * z[:, 1]
        z = z_next
        snapshots.append(Data(x=z.clone(), edge_index=edge_index))
    return GraphSnapshotSequence(snapshots)


def test_planted_discrete_recovery() -> None:
    """STLSQ recovers a planted sparse quadratic map; inactive terms are zero."""
    sequence = _planted_discrete_sequence()
    model = _identity_latent_model(2)
    report = identify_sparse_dynamics(
        model,
        sequence,
        library="poly",
        degree=2,
        threshold=0.05,
        mode="discrete",
    )
    assert isinstance(report, SINDyReport)
    names = list(report.term_names)
    assert "z0" in names and "z0^2" in names and "z1" in names
    i_z0 = names.index("z0")
    i_z0sq = names.index("z0^2")
    i_z1 = names.index("z1")

    # Column 0: z0' depends on z0 and z0^2
    assert report.coefficients[i_z0, 0].item() == pytest.approx(0.8, abs=0.05)
    assert report.coefficients[i_z0sq, 0].item() == pytest.approx(0.15, abs=0.05)
    # Column 1: z1' depends only on z1
    assert report.coefficients[i_z1, 1].item() == pytest.approx(0.7, abs=0.05)

    # Inactive coefficients are exactly zero after thresholding.
    inactive = ~report.active_mask
    assert torch.all(report.coefficients[inactive] == 0)
    assert report.residual < 1e-6


def test_term_names_align_with_coefficient_rows() -> None:
    """Library term names match coefficient row count for poly and graph."""
    z = torch.randn(5, 2)
    edge_index = _path_edge_index(5)
    theta_poly, names_poly = _build_poly_library(z, degree=2)
    assert len(names_poly) == theta_poly.shape[1]
    assert names_poly[0] == "1"
    assert "z0*z1" in names_poly or "z0^2" in names_poly

    theta_graph, names_graph = _build_graph_library(z, edge_index, None, degree=2)
    assert len(names_graph) == theta_graph.shape[1]
    assert any(name.startswith("Lsym:") for name in names_graph)
    assert names_graph[0] == "1"


def test_derivative_mode_recovers_linear_ode() -> None:
    """Derivative mode with timestamps recovers a sparse linear generator."""
    torch.manual_seed(1)
    num_nodes = 3
    num_timesteps = 50
    dt = 0.05
    edge_index = _path_edge_index(num_nodes)
    # \dot z0 = -0.5 z0 ; \dot z1 = -0.2 z1
    z = torch.randn(num_nodes, 2)
    snapshots: list[Data] = []
    times = []
    for step in range(num_timesteps):
        snapshots.append(Data(x=z.clone(), edge_index=edge_index))
        times.append(step * dt)
        z = z + dt * torch.stack([-0.5 * z[:, 0], -0.2 * z[:, 1]], dim=-1)

    sequence = GraphSnapshotSequence(
        snapshots,
        timestamps=torch.tensor(times, dtype=torch.float32),
    )
    model = _identity_latent_model(2)
    report = identify_sparse_dynamics(
        model,
        sequence,
        library="poly",
        degree=1,
        threshold=0.05,
        mode="derivative",
    )
    names = list(report.term_names)
    i_z0 = names.index("z0")
    i_z1 = names.index("z1")
    assert report.coefficients[i_z0, 0].item() == pytest.approx(-0.5, abs=0.08)
    assert report.coefficients[i_z1, 1].item() == pytest.approx(-0.2, abs=0.08)
    inactive = ~report.active_mask
    assert torch.all(report.coefficients[inactive] == 0)


def test_graph_library_end_to_end() -> None:
    """Graph library runs through the public API and exposes Lsym terms."""
    sequence = _planted_discrete_sequence(num_timesteps=20)
    model = _identity_latent_model(2)
    report = identify_sparse_dynamics(
        model,
        sequence,
        library="graph",
        degree=1,
        threshold=0.2,
        mode="discrete",
    )
    assert report.library == "graph"
    assert len(report.term_names) == report.coefficients.shape[0]
    assert any(name.startswith("Lsym:") for name in report.term_names)


def test_sindy_report_is_frozen() -> None:
    """SINDyReport rejects attribute assignment."""
    sequence = _planted_discrete_sequence(num_timesteps=8)
    model = _identity_latent_model(2)
    report = identify_sparse_dynamics(
        model,
        sequence,
        library="poly",
        degree=1,
        threshold=0.1,
    )
    with pytest.raises(FrozenInstanceError):
        report.residual = 0.0  # type: ignore[misc]
