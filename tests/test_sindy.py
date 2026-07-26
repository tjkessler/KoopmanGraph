"""Tests for SINDy latent sparse identification (TASK-1311)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.analysis import SINDyReport, identify_sparse_dynamics
from koopman_graph.analysis.sindy import (
    _build_graph_library,
    _build_poly_library,
    _discrete_pairs,
    _finite_difference_targets,
    _library_for_frames,
    _monomial_name,
    _stlsq,
)
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


def test_monomial_name_empty_and_powers() -> None:
    """Monomial naming covers the constant term and repeated powers."""
    assert _monomial_name(()) == "1"
    assert _monomial_name((0,)) == "z0"
    assert _monomial_name((0, 0, 1)) == "z0^2*z1"


def test_build_poly_library_validation() -> None:
    """Polynomial library rejects invalid shapes and degrees."""
    with pytest.raises(ValueError, match="num_samples, latent_dim"):
        _build_poly_library(torch.ones(2, 2, 2), degree=1)
    with pytest.raises(ValueError, match="degree must be >= 1"):
        _build_poly_library(torch.ones(3, 2), degree=0)


def test_build_graph_library_validation() -> None:
    """Graph library validates node-matrix rank before Laplacian lifting."""
    edge_index = _path_edge_index(3)
    with pytest.raises(ValueError, match="num_nodes, latent_dim"):
        _build_graph_library(torch.ones(2, 2, 2), edge_index, None, degree=1)


def test_stlsq_validation_and_rank_one_solution() -> None:
    """STLSQ validates inputs and handles rank-one least-squares solutions."""
    theta = torch.tensor([[1.0], [1.0], [1.0]])
    target = torch.tensor([[2.0], [2.0], [2.0]])
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        _stlsq(theta, target, threshold=-0.1, max_iter=1)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        _stlsq(theta, target, threshold=0.1, max_iter=0)
    with pytest.raises(ValueError, match="theta and target must be 2D"):
        _stlsq(torch.ones(3), target, threshold=0.1, max_iter=1)
    with pytest.raises(ValueError, match="share the sample axis"):
        _stlsq(theta, target[:2], threshold=0.1, max_iter=1)
    with pytest.raises(ValueError, match="at least one sample"):
        _stlsq(torch.zeros(0, 1), torch.zeros(0, 1), threshold=0.1, max_iter=1)

    coefficients, active_mask, history = _stlsq(
        theta,
        target,
        threshold=0.5,
        max_iter=2,
    )
    assert coefficients.shape == (1, 1)
    assert history[0] == pytest.approx(0.5)
    assert active_mask.shape == coefficients.shape

    wide_theta = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    wide_target = torch.tensor([[0.01, 5.0], [0.01, 5.0], [0.01, 5.0]])
    wide_coefficients, _, _ = _stlsq(
        wide_theta,
        wide_target,
        threshold=1.0,
        max_iter=2,
    )
    assert wide_coefficients.shape == (2, 2)
    assert wide_coefficients[0, 0].item() == pytest.approx(0.0)


def test_finite_difference_and_discrete_pair_validation() -> None:
    """Target builders require at least two latent frames and positive ``dt``."""
    latent = torch.ones(3, 2)
    with pytest.raises(ValueError, match="at least two latent frames"):
        _finite_difference_targets([latent], timestamps=None, time_step=0.1)
    with pytest.raises(ValueError, match="at least two latent frames"):
        _discrete_pairs([latent])
    with pytest.raises(ValueError, match="non-positive time increment"):
        _finite_difference_targets(
            [latent, latent + 1.0],
            timestamps=torch.tensor([0.0, 0.0]),
            time_step=0.1,
        )


def test_library_for_frames_validation() -> None:
    """Frame-wise library construction validates graph topology and indices."""
    latents = [torch.ones(3, 2), torch.ones(3, 2) * 2.0]
    with pytest.raises(ValueError, match="graph library requires edge_index"):
        _library_for_frames(
            latents,
            library="graph",
            degree=1,
            edge_index=None,
            edge_weight=None,
            frame_indices=[0],
        )
    with pytest.raises(ValueError, match="no frames provided"):
        _library_for_frames(
            latents,
            library="poly",
            degree=1,
            edge_index=None,
            edge_weight=None,
            frame_indices=[],
        )


def test_identify_sparse_dynamics_validation_branches() -> None:
    """Public API rejects invalid libraries, modes, degrees, and sequences."""
    sequence = _planted_discrete_sequence(num_timesteps=8)
    model = _identity_latent_model(2)
    with pytest.raises(ValueError, match="library must be"):
        identify_sparse_dynamics(
            model,
            sequence,
            library="invalid",
            threshold=0.1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="mode must be"):
        identify_sparse_dynamics(
            model,
            sequence,
            mode="invalid",
            threshold=0.1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="degree must be >= 1"):
        identify_sparse_dynamics(model, sequence, degree=0, threshold=0.1)
    single_snapshot = Data(x=torch.ones(3, 2), edge_index=_path_edge_index(3))
    with pytest.raises(ValueError, match="at least two snapshots"):
        identify_sparse_dynamics(
            model,
            GraphSnapshotSequence([single_snapshot]),
            threshold=0.1,
        )


def test_identify_sparse_dynamics_latent_dim_mismatch(monkeypatch) -> None:
    """Encoded latent width must stay constant across timesteps."""
    sequence = _planted_discrete_sequence(num_timesteps=8)
    model = _identity_latent_model(2)

    def _inconsistent_latents(_model, _sequence):
        return [torch.ones(4, 2), torch.ones(4, 3)]

    monkeypatch.setattr(
        "koopman_graph.analysis.sindy._encode_latent_trajectory",
        _inconsistent_latents,
    )
    with pytest.raises(ValueError, match="inconsistent across timesteps"):
        identify_sparse_dynamics(model, sequence, threshold=0.1)


def test_identify_sparse_dynamics_sample_count_mismatch(monkeypatch) -> None:
    """Library and target row counts must agree before STLSQ."""
    sequence = _planted_discrete_sequence(num_timesteps=8)
    model = _identity_latent_model(2)

    def _short_discrete(_latents):
        return torch.ones(1, 2), torch.ones(1, 2)

    monkeypatch.setattr(
        "koopman_graph.analysis.sindy._discrete_pairs",
        _short_discrete,
    )
    with pytest.raises(ValueError, match="sample counts disagree"):
        identify_sparse_dynamics(model, sequence, threshold=0.1)
