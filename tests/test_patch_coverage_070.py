"""Patch-coverage gaps for the 0.7.0 Codecov patch gate (target ≥ 90%)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphKoopmanOperator,
    GraphSnapshotSequence,
)
from koopman_graph.analysis.plotting import plot_spectrum
from koopman_graph.analysis.residuals import (
    _compute_mode_residuals,
    _prepare_amplitude_state,
    _resolve_spectrum,
    spectral_residuals,
)
from koopman_graph.baselines import DMDBaseline
from koopman_graph.baselines.base import (
    gavish_donoho_omega,
    optimal_hard_threshold_rank,
    resolve_fit_rank,
)
from koopman_graph.graph_utils.topology import random_walk_normalized_adjacency_matvec
from koopman_graph.operators.graph_inverse import block_diagonal_graph_inverse_advance
from koopman_graph.serialization import _resolve_checkpoint_adjacency
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum
from koopman_graph.statistics import _hann_window, largest_lyapunov_exponent


def _spectrum_from_eigenvalues(eigenvalues: torch.Tensor) -> KoopmanSpectrum:
    magnitudes = eigenvalues.abs()
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=torch.eye(eigenvalues.numel(), dtype=torch.complex128),
        magnitudes=magnitudes,
        growth_rates=torch.log(magnitudes.clamp_min(1e-12)),
        frequencies=torch.angle(eigenvalues) / (2 * torch.pi),
        time_step=1.0,
    )


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_residuals_hypergraph_and_encode_override_paths() -> None:
    """Hypergraph spectrum kwargs, encode overrides, and delta_t guards."""
    edge_index = _path_edges(3)
    # Missing hyperedge_index on a hypergraph model.
    hyp = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        koopman="hypergraph",
        time_step=1.0,
    )
    seq = GraphSnapshotSequence(
        [
            Data(x=torch.randn(3, 2), edge_index=edge_index),
            Data(x=torch.randn(3, 2), edge_index=edge_index),
        ]
    )
    with pytest.raises(ValueError, match="hyperedge_index"):
        spectral_residuals(hyp, seq)

    # Continuous graph with non-positive delta_t.
    cg = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        dynamics_mode="continuous",
        koopman="continuous_graph",
        time_step=0.1,
    )
    with pytest.raises(ValueError, match="delta_t must be positive"):
        spectral_residuals(cg, seq, delta_t=0.0)

    # Edge override forces the encode(snapshot, edges, weights) branch.
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
    )
    report = spectral_residuals(model, seq, edge_index=edge_index)
    assert report.num_pairs == 1
    assert torch.all(torch.isfinite(report.residuals))


def test_residuals_private_helpers_validation() -> None:
    """Cover residual helper ValueError branches."""
    spectrum = compute_spectrum(0.8 * torch.eye(2), time_step=1.0)
    with pytest.raises(ValueError, match="at least two latent frames"):
        _compute_mode_residuals(spectrum, [torch.randn(3, 2)])
    with pytest.raises(ValueError, match="at least 1-D"):
        _prepare_amplitude_state(torch.tensor(1.0), 2)
    with pytest.raises(ValueError, match="incompatible with spectrum"):
        _prepare_amplitude_state(torch.randn(5), 2)

    # Hypergraph kwargs assembly when hyperedge_index is present.
    hyp = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        koopman="hypergraph",
        time_step=1.0,
    )
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyp_seq = GraphSnapshotSequence(
        [
            Data(
                x=torch.randn(3, 2),
                edge_index=_path_edges(3),
                hyperedge_index=hyperedge_index,
            ),
            Data(
                x=torch.randn(3, 2),
                edge_index=_path_edges(3),
                hyperedge_index=hyperedge_index,
            ),
        ]
    )
    resolved = _resolve_spectrum(
        hyp, hyp_seq, edge_index=None, edge_weight=None, delta_t=None
    )
    assert resolved.eigenvalues.numel() == 6


def test_rank_auto_validation_branches() -> None:
    """Gavish–Donoho / resolve_fit_rank / DMDBaseline constructor guards."""
    with pytest.raises(ValueError, match="beta must lie"):
        gavish_donoho_omega(0.0)
    with pytest.raises(ValueError, match="beta must lie"):
        gavish_donoho_omega(1.5)
    with pytest.raises(ValueError, match="1-D"):
        optimal_hard_threshold_rank(torch.ones(2, 2), num_rows=2, num_cols=2)
    with pytest.raises(ValueError, match="num_rows and num_cols"):
        optimal_hard_threshold_rank(torch.ones(3), num_rows=0, num_cols=2)
    assert optimal_hard_threshold_rank(torch.tensor([]), num_rows=2, num_cols=2) == 0
    with pytest.raises(ValueError, match="finite"):
        optimal_hard_threshold_rank(
            torch.tensor([1.0, float("nan")]), num_rows=2, num_cols=2
        )
    assert optimal_hard_threshold_rank(torch.zeros(4), num_rows=4, num_cols=4) == 0

    assert resolve_fit_rank(torch.randn(8, 4), None) is None
    with pytest.raises(ValueError, match="2-D"):
        resolve_fit_rank(torch.randn(4), "auto")
    with pytest.raises(ValueError, match="int, None, or 'auto'"):
        resolve_fit_rank(torch.randn(8, 4), "full")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must be >= 1"):
        resolve_fit_rank(torch.randn(8, 4), 0)
    with pytest.raises(ValueError, match="rank must be <="):
        resolve_fit_rank(torch.randn(8, 4), 20)

    with pytest.raises(ValueError, match="int, None, or 'auto'"):
        DMDBaseline(rank="full")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must be >= 1"):
        DMDBaseline(rank=-1)


def test_checkpoint_adjacency_resolver_and_factory() -> None:
    """Serialization adjacency resolver + factory invalid adjacency."""
    assert (
        _resolve_checkpoint_adjacency("random_walk", koopman_kind="graph")
        == "random_walk"
    )
    with pytest.raises(ValueError, match="adjacency is required"):
        _resolve_checkpoint_adjacency(None, koopman_kind="graph")
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_checkpoint_adjacency("bogus", koopman_kind="graph")
    with pytest.raises(ValueError, match="must be null"):
        _resolve_checkpoint_adjacency("symmetric", koopman_kind="dense")
    assert _resolve_checkpoint_adjacency(None, koopman_kind="dense") == "symmetric"

    with pytest.raises(ValueError, match="koopman_adjacency must be one of"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 2, num_layers=1),
            decoder=GNNDecoder(2, 4, 2, num_layers=1),
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            koopman_adjacency="bogus",  # type: ignore[arg-type]
        )


def test_graph_inverse_and_operator_adjacency_guards() -> None:
    """Inverse helper / dual reset / bound_metric / set_dense k_bwd guards."""
    edge_index = _path_edges(3)
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="k_bwd is required"):
        block_diagonal_graph_inverse_advance(
            z,
            k_self=torch.eye(2),
            k_nbr=0.1 * torch.eye(2),
            edge_index=edge_index,
            adjacency="dual_random_walk",
        )
    with pytest.raises(ValueError, match="adjacency must be"):
        block_diagonal_graph_inverse_advance(
            z,
            k_self=torch.eye(2),
            k_nbr=0.1 * torch.eye(2),
            edge_index=edge_index,
            adjacency="bogus",  # type: ignore[arg-type]
        )

    dual = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    dual.reset_parameters()
    assert torch.isfinite(dual.bound_metric())

    plain = GraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="k_bwd is only valid"):
        plain.set_dense_matrices(
            0.8 * torch.eye(2), 0.1 * torch.eye(2), k_bwd=0.05 * torch.eye(2)
        )


def test_topology_matvec_validation() -> None:
    """Row-walk matvec rejects bad feature rank and direction."""
    edge_index = _path_edges(3)
    with pytest.raises(ValueError, match="x must be 2D"):
        random_walk_normalized_adjacency_matvec(edge_index, torch.randn(3))
    with pytest.raises(ValueError, match="direction must be"):
        random_walk_normalized_adjacency_matvec(
            edge_index,
            torch.randn(3, 2),
            direction="sideways",  # type: ignore[arg-type]
        )


def test_plot_annotate_all_trustworthy_is_noop_overlay() -> None:
    """annotate_untrustworthy with all residuals below tolerance skips overlay."""
    import matplotlib.pyplot as plt

    annotated = replace(
        _spectrum_from_eigenvalues(
            torch.tensor([0.7 + 0.0j, 0.5 + 0.1j], dtype=torch.complex128)
        ),
        residuals=torch.tensor([1e-6, 1e-6], dtype=torch.float64),
    )
    fig, ax = plt.subplots()
    try:
        plot_spectrum(
            annotated,
            ax=ax,
            annotate_untrustworthy=True,
            residual_tolerance=1e-2,
        )
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)


def test_statistics_hann_and_constant_theiler_power() -> None:
    """Direct Hann guard + constant-series Theiler power fallback."""
    with pytest.raises(ValueError, match="segment_length must be >= 2"):
        _hann_window(1, dtype=torch.float32, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="finite divergence|Theiler|too short"):
        largest_lyapunov_exponent(
            torch.ones(64),
            embedding_dim=3,
            delay=1,
            theiler=None,
            trajectory_len=4,
            min_neighbors=5,
        )
