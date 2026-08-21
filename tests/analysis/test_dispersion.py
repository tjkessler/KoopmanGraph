"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.analysis import (
    discrete_lyapunov_lmi,
    granger_latent_influence,
    graph_dispersion,
)
from koopman_graph.operators import (
    GraphKoopmanOperator,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _path_edges_v014_remaining(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(*, koopman: str = "pernode", parameterization: str = "dense", **kwargs):
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,
        koopman_parameterization=parameterization,
        **kwargs,
    )


def test_dispersion_rejects_ineligible_and_growth_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual-walk operators are ineligible; a mocked growth shape is reshaped."""
    operator = GraphKoopmanOperator(
        2, init_mode="identity", adjacency="dual_random_walk"
    )
    with pytest.raises(ValueError, match="Dual random-walk|not eligible"):
        graph_dispersion(operator, _path_edges(4), 4)

    import koopman_graph.analysis.dispersion as disp_mod

    spectrum = MagicMock()
    spectrum.eigenvalues = torch.ones(4)
    growth = MagicMock()
    growth.reshape.return_value.mean.return_value = torch.tensor([1.0])
    growth.__getitem__.return_value = torch.ones(4)
    spectrum.growth_rates.real = growth
    monkeypatch.setattr(disp_mod, "spectrum_k_eff_kronecker_sum", lambda **_: spectrum)
    eligible = GraphKoopmanOperator(2, init_mode="identity")
    relation = graph_dispersion(eligible, _path_edges(4), 4)
    assert relation.growth_rates.numel() == 4


def test_dispersion_granger_lmi() -> None:
    """Kronecker dispersion, Granger scores, and Lyapunov LMI smoke."""
    model = _tiny_model(koopman="graph")
    relation = graph_dispersion(model.koopman, _path_edges_v014_remaining(4), 4)
    assert relation.wavenumbers.shape[0] == 4
    latents = torch.randn(8, 3, 2)
    report = granger_latent_influence(latents)
    assert report.scores.shape == (3, 3)
    lmi = discrete_lyapunov_lmi(0.5 * torch.eye(3))
    assert lmi.lyapunov_matrix.shape == (3, 3)
