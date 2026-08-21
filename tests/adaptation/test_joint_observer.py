"""Joint state–topology observer: separable toy and theorem-flag guards."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.adaptation import JointObserverResult, JointStateTopologyObserver
from koopman_graph.nn import (
    SeparableDictionaryDecoder,
    SeparableDictionaryEncoder,
)
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path.

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


def _sequence(
    *,
    in_channels: int = 2,
    num_nodes: int = 4,
    length: int = 4,
) -> GraphSnapshotSequence:
    """Build a short homogeneous path sequence.

    Parameters
    ----------
    in_channels : int, optional
        Feature width.
    num_nodes : int, optional
        Node count.
    length : int, optional
        Snapshot count.

    Returns
    -------
    GraphSnapshotSequence
        Path-graph trajectory.
    """
    torch.manual_seed(0)
    edges = _path_edge_index(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, in_channels), edge_index=edges)
        for _ in range(length)
    ]
    return GraphSnapshotSequence(snapshots)


def _separable_graph_model() -> GraphKoopmanModel:
    """Return an untrained separable graph model.

    Returns
    -------
    GraphKoopmanModel
        Node-wise encoder/decoder with ``koopman="graph"``.
    """
    return GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
    )


def test_separable_graph_observer_runs_on_toy() -> None:
    """Filter plus group-sparse factor write-back returns finite factors."""
    model = _separable_graph_model()
    sequence = _sequence()
    observer = JointStateTopologyObserver(model, claim_homomorphism=True)
    result = observer.filter_and_adapt(sequence)
    assert result.sparse_factors is not None
    assert result.filter.latents.shape[0] == len(sequence)
    assert torch.isfinite(result.sparse_factors.K_self).all()
    assert torch.isfinite(result.sparse_factors.K_nbr).all()
    assert isinstance(model.koopman, GraphKoopmanOperator)
    torch.testing.assert_close(
        model.koopman.K_self,
        result.sparse_factors.K_self.to(
            device=model.koopman.K_self.device,
            dtype=model.koopman.K_self.dtype,
        ),
        atol=1e-6,
        rtol=0.0,
    )
    assert result.rls_steps == ()


def test_gnn_encoder_theorem_flag_raises() -> None:
    """Neighbor-mixing GNN encoders cannot carry homomorphism claims."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 8, 3, num_layers=1),
        GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
    )
    with pytest.raises(ValueError, match="encoder_kind='separable'"):
        JointStateTopologyObserver(model, claim_homomorphism=True)
    observer = JointStateTopologyObserver(model, claim_homomorphism=False)
    result = observer.filter_and_adapt(_sequence())
    assert result.sparse_factors is not None


def test_dense_operator_theorem_flag_raises() -> None:
    """Homomorphism claims require a graph operator, not dense per-node ``K``."""
    model = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman=None,
    )
    with pytest.raises(ValueError, match="GraphKoopmanOperator"):
        JointStateTopologyObserver(model, claim_homomorphism=True)


def test_dense_path_runs_rls() -> None:
    """Per-node dense ``K`` uses RLS write-back without a theorem tag."""
    model = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman=None,
    )
    assert isinstance(model.koopman, KoopmanOperator)
    observer = JointStateTopologyObserver(model, claim_homomorphism=False)
    result = observer.filter_and_adapt(_sequence())
    assert result.sparse_factors is None
    assert len(result.rls_steps) == 3
    assert all(torch.isfinite(step.operator_change_norm) for step in result.rls_steps)


def test_short_sequence_raises() -> None:
    """A single snapshot cannot form latent pairs."""
    model = _separable_graph_model()
    observer = JointStateTopologyObserver(model, claim_homomorphism=True)
    with pytest.raises(ValueError, match="T >= 2"):
        observer.filter_and_adapt(_sequence(length=1))


def test_polynomial_and_dual_walk_graph_paths_raise() -> None:
    """One-tap dense symmetric / random-walk factorization only."""
    dual = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_adjacency="dual_random_walk",
    )
    with pytest.raises(ValueError, match="dual_random_walk"):
        JointStateTopologyObserver(dual, claim_homomorphism=True)
    polynomial = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_filter_degree=2,
    )
    with pytest.raises(ValueError, match="filter_degree=1"):
        JointStateTopologyObserver(polynomial, claim_homomorphism=True)


def test_rls_and_observer_import_does_not_load_identification() -> None:
    """Package import of RLS / Kalman observer stays identification-free."""
    script = (
        "import sys\n"
        "from koopman_graph.adaptation import (\n"
        "    KoopmanObserver, RecursiveKoopmanAdapter)\n"
        "assert KoopmanObserver is not None\n"
        "assert RecursiveKoopmanAdapter is not None\n"
        "assert 'koopman_graph.identification' not in sys.modules, "
        "sorted(k for k in sys.modules if 'identification' in k)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_joint_observer_types_are_exported() -> None:
    """Lazy package export resolves joint-observer types."""
    assert JointObserverResult.__name__ == "JointObserverResult"
    assert JointStateTopologyObserver.__name__ == "JointStateTopologyObserver"
