"""Coverage and error-path tests for :mod:`koopman_graph.nn.predicted_topology`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphDynamicsConfig
from koopman_graph.nn import (
    DENSE_TOPOLOGY_MAX_NODES,
    PredictedTopologyHead,
    SparseCandidateTopologyHead,
    build_candidate_index,
)
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_model_config,
    reconstruct_model,
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
        COO index ``(2, 2*(N-1))``.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(
    *,
    graph_dynamics: GraphDynamicsConfig | None = None,
    learn_topology: str | None = None,
) -> GraphKoopmanModel:
    """Return a small GCN model.

    Parameters
    ----------
    graph_dynamics : GraphDynamicsConfig or None, optional
        Opt-in graph-state config.
    learn_topology : str or None, optional
        Adaptive-adjacency flag.

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
        graph_dynamics=graph_dynamics,
        learn_topology=learn_topology,
    )


def test_predicted_topology_validation_empty_edges_and_forward() -> None:
    """Topology head covers invalid inputs, empty thresholding, and ``forward``."""
    with pytest.raises(ValueError, match="latent_dim"):
        PredictedTopologyHead(0)
    with pytest.raises(ValueError, match="hidden_dim"):
        PredictedTopologyHead(3, hidden_dim=0)
    with pytest.raises(ValueError, match="max_nodes"):
        PredictedTopologyHead(3, max_nodes=1)
    head = PredictedTopologyHead(3, hidden_dim=8)
    with pytest.raises(ValueError, match="shape \\(N, d\\)"):
        head.pairwise_logits(torch.randn(3))
    z = torch.randn(4, 3)
    empty = head.edge_index(z, threshold=1e9)
    assert empty.shape == (2, 4)
    assert head(z).shape == (4, 4)


def test_predicted_topology_head_logits() -> None:
    """Predicted topology emits pairwise logits and a COO index."""
    head = PredictedTopologyHead(4, hidden_dim=8)
    z = torch.randn(5, 4)
    logits = head.pairwise_logits(z)
    assert logits.shape == (5, 5)
    edges = head.edge_index(z, top_k=2)
    assert edges.shape[0] == 2
    assert edges.shape[1] == 10


def test_predicted_topology_dense_ceiling() -> None:
    """Dense MLP refuses :math:`N` above the documented ceiling."""
    assert DENSE_TOPOLOGY_MAX_NODES == 64
    head = PredictedTopologyHead(2, hidden_dim=4)
    z_ok = torch.randn(5, 2)
    assert head.pairwise_logits(z_ok).shape == (5, 5)
    with pytest.raises(ValueError, match="max_nodes=64"):
        head.pairwise_logits(torch.randn(65, 2))
    tight = PredictedTopologyHead(2, hidden_dim=4, max_nodes=4)
    with pytest.raises(ValueError, match="max_nodes=4"):
        tight.pairwise_logits(torch.randn(5, 2))


def test_build_candidate_index_caps_destinations() -> None:
    """Candidate COO stays at most ``N * k`` and prefers existing edges."""
    with pytest.raises(ValueError, match="num_nodes"):
        build_candidate_index(1, 2)
    with pytest.raises(ValueError, match="candidate_k"):
        build_candidate_index(4, 0)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        build_candidate_index(4, 2, torch.arange(4))
    generator = torch.Generator().manual_seed(0)
    num_nodes = 32
    candidate_k = 8
    index = build_candidate_index(
        num_nodes,
        candidate_k,
        _path_edge_index(num_nodes),
        generator=generator,
    )
    assert index.shape[0] == 2
    assert int(index.shape[1]) <= num_nodes * candidate_k
    src, dst = index[0], index[1]
    assert int((src == dst).sum()) == 0
    for source in range(num_nodes):
        assert int((src == source).sum()) <= candidate_k
    filled = build_candidate_index(5, 8, generator=torch.Generator().manual_seed(1))
    assert int(filled.shape[1]) == 5 * 4


def test_sparse_candidate_head_scores_pairs_not_dense() -> None:
    """Sparse head returns ``(E,)`` logits with ``E <= N k``."""
    with pytest.raises(ValueError, match="latent_dim"):
        SparseCandidateTopologyHead(0)
    with pytest.raises(ValueError, match="hidden_dim"):
        SparseCandidateTopologyHead(3, hidden_dim=0)
    with pytest.raises(ValueError, match="candidate_k"):
        SparseCandidateTopologyHead(3, candidate_k=0)
    head = SparseCandidateTopologyHead(4, hidden_dim=8, candidate_k=8)
    z = torch.randn(32, 4)
    candidates = build_candidate_index(
        32, 8, _path_edge_index(32), generator=torch.Generator().manual_seed(2)
    )
    logits = head.pair_logits(z, candidates)
    assert logits.ndim == 1
    assert logits.shape == (int(candidates.shape[1]),)
    assert logits.shape[0] <= 32 * 8
    assert head(z, candidates).shape == logits.shape
    kept = head.edge_index(z, candidates, threshold=-1e9)
    assert kept.shape[0] == 2
    with pytest.raises(ValueError, match="shape \\(N, d\\)"):
        head.pair_logits(torch.randn(4), candidates)
    too_many = torch.zeros(2, 32 * 31 + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="N\\*\\(N-1\\)"):
        head.pair_logits(z, too_many)
    oversize_ok = torch.zeros(2, 32 * 8 + 1, dtype=torch.long)
    logits = head.pair_logits(z, oversize_ok)
    assert logits.shape == (32 * 8 + 1,)
    bad = torch.tensor([[0, 99], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        head.pair_logits(z[:5], bad)


def test_graph_koopman_model_attaches_sparse_head_without_using_it() -> None:
    """Default opt-in config attaches a sparse head unused by ``forward``."""
    default = _tiny_model()
    assert default.graph_dynamics is None
    assert default.predicted_topology is None
    assert not any(
        name.startswith("predicted_topology") for name in default.state_dict()
    )
    with pytest.raises(TypeError, match="GraphDynamicsConfig"):
        GraphKoopmanModel(
            GNNEncoder(2, 8, 3, num_layers=1),
            GNNDecoder(3, 8, 2, num_layers=1),
            latent_dim=3,
            time_step=0.1,
            graph_dynamics="sparse_candidate",  # type: ignore[arg-type]
        )
    none_head = _tiny_model(graph_dynamics=GraphDynamicsConfig(topology_head="none"))
    assert none_head.predicted_topology is None
    assert none_head.presence_head is not None
    sparse = _tiny_model(graph_dynamics=GraphDynamicsConfig())
    assert sparse.graph_dynamics is not None
    assert sparse.graph_dynamics.topology_head == "sparse_candidate"
    assert isinstance(sparse.predicted_topology, SparseCandidateTopologyHead)
    assert sparse.presence_head is not None
    assert sparse.predicted_topology.candidate_k == 8
    dense = _tiny_model(graph_dynamics=GraphDynamicsConfig(topology_head="dense_mlp"))
    assert isinstance(dense.predicted_topology, PredictedTopologyHead)
    assert dense.predicted_topology.max_nodes == DENSE_TOPOLOGY_MAX_NODES
    with pytest.raises(ValueError, match="self_adaptive"):
        _tiny_model(
            graph_dynamics=GraphDynamicsConfig(),
            learn_topology="self_adaptive",
        )
    torch.manual_seed(0)
    baseline = _tiny_model()
    configured = _tiny_model(graph_dynamics=GraphDynamicsConfig())
    configured.encoder.load_state_dict(baseline.encoder.state_dict())
    configured.decoder.load_state_dict(baseline.decoder.state_dict())
    configured.koopman.load_state_dict(baseline.koopman.state_dict())
    data = Data(x=torch.randn(4, 2), edge_index=_path_edge_index(4))
    baseline.eval()
    configured.eval()
    with torch.no_grad():
        assert torch.allclose(baseline(data), configured(data), atol=1e-6)


def test_graph_dynamics_format1_round_trip(tmp_path: Path) -> None:
    """Additive ``graph_dynamics`` round-trips; absent keys stay 0.14."""
    assert FORMAT_VERSION == 1
    model = _tiny_model(graph_dynamics=GraphDynamicsConfig(candidate_k=3))
    assert model.predicted_topology is not None
    before = {
        key: tensor.detach().clone() for key, tensor in model.state_dict().items()
    }
    path = tmp_path / "graph_dynamics"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.graph_dynamics == model.graph_dynamics
    assert isinstance(loaded.predicted_topology, SparseCandidateTopologyHead)
    assert loaded.predicted_topology.candidate_k == 3
    for key, tensor in before.items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)
    config = build_model_config(_tiny_model())
    assert config["graph_dynamics"] is None
    absent = dict(config)
    absent.pop("graph_dynamics", None)
    restored = reconstruct_model(absent)
    assert restored.graph_dynamics is None
    assert restored.predicted_topology is None
