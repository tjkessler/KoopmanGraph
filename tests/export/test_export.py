"""Coverage and error-path tests for :mod:`koopman_graph.export`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.export import (
    _reject_unsupported_export,
    compress_operator_svd,
    export_inference_module,
    export_onnx,
)
from koopman_graph.federated import federated_average
from koopman_graph.probabilistic import KoopmanVAEEncoder
from koopman_graph.robustness import corrupt_node_features


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(**kwargs: Any) -> GraphKoopmanModel:
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        **kwargs,
    )


def _path_edges_v014_remaining(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model_v014_remaining(
    *, koopman: str = "pernode", parameterization: str = "dense", **kwargs
):
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


def test_export_reject_paths_trace_fallback_onnx_and_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable export rejects unsupported models and falls back to tracing."""
    model = _tiny_model()
    graph = Data(x=torch.randn(4, 2), edge_index=_path_edges(4))
    flags = SimpleNamespace(
        control_dim=1,
        n_delays=1,
        uses_hetero_koopman=False,
        uses_hypergraph_koopman=False,
        learns_pairwise_topology=False,
        koopman=model.koopman,
    )
    with pytest.raises(ValueError, match="control_dim"):
        _reject_unsupported_export(flags)  # type: ignore[arg-type]
    flags.control_dim = 0
    flags.n_delays = 2
    with pytest.raises(ValueError, match="n_delays"):
        _reject_unsupported_export(flags)  # type: ignore[arg-type]
    flags.n_delays = 1
    flags.uses_hetero_koopman = True
    with pytest.raises(ValueError, match="hetero"):
        _reject_unsupported_export(flags)  # type: ignore[arg-type]
    flags.uses_hetero_koopman = False
    flags.uses_hypergraph_koopman = True
    with pytest.raises(ValueError, match="hetero"):
        _reject_unsupported_export(flags)  # type: ignore[arg-type]
    flags.uses_hypergraph_koopman = False
    flags.learns_pairwise_topology = True
    with pytest.raises(ValueError, match="adaptive"):
        _reject_unsupported_export(flags)  # type: ignore[arg-type]
    flags.learns_pairwise_topology = False
    with pytest.raises(ValueError, match="x and edge_index"):
        export_inference_module(model, Data(edge_index=_path_edges(4)))
    with pytest.raises(ValueError, match="rank"):
        compress_operator_svd(torch.eye(3), rank=0)

    monkeypatch.setattr(
        torch.export,
        "export",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no export")),
    )
    traced = export_inference_module(model, graph)
    assert traced(graph.x).shape == graph.x.shape

    linear = nn.Linear(2, 2)
    onnx_path = str(tmp_path / "linear.onnx")
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)
    written = export_onnx(linear, torch.randn(3, 2), onnx_path)
    assert written == onnx_path


def test_export_compress_federated_fdi_vae() -> None:
    """Export smoke, SVD compression, FedAvg, FDI, and VAE encoder."""
    model = _tiny_model_v014_remaining()
    graph = Data(x=torch.randn(4, 2), edge_index=_path_edges_v014_remaining(4))
    exported = export_inference_module(model, graph)
    out = exported(graph.x)
    assert out.shape == graph.x.shape
    truncated = compress_operator_svd(torch.eye(4), rank=2)
    assert truncated.shape == (4, 4)
    averaged = federated_average([{"w": torch.ones(2)}, {"w": torch.zeros(2)}])
    assert torch.allclose(averaged["w"], torch.full((2,), 0.5))
    corrupted = corrupt_node_features(graph, magnitude=0.1)
    assert corrupted.x is not None
    vae = KoopmanVAEEncoder(2, 4, 3)
    latent = vae.encode(graph)
    assert latent.z.shape[-1] == 3
    assert vae.advance(latent.z).shape == latent.z.shape


def test_export_rejects_switched() -> None:
    """Portable export refuses switched operators."""
    model = _tiny_model_v014_remaining(koopman="switched")
    graph = Data(x=torch.randn(4, 2), edge_index=_path_edges_v014_remaining(4))
    with pytest.raises(ValueError, match="switched"):
        export_inference_module(model, graph)
