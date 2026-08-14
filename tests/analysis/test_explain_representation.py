"""Tests for representation-explanation result types and GNNExplainer path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph import analysis as kg_analysis
from koopman_graph.analysis import RepresentationExplanation, explain_representation
from koopman_graph.analysis.explain import RepresentationExplanation as direct


def _path_data(*, num_nodes: int = 6, in_channels: int = 3) -> Data:
    """Build a small path-graph snapshot."""
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    x = torch.randn(num_nodes, in_channels)
    return Data(x=x, edge_index=edge_index)


def _tiny_model(*, in_channels: int = 3, latent_dim: int = 4) -> GraphKoopmanModel:
    """Build a tiny homogeneous GCN Koopman model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels, 8, latent_dim, num_layers=2),
        decoder=GNNDecoder(latent_dim, 8, in_channels, num_layers=2),
        latent_dim=latent_dim,
        time_step=0.1,
    )


def test_representation_explanation_exported_from_analysis() -> None:
    """Result type is on the analysis public surface (not root ``__all__``)."""
    import koopman_graph

    assert RepresentationExplanation is direct
    assert "RepresentationExplanation" in kg_analysis.__all__
    assert "explain_representation" in kg_analysis.__all__
    assert "RepresentationExplanation" not in koopman_graph.__all__
    assert "explain_representation" not in koopman_graph.__all__
    assert not hasattr(koopman_graph, "RepresentationExplanation")
    assert not hasattr(koopman_graph, "explain_representation")
    with pytest.raises(ImportError):
        exec("from koopman_graph import explain_representation")
    with pytest.raises(ImportError):
        exec("from koopman_graph import RepresentationExplanation")


def test_representation_explanation_fields_match_design() -> None:
    """Frozen dataclass matches the design §3.5 field sketch."""
    node_mask = torch.ones(4)
    edge_mask = torch.zeros(6)
    explanation = RepresentationExplanation(
        target="one_step_forecast",
        node_mask=node_mask,
        edge_mask=edge_mask,
        feature_mask=None,
        algorithm="gnn_explainer",
        index=2,
    )
    assert explanation.target == "one_step_forecast"
    assert explanation.node_mask is node_mask
    assert explanation.edge_mask is edge_mask
    assert explanation.feature_mask is None
    assert explanation.algorithm == "gnn_explainer"
    assert explanation.index == 2
    with pytest.raises(FrozenInstanceError):
        explanation.algorithm = "integrated_gradients"  # type: ignore[misc]


def test_representation_explanation_allows_all_none_masks() -> None:
    """Masks may be entirely absent before algorithm wrappers ship."""
    explanation = RepresentationExplanation(
        target="latent",
        node_mask=None,
        edge_mask=None,
        feature_mask=None,
        algorithm="pending",
        index=None,
    )
    assert explanation.node_mask is None
    assert explanation.edge_mask is None
    assert explanation.feature_mask is None
    assert explanation.index is None


def test_honesty_docstrings_non_causal_not_resdmd_not_mode_energy() -> None:
    """Module and class docs must state interpretive / non-causal honesty."""
    import koopman_graph.analysis.explain as explain_mod

    module_doc = explain_mod.__doc__ or ""
    class_doc = RepresentationExplanation.__doc__ or ""
    fn_doc = explain_representation.__doc__ or ""
    forbidden = (
        "causal discovery",
        "guarantees causality",
        "interventional ground truth",
        "proves causality",
        "causal topology recovery",
    )
    for doc in (module_doc, class_doc, fn_doc):
        lowered = doc.lower()
        assert "interpretive" in lowered
        assert "non-causal" in lowered or "not causal" in lowered
        assert "not" in lowered
        assert "ResDMD" in doc or "resdmd" in lowered
        assert "ModeEnergyAttribution" in doc
        for phrase in forbidden:
            assert phrase not in lowered


def test_explain_representation_gnn_explainer_shape_contract_all_targets() -> None:
    """GNNExplainer object masks match documented ``(N, 1)`` / ``(E,)`` shapes."""
    torch.manual_seed(0)
    model = _tiny_model()
    data = _path_data()
    num_nodes = int(data.x.shape[0])
    num_edges = int(data.edge_index.shape[1])

    for target in ("latent", "one_step_forecast", "reconstruction"):
        result = explain_representation(
            model,
            data,
            target=target,  # type: ignore[arg-type]
            algorithm="gnn_explainer",
            epochs=5,
            lr=0.05,
        )
        assert result.node_mask is not None
        assert result.edge_mask is not None
        assert result.node_mask.shape[0] == num_nodes
        assert result.node_mask.ndim == 2
        assert result.node_mask.shape[-1] == 1
        assert result.edge_mask.shape == (num_edges,)
        assert torch.all(torch.isfinite(result.node_mask))
        assert torch.all(torch.isfinite(result.edge_mask))


def test_explain_representation_rejects_unknown_algorithm_and_target() -> None:
    """Unknown algorithm / target names raise actionable ValueError."""
    model = _tiny_model()
    data = _path_data()
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        explain_representation(
            model,
            data,
            algorithm="not_a_real_algorithm",  # type: ignore[arg-type]
            epochs=1,
        )
    with pytest.raises(ValueError, match="Unsupported target"):
        explain_representation(
            model,
            data,
            target="not_a_real_target",  # type: ignore[arg-type]
            epochs=1,
        )


def test_explain_representation_accepts_delay_models() -> None:
    """Delay embeddings are in the homogeneous explain surface (0.14)."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(6, 8, 4, num_layers=2),
        decoder=GNNDecoder(4, 8, 3, num_layers=2),
        latent_dim=4,
        time_step=0.1,
        n_delays=2,
    )
    data = _path_data(in_channels=6)
    result = explain_representation(model, data, epochs=1)
    assert result.node_mask is not None


def test_explain_representation_ig_attribute_shape_matches_features() -> None:
    """IG attribute masks match ``Data.x`` shape when Captum is installed."""
    pytest.importorskip("captum")
    torch.manual_seed(3)
    model = _tiny_model(in_channels=3)
    data = _path_data(in_channels=3)
    result = explain_representation(
        model,
        data,
        target="one_step_forecast",
        algorithm="integrated_gradients",
        index=0,
    )
    assert result.node_mask is not None
    assert result.node_mask.shape == tuple(data.x.shape)
    assert result.feature_mask is not None
    assert result.feature_mask.shape == tuple(data.x.shape)


def test_explain_representation_gnn_explainer_seed_stability() -> None:
    """Same seed + model/data yields close GNNExplainer masks across runs."""

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        model = _tiny_model()
        data = _path_data()
        # Freeze data randomness by rebuilding under the same seed above.
        result = explain_representation(
            model,
            data,
            target="latent",
            algorithm="gnn_explainer",
            epochs=8,
            lr=0.05,
            index=0,
        )
        assert result.node_mask is not None
        assert result.edge_mask is not None
        return result.node_mask.detach().clone(), result.edge_mask.detach().clone()

    node_a, edge_a = _run()
    node_b, edge_b = _run()
    assert torch.allclose(node_a, node_b, atol=1e-5, rtol=1e-4)
    assert torch.allclose(edge_a, edge_b, atol=1e-5, rtol=1e-4)


def test_explain_representation_gnn_explainer_latent_and_forecast() -> None:
    """GNNExplainer returns finite masks with documented shapes."""
    torch.manual_seed(0)
    model = _tiny_model()
    data = _path_data()
    num_nodes = int(data.x.shape[0])
    num_edges = int(data.edge_index.shape[1])

    latent = explain_representation(
        model,
        data,
        target="latent",
        algorithm="gnn_explainer",
        epochs=5,
        lr=0.05,
    )
    assert latent.algorithm == "gnn_explainer"
    assert latent.target == "latent"
    assert latent.node_mask is not None
    assert latent.edge_mask is not None
    assert latent.node_mask.shape[0] == num_nodes
    assert latent.edge_mask.shape == (num_edges,)
    assert torch.all(torch.isfinite(latent.node_mask))
    assert torch.all(torch.isfinite(latent.edge_mask))

    forecast = explain_representation(
        model,
        data,
        target="one_step_forecast",
        algorithm="gnn_explainer",
        epochs=5,
        lr=0.05,
        index=0,
    )
    assert forecast.target == "one_step_forecast"
    assert forecast.index == 0
    assert forecast.node_mask is not None
    assert forecast.edge_mask is not None
    assert forecast.node_mask.shape[0] == num_nodes
    assert forecast.edge_mask.shape == (num_edges,)


def test_explain_representation_reconstruction_target() -> None:
    """Reconstruction target uses encode → decode via PyG Explainer."""
    torch.manual_seed(1)
    model = _tiny_model()
    data = _path_data()
    result = explain_representation(
        model,
        data,
        target="reconstruction",
        algorithm="gnn_explainer",
        epochs=5,
        lr=0.05,
    )
    assert result.target == "reconstruction"
    assert result.node_mask is not None
    assert result.edge_mask is not None


def test_explain_representation_rejects_hetero_data() -> None:
    """HeteroData is rejected with an actionable message."""
    model = _tiny_model()
    hetero = HeteroData()
    hetero["n"].x = torch.randn(3, 3)
    hetero["n", "to", "n"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(TypeError, match="homogeneous"):
        explain_representation(model, hetero, epochs=2)


def test_explain_representation_rejects_integrated_gradients_without_captum() -> None:
    """Missing Captum raises ImportError with the ``[explain]`` install hint."""
    from koopman_graph.analysis import explain as explain_mod

    model = _tiny_model()
    data = _path_data()
    with (
        patch.object(
            explain_mod,
            "_import_captum_stack",
            side_effect=ImportError(
                "algorithm='integrated_gradients' requires Captum; "
                'install with: pip install "koopman-graph[explain]"'
            ),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[explain\]"),
    ):
        explain_representation(
            model,
            data,
            algorithm="integrated_gradients",
        )


def test_explain_representation_integrated_gradients_when_captum_installed() -> None:
    """IG path returns attribute masks when Captum is installed."""
    pytest.importorskip("captum")
    torch.manual_seed(2)
    model = _tiny_model()
    data = _path_data()
    result = explain_representation(
        model,
        data,
        target="latent",
        algorithm="integrated_gradients",
        index=1,
    )
    assert result.algorithm == "integrated_gradients"
    assert result.target == "latent"
    assert result.index == 1
    assert result.node_mask is not None
    assert result.node_mask.ndim == 2
    assert result.node_mask.shape[0] == data.num_nodes
    assert result.feature_mask is not None
    assert torch.equal(result.feature_mask, result.node_mask)


def test_explain_extra_declares_captum() -> None:
    """``[explain]`` optional dependency pins Captum."""
    root = REPO_ROOT
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "\nexplain = [\n" in text
    assert '"captum>=0.7"' in text


def test_explain_representation_uses_torch_geometric_explainer() -> None:
    """Implementation routes through PyG Explainer rather than a custom loop."""
    model = _tiny_model()
    data = _path_data()
    fake_explanation = MagicMock()
    fake_explanation.node_mask = torch.ones(data.num_nodes, 1)
    fake_explanation.edge_mask = torch.ones(data.num_edges)
    fake_explanation.feature_mask = None

    with patch("torch_geometric.explain.Explainer") as explainer_cls:
        instance = MagicMock(return_value=fake_explanation)
        explainer_cls.return_value = instance
        with patch("torch_geometric.explain.GNNExplainer") as gnn_cls:
            gnn_cls.return_value = MagicMock()
            result = explain_representation(
                model,
                data,
                target="latent",
                algorithm="gnn_explainer",
                epochs=3,
            )
    explainer_cls.assert_called_once()
    gnn_cls.assert_called_once()
    instance.assert_called_once()
    assert result.node_mask is not None
    assert torch.equal(result.node_mask, fake_explanation.node_mask)
