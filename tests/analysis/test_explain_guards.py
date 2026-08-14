"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import koopman_graph.analysis.explain as explain_mod


def _unsupported_explain_model(**overrides: object) -> SimpleNamespace:
    """Return a model-shaped object for explain-MVP validation."""
    attributes: dict[str, object] = {
        "n_delays": 1,
        "control_dim": 0,
        "uses_hetero_koopman": False,
        "_uses_relgraph_encode": lambda: False,
        "uses_hypergraph_koopman": False,
        "_uses_hypergraph_encode": lambda: False,
        "learns_pairwise_topology": False,
        "adaptive_topology": None,
    }
    attributes.update(overrides)
    return SimpleNamespace(**attributes)


class _NonTensorForecast:
    """Model-shaped callable that returns a non-tensor forecast."""

    def __call__(self, data: Data) -> dict[str, Data]:
        return {"decoded": data}


def test_explain_wrapper_rejects_non_tensor_forecast() -> None:
    """One-step explanation requires a tensor model decode."""
    wrapper = explain_mod._HomogeneousExplainModule(
        _NonTensorForecast(),  # type: ignore[arg-type]
        target="one_step_forecast",
    )
    with pytest.raises(TypeError, match="requires a tensor decode"):
        wrapper(
            torch.ones(2, 2),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            _unsupported_explain_model(uses_hetero_koopman=True),
            "RelGraph / hetero_graph",
        ),
        (
            _unsupported_explain_model(uses_hypergraph_koopman=True),
            "hypergraph",
        ),
        (
            _unsupported_explain_model(adaptive_topology=object()),
            "adaptive / learned",
        ),
    ],
)
def test_explain_rejects_unsupported_model_modes(
    model: SimpleNamespace,
    message: str,
) -> None:
    """Explain-MVP validation covers hetero, hypergraph, and adaptive."""
    with pytest.raises(ValueError, match=message):
        explain_mod._reject_unsupported_model(model)  # type: ignore[arg-type]


def test_explain_rejects_invalid_data_inputs() -> None:
    """Explain data validation rejects non-Data and missing graph fields."""
    with pytest.raises(TypeError, match="torch_geometric.data.Data"):
        explain_mod._reject_unsupported_data(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Data.x and Data.edge_index"):
        explain_mod._reject_unsupported_data(Data())


@pytest.mark.parametrize(
    ("epochs", "lr", "message"),
    [(0, 0.01, "epochs"), (1, 0.0, "lr")],
)
def test_explain_gnn_explainer_validates_hyperparameters(
    epochs: int,
    lr: float,
    message: str,
) -> None:
    """GNNExplainer rejects invalid epochs and learning rates."""
    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match=message):
        explain_mod._run_gnn_explainer(
            MagicMock(),
            data,
            target="latent",
            index=None,
            epochs=epochs,
            lr=lr,
        )


def test_explain_captum_import_error_hint() -> None:
    """The direct Captum importer adds the optional-extra install hint."""
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "captum":
            raise ImportError("simulated missing captum")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=blocked_import),
        pytest.raises(ImportError, match=r"koopman-graph\[explain\]"),
    ):
        explain_mod._import_captum_stack()


def test_explain_integrated_gradients_rejects_negative_index() -> None:
    """Integrated gradients validates the selected node index."""
    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with (
        patch.object(
            explain_mod,
            "_import_captum_stack",
            return_value=(MagicMock(), MagicMock(), MagicMock()),
        ),
        pytest.raises(ValueError, match="index must be >= 0"),
    ):
        explain_mod._run_integrated_gradients(
            nn.Identity(),  # type: ignore[arg-type]
            data,
            target="latent",
            index=-1,
        )


def test_explain_integrated_gradients_discards_non_tensor_edge_mask() -> None:
    """Integrated gradients normalizes a non-tensor edge mask to None."""

    class FakeExplainer:
        """Return a deterministic explanation without invoking the wrapper."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            *,
            index: int,
        ) -> SimpleNamespace:
            del edge_index, index
            return SimpleNamespace(node_mask=torch.ones_like(x), edge_mask=object())

    data = Data(
        x=torch.ones(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with patch.object(
        explain_mod,
        "_import_captum_stack",
        return_value=(MagicMock(), FakeExplainer, MagicMock()),
    ):
        result = explain_mod._run_integrated_gradients(
            nn.Identity(),  # type: ignore[arg-type]
            data,
            target="latent",
            index=1,
        )
    assert result.edge_mask is None
    assert result.node_mask is not None
