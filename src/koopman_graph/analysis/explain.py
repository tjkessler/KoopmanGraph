"""Representation-level node / edge / feature attribution masks.

Honesty contract
----------------
Masks produced by this module are **interpretive** attributions under a
chosen algorithm (for example PyG GNNExplainer or Captum integrated
gradients). They are **non-causal**: not causal or interventional topology
recovery, **not** certified ResDMD residuals, and **not**
:class:`~koopman_graph.analysis.ModeEnergyAttribution` (the complementary
operator-level energy diagnostic on assembled ``K_eff``).

Homogeneous MVP first; hetero / adaptive / delay explanation is rejected
with actionable errors. ``algorithm="integrated_gradients"`` requires the
optional ``[explain]`` extra (``pip install "koopman-graph[explain]"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData

if TYPE_CHECKING:
    from koopman_graph.model import GraphKoopmanModel

ExplanationTarget = Literal["latent", "one_step_forecast", "reconstruction"]
ExplanationAlgorithm = Literal["gnn_explainer", "integrated_gradients"]

_SUPPORTED_TARGETS: frozenset[str] = frozenset(
    {"latent", "one_step_forecast", "reconstruction"}
)
_EXPLAIN_INSTALL_HINT = 'pip install "koopman-graph[explain]"'


@dataclass(frozen=True)
class RepresentationExplanation:
    """Node / edge / feature masks for a chosen model quantity.

    Honesty contract
    ----------------
    This result is an **interpretive** attribution under the recorded
    ``algorithm``. It is **non-causal**: masks must not be read as causal
    topology discovery or interventional importance. It is **not** a
    ResDMD residual bound and is **not**
    :class:`~koopman_graph.analysis.ModeEnergyAttribution`.

    Attributes
    ----------
    target : {"latent", "one_step_forecast", "reconstruction"}
        Model quantity the masks were requested for.
    node_mask : Tensor or None
        Optional per-node attribution mask. GNNExplainer object masks are
        typically ``(num_nodes, 1)``; integrated-gradients attribute masks
        are typically ``(num_nodes, num_features)``.
    edge_mask : Tensor or None
        Optional per-edge attribution mask, typically shape ``(num_edges,)``
        (often ``None`` for the Captum IG path).
    feature_mask : Tensor or None
        Optional per-feature attribution mask. For IG attribute masks, this
        mirrors ``node_mask`` when shape is ``(num_nodes, num_features)``.
    algorithm : str
        Algorithm identifier (for example ``\"gnn_explainer\"`` or
        ``\"integrated_gradients\"``).
    index : int or None
        Optional node index passed to the explainer when applicable.
    """

    target: ExplanationTarget
    node_mask: Tensor | None
    edge_mask: Tensor | None
    feature_mask: Tensor | None
    algorithm: str
    index: int | None


class _HomogeneousExplainModule(nn.Module):
    """Thin ``(x, edge_index)`` adapter for PyG ``Explainer``.

    Notes
    -----
    Wraps :class:`~koopman_graph.model.GraphKoopmanModel` so PyG explainers
    can call ``forward(x, edge_index)`` without a full ``Data`` batch.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        target: ExplanationTarget,
        reduce_to_node_scalar: bool = False,
    ) -> None:
        """Store the model and explanation target.

        Parameters
        ----------
        model : GraphKoopmanModel
            Homogeneous model whose encode / decode paths are attributed.
        target : {"latent", "one_step_forecast", "reconstruction"}
            Quantity exposed by :meth:`forward`.
        reduce_to_node_scalar : bool, optional
            When ``True``, sum the last feature dim so Captum IG sees a
            per-node scalar (default ``False``).
        """
        super().__init__()
        self.model = model
        self.target = target
        self.reduce_to_node_scalar = reduce_to_node_scalar

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return the selected homogeneous quantity for attribution.

        Parameters
        ----------
        x : Tensor
            Node features, shape ``(num_nodes, num_features)``.
        edge_index : Tensor
            COO edge index, shape ``(2, num_edges)``.

        Returns
        -------
        Tensor
            Encoded latent, reconstruction, or one-step forecast tensor
            (optionally reduced to a per-node scalar).
        """
        data = Data(x=x, edge_index=edge_index)
        if self.target == "latent":
            out = self.model.encode(data)
        elif self.target == "reconstruction":
            latent = self.model.encode(data)
            out = self.model.decoder(latent, edge_index, None)
        else:
            # one_step_forecast: encode → advance → decode
            decoded = self.model(data)
            if not isinstance(decoded, Tensor):
                msg = (
                    "one_step_forecast explanation requires a tensor decode; "
                    f"got {type(decoded).__name__}"
                )
                raise TypeError(msg)
            out = decoded
        if self.reduce_to_node_scalar:
            # Captum IG needs a scalar output per selected node.
            return out.sum(dim=-1, keepdim=True)
        return out


def _reject_unsupported_model(model: GraphKoopmanModel) -> None:
    """Raise if the model is outside the homogeneous explain MVP.

    Parameters
    ----------
    model : GraphKoopmanModel
        Candidate model to validate.

    Raises
    ------
    ValueError
        If delays, controls, hetero / hypergraph, or adaptive topology
        modes are outside the MVP.
    """
    if getattr(model, "n_delays", 1) != 1:
        msg = (
            "explain_representation supports n_delays=1 only "
            f"(got n_delays={model.n_delays})"
        )
        raise ValueError(msg)
    if int(getattr(model, "control_dim", 0) or 0) > 0:
        raise ValueError(
            "explain_representation does not support control_dim > 0 in the MVP"
        )
    if model.uses_hetero_koopman or model._uses_relgraph_encode():
        raise ValueError(
            "explain_representation homogeneous MVP does not support "
            "RelGraph / hetero_graph models"
        )
    if model.uses_hypergraph_koopman or model._uses_hypergraph_encode():
        raise ValueError(
            "explain_representation homogeneous MVP does not support "
            "hypergraph encoders / operators"
        )
    if model.learns_pairwise_topology or model.adaptive_topology is not None:
        raise ValueError(
            "explain_representation homogeneous MVP does not support "
            "adaptive / learned pairwise topology"
        )


def _reject_unsupported_data(data: Data | HeteroData) -> Data:
    """Validate and return a homogeneous :class:`~torch_geometric.data.Data`.

    Parameters
    ----------
    data : Data or HeteroData
        Snapshot supplied by the caller.

    Returns
    -------
    Data
        The same homogeneous ``Data`` instance after validation.

    Raises
    ------
    TypeError, ValueError
        If ``data`` is hetero, not a ``Data``, or missing ``x`` /
        ``edge_index``.
    """
    if isinstance(data, HeteroData):
        raise TypeError(
            "explain_representation requires homogeneous torch_geometric.data.Data "
            "(HeteroData is not supported in the MVP)"
        )
    if not isinstance(data, Data):
        raise TypeError(
            "explain_representation requires torch_geometric.data.Data, "
            f"got {type(data).__name__}"
        )
    if data.x is None or data.edge_index is None:
        raise ValueError("explain_representation requires Data.x and Data.edge_index")
    return data


def _run_gnn_explainer(
    model: GraphKoopmanModel,
    data: Data,
    *,
    target: ExplanationTarget,
    index: int | None,
    epochs: int,
    lr: float,
) -> RepresentationExplanation:
    """Attribute ``target`` with PyG :class:`~torch_geometric.explain.GNNExplainer`.

    Parameters
    ----------
    model : GraphKoopmanModel
        Homogeneous model to explain.
    data : Data
        Snapshot with ``x`` and ``edge_index``.
    target : {"latent", "one_step_forecast", "reconstruction"}
        Quantity to attribute.
    index : int or None
        Optional node index forwarded to PyG ``Explainer``.
    epochs : int
        GNNExplainer mask-optimization epochs (must be >= 1).
    lr : float
        GNNExplainer learning rate (must be > 0).

    Returns
    -------
    RepresentationExplanation
        Node / edge masks and metadata for ``algorithm='gnn_explainer'``.
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    wrapper = _HomogeneousExplainModule(model, target=target)
    wrapper.train()  # GNNExplainer optimizes masks against a trainable model
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        model_config=ModelConfig(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
        node_mask_type="object",
        edge_mask_type="object",
    )
    explanation = explainer(
        data.x,
        data.edge_index,
        index=index,
    )
    feature_mask = getattr(explanation, "feature_mask", None)
    return RepresentationExplanation(
        target=target,
        node_mask=explanation.node_mask,
        edge_mask=explanation.edge_mask,
        feature_mask=feature_mask if isinstance(feature_mask, Tensor) else None,
        algorithm="gnn_explainer",
        index=index,
    )


def _import_captum_stack() -> tuple[type, type, type]:
    """Import Captum + PyG CaptumExplainer with an actionable install hint.

    Returns
    -------
    CaptumExplainer, Explainer, ModelConfig
        PyG / Captum classes needed for integrated gradients.

    Raises
    ------
    ImportError
        If Captum or PyG Captum helpers are not installed.
    """
    try:
        import captum  # noqa: F401
        from torch_geometric.explain import CaptumExplainer, Explainer
        from torch_geometric.explain.config import ModelConfig
    except ImportError as exc:
        msg = (
            "algorithm='integrated_gradients' requires Captum; "
            f"install with: {_EXPLAIN_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc
    return CaptumExplainer, Explainer, ModelConfig


def _run_integrated_gradients(
    model: GraphKoopmanModel,
    data: Data,
    *,
    target: ExplanationTarget,
    index: int | None,
) -> RepresentationExplanation:
    """Attribute ``target`` with PyG Captum ``IntegratedGradients``.

    Requires the optional ``[explain]`` extra (Captum).

    Parameters
    ----------
    model : GraphKoopmanModel
        Homogeneous model to explain.
    data : Data
        Snapshot with ``x`` and ``edge_index``.
    target : {"latent", "one_step_forecast", "reconstruction"}
        Quantity to attribute.
    index : int or None
        Node index for IG; defaults to ``0`` when ``None``.

    Returns
    -------
    RepresentationExplanation
        Attribute masks and metadata for ``algorithm='integrated_gradients'``.
    """
    captum_explainer_cls, explainer_cls, model_config_cls = _import_captum_stack()

    node_index = 0 if index is None else int(index)
    if node_index < 0:
        raise ValueError(f"index must be >= 0, got {node_index}")

    wrapper = _HomogeneousExplainModule(
        model,
        target=target,
        reduce_to_node_scalar=True,
    )
    wrapper.eval()
    explainer = explainer_cls(
        model=wrapper,
        algorithm=captum_explainer_cls("IntegratedGradients"),
        explanation_type="model",
        model_config=model_config_cls(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type=None,
    )
    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_index,
    )
    node_mask = explanation.node_mask
    feature_mask = node_mask if isinstance(node_mask, Tensor) else None
    edge_mask = getattr(explanation, "edge_mask", None)
    if edge_mask is not None and not isinstance(edge_mask, Tensor):
        edge_mask = None
    return RepresentationExplanation(
        target=target,
        node_mask=node_mask if isinstance(node_mask, Tensor) else None,
        edge_mask=edge_mask,
        feature_mask=feature_mask,
        algorithm="integrated_gradients",
        index=node_index,
    )


def explain_representation(
    model: GraphKoopmanModel,
    data: Data | HeteroData,
    *,
    target: ExplanationTarget = "one_step_forecast",
    algorithm: ExplanationAlgorithm = "gnn_explainer",
    index: int | None = None,
    epochs: int = 50,
    lr: float = 0.01,
) -> RepresentationExplanation:
    """Attribute a homogeneous model quantity with node / edge masks.

    Honesty contract
    ----------------
    Returned masks are **interpretive** under ``algorithm``. They are
    **non-causal**, **not** ResDMD residual bounds, and **not**
    :class:`~koopman_graph.analysis.ModeEnergyAttribution`.

    Parameters
    ----------
    model : GraphKoopmanModel
        Trained (or randomly initialized) homogeneous model.
    data : Data
        Homogeneous snapshot with ``x`` and ``edge_index``.
    target : {"latent", "one_step_forecast", "reconstruction"}, optional
        Quantity to attribute. ``latent`` uses the encoder output;
        ``one_step_forecast`` uses encode → Koopman advance → decode;
        ``reconstruction`` uses encode → decode.
    algorithm : {"gnn_explainer", "integrated_gradients"}, optional
        ``gnn_explainer`` wraps PyG :class:`~torch_geometric.explain.GNNExplainer`
        (core). ``integrated_gradients`` wraps PyG ``CaptumExplainer`` and
        requires ``pip install "koopman-graph[explain]"``.
    index : int or None, optional
        Optional node index forwarded to PyG ``Explainer``. For
        ``integrated_gradients``, defaults to ``0`` when omitted.
    epochs, lr : int / float, optional
        GNNExplainer mask-optimization hyperparameters (ignored for IG).

    Returns
    -------
    RepresentationExplanation
        Masks and metadata. GNNExplainer object masks are typically
        ``node_mask`` ``(num_nodes, 1)`` and ``edge_mask`` ``(num_edges,)``.
        IG attribute masks are typically ``node_mask`` /
        ``feature_mask`` ``(num_nodes, num_features)`` with ``edge_mask``
        often ``None``.

    Raises
    ------
    TypeError, ValueError
        Unsupported data / model modes, or unknown target / algorithm.
    ImportError
        If ``integrated_gradients`` is requested without Captum.
    """
    if target not in _SUPPORTED_TARGETS:
        allowed = ", ".join(sorted(_SUPPORTED_TARGETS))
        raise ValueError(f"Unsupported target {target!r}; allowed: {allowed}")
    if algorithm not in {"gnn_explainer", "integrated_gradients"}:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}; "
            "allowed: 'gnn_explainer', 'integrated_gradients'"
        )

    _reject_unsupported_model(model)
    snapshot = _reject_unsupported_data(data)
    was_training = model.training
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if algorithm == "integrated_gradients":
            return _run_integrated_gradients(
                model,
                snapshot,
                target=target,  # type: ignore[arg-type]
                index=index,
            )
        return _run_gnn_explainer(
            model,
            snapshot,
            target=target,  # type: ignore[arg-type]
            index=index,
            epochs=epochs,
            lr=lr,
        )
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.train(was_training)
