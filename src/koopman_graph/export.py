"""Restricted portable inference export (fixed-topology homogeneous MVP)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.model import GraphKoopmanModel


class _FixedTopologyModule(nn.Module):
    """Traceable encode → advance → decode with frozen edge_index.

    Parameters
    ----------
    model : GraphKoopmanModel
        Homogeneous discrete model.
    edge_index : Tensor
        Static COO topology.
    """

    def __init__(self, model: GraphKoopmanModel, edge_index: Tensor) -> None:
        """Store the model and frozen topology.

        Parameters
        ----------
        model : GraphKoopmanModel
            Model to wrap.
        edge_index : Tensor
            Static edges.
        """
        super().__init__()
        self.model = model
        self.register_buffer("edge_index", edge_index.clone())

    def forward(self, x: Tensor) -> Tensor:
        """Encode, advance one step, and decode.

        Parameters
        ----------
        x : Tensor
            Node features ``(N, F)``.

        Returns
        -------
        Tensor
            Next-step reconstructed features.
        """
        encoded = self.model.encoder(x, self.edge_index, None)
        advanced = self.model.koopman.advance(
            encoded,
            edge_index=self.edge_index,
        )
        return self.model.decoder(advanced, self.edge_index, None)


def _reject_unsupported_export(model: GraphKoopmanModel) -> None:
    """Refuse hetero/hypergraph/control/delay/adaptive/switched exports.

    Parameters
    ----------
    model : GraphKoopmanModel
        Candidate model.

    Raises
    ------
    ValueError
        If the model is outside the fixed-topology homogeneous MVP.
    """
    if int(getattr(model, "control_dim", 0) or 0) > 0:
        raise ValueError("export_inference_module refuses control_dim > 0")
    if getattr(model, "n_delays", 1) != 1:
        raise ValueError("export_inference_module requires n_delays=1")
    if getattr(model, "uses_hetero_koopman", False) or getattr(
        model, "uses_hypergraph_koopman", False
    ):
        raise ValueError("export_inference_module refuses hetero/hypergraph models")
    if getattr(model, "learns_pairwise_topology", False):
        raise ValueError("export_inference_module refuses adaptive topology")
    kind = type(model.koopman).__name__
    if kind in {"SwitchedKoopmanOperator", "MixtureKoopmanOperator"}:
        raise ValueError("export_inference_module refuses switched/mixture operators")
    if kind == "ParametricKoopmanOperator":
        raise ValueError(
            "export_inference_module refuses parametric interpolants K(mu)"
        )


def export_inference_module(
    model: GraphKoopmanModel,
    example_graph: Data,
) -> Callable[[Tensor], Tensor]:
    """Export a fixed-topology encode-advance-decode module.

    Tries ``torch.export`` then TorchScript. Returns a callable on node
    features.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted homogeneous discrete model.
    example_graph : Data
        Snapshot providing ``x`` and static ``edge_index``.

    Returns
    -------
    callable
        Maps node features to one-step reconstructions.

    Raises
    ------
    ValueError
        If the model or example graph is outside the MVP.
    """
    _reject_unsupported_export(model)
    if example_graph.x is None or example_graph.edge_index is None:
        raise ValueError("example_graph requires x and edge_index")
    module = _FixedTopologyModule(model.eval(), example_graph.edge_index)
    example = example_graph.x.detach()
    try:
        exported = torch.export.export(module, (example,))
        return exported.module()
    except Exception:
        scripted = torch.jit.trace(module, example)
        return scripted  # type: ignore[return-value]


def compress_operator_svd(matrix: Tensor, rank: int) -> Tensor:
    """Low-rank SVD truncation of an assembled operator.

    Parameters
    ----------
    matrix : Tensor
        Square ``K``.
    rank : int
        Retained singular values.

    Returns
    -------
    Tensor
        Rank-``k`` reconstruction.

    Raises
    ------
    ValueError
        If ``rank < 1``.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    k = min(int(rank), int(s.shape[0]))
    return (u[:, :k] * s[:k]) @ vh[:k]


def export_onnx(
    module: nn.Module,
    example: Tensor,
    path: str,
) -> str:
    """Export a traced module to ONNX (``[export]`` extra).

    Parameters
    ----------
    module : nn.Module
        Traceable module (typically from :func:`export_inference_module`).
    example : Tensor
        Example node-feature input.
    path : str
        Destination ``.onnx`` path.

    Returns
    -------
    str
        The written path.
    """
    try:
        import torch.onnx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ONNX export requires torch.onnx; install with: "
            "pip install 'koopman-graph[export]'"
        ) from exc
    torch.onnx.export(module, example, path, dynamo=False)
    return path
