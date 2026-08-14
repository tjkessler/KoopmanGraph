"""Geometry-aware encoders from ``Data.pos`` (Tier A / optional Tier B).

Tier A (:class:`InvariantGeometryEncoder`) builds rotation-/translation-invariant
distance and angle features, concatenates them with ``Data.x``, and lifts with a
standard GCN. Latent evolution still uses an ordinary linear Koopman operator —
**invariant features ≠ equivariant** :math:`K`.

Tier B (:class:`E3EquivariantEncoder`) is an optional steerable encode path via
``e3nn`` (``[equivariance]`` extra). Intermediate layers use E(3)-equivariant
tensor products; the encoder **projects to invariant scalar latents** of width
``latent_dim`` for the usual linear :math:`K`. This does **not** make
:math:`K` itself E(n)/SE(3) equivariant.

References
----------
Thomas, N. et al. Tensor field networks: Rotation- and translation-equivariant
neural networks for 3D point clouds. arXiv:1802.08219 (2018)
(``Thomas2018TFN``).
Geiger, M. & Smidt, T. e3nn: Euclidean Neural Networks. arXiv:2207.09453 (2022)
(``Geiger2022e3nn``).
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_gcn_convs,
    validate_positive_dims,
)

GEOM_CHANNELS = 4
_ANGLE_EPS = 1e-8
_EQUIVARIANCE_INSTALL_HINT = 'pip install "koopman-graph[equivariance]"'
_EDGE_LENGTH_EPS = 1e-6


def _import_e3nn_modules() -> tuple[Any, Any]:
    """Import ``e3nn.o3`` and ``e3nn.nn`` with an actionable install hint.

    Returns
    -------
    tuple
        ``(e3nn.o3, e3nn.nn)`` modules.

    Raises
    ------
    ImportError
        If ``e3nn`` is not installed.
    """
    try:
        o3 = importlib.import_module("e3nn.o3")
        enn = importlib.import_module("e3nn.nn")
    except ImportError as exc:
        msg = (
            "e3nn is required for E3EquivariantEncoder; "
            f"install with: {_EQUIVARIANCE_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc
    return o3, enn


def invariant_geometry_features(
    pos: Tensor,
    edge_index: Tensor,
    *,
    num_nodes: int | None = None,
) -> Tensor:
    """Build per-node invariant geometry features from coordinates.

    Channels (``GEOM_CHANNELS == 4``):

    0. mean Euclidean distance to undirected neighbors
    1. max Euclidean distance to undirected neighbors
    2. mean cosine of angles at the node over unordered neighbor pairs
       (``0`` when degree ``< 2``)
    3. ``log1p(degree)``

    Isolated nodes receive zeros in channels 0–2 and ``log1p(0) = 0``.

    Parameters
    ----------
    pos : Tensor
        Node coordinates with shape ``(num_nodes, D)`` where ``D`` is 2 or 3.
    edge_index : Tensor
        Edge index ``(2, num_edges)`` (undirected graphs may list both
        directions; neighbors are de-duplicated).
    num_nodes : int or None, optional
        Node count. Defaults to ``pos.size(0)``.

    Returns
    -------
    Tensor
        Features with shape ``(num_nodes, GEOM_CHANNELS)``.

    Raises
    ------
    ValueError
        If ``pos`` / ``edge_index`` shapes are invalid or values are non-finite.
    """
    if pos.ndim != 2 or pos.size(1) not in (2, 3):
        msg = (
            f"pos must have shape (num_nodes, 2) or (num_nodes, 3), "
            f"got {tuple(pos.shape)}"
        )
        raise ValueError(msg)
    if not torch.isfinite(pos).all():
        msg = "pos must contain only finite values"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        msg = (
            f"edge_index must have shape (2, num_edges), got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)

    resolved_nodes = int(pos.size(0) if num_nodes is None else num_nodes)
    if resolved_nodes != pos.size(0):
        msg = f"num_nodes={resolved_nodes} does not match pos.size(0)={pos.size(0)}"
        raise ValueError(msg)

    features = pos.new_zeros((resolved_nodes, GEOM_CHANNELS))
    if edge_index.numel() == 0:
        return features

    # Undirected neighbor sets (Python lists; graphs in smokes are small).
    neighbors: list[set[int]] = [set() for _ in range(resolved_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for a, b in zip(src, dst, strict=True):
        a_i, b_i = int(a), int(b)
        if a_i == b_i:
            continue
        if not (0 <= a_i < resolved_nodes and 0 <= b_i < resolved_nodes):
            msg = (
                f"edge_index node ids must lie in [0, {resolved_nodes}), "
                f"got edge ({a_i}, {b_i})"
            )
            raise ValueError(msg)
        neighbors[a_i].add(b_i)
        neighbors[b_i].add(a_i)

    for node, neigh in enumerate(neighbors):
        degree = len(neigh)
        features[node, 3] = math.log1p(float(degree))
        if degree == 0:
            continue
        neigh_list = sorted(neigh)
        dists = torch.stack([(pos[node] - pos[j]).norm(p=2) for j in neigh_list])
        features[node, 0] = dists.mean()
        features[node, 1] = dists.max()
        if degree < 2:
            continue
        cosines: list[Tensor] = []
        for i_idx in range(degree):
            for j_idx in range(i_idx + 1, degree):
                v1 = pos[neigh_list[i_idx]] - pos[node]
                v2 = pos[neigh_list[j_idx]] - pos[node]
                denom = v1.norm(p=2) * v2.norm(p=2)
                cos = (v1 * v2).sum() / denom.clamp_min(_ANGLE_EPS)
                cosines.append(cos.clamp(-1.0, 1.0))
        features[node, 2] = torch.stack(cosines).mean()
    return features


class InvariantGeometryEncoder(BaseGNNModule):
    """GCN encoder on ``[x | invariant_geometry_features(pos)]``.

    Requires PyG :class:`~torch_geometric.data.Data` with ``x``, ``pos``, and
    ``edge_index``. Geometry features are invariant to rigid motions of
    ``pos``; the Koopman operator applied downstream is **not** claimed to be
    E(n)/SE(3) equivariant.

    Attributes
    ----------
    in_channels : int
        Width of ``data.x`` (geometry channels are added internally).
    hidden_channels : int
        Hidden GCN width.
    latent_dim : int
        Output latent dimension per node.
    geom_channels : int
        Number of appended invariant geometry channels (``GEOM_CHANNELS``).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
    ) -> None:
        """Initialize the invariant-geometry GCN encoder.

        Parameters
        ----------
        in_channels : int
            ``data.x`` feature width (not including geometry channels).
        hidden_channels : int
            Hidden GCN channel width.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Number of GCN layers. Default ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default ``"relu"``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.geom_channels = GEOM_CHANNELS
        total_in = in_channels + GEOM_CHANNELS
        super().__init__(
            input_channels=total_in,
            input_dim_name="in_channels+geom",
            num_layers=num_layers,
            activation=activation,
            convs=build_gcn_convs(
                total_in,
                hidden_channels,
                latent_dim,
                num_layers,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode ``Data`` with invariant geometry features + GCN.

        Parameters
        ----------
        x_or_data : Data
            Snapshot carrying ``x``, ``pos``, and ``edge_index``.
        edge_index : Tensor or None, optional
            Ignored (topology is read from ``Data``).
        edge_weight : Tensor or None, optional
            Optional edge weights for ``GCNConv``.

        Returns
        -------
        Tensor
            Latent node features ``(num_nodes, latent_dim)``.

        Raises
        ------
        ValueError
            If ``x_or_data`` is not ``Data`` or required fields are missing.
        TypeError
            If ``convs`` contains a non-``GCNConv`` layer.
        """
        del edge_index  # Data-only API
        if not isinstance(x_or_data, Data):
            msg = (
                "InvariantGeometryEncoder requires a torch_geometric.data.Data "
                "input with x, pos, and edge_index (tensor-only forward is "
                "unsupported)"
            )
            raise ValueError(msg)
        data = x_or_data
        if data.x is None:
            msg = "data.x is required for InvariantGeometryEncoder"
            raise ValueError(msg)
        if data.edge_index is None:
            msg = "data.edge_index is required for InvariantGeometryEncoder"
            raise ValueError(msg)
        pos = getattr(data, "pos", None)
        if pos is None:
            msg = "data.pos is required for InvariantGeometryEncoder"
            raise ValueError(msg)
        if data.x.ndim != 2 or data.x.size(1) != self.in_channels:
            msg = (
                f"Expected data.x with shape (num_nodes, {self.in_channels}), "
                f"got {tuple(data.x.shape)}"
            )
            raise ValueError(msg)

        geom = invariant_geometry_features(
            pos,
            data.edge_index,
            num_nodes=int(data.x.size(0)),
        )
        x = torch.cat([data.x, geom], dim=-1)
        resolved_weight = edge_weight
        if resolved_weight is None:
            resolved_weight = getattr(data, "edge_weight", None)

        for layer_idx, conv in enumerate(self.convs):
            if not isinstance(conv, GCNConv):
                msg = (
                    f"expected GCNConv in InvariantGeometryEncoder stack, "
                    f"got {type(conv).__name__}"
                )
                raise TypeError(msg)
            if resolved_weight is not None:
                x = conv(x, data.edge_index, edge_weight=resolved_weight)
            else:
                x = conv(x, data.edge_index)
            if layer_idx < len(self.convs) - 1:
                x = self.activation(x)
        return x


class _E3MessagePassingLayer(nn.Module):
    """One e3nn tensor-product message-passing layer with a linear skip.

    Notes
    -----
    Internal or implementation helper.
    """

    def __init__(
        self,
        *,
        o3: Any,
        fully_connected_net: Any,
        irreps_in: Any,
        irreps_out: Any,
        irreps_sh: Any,
        edge_mlp_hidden: int,
    ) -> None:
        """Internal init helper.

        Parameters
        ----------
        o3
            Value for ``o3``.
        fully_connected_net
            Value for ``fully_connected_net``.
        irreps_in
            Value for ``irreps_in``.
        irreps_out
            Value for ``irreps_out``.
        irreps_sh
            Value for ``irreps_sh``.
        edge_mlp_hidden
            Value for ``edge_mlp_hidden``.
        """
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out,
            shared_weights=False,
        )
        self.fc = fully_connected_net(
            [1, edge_mlp_hidden, self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.sc = o3.Linear(irreps_in, irreps_out)
        self.irreps_sh = irreps_sh
        self._o3 = o3

    def forward(self, features: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        """Aggregate equivariant messages and add the linear self-connection.

        Parameters
        ----------
        features
            Value for ``features``.
        pos
            Value for ``pos``.
        edge_index
            Value for ``edge_index``.

        Returns
        -------
        object
            Function result.
        """
        skip = self.sc(features)
        if edge_index.numel() == 0:
            return skip
        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[src] - pos[dst]
        edge_sh = self._o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        edge_len = edge_vec.norm(dim=1, keepdim=True).clamp_min(_EDGE_LENGTH_EPS)
        messages = self.tp(features[src], edge_sh, self.fc(edge_len))
        aggregated = scatter(
            messages,
            dst,
            dim=0,
            dim_size=features.size(0),
            reduce="mean",
        )
        return aggregated + skip


class E3EquivariantEncoder(nn.Module):
    """Optional steerable encoder via ``e3nn`` (``[equivariance]`` extra).

    Requires PyG :class:`~torch_geometric.data.Data` with ``x``, ``pos``
    (3-D), and ``edge_index``. Intermediate features use scalar and vector
    irreps (Thomas et al. TFN-style tensor products via e3nn). The final
    layer maps to invariant scalars ``(num_nodes, latent_dim)`` for the
    ordinary linear Koopman operator — **steerable encode ≠ equivariant**
    :math:`K`.

    Attributes
    ----------
    in_channels : int
        Width of ``data.x`` (treated as scalar irreps).
    hidden_channels : int
        Scalar / vector multiplicity in hidden irreps.
    latent_dim : int
        Output invariant scalar width per node.
    num_layers : int
        Number of steerable message-passing layers.
    lmax : int
        Maximum spherical-harmonic degree on relative edge vectors.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        lmax: int = 1,
        edge_mlp_hidden: int = 16,
        project_invariants: bool = True,
        n_vectors: int = 0,
    ) -> None:
        """Initialize the optional e3nn steerable encoder.

        Parameters
        ----------
        in_channels : int
            ``data.x`` feature width (scalar channels).
        hidden_channels : int
            Multiplicity of hidden ``0e`` and ``1o`` irreps.
        latent_dim : int
            Output invariant scalar dimension per node.
        num_layers : int, optional
            Steerable message-passing depth. Default ``2``.
        lmax : int, optional
            Spherical-harmonic degree for edge attributes. Default ``1``.
        edge_mlp_hidden : int, optional
            Hidden width of the edge-length MLP that produces tensor-product
            weights. Default ``16``.
        project_invariants : bool, optional
            When ``True`` (default), map to invariant scalars of width
            ``latent_dim``. When ``False``, keep ``n_vectors`` vector irreps
            concatenated after the scalars for an equivariant ``K``.
        n_vectors : int, optional
            Number of vector channels when ``project_invariants`` is
            ``False``. Default ``0``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive or ``lmax < 0``.
        ImportError
            If the ``[equivariance]`` / ``e3nn`` extra is not installed.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
            edge_mlp_hidden=edge_mlp_hidden,
        )
        if lmax < 0:
            msg = f"lmax must be >= 0, got {lmax}"
            raise ValueError(msg)

        o3, enn = _import_e3nn_modules()
        fully_connected_net = enn.FullyConnectedNet

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lmax = lmax
        self.edge_mlp_hidden = edge_mlp_hidden

        irreps_in = o3.Irreps(f"{in_channels}x0e")
        irreps_hidden = o3.Irreps(f"{hidden_channels}x0e+{hidden_channels}x1o")
        self.project_invariants = bool(project_invariants)
        self.n_vectors = int(n_vectors)
        if self.n_vectors < 0:
            raise ValueError(f"n_vectors must be >= 0, got {n_vectors}")
        if self.project_invariants:
            irreps_out = o3.Irreps(f"{latent_dim}x0e")
            self.output_dim = latent_dim
        else:
            irreps_out = o3.Irreps(f"{latent_dim}x0e+{self.n_vectors}x1o")
            self.output_dim = latent_dim + 3 * self.n_vectors
        irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            layer_in = irreps_in if layer_idx == 0 else irreps_hidden
            layers.append(
                _E3MessagePassingLayer(
                    o3=o3,
                    fully_connected_net=fully_connected_net,
                    irreps_in=layer_in,
                    irreps_out=irreps_hidden,
                    irreps_sh=irreps_sh,
                    edge_mlp_hidden=edge_mlp_hidden,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.project = o3.Linear(irreps_hidden, irreps_out)

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode ``Data`` with steerable e3nn layers to invariant latents.

        Parameters
        ----------
        x_or_data : Data
            Snapshot carrying ``x``, ``pos`` (shape ``(N, 3)``), and
            ``edge_index``.
        edge_index : Tensor or None, optional
            Ignored (topology is read from ``Data``).
        edge_weight : Tensor or None, optional
            Unused (edge geometry comes from ``pos``); accepted for API parity.

        Returns
        -------
        Tensor
            Invariant latent node features ``(num_nodes, latent_dim)``.

        Raises
        ------
        ValueError
            If ``x_or_data`` is not ``Data`` or required fields are missing /
            mistyped.
        """
        del edge_index, edge_weight  # Data-only API; geometry from pos
        if not isinstance(x_or_data, Data):
            msg = (
                "E3EquivariantEncoder requires a torch_geometric.data.Data "
                "input with x, pos, and edge_index (tensor-only forward is "
                "unsupported)"
            )
            raise ValueError(msg)
        data = x_or_data
        if data.x is None:
            msg = "data.x is required for E3EquivariantEncoder"
            raise ValueError(msg)
        if data.edge_index is None:
            msg = "data.edge_index is required for E3EquivariantEncoder"
            raise ValueError(msg)
        pos = getattr(data, "pos", None)
        if pos is None:
            msg = "data.pos is required for E3EquivariantEncoder"
            raise ValueError(msg)
        if pos.ndim != 2 or pos.size(1) != 3:
            msg = (
                f"E3EquivariantEncoder requires data.pos with shape "
                f"(num_nodes, 3), got {tuple(pos.shape)}"
            )
            raise ValueError(msg)
        if not torch.isfinite(pos).all():
            msg = "data.pos must contain only finite values"
            raise ValueError(msg)
        if data.x.ndim != 2 or data.x.size(1) != self.in_channels:
            msg = (
                f"Expected data.x with shape (num_nodes, {self.in_channels}), "
                f"got {tuple(data.x.shape)}"
            )
            raise ValueError(msg)
        if data.x.size(0) != pos.size(0):
            msg = (
                f"data.x and data.pos node counts differ: "
                f"{data.x.size(0)} vs {pos.size(0)}"
            )
            raise ValueError(msg)

        features = data.x
        for layer in self.layers:
            features = layer(features, pos, data.edge_index)
        latents = self.project(features)
        if latents.size(-1) != self.output_dim:
            msg = (
                f"internal latent width {latents.size(-1)} != "
                f"output_dim={self.output_dim}"
            )
            raise RuntimeError(msg)
        return latents
