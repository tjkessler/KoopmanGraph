"""GraphCast-style mesh teaching forecaster baseline.

Teaching-scale pure-PyTorch adapter inspired by Lam et al., Science 2023
("Learning skillful medium-range global weather forecasting" / GraphCast): a
tiny encoder–processor–decoder on a documented latitude–longitude mesh slice.

This is a **mesh-weather teaching slice**, not a sensor-traffic (PEMS/METR)
forecaster and not DeepMind's JAX GraphCast, ERA5 training, or multi-mesh
hierarchy. No ``eccodes``, no production weather stack. The optional
``[baselines-graphcast]`` extra is reserved for future mesh/weather helper
deps; the teaching path needs only the core install.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.base import (
    GNNForecasterBaseline,
    dense_adjacency,
    random_walk_normalize,
)
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils import snapshot_edge_weight


def build_teaching_mesh_edge_index(
    num_lat: int = 3,
    num_lon: int = 4,
) -> Tensor:
    """Build an undirected lat–lon mesh edge index (longitude wraps).

    Parameters
    ----------
    num_lat : int, optional
        Number of latitude bands. Default is ``3``.
    num_lon : int, optional
        Number of longitude samples per band. Default is ``4``.

    Returns
    -------
    Tensor
        COO edge index with shape ``(2, E)`` for ``num_lat * num_lon`` nodes
        indexed row-major as ``lat * num_lon + lon``.

    Raises
    ------
    ValueError
        If ``num_lat`` or ``num_lon`` is not at least ``2``.
    """
    if num_lat < 2 or num_lon < 2:
        msg = (
            "teaching mesh requires num_lat >= 2 and num_lon >= 2, "
            f"got num_lat={num_lat}, num_lon={num_lon}"
        )
        raise ValueError(msg)

    edges: list[tuple[int, int]] = []
    for lat in range(num_lat):
        for lon in range(num_lon):
            node = lat * num_lon + lon
            # East neighbor (periodic in longitude).
            east = lat * num_lon + (lon + 1) % num_lon
            edges.append((node, east))
            # North neighbor (no wrap at poles for this teaching slice).
            if lat + 1 < num_lat:
                north = (lat + 1) * num_lon + lon
                edges.append((node, north))
    directed = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([directed, directed.flip(0)], dim=1)


class _MeshProcessorLayer(nn.Module):
    """Residual dense graph mix (teaching stand-in for a GraphCast processor).

    Notes
    -----
    See class definition.
    """

    def __init__(self, channels: int) -> None:
        """Initialize a residual spatial processor layer.

        Parameters
        ----------
        channels : int
            Latent channel width.
        """
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, hidden: Tensor, adjacency: Tensor) -> Tensor:
        """Apply RW-normalized mix with residual and LayerNorm.

        Parameters
        ----------
        hidden : Tensor
            Latent features ``(batch, num_nodes, channels)``.
        adjacency : Tensor
            Dense normalized adjacency ``(num_nodes, num_nodes)``.

        Returns
        -------
        Tensor
            Updated latents with the same shape as ``hidden``.
        """
        mixed = torch.einsum("ij,bjc->bic", adjacency, hidden)
        return self.norm(hidden + torch.relu(self.linear(mixed)))


class GraphCastBaseline(GNNForecasterBaseline):
    """Pure-PyTorch GraphCast-style mesh teaching forecaster (Lam et al., 2023).

    Encoder–processor–decoder on a small mesh graph under the shared
    :class:`~koopman_graph.baselines.gnn.base.GNNForecasterBaseline` contract.

    **Problem-class honesty.** Mesh-weather forecasting (global lat–lon / multi-
    scale meshes, atmospheric state) is a different problem class from
    sensor-graph traffic forecasting (PEMS/METR). This adapter is a teaching
    slice only — **not** a PEMS/METR forecaster, **not** DeepMind GraphCast,
    and **not** an ERA5-scale weather model.

    The optional ``[baselines-graphcast]`` extra is reserved; this teaching
    path runs on the core install (no JAX / ``eccodes``).

    :meth:`spectrum` raises :class:`RuntimeError` (no linear Koopman operator).

    Notes
    -----
    Constructor arguments are documented on :meth:`__init__`. For a documented
    mesh fixture see :func:`build_teaching_mesh_edge_index`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_processor_layers: int = 2,
        history_len: int = 2,
        time_step: float = 1.0,
    ) -> None:
        """Initialize the GraphCast-style teaching baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Latent width for encoder / processor / decoder.
        out_channels : int
            Output node-feature dimension.
        num_processor_layers : int, optional
            Number of residual mesh processor layers. Default is ``2``.
        history_len : int, optional
            Encoder lookback. Default is ``2`` (GraphCast uses two states;
            this teaching port flattens the lookback into the encoder).
        time_step : float, optional
            Snapshot interval metadata. Default is ``1.0``.
        """
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            history_len=history_len,
            time_step=time_step,
        )
        if num_processor_layers < 1:
            msg = f"num_processor_layers must be positive, got {num_processor_layers}"
            raise ValueError(msg)
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * history_len, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.processors = nn.ModuleList(
            [_MeshProcessorLayer(hidden_channels) for _ in range(num_processor_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self._cached_rw_adj: Tensor | None = None

    def protocol(self) -> ForecasterProtocol:
        """Return the GraphCast teaching protocol (non-empty deviations).

        Returns
        -------
        ForecasterProtocol
            Lookback, claimed evaluation horizon, split ratios, and deviations
            versus Lam et al. GraphCast / operational weather stacks.
        """
        return self._teaching_protocol(
            name="graphcast",
            deviations=(
                "architecture: pure-PyTorch teaching encoder–processor–decoder "
                "on a tiny lat–lon mesh; not DeepMind JAX GraphCast or "
                "pretrained weights",
                "data: no ERA5 / HRES / eccodes / multi-mesh hierarchy or "
                "pressure-level atmospheric state vector",
                "problem class: mesh-weather teaching slice — not a PEMS/METR "
                "sensor-traffic forecaster; do not compare as traffic SOTA",
                "architecture: next-frame fit only; not 10-day autoregressive "
                "weather rollouts at operational resolution",
                "dependency: [baselines-graphcast] is reserved for future mesh "
                "helpers; teaching path needs only the core install",
            ),
        )

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> GraphCastBaseline:
        """Fit after caching RW-normalized adjacency for the mesh topology.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots on a fixed mesh (e.g. from
            :func:`build_teaching_mesh_edge_index`).
        epochs : int, optional
            Number of Adam epochs. Default is ``40``.
        lr : float, optional
            Adam learning rate. Default is ``1e-3``.
        batch_size : int or None, optional
            Mini-batch size over sliding windows.
        device : torch.device, str, or None, optional
            Training device.

        Returns
        -------
        GraphCastBaseline
            ``self`` for sklearn-style chaining.
        """
        resolved = resolve_sequence(sequence)
        train_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        edge_index = resolved.edge_index.to(train_device)
        edge_weight = snapshot_edge_weight(resolved[0])
        if edge_weight is not None:
            edge_weight = edge_weight.to(train_device)
        self._cached_rw_adj = random_walk_normalize(
            dense_adjacency(edge_index, edge_weight, resolved.num_nodes)
        )
        try:
            return super().fit(
                resolved,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
            )
        finally:
            self._cached_rw_adj = None

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next mesh frame with encoder–processor–decoder.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or
            ``(batch, history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Mesh connectivity.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)`` when ``history`` is batched.
        """
        squeeze = history.dim() == 3
        if squeeze:
            history = history.unsqueeze(0)
        elif history.dim() != 4:
            msg = (
                "history must have shape (history_len, N, C) or "
                f"(batch, history_len, N, C), got {tuple(history.shape)}"
            )
            raise ValueError(msg)

        batch, history_len, num_nodes, channels = history.shape
        if history_len != self.history_len:
            msg = (
                f"history_len mismatch: model expects {self.history_len}, "
                f"got {history_len}"
            )
            raise ValueError(msg)
        if channels != self.in_channels:
            msg = (
                f"in_channels mismatch: model expects {self.in_channels}, "
                f"got {channels}"
            )
            raise ValueError(msg)

        cached = self._cached_rw_adj
        if cached is not None and cached.shape[0] == num_nodes:
            adjacency = cached
        else:
            adjacency = random_walk_normalize(
                dense_adjacency(edge_index, edge_weight, num_nodes)
            )

        # Flatten lookback into features: (B, N, T*C).
        flat = history.permute(0, 2, 1, 3).reshape(batch, num_nodes, -1)
        hidden = self.encoder(flat)
        for layer in self.processors:
            hidden = layer(hidden, adjacency)
        output = self.decoder(hidden)
        return output[0] if squeeze else output
