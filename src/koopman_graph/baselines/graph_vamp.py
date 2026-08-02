"""Graph-aware VAMP-2 teaching baseline (contact-graph + GCN encode).

Builds a thin GCN encode → mean-pool trajectory of graph embeddings, then
scores lag pairs with the existing topology-blind
:func:`~koopman_graph.baselines.vamp2.vamp2_score` /
:func:`~koopman_graph.baselines.vamp2.vamp2_loss` mathematics (no
reimplementation). Contact graphs may be supplied or built via
:func:`~koopman_graph.datasets.molecular.contact_edge_index`.

This is a **diagnostic / teaching** baseline inspired by GraphVAMPNet
(Ghorbani et al., 2022). It is **not** a protocol-matched reproduction of
that paper, not a PyEMMA replacement, and not a Folding@home-scale MD
pipeline. Optional ``[msm]`` / deeptime is **not** required for the in-repo
score path. For discrete-time implied timescales from eigenvalues, use
:func:`~koopman_graph.analysis.implied_timescales` (analysis-owned; this
baseline does not fit a linear transfer operator).

References
----------
Ghorbani, M., Prasad, S., Klauda, J. B. & Brooks, B. R. GraphVAMPNet,
using graph neural networks and variational approach to Markov processes
for dynamical modeling of biomolecules. *J. Chem. Phys.* 156, 184103
(2022). https://doi.org/10.1063/5.0085607 (``Ghorbani2022GraphVAMPNet``)
Wu, H. & Noé, F. Variational approach for learning Markov processes from
time series data. *J. Nonlinear Sci.* 30, 23–66 (2020).
https://doi.org/10.1007/s00332-019-09567-y (``Wu2020VAMP``)
Mardt, A., Pasquali, L., Wu, H. & Noé, F. VAMPnets for deep learning of
molecular kinetics. *Nat. Commun.* 9, 5 (2018).
https://doi.org/10.1038/s41467-017-02388-1 (``Mardt2018VAMPnets``)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from koopman_graph.baselines.vamp2 import vamp2_loss, vamp2_score
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.molecular import contact_edge_index
from koopman_graph.nn import ActivationName, GNNEncoder


class GraphVAMPBaseline(nn.Module):
    """Contact-graph VAMP-2 teaching baseline with a thin GCN encode.

    Parameters
    ----------
    in_channels : int
        Node feature width.
    hidden_channels : int
        GCN hidden width.
    latent_dim : int
        Per-node latent width before mean pooling to a graph embedding.
    num_layers : int, optional
        GCN depth. Default is ``2``.
    activation : str, optional
        Hidden activation name. Default is ``\"relu\"``.
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
        """Initialize the GCN encoder used for graph embeddings.

        Parameters
        ----------
        in_channels : int
            Node feature width.
        hidden_channels : int
            GCN hidden width.
        latent_dim : int
            Per-node latent width before mean pooling.
        num_layers : int, optional
            GCN depth. Default is ``2``.
        activation : str, optional
            Hidden activation name. Default is ``\"relu\"``.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.latent_dim = int(latent_dim)
        self.encoder = GNNEncoder(
            in_channels,
            hidden_channels,
            latent_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self._edge_index: Tensor | None = None
        self._fitted = False

    @property
    def edge_index(self) -> Tensor | None:
        """Static oriented contact edges bound at the last successful fit.

        Returns
        -------
        Tensor or None
            Bound edges from :meth:`fit`, or ``None`` before fitting.
        """
        return self._edge_index

    def encode_frame(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Encode one frame to a mean-pooled graph embedding.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features ``(num_nodes, in_channels)`` or a PyG ``Data``.
        edge_index : Tensor or None, optional
            Oriented edges when ``x_or_data`` is a tensor. Defaults to the
            edges bound during :meth:`fit`.

        Returns
        -------
        Tensor
            Graph embedding with shape ``(latent_dim,)``.
        """
        resolved_edges = self._resolve_edge_index(x_or_data, edge_index)
        x = x_or_data.x if isinstance(x_or_data, Data) else x_or_data
        device = next(self.parameters()).device
        # Contact graphs use B1 orientation (i < j); GCN message passing needs
        # both directions. ``to_undirected`` is idempotent for already-symmetric
        # user-supplied edges.
        edges = to_undirected(resolved_edges.to(device=device))
        z_nodes = self.encoder(x.to(device=device), edges, None)
        return z_nodes.mean(dim=0)

    def encode_sequence(
        self,
        sequence: GraphSnapshotSequence,
        *,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Encode every snapshot to a trajectory of graph embeddings.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Time-ordered snapshots with shared node feature width.
        edge_index : Tensor or None, optional
            Override topology; defaults to bound fit edges or snapshot edges.

        Returns
        -------
        Tensor
            Embeddings with shape ``(num_timesteps, latent_dim)``.
        """
        frames = [
            self.encode_frame(sequence[t], edge_index=edge_index)
            for t in range(sequence.num_timesteps)
        ]
        return torch.stack(frames, dim=0)

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        lag: int = 1,
        epochs: int = 50,
        lr: float = 1e-3,
        epsilon: float = 1e-6,
        edge_index: Tensor | None = None,
        positions_nm: Tensor | None = None,
        cutoff_nm: float | None = None,
        device: str | torch.device | None = None,
    ) -> GraphVAMPBaseline:
        """Fit the GCN encode by maximizing VAMP-2 on lag pairs.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training trajectory.
        lag : int, optional
            Time lag in **snapshots** (integer steps). Default is ``1``.
        epochs : int, optional
            Gradient steps. Default is ``50``.
        lr : float, optional
            Adam learning rate. Default is ``1e-3``.
        epsilon : float, optional
            Ridge forwarded to :func:`vamp2_loss`. Default is ``1e-6``.
        edge_index : Tensor or None, optional
            Static oriented contact edges. If omitted, built from
            ``positions_nm`` / ``cutoff_nm`` or taken from ``sequence[0]``.
        positions_nm : Tensor or None, optional
            Atom coordinates in nanometres for
            :func:`~koopman_graph.datasets.molecular.contact_edge_index`.
        cutoff_nm : float or None, optional
            Contact cutoff in nanometres (required with ``positions_nm``).
        device : str or torch.device or None, optional
            Torch device for training. Default is the encoder device.

        Returns
        -------
        GraphVAMPBaseline
            ``self`` (sklearn-style).
        """
        traj = _as_sequence(sequence)
        resolved_edges = _resolve_fit_edges(
            traj,
            edge_index=edge_index,
            positions_nm=positions_nm,
            cutoff_nm=cutoff_nm,
        )
        if lag < 1:
            msg = f"lag must be >= 1 snapshot steps, got {lag}"
            raise ValueError(msg)
        if traj.num_timesteps <= lag:
            msg = f"need num_timesteps > lag ({lag}), got {traj.num_timesteps}"
            raise ValueError(msg)
        if epochs < 1:
            msg = f"epochs must be >= 1, got {epochs}"
            raise ValueError(msg)

        resolved_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        self.to(resolved_device)
        edges = resolved_edges.to(device=resolved_device)
        self._edge_index = edges.detach().cpu().clone()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            embeddings = self.encode_sequence(traj, edge_index=edges)
            x = embeddings[:-lag]
            y = embeddings[lag:]
            loss = vamp2_loss(x, y, epsilon=epsilon)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self._fitted = True
        self.eval()
        return self

    def score(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        lag: int = 1,
        epsilon: float = 1e-6,
        edge_index: Tensor | None = None,
    ) -> float:
        """Return the empirical VAMP-2 score on a trajectory.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Evaluation trajectory.
        lag : int, optional
            Time lag in snapshot steps. Default is ``1``.
        epsilon : float, optional
            Ridge forwarded to :func:`vamp2_score`.
        edge_index : Tensor or None, optional
            Topology override; defaults to edges bound during :meth:`fit`.

        Returns
        -------
        float
            Scalar VAMP-2 score.
        """
        if not self._fitted and self._edge_index is None and edge_index is None:
            msg = "score requires a prior fit or an explicit edge_index"
            raise RuntimeError(msg)
        traj = _as_sequence(sequence)
        if lag < 1:
            msg = f"lag must be >= 1 snapshot steps, got {lag}"
            raise ValueError(msg)
        if traj.num_timesteps <= lag:
            msg = f"need num_timesteps > lag ({lag}), got {traj.num_timesteps}"
            raise ValueError(msg)
        edges = edge_index if edge_index is not None else self._edge_index
        self.eval()
        with torch.no_grad():
            embeddings = self.encode_sequence(traj, edge_index=edges)
            score = vamp2_score(embeddings[:-lag], embeddings[lag:], epsilon=epsilon)
        return float(score.item())

    def _resolve_edge_index(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None,
    ) -> Tensor:
        """Resolve oriented edges for a single encode call.

        Parameters
        ----------
        x_or_data
            See signature.
        edge_index
            See signature.

        Returns
        -------
            See signature."""
        if edge_index is not None:
            return edge_index
        if isinstance(x_or_data, Data) and x_or_data.edge_index is not None:
            return x_or_data.edge_index
        if self._edge_index is not None:
            return self._edge_index
        msg = (
            "edge_index is required (pass explicitly, bind via fit, or "
            "provide Data.edge_index)"
        )
        raise ValueError(msg)


def _as_sequence(
    sequence: GraphSnapshotSequence | Sequence[Data],
) -> GraphSnapshotSequence:
    """Normalize a trajectory input to ``GraphSnapshotSequence``.

    Parameters
    ----------
    sequence
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(sequence, GraphSnapshotSequence):
        return sequence
    snapshots = list(sequence)
    if not snapshots:
        msg = "sequence must contain at least one snapshot"
        raise ValueError(msg)
    return GraphSnapshotSequence(snapshots)


def _resolve_fit_edges(
    sequence: GraphSnapshotSequence,
    *,
    edge_index: Tensor | None,
    positions_nm: Tensor | None,
    cutoff_nm: float | None,
) -> Tensor:
    """Resolve static contact edges for fit from args or the first snapshot.

    Parameters
    ----------
    sequence
        See signature.
    edge_index
        See signature.
    positions_nm
        See signature.
    cutoff_nm
        See signature.

    Returns
    -------
        See signature."""
    if edge_index is not None:
        if positions_nm is not None or cutoff_nm is not None:
            msg = "pass either edge_index or positions_nm/cutoff_nm, not both"
            raise ValueError(msg)
        return edge_index.to(dtype=torch.long)
    if positions_nm is not None:
        if cutoff_nm is None:
            msg = "cutoff_nm is required when positions_nm is provided"
            raise ValueError(msg)
        return contact_edge_index(positions_nm, cutoff_nm)
    if cutoff_nm is not None:
        msg = "positions_nm is required when cutoff_nm is provided"
        raise ValueError(msg)
    first = sequence[0]
    if first.edge_index is None:
        msg = (
            "could not resolve contact edges: pass edge_index or "
            "positions_nm/cutoff_nm, or provide sequence[0].edge_index"
        )
        raise ValueError(msg)
    return first.edge_index.to(dtype=torch.long)
