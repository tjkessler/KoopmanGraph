"""Koopman spectral clustering for dynamically coherent node groups.

Embeds nodes from leading Koopman eigenmodes (or caller-supplied mode
shapes), then runs seeded k-means. Cluster quality inherits the quality of
the underlying operator / spectrum — this is a structural diagnostic, not a
guarantee of ground-truth communities.

Eigenpairs follow the existing magnitude-sorted
:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` convention. Selected
mode columns are sign-canonicalized so the largest-magnitude entry has
positive real part. Networked ``N·d`` eigenvectors are reduced to per-node
coordinates by the signed mean over the latent block (preserving Fiedler
polarity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.analysis.similarity import SpectrumSource, resolve_spectrum
from koopman_graph.analysis.spectrum import decode_mode_shapes
from koopman_graph.spectrum_types import KoopmanSpectrum

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ClusteringResult:
    """Node clusters from Koopman spectral embedding + k-means.

    Attributes
    ----------
    labels : Tensor
        Integer cluster labels with shape ``(num_nodes,)``.
    embedding : Tensor
        Sign-canonicalized node embedding with shape ``(num_nodes, n_modes)``.
    eigen_indices : tuple of int
        Magnitude-sorted spectrum indices used for the embedding.
    inertia : float
        Final k-means within-cluster sum of squared distances.
    """

    labels: Tensor
    embedding: Tensor
    eigen_indices: tuple[int, ...]
    inertia: float


def _canonicalize_columns(matrix: Tensor) -> Tensor:
    """Flip each column so its largest-magnitude entry has positive real part.

    Parameters
    ----------

    matrix : Tensor
        See the function signature / summary for ``matrix``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if matrix.ndim != 2:
        msg = f"matrix must be 2D, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)
    out = matrix.clone()
    for col in range(out.shape[1]):
        column = out[:, col]
        if column.numel() == 0:
            continue
        idx = int(torch.argmax(column.abs()).item())
        pivot = column[idx]
        real = pivot.real if torch.is_complex(pivot) else pivot
        if real < 0:
            out[:, col] = -column
    return out.real if torch.is_complex(out) else out


def _normalize_modes_matrix(modes: Tensor, n_modes: int) -> Tensor:
    """Accept ``(N, n_modes)`` or ``(n_modes, N)`` and return ``(N, n_modes)``.

    Parameters
    ----------

    modes : Tensor
        See the function signature / summary for ``modes``.
    n_modes : int
        See the function signature / summary for ``n_modes``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if modes.ndim != 2:
        msg = f"modes must be 2D, got shape {tuple(modes.shape)}"
        raise ValueError(msg)
    if modes.shape[1] == n_modes:
        embedding = modes
    elif modes.shape[0] == n_modes:
        embedding = modes.T
    else:
        msg = (
            f"modes must have one axis equal to n_modes={n_modes}, "
            f"got shape {tuple(modes.shape)}"
        )
        raise ValueError(msg)
    return _canonicalize_columns(embedding.to(dtype=torch.float64)).to(
        dtype=torch.float32
    )


def _embedding_from_effective_eigenvectors(
    eigenvectors: Tensor,
    *,
    num_nodes: int,
    n_modes: int,
) -> Tensor:
    """Build ``(N, n_modes)`` embeddings from an ``N·d`` effective eigenbasis.

    Parameters
    ----------

    eigenvectors : Tensor
        See the function signature / summary for ``eigenvectors``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.
    n_modes : int
        See the function signature / summary for ``n_modes``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    state_dim = eigenvectors.shape[0]
    if state_dim % num_nodes != 0:
        msg = (
            f"eigenvector length {state_dim} is not divisible by num_nodes={num_nodes}"
        )
        raise ValueError(msg)
    latent_dim = state_dim // num_nodes
    if n_modes > eigenvectors.shape[1]:
        msg = f"n_modes={n_modes} exceeds spectrum size {eigenvectors.shape[1]}"
        raise ValueError(msg)

    columns: list[Tensor] = []
    for mode_id in range(n_modes):
        vec = eigenvectors[:, mode_id]
        reshaped = vec.reshape(num_nodes, latent_dim).real
        # Signed reduction preserves community-separating Fiedler structure;
        # L2 norms would discard essential sign information.
        columns.append(reshaped.mean(dim=-1))
    embedding = torch.stack(columns, dim=-1)
    return _canonicalize_columns(embedding)


def _embedding_from_mode_shapes(mode_shapes: Tensor, n_modes: int) -> Tensor:
    """Reduce ``(n_modes, N, C)`` complex mode shapes to ``(N, n_modes)``.

    Parameters
    ----------

    mode_shapes : Tensor
        See the function signature / summary for ``mode_shapes``.
    n_modes : int
        See the function signature / summary for ``n_modes``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if mode_shapes.ndim != 3:
        msg = (
            "mode_shapes must have shape (n_modes, num_nodes, channels), "
            f"got {tuple(mode_shapes.shape)}"
        )
        raise ValueError(msg)
    if mode_shapes.shape[0] < n_modes:
        msg = (
            f"mode_shapes provides {mode_shapes.shape[0]} modes, need n_modes={n_modes}"
        )
        raise ValueError(msg)
    selected = mode_shapes[:n_modes]
    # Signed mean over feature channels (keeps eigenfunction polarity).
    reduced = selected.real.mean(dim=-1)  # (n_modes, N)
    return _canonicalize_columns(reduced.T.contiguous())


def _relabel_by_embedding_mean(labels: Tensor, embedding: Tensor) -> Tensor:
    """Permute labels so cluster means of the first embedding coord ascend.

    Parameters
    ----------

    labels : Tensor
        See the function signature / summary for ``labels``.
    embedding : Tensor
        See the function signature / summary for ``embedding``.

    Returns
    -------

    Tensor
        See summary line."""
    n_clusters = int(labels.max().item()) + 1
    means = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if not torch.any(mask):
            means.append((float("inf"), cluster_id))
            continue
        means.append((float(embedding[mask, 0].mean().item()), cluster_id))
    means.sort(key=lambda item: item[0])
    mapping = {old: new for new, (_, old) in enumerate(means)}
    remapped = torch.empty_like(labels)
    for old, new in mapping.items():
        remapped[labels == old] = new
    return remapped


def _kmeans(
    embedding: Tensor,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int,
) -> tuple[Tensor, float]:
    """Seeded Lloyd k-means on rows of ``embedding``.

    Parameters
    ----------

    embedding : Tensor
        See the function signature / summary for ``embedding``.
    n_clusters : int
        See the function signature / summary for ``n_clusters``.
    seed : int
        See the function signature / summary for ``seed``.
    max_iter : int
        See the function signature / summary for ``max_iter``.

    Returns
    -------

    tuple[Tensor, float]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if embedding.ndim != 2:
        msg = f"embedding must be 2D, got shape {tuple(embedding.shape)}"
        raise ValueError(msg)
    num_nodes = embedding.shape[0]
    if n_clusters < 1:
        msg = f"n_clusters must be >= 1, got {n_clusters}"
        raise ValueError(msg)
    if n_clusters > num_nodes:
        msg = f"n_clusters ({n_clusters}) cannot exceed num_nodes ({num_nodes})"
        raise ValueError(msg)
    if max_iter < 1:
        msg = f"max_iter must be >= 1, got {max_iter}"
        raise ValueError(msg)

    generator = torch.Generator(device=embedding.device)
    generator.manual_seed(seed)
    # Deterministic init: pick n_clusters distinct rows by seeded permutation.
    perm = torch.randperm(num_nodes, generator=generator, device=embedding.device)
    centers = embedding[perm[:n_clusters]].clone()
    labels = torch.zeros(num_nodes, dtype=torch.long, device=embedding.device)

    for _ in range(max_iter):
        distances = torch.cdist(embedding, centers)
        new_labels = torch.argmin(distances, dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if torch.any(mask):
                centers[cluster_id] = embedding[mask].mean(dim=0)
            else:
                # Re-seed empty cluster from a random point.
                idx = int(
                    torch.randint(
                        0,
                        num_nodes,
                        (1,),
                        generator=generator,
                        device=embedding.device,
                    ).item()
                )
                centers[cluster_id] = embedding[idx]

    inertia = float(torch.sum((embedding - centers[labels]) ** 2).item())
    labels = _relabel_by_embedding_mean(labels, embedding)
    return labels, inertia


def _is_graph_koopman_model(source: object) -> bool:
    """Return whether ``source`` is a networked GraphKoopmanModel.

    Parameters
    ----------

    source : object
        See the function signature / summary for ``source``.

    Returns
    -------

    bool
        See summary line."""
    return bool(
        getattr(source, "uses_graph_koopman", False)
        or getattr(source, "uses_continuous_graph_koopman", False)
        or getattr(source, "uses_hypergraph_koopman", False)
    )


def _is_graph_koopman_model_type(source: object) -> bool:
    """Return whether ``source`` looks like GraphKoopmanModel (has encode).

    Parameters
    ----------

    source : object
        See the function signature / summary for ``source``.

    Returns
    -------

    bool
        See summary line."""
    return hasattr(source, "encode") and hasattr(source, "decode")


def koopman_spectral_clustering(
    source: SpectrumSource,
    n_clusters: int,
    *,
    n_modes: int | None = None,
    modes: Tensor | None = None,
    x_or_data: Tensor | Data | None = None,
    delta_t: float | None = None,
    edge_index: Tensor | None = None,
    num_nodes: int | None = None,
    edge_weight: Tensor | None = None,
    seed: int = 0,
    max_iter: int = 100,
) -> ClusteringResult:
    """Cluster nodes from leading Koopman eigenmodes via seeded k-means.

    Call patterns::

        koopman_spectral_clustering(
            graph_model, 2, edge_index=edges, num_nodes=n
        )
        koopman_spectral_clustering(
            pernode_model, 2, x_or_data=snapshot, n_modes=2
        )
        koopman_spectral_clustering(spectrum, 2, modes=node_modes)

    .. warning::

       Recovered communities reflect the supplied operator / spectrum. Poorly
       identified dynamics yield unreliable clusters.

    Parameters
    ----------
    source : KoopmanSpectrum or SpectrumProvider
        Precomputed spectrum or an object with ``spectrum`` (for example
        :class:`~koopman_graph.model.GraphKoopmanModel`).
    n_clusters : int
        Number of k-means clusters (``≥ 1``).
    n_modes : int or None, optional
        Number of leading magnitude-sorted modes in the embedding. Defaults
        to ``n_clusters``.
    modes : Tensor or None, optional
        Precomputed node embedding / mode matrix with shape ``(N, n_modes)``
        or ``(n_modes, N)``. Required when ``source`` is a bare
        :class:`~koopman_graph.spectrum_types.KoopmanSpectrum`.
    x_or_data : Tensor, Data, or None, optional
        Reference graph for per-node models (``decode_mode_shapes`` path).
    delta_t, edge_index, num_nodes, edge_weight
        Forwarded to :func:`~koopman_graph.analysis.resolve_spectrum` when
        accepted by the provider (same rules as
        :func:`~koopman_graph.analysis.dynamical_similarity`).
    seed : int, optional
        RNG seed for k-means initialization. Default ``0``.
    max_iter : int, optional
        Maximum Lloyd iterations. Default ``100``.

    Returns
    -------
    ClusteringResult
        Labels, embedding, eigen-indices, and inertia.

    Raises
    ------
    ValueError
        If arguments are inconsistent (missing modes / topology / reference
        graph, invalid sizes).
    TypeError
        If ``source`` is not a spectrum or spectrum provider.
    """
    if n_clusters < 1:
        msg = f"n_clusters must be >= 1, got {n_clusters}"
        raise ValueError(msg)
    resolved_n_modes = n_clusters if n_modes is None else n_modes
    if resolved_n_modes < 1:
        msg = f"n_modes must be >= 1, got {resolved_n_modes}"
        raise ValueError(msg)

    eigen_indices = tuple(range(resolved_n_modes))

    if modes is not None:
        embedding = _normalize_modes_matrix(modes, resolved_n_modes)
        # Still resolve spectrum when a provider is given so eigen_indices stay
        # aligned with a real spectrum size check.
        if isinstance(source, KoopmanSpectrum):
            if resolved_n_modes > source.eigenvalues.numel():
                msg = (
                    f"n_modes={resolved_n_modes} exceeds spectrum size "
                    f"{source.eigenvalues.numel()}"
                )
                raise ValueError(msg)
        else:
            spectrum = resolve_spectrum(
                source,
                delta_t=delta_t,
                edge_index=edge_index,
                num_nodes=num_nodes,
                edge_weight=edge_weight,
            )
            if resolved_n_modes > spectrum.eigenvalues.numel():
                msg = (
                    f"n_modes={resolved_n_modes} exceeds spectrum size "
                    f"{spectrum.eigenvalues.numel()}"
                )
                raise ValueError(msg)
    elif isinstance(source, KoopmanSpectrum):
        msg = (
            "modes is required when source is a KoopmanSpectrum "
            "(pass a (num_nodes, n_modes) mode matrix)"
        )
        raise ValueError(msg)
    elif _is_graph_koopman_model(source):
        if num_nodes is None:
            msg = "num_nodes is required for networked Koopman spectral clustering"
            raise ValueError(msg)
        spectrum = resolve_spectrum(
            source,
            delta_t=delta_t,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        )
        embedding = _embedding_from_effective_eigenvectors(
            spectrum.eigenvectors,
            num_nodes=num_nodes,
            n_modes=resolved_n_modes,
        )
    elif _is_graph_koopman_model_type(source):
        if x_or_data is None:
            msg = (
                "x_or_data is required for per-node Koopman spectral "
                "clustering (decode_mode_shapes path)"
            )
            raise ValueError(msg)
        model = source  # type: GraphKoopmanModel
        mode_shapes = decode_mode_shapes(
            model,
            x_or_data,
            mode_indices=list(eigen_indices),
            edge_index=edge_index,
        )
        embedding = _embedding_from_mode_shapes(mode_shapes, resolved_n_modes)
    else:
        # Generic SpectrumProvider without model encode/decode: require modes.
        msg = (
            "Unable to build a node embedding from this source; pass modes= "
            "explicitly, or supply a GraphKoopmanModel"
        )
        raise ValueError(msg)

    labels, inertia = _kmeans(
        embedding,
        n_clusters,
        seed=seed,
        max_iter=max_iter,
    )
    return ClusteringResult(
        labels=labels,
        embedding=embedding,
        eigen_indices=eigen_indices,
        inertia=inertia,
    )
