"""Shared Laplacian diffusion dynamics for benchmark datasets.

Uses the same pseudoinverse-normalized Laplacian ``L_sym = P - Â`` as
:func:`~koopman_graph.observables.graph_laplacian_features` (via
:mod:`koopman_graph.graph_utils`), but assembles a **dense** one-step diffusion
operator for offline benchmark rollouts. Prefer the sparse matvec path in
``observables`` for hybrid physics lifting at training time.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.graph_utils import dense_symmetric_normalized_laplacian

InitialStateName = Literal["random", "ones"]


def normalized_step_operator(
    edge_index: Tensor,
    num_nodes: int,
    diffusion_rate: float,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Build one-step Laplacian diffusion operator ``I - alpha * L_sym``.

    The symmetrically normalized Laplacian is
    ``L_sym = P - Â = (D^+)^{1/2} (D - A) (D^+)^{1/2}``, sharing its adjacency
    normalization with
    :func:`~koopman_graph.graph_utils.symmetric_normalized_adjacency_edge_weights`.
    Isolated nodes have ``L_sym`` row/column zero, so the step operator leaves
    their features unchanged (diagonal contribution ``1``). On graphs with no
    isolates this reduces to ``(1 - alpha) * I + alpha * Â``.

    The contract assumes an undirected, symmetrically represented adjacency.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    diffusion_rate : float
        Diffusion strength ``alpha`` in ``[0, 1]``.
    dtype : torch.dtype
        Floating dtype for the dense operator.

    Returns
    -------
    Tensor
        Step operator with shape ``(num_nodes, num_nodes)``.
    """
    laplacian = dense_symmetric_normalized_laplacian(
        edge_index,
        num_nodes,
        dtype=dtype,
    )
    eye = torch.eye(num_nodes, dtype=dtype, device=edge_index.device)
    return eye - diffusion_rate * laplacian


def make_generator(seed: int | None) -> torch.Generator | None:
    """Return a seeded torch generator when ``seed`` is provided.

    Parameters
    ----------
    seed : int or None
        Random seed. When ``None``, no generator is created.

    Returns
    -------
    torch.Generator or None
        Seeded generator, or ``None`` when ``seed`` is ``None``.
    """
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def validate_diffusion_generation_params(
    *,
    decay_rate: float,
    noise_std: float,
    diffusion_rate: float | None = None,
    initial_state: InitialStateName | None = None,
) -> None:
    """Validate shared diffusion benchmark generation parameters.

    Used by Laplacian-diffusion generators (synthetic, grid, IEEE 118). The
    shared ``decay_rate`` domain is ``> 0`` (values ``>= 1`` are allowed and
    amplify rather than damp). Anisotropic advection uses a stricter open
    interval via :func:`validate_advection_decay_rate`.

    Parameters
    ----------
    decay_rate : float
        Global amplitude decay applied each diffusion step. Must be ``> 0``.
    noise_std : float
        Standard deviation of additive Gaussian noise. Must be ``>= 0``.
    diffusion_rate : float or None, optional
        Laplacian diffusion strength in ``[0, 1]``. Validated when provided.
    initial_state : {"random", "ones"} or None, optional
        Initial node feature pattern. Validated when provided.

    Raises
    ------
    ValueError
        If any parameter is outside its allowed range.
    """
    if diffusion_rate is not None and not 0.0 <= diffusion_rate <= 1.0:
        msg = f"diffusion_rate must be in [0, 1], got {diffusion_rate}"
        raise ValueError(msg)
    if decay_rate <= 0.0:
        msg = f"decay_rate must be > 0, got {decay_rate}"
        raise ValueError(msg)
    if noise_std < 0.0:
        msg = f"noise_std must be >= 0, got {noise_std}"
        raise ValueError(msg)
    if initial_state is not None and initial_state not in {"random", "ones"}:
        msg = f"initial_state must be 'random' or 'ones', got {initial_state!r}"
        raise ValueError(msg)


def validate_advection_decay_rate(decay_rate: float) -> None:
    """Validate anisotropic-advection self-retention ``decay_rate``.

    Unlike Laplacian diffusion (where ``decay_rate > 0``), advection treats
    ``decay_rate`` as a self-retention factor that must lie in the open
    interval ``(0, 1)`` so neighbor mass remains strictly positive.

    Parameters
    ----------
    decay_rate : float
        Self-retention factor. Must satisfy ``0 < decay_rate < 1``.

    Raises
    ------
    ValueError
        If ``decay_rate`` is outside ``(0, 1)``.
    """
    if not 0.0 < decay_rate < 1.0:
        msg = f"decay_rate must be in (0, 1), got {decay_rate}"
        raise ValueError(msg)


def initial_node_features(
    num_nodes: int,
    in_channels: int,
    initial_state: InitialStateName,
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> Tensor:
    """Create the initial node feature matrix for diffusion benchmarks.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    in_channels : int
        Node feature dimension.
    initial_state : {"random", "ones"}
        Feature initialization pattern.
    generator : torch.Generator or None
        Optional RNG used when ``initial_state="random"``.
    dtype : torch.dtype
        Floating dtype for the returned tensor.

    Returns
    -------
    Tensor
        Initial node features with shape ``(num_nodes, in_channels)``.
    """
    if initial_state == "ones":
        return torch.ones((num_nodes, in_channels), dtype=dtype)
    return torch.randn(
        (num_nodes, in_channels),
        generator=generator,
        dtype=dtype,
    )


def apply_laplacian_diffusion_step(
    state: Tensor,
    step_operator: Tensor,
    decay_rate: float,
) -> Tensor:
    """Advance node features by one Laplacian diffusion step.

    Parameters
    ----------
    state : Tensor
        Current node features with shape ``(num_nodes, in_channels)``.
    step_operator : Tensor
        One-step diffusion operator with shape ``(num_nodes, num_nodes)``.
    decay_rate : float
        Global amplitude decay applied to the diffused state.

    Returns
    -------
    Tensor
        Updated node features with the same shape as ``state``.
    """
    return decay_rate * (step_operator @ state)


def add_gaussian_noise(
    state: Tensor,
    noise_std: float,
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> Tensor:
    """Add optional isotropic Gaussian noise to node features.

    Parameters
    ----------
    state : Tensor
        Node features to perturb.
    noise_std : float
        Standard deviation of additive Gaussian noise. No noise is added when
        ``noise_std <= 0``.
    generator : torch.Generator or None
        Optional RNG for drawing noise samples.
    dtype : torch.dtype
        Floating dtype for generated noise.

    Returns
    -------
    Tensor
        Perturbed node features with the same shape as ``state``.
    """
    if noise_std <= 0.0:
        return state
    noise = torch.randn(state.shape, generator=generator, dtype=dtype)
    return state + noise_std * noise


def laplacian_diffusion_rollout(
    *,
    edge_index: Tensor,
    num_nodes: int,
    num_timesteps: int,
    in_channels: int,
    diffusion_rate: float,
    decay_rate: float,
    noise_std: float,
    initial_state: InitialStateName,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    initial_features: Tensor | None = None,
) -> Tensor:
    """Simulate Laplacian diffusion dynamics and return stacked features.

    Parameters
    ----------
    edge_index : Tensor
        Shared graph topology.
    num_nodes : int
        Number of nodes in the graph.
    num_timesteps : int
        Number of temporal snapshots to generate.
    in_channels : int
        Node feature dimension.
    diffusion_rate : float
        Laplacian diffusion strength in ``[0, 1]``.
    decay_rate : float
        Global amplitude decay applied each step.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    initial_state : {"random", "ones"}
        Initial node feature pattern when ``initial_features`` is ``None``.
    dtype : torch.dtype
        Floating dtype for generated features.
    generator : torch.Generator or None
        Optional RNG for initial state and noise.
    initial_features : Tensor or None, optional
        Explicit initial node features with shape ``(num_nodes, in_channels)``.

    Returns
    -------
    Tensor
        Node features with shape ``(num_timesteps, num_nodes, in_channels)``.
    """
    step_operator = normalized_step_operator(
        edge_index,
        num_nodes,
        diffusion_rate,
        dtype=dtype,
    )
    state = (
        initial_features.clone()
        if initial_features is not None
        else initial_node_features(
            num_nodes,
            in_channels,
            initial_state,
            generator=generator,
            dtype=dtype,
        )
    )

    snapshots = [state.clone()]
    for _ in range(num_timesteps - 1):
        state = apply_laplacian_diffusion_step(state, step_operator, decay_rate)
        state = add_gaussian_noise(
            state,
            noise_std,
            generator=generator,
            dtype=dtype,
        )
        snapshots.append(state.clone())

    return torch.stack(snapshots, dim=0)


def diffusion_sequence_from_features(
    features: Tensor,
    edge_index: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> GraphSnapshotSequence:
    """Wrap rollout features and topology in a validated snapshot sequence.

    Parameters
    ----------
    features : Tensor
        Node features with shape ``(num_timesteps, num_nodes, in_channels)``.
    edge_index : Tensor
        Shared edge index with shape ``(2, num_edges)``.
    dtype : torch.dtype, optional
        Floating dtype used when building snapshots. Default is ``torch.float32``.

    Returns
    -------
    :class:`~koopman_graph.data.GraphSnapshotSequence`
        Validated time-ordered snapshot sequence.
    """
    return GraphSnapshotSequence.from_arrays(features, edge_index, dtype=dtype)
