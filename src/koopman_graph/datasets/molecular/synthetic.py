"""Seeded synthetic two-state molecular teaching trajectory.

Generates a fixed four-atom contact graph whose node features switch
between two patterns according to a symmetric discrete-time Markov chain.
The slow eigenvalue and implied timescale are known in closed form and are
recorded in the package-relative FAIR metadata JSON
(``data/synthetic_two_state_v1.json``).

This is a **CI / teaching oracle**, not experimental MD.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.molecular.contact_graph import contact_edge_index

_METADATA_RESOURCE = "data/synthetic_two_state_v1.json"


@dataclass(frozen=True)
class SyntheticTwoStateTrajectory:
    """Seeded two-state teaching trajectory with oracle timescale metadata.

    Attributes
    ----------
    sequence : GraphSnapshotSequence
        Time-ordered snapshots with shared contact topology.
    state_labels : Tensor
        Integer state id per frame with shape ``(num_timesteps,)``
        (``0`` or ``1``).
    edge_index : Tensor
        Static oriented contact edges ``(2, num_edges)``.
    positions_nm : Tensor
        Atom coordinates in nanometres with shape ``(num_nodes, 3)``.
    transition_prob : float
        Symmetric flip probability ``p`` per snapshot step.
    slow_eigenvalue : float
        Closed-form slow eigenvalue ``1 - 2p`` of the two-state chain.
    oracle_slow_timescale_steps : float
        Implied timescale ``-lag / ln(1 - 2p)`` at ``oracle_lag_steps``.
    oracle_lag_steps : int
        Lag (snapshot steps) for which the oracle timescale is stated.
    timestep_ps : float
        Teaching timestep in picoseconds (one snapshot step).
    dataset_id : str
        Stable dataset identifier (``synthetic_two_state_v1``).
    version : str
        Dataset card version string.
    seed : int
        RNG seed used to generate this realization.
    """

    sequence: GraphSnapshotSequence
    state_labels: Tensor
    edge_index: Tensor
    positions_nm: Tensor
    transition_prob: float
    slow_eigenvalue: float
    oracle_slow_timescale_steps: float
    oracle_lag_steps: int
    timestep_ps: float
    dataset_id: str
    version: str
    seed: int


def load_synthetic_two_state_metadata() -> dict[str, Any]:
    """Load FAIR oracle metadata from the package-relative JSON resource.

    Returns
    -------
    dict
        Parsed contents of ``data/synthetic_two_state_v1.json``.

    Raises
    ------
    FileNotFoundError
        If the package resource is missing from the install.
    """
    resource = resources.files("koopman_graph.datasets.molecular").joinpath(
        _METADATA_RESOURCE
    )
    with resources.as_file(resource) as path:
        if not path.is_file():
            msg = (
                f"missing package resource {_METADATA_RESOURCE!r} under "
                f"koopman_graph.datasets.molecular"
            )
            raise FileNotFoundError(msg)
        return json.loads(path.read_text(encoding="utf-8"))


def synthetic_two_state_slow_eigenvalue(transition_prob: float) -> float:
    """Return the closed-form slow eigenvalue ``1 - 2p``.

    Parameters
    ----------
    transition_prob : float
        Symmetric flip probability ``p`` in ``(0, 0.5)``.

    Returns
    -------
    float
        Slow eigenvalue of the two-state transition matrix.
    """
    p = _validate_transition_prob(transition_prob)
    return 1.0 - 2.0 * p


def synthetic_two_state_oracle_timescale(
    transition_prob: float,
    *,
    lag_steps: int = 1,
) -> float:
    """Return the closed-form implied timescale in snapshot steps.

    Parameters
    ----------
    transition_prob : float
        Symmetric flip probability ``p`` in ``(0, 0.5)``.
    lag_steps : int, optional
        Positive lag ``τ`` in snapshot steps. Default is ``1``.

    Returns
    -------
    float
        ``-τ / ln(1 - 2p)``.
    """
    if lag_steps < 1:
        msg = f"lag_steps must be >= 1, got {lag_steps}"
        raise ValueError(msg)
    lam = synthetic_two_state_slow_eigenvalue(transition_prob)
    return -float(lag_steps) / math.log(lam)


def generate_synthetic_two_state(
    *,
    num_timesteps: int | None = None,
    seed: int | None = None,
    transition_prob: float | None = None,
    noise_std: float = 0.05,
) -> SyntheticTwoStateTrajectory:
    """Generate a seeded two-state contact-graph teaching trajectory.

    Parameters
    ----------
    num_timesteps : int or None, optional
        Trajectory length. Defaults to the metadata
        ``default_num_timesteps``.
    seed : int or None, optional
        RNG seed. Defaults to the metadata ``default_seed``.
    transition_prob : float or None, optional
        Symmetric flip probability. Defaults to the metadata value
        (``0.05`` → slow eigenvalue ``0.9``).
    noise_std : float, optional
        Gaussian feature noise standard deviation. Default is ``0.05``.

    Returns
    -------
    SyntheticTwoStateTrajectory
        Snapshots, state labels, and oracle timescale metadata.

    Raises
    ------
    ValueError
        If length, seed, noise, or transition probability are invalid.
    """
    meta = load_synthetic_two_state_metadata()
    resolved_steps = (
        int(meta["default_num_timesteps"])
        if num_timesteps is None
        else int(num_timesteps)
    )
    resolved_seed = int(meta["default_seed"]) if seed is None else int(seed)
    resolved_p = (
        float(meta["transition_prob"])
        if transition_prob is None
        else float(transition_prob)
    )
    if resolved_steps < 2:
        msg = f"num_timesteps must be >= 2, got {resolved_steps}"
        raise ValueError(msg)
    if noise_std < 0.0 or noise_std != noise_std:
        msg = f"noise_std must be finite and >= 0, got {noise_std}"
        raise ValueError(msg)

    p = _validate_transition_prob(resolved_p)
    lag_steps = int(meta["oracle_lag_steps"])
    slow_eigenvalue = synthetic_two_state_slow_eigenvalue(p)
    oracle_timescale = synthetic_two_state_oracle_timescale(p, lag_steps=lag_steps)

    positions = torch.tensor(meta["positions_nm"], dtype=torch.float64)
    cutoff_nm = float(meta["cutoff_nm"])
    edge_index = contact_edge_index(positions, cutoff_nm)
    patterns = _state_patterns(num_nodes=int(positions.shape[0]), num_features=2)

    generator = torch.Generator().manual_seed(resolved_seed)
    states = _sample_symmetric_two_state(
        num_timesteps=resolved_steps,
        transition_prob=p,
        generator=generator,
    )
    snapshots: list[Data] = []
    num_nodes, num_features = int(patterns.shape[1]), int(patterns.shape[2])
    for t in range(resolved_steps):
        state = int(states[t].item())
        noise = noise_std * torch.randn(
            (num_nodes, num_features),
            generator=generator,
            dtype=torch.float32,
        )
        x = patterns[state] + noise
        snapshots.append(Data(x=x, edge_index=edge_index.clone()))

    return SyntheticTwoStateTrajectory(
        sequence=GraphSnapshotSequence(snapshots),
        state_labels=states,
        edge_index=edge_index.clone(),
        positions_nm=positions.to(dtype=torch.float32),
        transition_prob=p,
        slow_eigenvalue=slow_eigenvalue,
        oracle_slow_timescale_steps=oracle_timescale,
        oracle_lag_steps=lag_steps,
        timestep_ps=float(meta["timestep_ps"]),
        dataset_id=str(meta["dataset_id"]),
        version=str(meta["version"]),
        seed=resolved_seed,
    )


def _state_patterns(*, num_nodes: int, num_features: int) -> Tensor:
    """Return the two noise-free node-feature patterns ``(2, N, F)``.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_features : int
        Feature width per node.

    Returns
    -------
    Tensor
        Stacked patterns with shape ``(2, num_nodes, num_features)``.
    """
    pattern_a = torch.zeros(num_nodes, num_features, dtype=torch.float32)
    pattern_a[:, 0] = 1.0
    pattern_b = torch.zeros(num_nodes, num_features, dtype=torch.float32)
    pattern_b[:, 1] = 1.0
    return torch.stack([pattern_a, pattern_b], dim=0)


def _sample_symmetric_two_state(
    *,
    num_timesteps: int,
    transition_prob: float,
    generator: torch.Generator,
) -> Tensor:
    """Sample a symmetric two-state Markov chain.

    Parameters
    ----------
    num_timesteps
        See signature.
    transition_prob
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    states = torch.empty(num_timesteps, dtype=torch.long)
    states[0] = 0
    for t in range(1, num_timesteps):
        flip = torch.rand(1, generator=generator).item() < transition_prob
        states[t] = 1 - states[t - 1] if flip else states[t - 1]
    return states


def _validate_transition_prob(transition_prob: float) -> float:
    """Validate the symmetric flip probability.

    Parameters
    ----------
    transition_prob
        See signature.

    Returns
    -------
        See signature."""
    try:
        p = float(transition_prob)
    except (TypeError, ValueError) as exc:
        msg = f"transition_prob must be a real number, got {transition_prob!r}"
        raise ValueError(msg) from exc
    if not (0.0 < p < 0.5) or p != p:
        msg = (
            "transition_prob must be in (0, 0.5) for a distinct slow "
            f"eigenvalue in (0, 1), got {transition_prob!r}"
        )
        raise ValueError(msg)
    return p
