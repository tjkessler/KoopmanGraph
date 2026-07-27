"""Shared teacher-forced latent trajectory cache for training windows.

Encodes each timestep once so reconstruction / consistency losses (and later
helpers) can reuse latents without redundant GNN forwards. Topology, controls,
and observation masks stay on the
:class:`~koopman_graph.data.GraphSnapshotSequence` and are read at pair time —
they are intentionally **not** stored on the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.operators.global_local import stack_latent_window
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.timestep_encode import encode_at_timestep


@dataclass(frozen=True)
class SequenceLatentCache:
    """Teacher-forced encodings for one training window or trajectory.

    Parameters
    ----------
    z : list of Tensor
        Latent node features per timestep; ``z[t]`` is
        :func:`~koopman_graph.training.timestep_encode.encode_at_timestep`
        at index ``t``. Length equals ``sequence.num_timesteps``.

    Notes
    -----
    Does not store ``edge_index``, controls, or observation masks. Pair losses
    must continue to read those from the source sequence so topology kwargs stay
    synchronized with the snapshot being advanced.
    """

    z: list[Tensor]

    @property
    def num_timesteps(self) -> int:
        """Return the number of encoded timesteps ``T``.

        Returns
        -------
        int
            ``len(self.z)``.
        """
        return len(self.z)


def encode_sequence_latents(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> SequenceLatentCache:
    """Encode every timestep of ``sequence`` exactly once.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model providing ``encode`` / ``encode_at``.
    sequence : GraphSnapshotSequence
        Time-ordered snapshots (training window or full trajectory).

    Returns
    -------
    SequenceLatentCache
        Cache with ``len(z) == sequence.num_timesteps``.

    Notes
    -----
    Rollout losses may reuse ``cache.z[start]`` as the origin latent but must
    **not** replace autoregressive advanced latents with teacher-forced
    ``cache.z[t]`` after the first step.
    """
    latents = [
        encode_at_timestep(model, sequence, index)
        for index in range(sequence.num_timesteps)
    ]
    return SequenceLatentCache(z=latents)


def latent_window_from_cache(
    cache: SequenceLatentCache,
    timestep: int,
    local_window: int,
) -> Tensor:
    """Build a teacher-forced latent window from cached encodings.

    Stacks already-encoded frames ``[t-w+1, …, t]`` (left-padded by the
    earliest available frame in the slice) without calling the encoder.
    Matches
    :func:`~koopman_graph.training.pair_objectives.teacher_forced_latent_window`
    when the cache was built from the same model and sequence.

    Parameters
    ----------
    cache : SequenceLatentCache
        Latents from :func:`encode_sequence_latents`.
    timestep : int
        Index ``t`` of the state being advanced (``0 <= t < T``).
    local_window : int
        Window length ``w`` (must be ``>= 1``).

    Returns
    -------
    Tensor
        Window with shape ``(w, ..., d)``.

    Raises
    ------
    ValueError
        If ``timestep`` is out of range or ``local_window < 1``.
    """
    if local_window < 1:
        msg = f"local_window must be >= 1, got {local_window}"
        raise ValueError(msg)
    num_timesteps = cache.num_timesteps
    if timestep < 0 or timestep >= num_timesteps:
        msg = f"timestep must be in [0, {num_timesteps}), got {timestep}"
        raise ValueError(msg)

    start = max(0, timestep - local_window + 1)
    frames = cache.z[start : timestep + 1]
    return stack_latent_window(frames[:-1], window=local_window, current=frames[-1])
