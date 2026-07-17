"""Shared rollout-origin helpers for training and forecast evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from koopman_graph.data.containers import GraphSnapshotSequence

RolloutStartIndices = Sequence[int] | Literal["all"] | None


def resolve_rollout_start_indices(
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
    epoch: int = 0,
) -> list[int]:
    """Resolve rollout origin indices for training or forecast evaluation.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Snapshot sequence used for rollout.
    horizon : int
        Rollout horizon in steps.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Explicit origins, all valid origins, or ``None`` for the default
        ``[0]``.
    rollout_starts_per_epoch : int or None, optional
        When set, randomly sample this many valid origins each epoch.
        Overrides ``rollout_start_indices`` except when that argument is an
        explicit non-empty index list.
    rollout_start_seed : int or None, optional
        Base seed for random origin sampling. The effective seed is
        ``rollout_start_seed + epoch`` when provided.
    epoch : int, optional
        Zero-based epoch index mixed into the random seed. Default is ``0``.

    Returns
    -------
    list of int
        Valid zero-based rollout origin indices.

    Raises
    ------
    ValueError
        If ``horizon`` is invalid, no origins are available, or an origin is
        out of range.
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if sequence.num_timesteps <= horizon:
        msg = (
            f"sequence has {sequence.num_timesteps} timesteps but needs more "
            f"than {horizon} for rollout loss"
        )
        raise ValueError(msg)

    upper = sequence.num_timesteps - horizon
    valid_origins = list(range(upper))

    if rollout_start_indices == "all":
        origins = valid_origins
    elif rollout_start_indices is not None:
        origins = []
        for start in rollout_start_indices:
            if start < 0 or start >= upper:
                msg = (
                    f"start index {start} is out of range for horizon "
                    f"{horizon}; valid range is [0, {upper - 1}]"
                )
                raise ValueError(msg)
            origins.append(start)
        if not origins:
            msg = "rollout_start_indices must contain at least one valid origin"
            raise ValueError(msg)
    else:
        origins = [0]

    if rollout_starts_per_epoch is not None:
        if rollout_starts_per_epoch < 1:
            msg = (
                f"rollout_starts_per_epoch must be >= 1 when set, "
                f"got {rollout_starts_per_epoch}"
            )
            raise ValueError(msg)
        generator: torch.Generator | None = None
        if rollout_start_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(rollout_start_seed + epoch)
        sampled = torch.randint(
            low=0,
            high=len(valid_origins),
            size=(rollout_starts_per_epoch,),
            generator=generator,
        )
        origins = [valid_origins[int(index)] for index in sampled.tolist()]

    return origins
