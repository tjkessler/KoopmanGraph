"""Timestep encode helpers shared by pair losses and the latent cache.

Kept as a separate training peer so :mod:`~koopman_graph.training.latent_cache`
and :mod:`~koopman_graph.training.pair_objectives` do not import each other
for encoding.
"""

from __future__ import annotations

from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.protocols import TrainableKoopmanModel


def encode_at_timestep(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    index: int,
) -> Tensor:
    """Encode with delay history when the model exposes ``encode_at``.

    Shared non-private entry point for pair losses and the sequence latent
    cache (:mod:`~koopman_graph.training.latent_cache`). Prefer this name over
    leading-underscore helpers for cross-module imports.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable model; may implement ``encode_at(sequence, index)``.
    sequence : GraphSnapshotSequence
        Source trajectory.
    index : int
        Timestep to encode (window end).

    Returns
    -------
    Tensor
        Latent node features.
    """
    encode_at = getattr(model, "encode_at", None)
    if callable(encode_at):
        return encode_at(sequence, index)
    return model.encode(sequence[index])
