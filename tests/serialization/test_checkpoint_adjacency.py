"""Coverage and error-path tests for :mod:`koopman_graph.serialization`."""

from __future__ import annotations

import pytest

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.serialization import _resolve_checkpoint_adjacency


def test_checkpoint_adjacency_resolver_and_factory() -> None:
    """Serialization adjacency resolver + factory invalid adjacency."""
    assert (
        _resolve_checkpoint_adjacency("random_walk", koopman_kind="graph")
        == "random_walk"
    )
    with pytest.raises(ValueError, match="adjacency is required"):
        _resolve_checkpoint_adjacency(None, koopman_kind="graph")
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_checkpoint_adjacency("bogus", koopman_kind="graph")
    with pytest.raises(ValueError, match="must be null"):
        _resolve_checkpoint_adjacency("symmetric", koopman_kind="dense")
    assert _resolve_checkpoint_adjacency(None, koopman_kind="dense") == "symmetric"

    with pytest.raises(ValueError, match="koopman_adjacency must be one of"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 2, num_layers=1),
            decoder=GNNDecoder(2, 4, 2, num_layers=1),
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            koopman_adjacency="bogus",  # type: ignore[arg-type]
        )
