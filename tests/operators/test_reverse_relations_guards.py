"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.graph_utils.topology import (
    materialize_reverse_relation_edges,
    synthesize_reverse_edge_types,
)

_EDGE_TYPES = (("a", "r", "b"),)


def test_reverse_relation_validation_and_materialization_guards() -> None:
    """Reverse-relation helpers reject malformed schemas and absent forwards."""
    for bad_prefix in ("",):
        with pytest.raises(ValueError, match="non-empty"):
            synthesize_reverse_edge_types(_EDGE_TYPES, reverse_prefix=bad_prefix)
        with pytest.raises(ValueError, match="non-empty"):
            materialize_reverse_relation_edges(
                HeteroData(), _EDGE_TYPES, reverse_prefix=bad_prefix
            )

    with pytest.raises(ValueError, match="triples"):
        synthesize_reverse_edge_types((("a", "r"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        synthesize_reverse_edge_types((("a", "", "b"),))
    with pytest.raises(ValueError, match="unique"):
        synthesize_reverse_edge_types((_EDGE_TYPES[0], _EDGE_TYPES[0]))
    with pytest.raises(ValueError, match="not the geometric reverse"):
        synthesize_reverse_edge_types((("a", "r", "b"), ("a", "rev_r", "a")))

    with pytest.raises(ValueError, match="triples"):
        materialize_reverse_relation_edges(
            HeteroData(),
            (("a", "r"),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        materialize_reverse_relation_edges(HeteroData(), (("a", "", "b"),))
    schema = (("a", "r", "b"), ("b", "rev_r", "a"))
    with pytest.raises(ValueError, match="missing forward edge type"):
        materialize_reverse_relation_edges(HeteroData(), schema)

    snapshot = HeteroData()
    snapshot["a"].x = torch.zeros(1, 1)
    snapshot["b"].x = torch.zeros(1, 1)
    snapshot["a", "r", "b"].edge_index = torch.tensor([[0], [0]])
    snapshot["b", "rev_r", "a"].edge_index = torch.tensor([[0], [0]])
    preserved = materialize_reverse_relation_edges(snapshot, schema)
    torch.testing.assert_close(
        preserved["b", "rev_r", "a"].edge_index,
        snapshot["b", "rev_r", "a"].edge_index,
    )
