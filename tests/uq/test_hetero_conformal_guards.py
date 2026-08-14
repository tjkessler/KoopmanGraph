"""Coverage and error-path tests for :mod:`koopman_graph.uq`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.operators import (
    HeteroGraphKoopmanOperator,
)
from koopman_graph.uq.conformal import (
    ConformalKoopmanUQ,
    _is_hetero_sequence,
    _stack_hetero_features,
    _union_relation_edge_index,
)

_TYPES = ("a", "b")

_EDGE_TYPES = (("a", "r", "b"),)


def test_hetero_conformal_helper_guards() -> None:
    """Heterogeneous conformal helpers validate sequences, features, and origins."""
    hetero = HeteroData()
    hetero["a"].x = torch.zeros(1, 2)
    assert _is_hetero_sequence([hetero])
    assert not _is_hetero_sequence([])

    with pytest.raises(ValueError, match="missing node type"):
        _stack_hetero_features(hetero, _TYPES)

    class _MissingFeatures:
        node_types = _TYPES

        def __getitem__(self, name: str) -> SimpleNamespace:
            features = torch.zeros(1, 2) if name == "a" else None
            return SimpleNamespace(x=features)

    with pytest.raises(ValueError, match="missing feature matrix"):
        _stack_hetero_features(_MissingFeatures(), _TYPES)  # type: ignore[arg-type]

    empty = HeteroData()
    empty["a"].x = torch.zeros(1, 2)
    empty["b"].x = torch.zeros(1, 2)
    empty["a", "r", "b"].edge_index = torch.empty(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty relation"):
        _union_relation_edge_index(empty, _EDGE_TYPES, _TYPES)

    bad_uq = ConformalKoopmanUQ(
        SimpleNamespace(koopman=nn.Linear(2, 2))  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="HeteroGraphKoopmanOperator"):
        bad_uq._hetero_operator()

    hetero_model = SimpleNamespace(
        koopman=HeteroGraphKoopmanOperator(2, 1),
        uses_hetero_koopman=True,
    )
    uq = ConformalKoopmanUQ(hetero_model)  # type: ignore[arg-type]
    uq._quantiles = torch.ones(1)
    uq._alpha = 0.1
    uq._calibrated_steps = 1
    with pytest.raises(TypeError, match="requires a HeteroData origin"):
        uq.predict_interval(Data(x=torch.zeros(2, 2)), steps=1, level=0.9)
