"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from koopman_graph.baselines.gnn import (
    SpatiotemporalSSMBaseline,
    STAEformerBaseline,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _path_edges_v014_remaining(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_attention_ssm_batched_and_head_validation() -> None:
    """Teaching attention/SSM cover batched predict and invalid heads."""
    with pytest.raises(ValueError, match="n_heads"):
        STAEformerBaseline(3, 8, 3, history_len=2, n_heads=0)
    history = torch.randn(2, 2, 4, 3)
    edge_index = _path_edges(4)
    attn = STAEformerBaseline(3, 8, 3, history_len=2, n_heads=1)
    assert attn.predict_next(history, edge_index).shape == (2, 4, 3)
    ssm = SpatiotemporalSSMBaseline(3, 8, 3, history_len=2)
    assert ssm.predict_next(history, edge_index).shape == (2, 4, 3)


def test_teaching_attention_and_ssm_predict() -> None:
    """New teaching baselines declare deviations and predict one step."""
    history = torch.randn(2, 4, 3)
    edge_index = _path_edges_v014_remaining(4)
    attn = STAEformerBaseline(3, 8, 3, history_len=2, n_heads=1)
    assert attn.protocol().deviations
    pred = attn.predict_next(history, edge_index)
    assert pred.shape == (4, 3)
    ssm = SpatiotemporalSSMBaseline(3, 8, 3, history_len=2)
    assert ssm.protocol().deviations
    assert ssm.predict_next(history, edge_index).shape == (4, 3)
