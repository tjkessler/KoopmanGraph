"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.stgode import STGODEBaseline, _GraphODEFunc
from koopman_graph.data import (
    GraphSnapshotSequence,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_stgode_constructor_and_ode_guards() -> None:
    """Constructor / ODE-func / history-shape validation branches."""
    with pytest.raises(ValueError, match="num_layers must be positive"):
        STGODEBaseline(1, 4, 1, num_layers=0)
    with pytest.raises(ValueError, match="integration_time must be positive"):
        STGODEBaseline(1, 4, 1, integration_time=0.0)
    with pytest.raises(ValueError, match="solver_rtol and solver_atol"):
        STGODEBaseline(1, 4, 1, solver_rtol=0.0)

    ode = _GraphODEFunc(2)
    with pytest.raises(RuntimeError, match="adjacency must be set"):
        ode.forward(torch.tensor(0.0), torch.zeros(1, 3, 2))

    pytest.importorskip("torchdiffeq")
    model = STGODEBaseline(1, 4, 1, history_len=2, num_layers=1)
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="history must have shape"):
        model.predict_next(torch.randn(2, 1), edge, None)


def test_stgode_mocked_odeint_fit_predict_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STGODE teaching paths run when ``odeint`` is a local Euler stand-in."""
    import koopman_graph.baselines.gnn.stgode as stgode_mod

    def _fake_odeint(
        func: Any,
        hidden0: Tensor,
        times: Tensor,
        **_kwargs: Any,
    ) -> Tensor:
        del times
        stepped = hidden0 + 0.01 * func(hidden0.new_zeros(()), hidden0)
        return torch.stack([hidden0, stepped])

    monkeypatch.setattr(stgode_mod, "_odeint", _fake_odeint)
    edge_index = _path_edges(4)
    weight = torch.ones(edge_index.shape[1])
    snapshots = [
        Data(x=torch.randn(4, 1), edge_index=edge_index, edge_weight=weight)
        for _ in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    model = STGODEBaseline(1, 4, 1, history_len=2, num_layers=1)
    model.fit(sequence, epochs=1, lr=1e-2, device="cpu")
    history = torch.stack([snap.x for snap in sequence[:2]])
    pred = model.predict_next(history, edge_index, weight)
    assert pred.shape == (4, 1)
    batched = model.predict_next(history.unsqueeze(0), edge_index, weight)
    assert batched.shape == (1, 4, 1)
    with pytest.raises(ValueError, match="history must have shape"):
        model.predict_next(history[0], edge_index)
    ode = _GraphODEFunc(3)
    with pytest.raises(RuntimeError, match="adjacency must be set"):
        ode(torch.tensor(0.0), torch.randn(1, 4, 3))
