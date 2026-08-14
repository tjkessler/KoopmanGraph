"""Coverage and error-path tests for :mod:`koopman_graph.mpc`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from koopman_graph import GraphSnapshotSequence
from koopman_graph.mpc.controller import _validate_mpc_model
from koopman_graph.operators import (
    ContinuousHeteroGraphKoopmanOperator,
)
from koopman_graph.training.objectives import compute_eigenvalue_regularization_loss


def test_mpc_and_continuous_hetero_objective_guards() -> None:
    """MPC and continuous hetero regularization reject incompatible surfaces."""
    model = SimpleNamespace(koopman=nn.Linear(2, 2), dynamics_mode="continuous")
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        _validate_mpc_model(model)  # type: ignore[arg-type]

    operator = ContinuousHeteroGraphKoopmanOperator(2, 1, parameterization="dense")
    continuous = SimpleNamespace(koopman=operator, dynamics_mode="continuous")
    with pytest.raises(ValueError, match="sequence is required"):
        compute_eigenvalue_regularization_loss(continuous, None)  # type: ignore[arg-type]
    sequence = GraphSnapshotSequence(
        [Data(x=torch.zeros(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long))]
    )
    with pytest.raises(ValueError, match="HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(continuous, sequence)  # type: ignore[arg-type]
