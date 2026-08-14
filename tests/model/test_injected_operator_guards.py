"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

from typing import Any

import pytest
from torch import nn

from koopman_graph.model.factory import (
    resolve_injected_koopman,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    GlobalLocalKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)

_TYPES = ("a", "b")

_EDGE_TYPES = (("a", "r", "b"),)

_LATENT_DIMS = {"a": 2, "b": 3}


def _resolve_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "latent_dim": 2,
        "control_dim": 0,
        "control_mode": "additive",
        "bilinear_rank": None,
        "dynamics_mode": "discrete",
        "koopman_init_mode": "identity_noise",
        "koopman_init_scale": 0.01,
        "koopman_parameterization": "dense",
        "koopman_max_spectral_radius": 1.0,
        "koopman_auxiliary_hidden_dims": None,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("operator", "mode", "message"),
    [
        (HypergraphKoopmanOperator(2), "stochastic", "does not support hypergraph"),
        (HypergraphKoopmanOperator(2), "continuous", "requires dynamics_mode"),
        (HeteroGraphKoopmanOperator(2, 1), "continuous", "discrete.*stochastic"),
        (
            HeteroGraphKoopmanOperator(
                2,
                1,
                node_types=_TYPES,
                edge_types=_EDGE_TYPES,
                latent_dims=_LATENT_DIMS,
            ),
            "stochastic",
            "rectangular",
        ),
        (
            ContinuousHeteroGraphKoopmanOperator(2, 1),
            "discrete",
            "requires dynamics_mode",
        ),
        (GlobalLocalKoopmanOperator(2), "stochastic", "global_local"),
        (GlobalLocalKoopmanOperator(2), "continuous", "requires dynamics_mode"),
        (ContinuousGraphKoopmanOperator(2), "discrete", "requires dynamics_mode"),
    ],
)
def test_injected_operator_dynamics_guards(
    operator: nn.Module, mode: str, message: str
) -> None:
    """Injected operator families enforce their supported dynamics modes."""
    with pytest.raises(ValueError, match=message):
        resolve_injected_koopman(
            operator,  # type: ignore[arg-type]
            **_resolve_kwargs(dynamics_mode=mode),
        )
