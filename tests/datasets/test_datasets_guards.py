"""Coverage and error-path tests for :mod:`koopman_graph.datasets`."""

from __future__ import annotations

import pytest

from koopman_graph.datasets import (
    EpidemicNetworkBenchmark,
    KuramotoSivashinskyBenchmark,
    Lorenz96GraphBenchmark,
)


def test_dataset_validation_gap_branches() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        Lorenz96GraphBenchmark.generate(num_nodes=3, num_timesteps=5, burn_in=0)
    with pytest.raises(ValueError, match="dt"):
        Lorenz96GraphBenchmark.generate(num_nodes=4, num_timesteps=5, dt=0.0, burn_in=0)
    with pytest.raises(ValueError, match="burn_in"):
        Lorenz96GraphBenchmark.generate(num_nodes=4, num_timesteps=5, burn_in=-1)
    with pytest.raises(ValueError, match="num_nodes"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=3,
            num_timesteps=5,
            burn_in=0,
        )
    with pytest.raises(ValueError, match="beta"):
        EpidemicNetworkBenchmark.generate(beta=-0.1)
