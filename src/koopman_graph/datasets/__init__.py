"""Dataset utilities for KoopmanGraph benchmarks and examples.

Public benchmarks
-----------------
SyntheticDynamicGraphBenchmark
    Reproducible Laplacian-diffusion dynamics on path/ring topologies.
GridDynamicGraphBenchmark
    Laplacian diffusion on a 4-connected 2D lattice graph.
AnisotropicAdvectionGridBenchmark
    Directional advection on a grid with asymmetric neighbor weights.
IEEE118DynamicBenchmark
    IEEE 118-bus MATPOWER topology with simulated voltage/load dynamics.
MetrLaTrafficBenchmark
    METR-LA traffic-speed sensor graph with cached speed snapshots.
"""

from koopman_graph.datasets.grid import (
    AnisotropicAdvectionGridBenchmark,
    GridDynamicGraphBenchmark,
)
from koopman_graph.datasets.ieee118 import IEEE118DynamicBenchmark
from koopman_graph.datasets.metr_la import MetrLaTrafficBenchmark
from koopman_graph.datasets.synthetic import SyntheticDynamicGraphBenchmark

__all__ = [
    "AnisotropicAdvectionGridBenchmark",
    "GridDynamicGraphBenchmark",
    "IEEE118DynamicBenchmark",
    "MetrLaTrafficBenchmark",
    "SyntheticDynamicGraphBenchmark",
]
