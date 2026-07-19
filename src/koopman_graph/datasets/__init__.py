"""Public dataset package exports for KoopmanGraph benchmarks.

Factory idioms
--------------
Simulated benchmarks expose ``ClassName.generate(...)`` and return a
:class:`~koopman_graph.data.GraphSnapshotSequence` built from Laplacian (or
related) dynamics on a fixed topology. Real telemetry and cached teaching
benchmarks expose ``load_topology`` / ``load_sequence`` (or equivalent cache
loaders) because the graph and time series are downloaded or precomputed
artifacts rather than synthesized in process.

Prefer the classmethod entry points on the public benchmark classes below.
Module-level helpers in individual dataset modules are implementation details
or download-script shims unless documented otherwise.

Seed defaults
-------------
All simulated ``generate`` methods default ``seed=None`` (unseeded RNG). Pass an
explicit seed for reproducible runs; tutorials and docs use ``seed=42``.

Generation validators
---------------------
Laplacian-diffusion generators share
``validate_diffusion_generation_params`` (``decay_rate > 0``). Anisotropic
advection uses ``validate_advection_decay_rate`` for the stricter
``decay_rate ∈ (0, 1)`` self-retention interval. See
:mod:`koopman_graph.datasets.dynamics`.

Topology payloads
-----------------
``load_topology`` returns a frozen :class:`TopologyPayload` (attribute access
preferred; mapping-style ``payload["edge_index"]`` remains supported).

Public benchmarks
-----------------
SyntheticDynamicGraphBenchmark
    Reproducible Laplacian-diffusion dynamics on path/ring topologies.
GridDynamicGraphBenchmark
    Laplacian diffusion on a 4-connected 2D lattice graph.
AnisotropicAdvectionGridBenchmark
    Directional advection on a grid with asymmetric neighbor weights.
IEEE118DynamicBenchmark
    IEEE 118-bus MATPOWER topology with simulated voltage/load dynamics
    (``generate``).
MetrLaTrafficBenchmark
    METR-LA traffic-speed sensor graph with cached speed snapshots
    (``load_topology`` / ``load_sequence``).
EpidemicNetworkBenchmark
    Networked SIR epidemic on ring / small-world / custom contact graphs.
Lorenz96GraphBenchmark
    Lorenz-96 chaotic ODE on a ring graph.
KuramotoSivashinskyBenchmark
    1D Kuramoto–Sivashinsky PDE on a path/ring discretization.
CylinderWakeBenchmark
    Cached Hopf/Stuart–Landau cylinder-wake teaching surrogate
    (``load_topology`` / ``load_sequence``).
"""

from koopman_graph.datasets.cylinder_wake import CylinderWakeBenchmark
from koopman_graph.datasets.epidemic import EpidemicNetworkBenchmark
from koopman_graph.datasets.grid import (
    AnisotropicAdvectionGridBenchmark,
    GridDynamicGraphBenchmark,
)
from koopman_graph.datasets.ieee118 import IEEE118DynamicBenchmark
from koopman_graph.datasets.kuramoto_sivashinsky import KuramotoSivashinskyBenchmark
from koopman_graph.datasets.lorenz96 import Lorenz96GraphBenchmark
from koopman_graph.datasets.metr_la import MetrLaTrafficBenchmark
from koopman_graph.datasets.synthetic import SyntheticDynamicGraphBenchmark
from koopman_graph.datasets.topology import TopologyPayload

__all__ = [
    "AnisotropicAdvectionGridBenchmark",
    "CylinderWakeBenchmark",
    "EpidemicNetworkBenchmark",
    "GridDynamicGraphBenchmark",
    "IEEE118DynamicBenchmark",
    "KuramotoSivashinskyBenchmark",
    "Lorenz96GraphBenchmark",
    "MetrLaTrafficBenchmark",
    "SyntheticDynamicGraphBenchmark",
    "TopologyPayload",
]
