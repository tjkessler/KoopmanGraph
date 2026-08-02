"""Molecular / MD dataset helpers (optional ``[md]`` toolchain).

Capability layout
-----------------
``contact_graph``
    Unit-aware contact ``edge_index`` builders from coordinates in
    nanometres (no mdtraj required).
``synthetic``
    Seeded two-state Markov teaching trajectory and package-relative FAIR
    metadata
    (:func:`~koopman_graph.datasets.molecular.generate_synthetic_two_state`).
``md_io``
    Lazy ``[md]`` boundary for mdtraj-backed trajectory I/O stubs.
    Importing this package does **not** import mdtraj.

Positioning: diagnostic / teaching path — not Folding@home scale and not a
PyEMMA replacement.
"""

from koopman_graph.datasets.molecular.contact_graph import (
    ContactGranularity,
    contact_edge_index,
)
from koopman_graph.datasets.molecular.md_io import load_md_trajectory, require_mdtraj
from koopman_graph.datasets.molecular.synthetic import (
    SyntheticTwoStateTrajectory,
    generate_synthetic_two_state,
    load_synthetic_two_state_metadata,
    synthetic_two_state_oracle_timescale,
    synthetic_two_state_slow_eigenvalue,
)

__all__ = [
    "ContactGranularity",
    "SyntheticTwoStateTrajectory",
    "contact_edge_index",
    "generate_synthetic_two_state",
    "load_md_trajectory",
    "load_synthetic_two_state_metadata",
    "require_mdtraj",
    "synthetic_two_state_oracle_timescale",
    "synthetic_two_state_slow_eigenvalue",
]
