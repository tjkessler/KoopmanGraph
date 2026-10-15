"""Mass-action reaction-network dataset helpers (optional ``[reaction]`` toolchain).

Capability layout
-----------------
``system``
    ``ReactionSystem``, ``ArrheniusParams``, and element-balance checks.
``graphs``
    Species-graph, hyperedge, and bipartite species-reaction encodings
    derived from stoichiometry.
``transform``
    Log-concentration transform with a recorded floor, its inverse, and
    element-balance violation.
``integrate``
    Exact matrix-exponential propagation for linear systems, torch RK4 for
    non-stiff systems, and SciPy stiff solvers behind a lazy import.
``catalog``
    Small reference systems (first-order chains, Michaelis-Menten,
    oscillators) with frozen defaults.
``random_network``
    Seeded sampler for random mass-action networks with a
    composition-consistent element matrix.
``cstr``
    Van de Vusse CSTR with feed and temperature inputs.
``gillespie``
    Seeded stochastic simulation sampler and grid binning.
``cantera_io``
    Lazy ``[cantera]`` boundary for Cantera mechanism loading and reactor
    simulation.
``benchmark``
    ``ReactionNetworkBenchmark`` (generate, split, payload I/O, cards).
``oracles``
    Equilibrium, Jacobian, oracle timescales, and spectrum.

Importing this package does **not** import scipy or cantera.

Positioning: diagnostic / teaching path for checking whether a learned latent
operator recovers slow kinetic timescales and respects conservation — not a
replacement for a production kinetics solver.
"""

from koopman_graph.datasets.reaction.graphs import (
    PRODUCT_RELATION,
    REACTANT_RELATION,
    REACTION_NODE,
    SPECIES_NODE,
    reaction_directed_hyperedges,
    reaction_hyperedge_index,
    species_influence_graph,
    species_reaction_bipartite,
)
from koopman_graph.datasets.reaction.integrate import (
    IntegrationMethod,
    integrate_expm,
    integrate_reaction_system,
    integrate_rk4,
    integrate_stiff,
    linear_rate_equations,
    mass_action_jacobian,
    require_scipy,
)
from koopman_graph.datasets.reaction.system import (
    GAS_CONSTANT_J_PER_MOL_K,
    ArrheniusParams,
    ElementBalanceReport,
    ReactionSystem,
    check_element_balance,
)
from koopman_graph.datasets.reaction.transform import (
    DEFAULT_LOG_FLOOR,
    LogConcentrationTransform,
    element_balance_violation,
)

__all__ = [
    "DEFAULT_LOG_FLOOR",
    "GAS_CONSTANT_J_PER_MOL_K",
    "PRODUCT_RELATION",
    "REACTANT_RELATION",
    "REACTION_NODE",
    "SPECIES_NODE",
    "ArrheniusParams",
    "ElementBalanceReport",
    "IntegrationMethod",
    "LogConcentrationTransform",
    "ReactionSystem",
    "check_element_balance",
    "element_balance_violation",
    "integrate_expm",
    "integrate_reaction_system",
    "integrate_rk4",
    "integrate_stiff",
    "linear_rate_equations",
    "mass_action_jacobian",
    "reaction_directed_hyperedges",
    "reaction_hyperedge_index",
    "require_scipy",
    "species_influence_graph",
    "species_reaction_bipartite",
]
