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

from koopman_graph.datasets.reaction.cantera_io import (
    CANTERA_CITATION_DOI,
    CANTERA_ILLUSTRATION_NOTE,
    DEFAULT_MECHANISM,
    CanteraMechanism,
    CanteraProvenance,
    CanteraReactorTrajectory,
    StateDependentPolicy,
    load_cantera_mechanism,
    mechanism_sha256,
    require_cantera,
    resolve_mechanism_path,
    simulate_cantera_reactor,
)
from koopman_graph.datasets.reaction.catalog import (
    DEFAULT_BRUSSELATOR_PARAMS,
    DEFAULT_CHAIN_RATE,
    DEFAULT_CHAIN_RATIO,
    DEFAULT_DIMERIZATION_RATES,
    DEFAULT_LOTKA_VOLTERRA_RATES,
    DEFAULT_MICHAELIS_MENTEN_RATES,
    brusselator,
    first_order_chain,
    lotka_volterra,
    michaelis_menten,
    reversible_dimerization,
)
from koopman_graph.datasets.reaction.cstr import (
    VAN_DE_VUSSE_FEED_A,
    VAN_DE_VUSSE_NOMINAL_DILUTION,
    VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K,
    VanDeVusseCSTR,
    van_de_vusse_cstr,
)
from koopman_graph.datasets.reaction.gillespie import (
    DEFAULT_MAX_EVENTS,
    SSATrajectory,
    bin_to_grid,
    gillespie_ssa,
    propensities,
)
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
from koopman_graph.datasets.reaction.oracles import (
    OracleSpectrum,
    conservation_laws,
    equilibrium,
    oracle_spectrum,
    oracle_timescales,
    stoichiometric_rank,
)
from koopman_graph.datasets.reaction.random_network import (
    SampledReactionNetwork,
    sample_random_network,
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
    "CANTERA_CITATION_DOI",
    "CANTERA_ILLUSTRATION_NOTE",
    "DEFAULT_BRUSSELATOR_PARAMS",
    "DEFAULT_CHAIN_RATE",
    "DEFAULT_CHAIN_RATIO",
    "DEFAULT_DIMERIZATION_RATES",
    "DEFAULT_LOG_FLOOR",
    "DEFAULT_LOTKA_VOLTERRA_RATES",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_MECHANISM",
    "DEFAULT_MICHAELIS_MENTEN_RATES",
    "GAS_CONSTANT_J_PER_MOL_K",
    "PRODUCT_RELATION",
    "REACTANT_RELATION",
    "REACTION_NODE",
    "SPECIES_NODE",
    "VAN_DE_VUSSE_FEED_A",
    "VAN_DE_VUSSE_NOMINAL_DILUTION",
    "VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K",
    "ArrheniusParams",
    "CanteraMechanism",
    "CanteraProvenance",
    "CanteraReactorTrajectory",
    "ElementBalanceReport",
    "IntegrationMethod",
    "LogConcentrationTransform",
    "OracleSpectrum",
    "ReactionSystem",
    "SSATrajectory",
    "SampledReactionNetwork",
    "StateDependentPolicy",
    "VanDeVusseCSTR",
    "bin_to_grid",
    "brusselator",
    "check_element_balance",
    "conservation_laws",
    "element_balance_violation",
    "equilibrium",
    "first_order_chain",
    "gillespie_ssa",
    "integrate_expm",
    "integrate_reaction_system",
    "integrate_rk4",
    "integrate_stiff",
    "linear_rate_equations",
    "load_cantera_mechanism",
    "lotka_volterra",
    "mass_action_jacobian",
    "mechanism_sha256",
    "michaelis_menten",
    "oracle_spectrum",
    "oracle_timescales",
    "propensities",
    "reaction_directed_hyperedges",
    "reaction_hyperedge_index",
    "require_cantera",
    "require_scipy",
    "resolve_mechanism_path",
    "reversible_dimerization",
    "sample_random_network",
    "simulate_cantera_reactor",
    "species_influence_graph",
    "species_reaction_bipartite",
    "stoichiometric_rank",
    "van_de_vusse_cstr",
]
