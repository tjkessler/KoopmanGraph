"""Small reference reaction systems with frozen defaults.

Each constructor returns a validated
:class:`~koopman_graph.datasets.reaction.ReactionSystem`. The default
parameters are part of the dataset contract: changing them changes every
dataset generated from the catalog, so they are module constants. Closed
systems carry an element matrix; open systems (sources, sinks, or a
reservoir held constant) do not, and :func:`check_element_balance` is not
meaningful for them.

* :func:`first_order_chain`: closed, linear ``S0 → S1 → … → S(n-1)``.
* :func:`reversible_dimerization`: closed, nonlinear ``2 A ⇌ A2``.
* :func:`michaelis_menten`: closed ``E + S ⇌ ES → E + P``.
* :func:`brusselator`: open; limit cycle when ``b > 1 + a²``.
* :func:`lotka_volterra`: open; neutral centre (purely imaginary pair).

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math

import torch

from koopman_graph.datasets.reaction.system import ReactionSystem

DEFAULT_CHAIN_RATE = 1.0
"""Rate of the first step of :func:`first_order_chain`."""

DEFAULT_CHAIN_RATIO = 0.5
"""Geometric ratio between successive :func:`first_order_chain` rates."""

DEFAULT_DIMERIZATION_RATES = (3.0, 0.5)
"""Default ``(k_forward, k_reverse)`` for :func:`reversible_dimerization`."""

DEFAULT_MICHAELIS_MENTEN_RATES = (10.0, 1.0, 0.5)
"""Default ``(k_on, k_off, k_cat)`` for :func:`michaelis_menten`."""

DEFAULT_BRUSSELATOR_PARAMS = (1.0, 3.0)
"""Default ``(a, b)`` for :func:`brusselator` (oscillatory: ``3 > 1 + 1``)."""

DEFAULT_LOTKA_VOLTERRA_RATES = (1.0, 1.0, 1.0)
"""Default ``(alpha, beta, gamma)`` for :func:`lotka_volterra`."""

_MAX_CHAIN_SPECIES = 64


def first_order_chain(
    num_species: int = 3,
    *,
    rate_constants: tuple[float, ...] | None = None,
) -> ReactionSystem:
    """Return the linear chain ``S0 → S1 → … → S(n-1)``.

    Parameters
    ----------
    num_species : int, optional
        Chain length ``n`` in ``[2, 64]``. Default is ``3``.
    rate_constants : tuple of float or None, optional
        ``n - 1`` step rates. Default is the geometric sequence
        ``DEFAULT_CHAIN_RATE * DEFAULT_CHAIN_RATIO**i`` (``1.0, 0.5, …``),
        which keeps every Jacobian eigenvalue distinct.

    Returns
    -------
    ReactionSystem
        Closed, linear system with one element ``M`` (total mass). The
        Jacobian eigenvalues are ``-k_i`` and one conservation zero.

    Raises
    ------
    ValueError
        If ``num_species`` is out of range, the rate count is wrong, or a
        rate is not finite and positive.
    """
    if (
        isinstance(num_species, bool)
        or not isinstance(num_species, int)
        or not 2 <= num_species <= _MAX_CHAIN_SPECIES
    ):
        msg = (
            f"num_species must be an int in [2, {_MAX_CHAIN_SPECIES}], got "
            f"{num_species!r}"
        )
        raise ValueError(msg)
    num_reactions = num_species - 1
    if rate_constants is None:
        rates = tuple(
            DEFAULT_CHAIN_RATE * DEFAULT_CHAIN_RATIO**i for i in range(num_reactions)
        )
    else:
        rates = tuple(float(k) for k in rate_constants)
        if len(rates) != num_reactions:
            msg = (
                f"rate_constants must have {num_reactions} entries for "
                f"{num_species} species, got {len(rates)}"
            )
            raise ValueError(msg)
        _require_positive(
            *((f"rate_constants[{i}]", k) for i, k in enumerate(rate_constants))
        )
    reactants = torch.zeros(num_reactions, num_species, dtype=torch.float64)
    products = torch.zeros(num_reactions, num_species, dtype=torch.float64)
    index = torch.arange(num_reactions)
    reactants[index, index] = 1.0
    products[index, index + 1] = 1.0
    species = tuple(f"S{i}" for i in range(num_species))
    return ReactionSystem(
        species=species,
        reactant_stoichiometry=reactants,
        product_stoichiometry=products,
        rate_constants=torch.tensor(rates, dtype=torch.float64),
        element_matrix=torch.ones(1, num_species, dtype=torch.float64),
        elements=("M",),
        reaction_names=tuple(
            f"{species[i]} -> {species[i + 1]}" for i in range(num_reactions)
        ),
        name=f"first_order_chain_{num_species}",
    )


def reversible_dimerization(
    k_forward: float = DEFAULT_DIMERIZATION_RATES[0],
    k_reverse: float = DEFAULT_DIMERIZATION_RATES[1],
) -> ReactionSystem:
    """Return ``2 A ⇌ A2``.

    Parameters
    ----------
    k_forward : float, optional
        Rate constant of ``2 A → A2``. Default is ``3.0``.
    k_reverse : float, optional
        Rate constant of ``A2 → 2 A``. Default is ``0.5``.

    Returns
    -------
    ReactionSystem
        Closed, nonlinear system; element ``X`` counts monomer units
        (``A: 1``, ``A2: 2``). Equilibrium satisfies
        ``k_forward A² = k_reverse A2``.

    Raises
    ------
    ValueError
        If a rate constant is not finite and positive.
    """
    _require_positive(("k_forward", k_forward), ("k_reverse", k_reverse))
    return ReactionSystem(
        species=("A", "A2"),
        reactant_stoichiometry=torch.tensor(
            [[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64
        ),
        rate_constants=torch.tensor([k_forward, k_reverse], dtype=torch.float64),
        element_matrix=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        elements=("X",),
        reaction_names=("2 A -> A2", "A2 -> 2 A"),
        name="reversible_dimerization",
    )


def michaelis_menten(
    k_on: float = DEFAULT_MICHAELIS_MENTEN_RATES[0],
    k_off: float = DEFAULT_MICHAELIS_MENTEN_RATES[1],
    k_cat: float = DEFAULT_MICHAELIS_MENTEN_RATES[2],
) -> ReactionSystem:
    """Return the mass-action Michaelis-Menten mechanism ``E + S ⇌ ES → E + P``.

    Parameters
    ----------
    k_on : float, optional
        Binding rate constant. Default is ``10.0``.
    k_off : float, optional
        Unbinding rate constant. Default is ``1.0``.
    k_cat : float, optional
        Catalytic rate constant. Default is ``0.5``.

    Returns
    -------
    ReactionSystem
        Closed system over species ``(E, S, ES, P)`` with elements ``Enz``
        (enzyme units) and ``Sub`` (substrate backbone). All substrate ends
        up as ``P``, a boundary equilibrium with ``S = ES = 0``.

    Raises
    ------
    ValueError
        If a rate constant is not finite and positive.
    """
    _require_positive(("k_on", k_on), ("k_off", k_off), ("k_cat", k_cat))
    return ReactionSystem(
        species=("E", "S", "ES", "P"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        rate_constants=torch.tensor([k_on, k_off, k_cat], dtype=torch.float64),
        element_matrix=torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]], dtype=torch.float64
        ),
        elements=("Enz", "Sub"),
        reaction_names=("E + S -> ES", "ES -> E + S", "ES -> E + P"),
        name="michaelis_menten",
    )


def brusselator(
    a: float = DEFAULT_BRUSSELATOR_PARAMS[0],
    b: float = DEFAULT_BRUSSELATOR_PARAMS[1],
) -> ReactionSystem:
    """Return the Brusselator with reservoir species folded into rates.

    Parameters
    ----------
    a : float, optional
        Feed rate of ``∅ → X`` (reservoir ``A`` held constant). Default is
        ``1.0``.
    b : float, optional
        Rate of ``X → Y`` (reservoir ``B`` held constant). Default is
        ``3.0``.

    Returns
    -------
    ReactionSystem
        Open system over ``(X, Y)``: ``∅ → X`` (``a``), ``X → Y`` (``b``),
        ``2 X + Y → 3 X`` (``1``), ``X → ∅`` (``1``). The unique fixed point
        ``(a, b / a)`` is unstable, with a limit cycle around it, when
        ``b > 1 + a²``.

    Raises
    ------
    ValueError
        If ``a`` or ``b`` is not finite and positive.
    """
    _require_positive(("a", a), ("b", b))
    return ReactionSystem(
        species=("X", "Y"),
        reactant_stoichiometry=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [1.0, 0.0]], dtype=torch.float64
        ),
        product_stoichiometry=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        ),
        rate_constants=torch.tensor([a, b, 1.0, 1.0], dtype=torch.float64),
        reaction_names=("-> X", "X -> Y", "2 X + Y -> 3 X", "X ->"),
        name="brusselator",
    )


def lotka_volterra(
    alpha: float = DEFAULT_LOTKA_VOLTERRA_RATES[0],
    beta: float = DEFAULT_LOTKA_VOLTERRA_RATES[1],
    gamma: float = DEFAULT_LOTKA_VOLTERRA_RATES[2],
) -> ReactionSystem:
    """Return the mass-action Lotka-Volterra predator-prey network.

    Parameters
    ----------
    alpha : float, optional
        Prey birth ``X → 2 X``. Default is ``1.0``.
    beta : float, optional
        Predation ``X + Y → 2 Y``. Default is ``1.0``.
    gamma : float, optional
        Predator death ``Y → ∅``. Default is ``1.0``.

    Returns
    -------
    ReactionSystem
        Open system over ``(X, Y)``. The interior fixed point
        ``(gamma / beta, alpha / beta)`` is a neutral centre with Jacobian
        eigenvalues ``± i sqrt(alpha gamma)``, so trajectories are closed
        orbits and the oracle timescale is infinite.

    Raises
    ------
    ValueError
        If a rate constant is not finite and positive.
    """
    _require_positive(("alpha", alpha), ("beta", beta), ("gamma", gamma))
    return ReactionSystem(
        species=("X", "Y"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        ),
        product_stoichiometry=torch.tensor(
            [[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=torch.float64
        ),
        rate_constants=torch.tensor([alpha, beta, gamma], dtype=torch.float64),
        reaction_names=("X -> 2 X", "X + Y -> 2 Y", "Y ->"),
        name="lotka_volterra",
    )


def _require_positive(*params: tuple[str, float]) -> None:
    """Raise unless every ``(name, value)`` is finite and positive.

    Parameters
    ----------
    *params
        See signature.

    Raises
    ------
        See signature."""
    for name, value in params:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            msg = f"{name} must be finite and > 0, got {value!r}"
            raise ValueError(msg)
