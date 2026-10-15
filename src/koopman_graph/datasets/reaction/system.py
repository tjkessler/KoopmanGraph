"""Mass-action reaction systems, Arrhenius rate laws, and element balance.

A :class:`ReactionSystem` stores reactant and product stoichiometry as
``(num_reactions, num_species)`` tensors. Concentrations evolve under
mass-action kinetics

.. math::

    \\frac{dc}{dt} = N r(c), \\qquad
    r_j(c) = k_j \\prod_s c_s^{\\alpha_{js}},

where :math:`N = (\\beta - \\alpha)^\\top` is the ``(num_species,
num_reactions)`` stoichiometric matrix, :math:`\\alpha` / :math:`\\beta`
are the reactant / product coefficients, and :math:`k_j` the rate
constants. An optional ``(num_elements, num_species)`` element matrix
records composition; :func:`check_element_balance` reports reactions that
do not conserve it (open-system source / sink reactions are expected to).

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
"""Molar gas constant ``R`` in J/(mol K) (exact since the 2019 SI redefinition)."""


@dataclass(frozen=True)
class ArrheniusParams:
    """Modified Arrhenius rate law ``k(T) = A T^b exp(-E_a / (R T))``.

    Attributes
    ----------
    pre_exponential : float
        Pre-exponential factor ``A`` (> 0), in the reaction's rate units.
    activation_energy : float
        Activation energy ``E_a`` in J/mol (finite; may be negative for
        barrierless reactions).
    temperature_exponent : float
        Temperature exponent ``b``. Default is ``0.0`` (plain Arrhenius).
    """

    pre_exponential: float
    activation_energy: float
    temperature_exponent: float = 0.0

    def __post_init__(self) -> None:
        """Validate that all parameters are finite and ``A > 0``.

        Raises
        ------
        ValueError
            If any parameter is non-finite or ``pre_exponential <= 0``.
        """
        a = _finite_float(self.pre_exponential, "pre_exponential")
        if a <= 0.0:
            msg = f"pre_exponential must be > 0, got {self.pre_exponential!r}"
            raise ValueError(msg)
        object.__setattr__(self, "pre_exponential", a)
        object.__setattr__(
            self,
            "activation_energy",
            _finite_float(self.activation_energy, "activation_energy"),
        )
        object.__setattr__(
            self,
            "temperature_exponent",
            _finite_float(self.temperature_exponent, "temperature_exponent"),
        )

    def rate_constant(self, temperature_k: float | Tensor) -> Tensor:
        """Evaluate ``k(T)`` at one or more temperatures.

        Parameters
        ----------
        temperature_k : float or Tensor
            Absolute temperature(s) in kelvin (finite, > 0).

        Returns
        -------
        Tensor
            Rate constant(s) with the shape of ``temperature_k``
            (``float64`` unless a floating tensor is passed).

        Raises
        ------
        ValueError
            If any temperature is non-finite or not positive.
        """
        temperature = _validate_temperature(temperature_k)
        return (
            self.pre_exponential
            * temperature.pow(self.temperature_exponent)
            * torch.exp(
                -self.activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature)
            )
        )


@dataclass(frozen=True)
class ElementBalanceReport:
    """Per-reaction element-balance residuals of a :class:`ReactionSystem`.

    Attributes
    ----------
    residual : Tensor
        ``element_matrix @ stoichiometric_matrix`` with shape
        ``(num_elements, num_reactions)``; entry ``(e, j)`` is the net
        change in element ``e`` per unit extent of reaction ``j``.
    elements : tuple of str
        Element names, one per residual row.
    unbalanced_reactions : tuple of int
        Indices of reactions with any ``|residual| > atol``.
    atol : float
        Absolute tolerance used for the check.
    """

    residual: Tensor
    elements: tuple[str, ...]
    unbalanced_reactions: tuple[int, ...]
    atol: float

    @property
    def balanced(self) -> bool:
        """Return whether every reaction conserves every element.

        Returns
        -------
        bool
            ``True`` when ``unbalanced_reactions`` is empty.
        """
        return not self.unbalanced_reactions

    @property
    def max_abs_residual(self) -> float:
        """Return the largest absolute residual over elements and reactions.

        Returns
        -------
        float
            ``max |residual|`` (``0.0`` for an empty system).
        """
        if self.residual.numel() == 0:
            return 0.0
        return float(self.residual.abs().max().item())


@dataclass(frozen=True, eq=False)
class ReactionSystem:
    """Mass-action reaction network with optional composition and Arrhenius data.

    Tensors are validated and stored as owned ``float64`` copies; the
    instance is immutable. Use :meth:`with_rate_constants` or
    :meth:`at_temperature` to derive a system with new rate constants.

    Attributes
    ----------
    species : tuple of str
        Unique, non-empty species names (length ``num_species``).
    reactant_stoichiometry : Tensor
        Non-negative integer-valued reactant coefficients ``α`` with shape
        ``(num_reactions, num_species)``. Row ``j`` also gives the
        mass-action reaction orders of reaction ``j``.
    product_stoichiometry : Tensor
        Non-negative integer-valued product coefficients ``β`` with shape
        ``(num_reactions, num_species)``.
    rate_constants : Tensor
        Finite, non-negative rate constants ``k`` with shape
        ``(num_reactions,)``.
    element_matrix : Tensor or None, optional
        Non-negative integer-valued composition with shape
        ``(num_elements, num_species)``: entry ``(e, s)`` is the count of
        element ``e`` in species ``s``. Requires ``elements``.
    elements : tuple of str or None, optional
        Unique element names (length ``num_elements``). Required exactly
        when ``element_matrix`` is given.
    reaction_names : tuple of str or None, optional
        Optional labels, one per reaction.
    arrhenius : tuple of ArrheniusParams or None, optional
        Optional per-reaction rate laws used by :meth:`at_temperature`.
    name : str, optional
        Human-readable system name. Default is ``"reaction_system"``.
    """

    species: tuple[str, ...]
    reactant_stoichiometry: Tensor
    product_stoichiometry: Tensor
    rate_constants: Tensor
    element_matrix: Tensor | None = None
    elements: tuple[str, ...] | None = None
    reaction_names: tuple[str, ...] | None = None
    arrhenius: tuple[ArrheniusParams, ...] | None = None
    name: str = "reaction_system"

    def __post_init__(self) -> None:
        """Validate shapes, values, and names; store owned ``float64`` copies.

        Raises
        ------
        TypeError
            If a tensor field is not a Tensor or an Arrhenius entry has the
            wrong type.
        ValueError
            If shapes, values, or names are inconsistent.
        """
        species = _validate_names(self.species, "species")
        if not species:
            msg = "species must contain at least one name"
            raise ValueError(msg)
        num_species = len(species)

        reactants = _integer_matrix(
            self.reactant_stoichiometry, "reactant_stoichiometry", num_species
        )
        products = _integer_matrix(
            self.product_stoichiometry, "product_stoichiometry", num_species
        )
        if reactants.shape != products.shape:
            msg = (
                "reactant_stoichiometry and product_stoichiometry must have the "
                f"same shape, got {tuple(reactants.shape)} and "
                f"{tuple(products.shape)}"
            )
            raise ValueError(msg)
        num_reactions = int(reactants.shape[0])
        if num_reactions == 0:
            msg = "a reaction system needs at least one reaction"
            raise ValueError(msg)
        if bool(torch.all(reactants == products, dim=1).any()):
            index = int(torch.all(reactants == products, dim=1).nonzero()[0].item())
            msg = f"reaction {index} has identical reactants and products"
            raise ValueError(msg)

        rate_constants = _rate_constant_vector(self.rate_constants, num_reactions)

        element_matrix: Tensor | None = None
        elements: tuple[str, ...] | None = None
        if (self.element_matrix is None) != (self.elements is None):
            msg = "element_matrix and elements must be given together"
            raise ValueError(msg)
        if self.element_matrix is not None and self.elements is not None:
            elements = _validate_names(self.elements, "elements")
            element_matrix = _integer_matrix(
                self.element_matrix, "element_matrix", num_species
            )
            if int(element_matrix.shape[0]) != len(elements):
                msg = (
                    f"element_matrix has {int(element_matrix.shape[0])} rows but "
                    f"{len(elements)} elements were named"
                )
                raise ValueError(msg)

        reaction_names: tuple[str, ...] | None = None
        if self.reaction_names is not None:
            reaction_names = _validate_names(
                self.reaction_names, "reaction_names", unique=False
            )
            if len(reaction_names) != num_reactions:
                msg = (
                    f"reaction_names has {len(reaction_names)} entries for "
                    f"{num_reactions} reactions"
                )
                raise ValueError(msg)

        arrhenius: tuple[ArrheniusParams, ...] | None = None
        if self.arrhenius is not None:
            arrhenius = tuple(self.arrhenius)
            if len(arrhenius) != num_reactions:
                msg = (
                    f"arrhenius has {len(arrhenius)} entries for "
                    f"{num_reactions} reactions"
                )
                raise ValueError(msg)
            for index, params in enumerate(arrhenius):
                if not isinstance(params, ArrheniusParams):
                    msg = (
                        f"arrhenius[{index}] must be ArrheniusParams, got "
                        f"{type(params).__name__}"
                    )
                    raise TypeError(msg)

        if not isinstance(self.name, str) or not self.name:
            msg = f"name must be a non-empty string, got {self.name!r}"
            raise ValueError(msg)

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "reactant_stoichiometry", reactants)
        object.__setattr__(self, "product_stoichiometry", products)
        object.__setattr__(self, "rate_constants", rate_constants)
        object.__setattr__(self, "element_matrix", element_matrix)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "reaction_names", reaction_names)
        object.__setattr__(self, "arrhenius", arrhenius)

    @property
    def num_species(self) -> int:
        """Return the number of species.

        Returns
        -------
        int
            ``len(species)``.
        """
        return len(self.species)

    @property
    def num_reactions(self) -> int:
        """Return the number of reactions.

        Returns
        -------
        int
            Row count of the stoichiometry tensors.
        """
        return int(self.reactant_stoichiometry.shape[0])

    @property
    def stoichiometric_matrix(self) -> Tensor:
        """Return the net stoichiometric matrix ``N = (β - α)^T``.

        Returns
        -------
        Tensor
            ``float64`` tensor with shape ``(num_species, num_reactions)``.
        """
        return (self.product_stoichiometry - self.reactant_stoichiometry).T

    @property
    def reaction_orders(self) -> Tensor:
        """Return the total mass-action order of each reaction.

        Returns
        -------
        Tensor
            ``float64`` tensor ``α.sum(dim=1)`` with shape
            ``(num_reactions,)``.
        """
        return self.reactant_stoichiometry.sum(dim=1)

    @property
    def is_linear(self) -> bool:
        """Return whether the rate equations are affine in concentration.

        Returns
        -------
        bool
            ``True`` when every reaction has total order at most one
            (zeroth-order sources and first-order conversions only).
        """
        return bool((self.reaction_orders <= 1).all())

    def reaction_rates(
        self,
        concentrations: Tensor,
        *,
        rate_constants: Tensor | None = None,
    ) -> Tensor:
        """Evaluate mass-action reaction rates ``r_j = k_j ∏_s c_s^{α_js}``.

        Parameters
        ----------
        concentrations : Tensor
            Floating tensor with shape ``(..., num_species)``.
        rate_constants : Tensor or None, optional
            Override rate constants with shape ``(num_reactions,)`` or
            ``(..., num_reactions)``. Default uses :attr:`rate_constants`.

        Returns
        -------
        Tensor
            Rates with shape ``(..., num_reactions)`` in the dtype and on the
            device of ``concentrations``.

        Raises
        ------
        TypeError
            If ``concentrations`` is not a floating Tensor.
        ValueError
            If the trailing dimension is not ``num_species``.
        """
        c = self._validate_concentrations(concentrations)
        alpha = self.reactant_stoichiometry.to(dtype=c.dtype, device=c.device)
        # c^α with 0^0 = 1, so species absent from a reaction never zero it.
        monomials = torch.pow(c.unsqueeze(-2), alpha).prod(dim=-1)
        k = self.rate_constants if rate_constants is None else rate_constants
        return k.to(dtype=c.dtype, device=c.device) * monomials

    def time_derivative(
        self,
        concentrations: Tensor,
        *,
        rate_constants: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the mass-action right-hand side ``dc/dt = N r(c)``.

        Parameters
        ----------
        concentrations : Tensor
            Floating tensor with shape ``(..., num_species)``.
        rate_constants : Tensor or None, optional
            Override rate constants; see :meth:`reaction_rates`.

        Returns
        -------
        Tensor
            Time derivative with the shape of ``concentrations``.
        """
        rates = self.reaction_rates(concentrations, rate_constants=rate_constants)
        n = self.stoichiometric_matrix.to(dtype=rates.dtype, device=rates.device)
        return rates @ n.T

    def rate_constants_at(self, temperature_k: float) -> Tensor:
        """Evaluate the Arrhenius rate constants at one temperature.

        Parameters
        ----------
        temperature_k : float
            Absolute temperature in kelvin (finite, > 0).

        Returns
        -------
        Tensor
            ``float64`` rate constants with shape ``(num_reactions,)``.

        Raises
        ------
        ValueError
            If the system has no Arrhenius parameters or the temperature is
            invalid.
        """
        if self.arrhenius is None:
            msg = f"{self.name!r} has no Arrhenius parameters"
            raise ValueError(msg)
        temperature = _finite_float(temperature_k, "temperature_k")
        return torch.stack(
            [params.rate_constant(temperature) for params in self.arrhenius]
        ).to(dtype=torch.float64)

    def at_temperature(self, temperature_k: float) -> ReactionSystem:
        """Return a copy whose rate constants are evaluated at ``temperature_k``.

        Parameters
        ----------
        temperature_k : float
            Absolute temperature in kelvin (finite, > 0).

        Returns
        -------
        ReactionSystem
            New system with ``rate_constants = rate_constants_at(T)``.
        """
        return self.with_rate_constants(self.rate_constants_at(temperature_k))

    def with_rate_constants(self, rate_constants: Tensor) -> ReactionSystem:
        """Return a copy with replaced rate constants.

        Parameters
        ----------
        rate_constants : Tensor
            Finite, non-negative tensor with shape ``(num_reactions,)``.

        Returns
        -------
        ReactionSystem
            New validated system; other fields are shared (they are
            immutable owned copies).
        """
        return replace(self, rate_constants=rate_constants)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system to JSON-compatible primitives.

        Returns
        -------
        dict
            Names, stoichiometry, rate constants, and optional composition /
            Arrhenius data as nested lists and floats.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "species": list(self.species),
            "reactant_stoichiometry": self.reactant_stoichiometry.tolist(),
            "product_stoichiometry": self.product_stoichiometry.tolist(),
            "rate_constants": self.rate_constants.tolist(),
        }
        if self.element_matrix is not None and self.elements is not None:
            payload["elements"] = list(self.elements)
            payload["element_matrix"] = self.element_matrix.tolist()
        if self.reaction_names is not None:
            payload["reaction_names"] = list(self.reaction_names)
        if self.arrhenius is not None:
            payload["arrhenius"] = [
                {
                    "pre_exponential": params.pre_exponential,
                    "activation_energy": params.activation_energy,
                    "temperature_exponent": params.temperature_exponent,
                }
                for params in self.arrhenius
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReactionSystem:
        """Rebuild a system from :meth:`to_dict` output.

        Parameters
        ----------
        payload : dict
            Mapping with at least ``species``, ``reactant_stoichiometry``,
            ``product_stoichiometry``, and ``rate_constants``.

        Returns
        -------
        ReactionSystem
            Validated system.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        element_matrix = payload.get("element_matrix")
        elements = payload.get("elements")
        reaction_names = payload.get("reaction_names")
        arrhenius = payload.get("arrhenius")
        return cls(
            species=tuple(payload["species"]),
            reactant_stoichiometry=torch.tensor(
                payload["reactant_stoichiometry"], dtype=torch.float64
            ),
            product_stoichiometry=torch.tensor(
                payload["product_stoichiometry"], dtype=torch.float64
            ),
            rate_constants=torch.tensor(payload["rate_constants"], dtype=torch.float64),
            element_matrix=(
                None
                if element_matrix is None
                else torch.tensor(element_matrix, dtype=torch.float64)
            ),
            elements=None if elements is None else tuple(elements),
            reaction_names=None if reaction_names is None else tuple(reaction_names),
            arrhenius=(
                None
                if arrhenius is None
                else tuple(ArrheniusParams(**params) for params in arrhenius)
            ),
            name=str(payload.get("name", "reaction_system")),
        )

    def _validate_concentrations(self, concentrations: Tensor) -> Tensor:
        """Check a concentration tensor's type and trailing dimension.

        Parameters
        ----------
        concentrations
            See signature.

        Returns
        -------
            See signature."""
        if not isinstance(concentrations, Tensor):
            msg = (
                f"concentrations must be a Tensor, got {type(concentrations).__name__}"
            )
            raise TypeError(msg)
        if not concentrations.is_floating_point():
            msg = f"concentrations must be floating point, got {concentrations.dtype}"
            raise TypeError(msg)
        if concentrations.ndim == 0 or concentrations.shape[-1] != self.num_species:
            msg = (
                f"concentrations must have shape (..., {self.num_species}), got "
                f"{tuple(concentrations.shape)}"
            )
            raise ValueError(msg)
        return concentrations


def check_element_balance(
    system: ReactionSystem,
    *,
    atol: float = 1e-9,
) -> ElementBalanceReport:
    """Report which reactions fail to conserve the system's elements.

    Parameters
    ----------
    system : ReactionSystem
        System with an ``element_matrix``.
    atol : float, optional
        Absolute tolerance on each residual entry. Default is ``1e-9``.

    Returns
    -------
    ElementBalanceReport
        Residual matrix and the indices of unbalanced reactions. Open-system
        source / sink reactions (e.g. ``∅ → A``) are reported as unbalanced
        by design; whether that is acceptable is the caller's decision.

    Raises
    ------
    ValueError
        If ``system`` has no element matrix or ``atol`` is invalid.
    """
    if system.element_matrix is None or system.elements is None:
        msg = f"{system.name!r} has no element_matrix to check"
        raise ValueError(msg)
    tol = _finite_float(atol, "atol")
    if tol < 0.0:
        msg = f"atol must be >= 0, got {atol!r}"
        raise ValueError(msg)
    residual = system.element_matrix @ system.stoichiometric_matrix
    bad = (residual.abs() > tol).any(dim=0)
    unbalanced = tuple(int(i) for i in bad.nonzero().flatten().tolist())
    return ElementBalanceReport(
        residual=residual,
        elements=system.elements,
        unbalanced_reactions=unbalanced,
        atol=tol,
    )


def _finite_float(value: Any, name: str) -> float:
    """Coerce ``value`` to a finite float or raise ``ValueError``.

    Parameters
    ----------
    value
        See signature.
    name
        See signature.

    Returns
    -------
        See signature."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a real number, got {value!r}"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    return result


def _validate_temperature(temperature_k: float | Tensor) -> Tensor:
    """Return temperatures as a floating tensor, checking finite and > 0.

    Parameters
    ----------
    temperature_k
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(temperature_k, Tensor):
        temperature = (
            temperature_k
            if temperature_k.is_floating_point()
            else temperature_k.to(dtype=torch.float64)
        )
    else:
        temperature = torch.tensor(
            _finite_float(temperature_k, "temperature_k"), dtype=torch.float64
        )
    if not bool(torch.isfinite(temperature).all()) or bool((temperature <= 0).any()):
        msg = "temperature_k must be finite and > 0 kelvin"
        raise ValueError(msg)
    return temperature


def _validate_names(
    names: Any,
    field: str,
    *,
    unique: bool = True,
) -> tuple[str, ...]:
    """Return ``names`` as a tuple of non-empty strings.

    Parameters
    ----------
    names
        See signature.
    field
        See signature.
    unique
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(names, str):
        msg = f"{field} must be a sequence of strings, not a single string"
        raise TypeError(msg)
    result = tuple(names)
    for name in result:
        if not isinstance(name, str) or not name:
            msg = f"{field} entries must be non-empty strings, got {name!r}"
            raise ValueError(msg)
    if unique and len(set(result)) != len(result):
        msg = f"{field} must be unique, got {list(result)}"
        raise ValueError(msg)
    return result


def _integer_matrix(value: Any, field: str, num_columns: int) -> Tensor:
    """Validate a non-negative integer-valued ``(rows, num_columns)`` matrix.

    Parameters
    ----------
    value
        See signature.
    field
        See signature.
    num_columns
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(value, Tensor):
        msg = f"{field} must be a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    matrix = value.detach().to(dtype=torch.float64, device="cpu").clone()
    if matrix.ndim != 2 or int(matrix.shape[1]) != num_columns:
        msg = f"{field} must have shape (rows, {num_columns}), got {tuple(value.shape)}"
        raise ValueError(msg)
    if not bool(torch.isfinite(matrix).all()):
        msg = f"{field} must be finite"
        raise ValueError(msg)
    if bool((matrix < 0).any()) or not bool((matrix == matrix.round()).all()):
        msg = f"{field} must contain non-negative integers"
        raise ValueError(msg)
    return matrix


def _rate_constant_vector(value: Any, num_reactions: int) -> Tensor:
    """Validate finite, non-negative rate constants of length ``num_reactions``.

    Parameters
    ----------
    value
        See signature.
    num_reactions
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(value, Tensor):
        msg = f"rate_constants must be a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    k = value.detach().to(dtype=torch.float64, device="cpu").clone()
    if k.shape != (num_reactions,):
        msg = (
            f"rate_constants must have shape ({num_reactions},), got "
            f"{tuple(value.shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(k).all()) or bool((k < 0).any()):
        msg = "rate_constants must be finite and >= 0"
        raise ValueError(msg)
    return k
