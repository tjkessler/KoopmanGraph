"""Contact-graph construction from Cartesian coordinates (teaching MD path).

Canonical length unit
---------------------
All positions and cutoffs are interpreted in **nanometres** (nm). Parameter
names carry the unit suffix (``positions_nm``, ``cutoff_nm``). Callers that
hold ångströms must convert (``1 Å = 0.1 nm``) before calling.

Granularity
-----------
* ``\"atom\"`` (default) — one node per coordinate row; an undirected edge
  exists when the Euclidean distance is at most ``cutoff_nm``.
* ``\"residue\"`` — one node per unique residue id; an undirected edge exists
  when **any** atom pair belonging to the two residues lies within
  ``cutoff_nm``.

Edge orientation matches the package ``B_1`` convention: one column per
undirected contact with ``i < j`` stored as ``(i → j)`` (tail ``i``, head
``j``). Columns are sorted by ``(i, j)`` for deterministic fixtures.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

ContactGranularity = Literal["atom", "residue"]

# Soft upper bound to catch unit mistakes (e.g. passing ångströms as nm).
_MAX_REASONABLE_CUTOFF_NM = 5.0


def contact_edge_index(
    positions_nm: Tensor,
    cutoff_nm: float,
    *,
    granularity: ContactGranularity = "atom",
    residue_ids: Tensor | None = None,
) -> Tensor:
    """Build an oriented contact ``edge_index`` from coordinates in nanometres.

    Parameters
    ----------
    positions_nm : Tensor
        Cartesian coordinates with shape ``(num_atoms, 3)`` in **nanometres**.
    cutoff_nm : float
        Inclusive contact distance in **nanometres**. Must be finite and
        strictly positive; values above ``5.0`` nm are rejected as likely
        unit errors for biomolecular contacts.
    granularity : {"atom", "residue"}, optional
        ``\"atom\"`` (default) connects atom nodes. ``\"residue\"`` connects
        residue nodes when any inter-residue atom pair is within cutoff
        (requires ``residue_ids``).
    residue_ids : Tensor or None, optional
        Integer residue label per atom with shape ``(num_atoms,)``. Required
        when ``granularity=\"residue\"``.

    Returns
    -------
    Tensor
        Oriented edges ``(2, num_edges)`` with one column per undirected
        contact, ``i < j``, sorted by ``(i, j)``. Empty ``(2, 0)`` when no
        contacts exist.

    Raises
    ------
    ValueError
        If shapes, cutoff, granularity, or residue ids are invalid.
    """
    cutoff = _validate_cutoff_nm(cutoff_nm)
    positions = _validate_positions_nm(positions_nm)
    kind = _validate_granularity(granularity)

    if kind == "atom":
        if residue_ids is not None:
            msg = "residue_ids must be None when granularity='atom'"
            raise ValueError(msg)
        return _atom_contact_edges(positions, cutoff)

    ids = _validate_residue_ids(residue_ids, num_atoms=int(positions.shape[0]))
    return _residue_contact_edges(positions, ids, cutoff)


def _validate_cutoff_nm(cutoff_nm: float) -> float:
    """Validate a contact cutoff expressed in nanometres.

    Parameters
    ----------
    cutoff_nm : float
        Contact distance in nanometres.

    Returns
    -------
    float
        Validated cutoff value.

    Raises
    ------
    ValueError
        If the cutoff is non-finite, non-positive, or unreasonably large.
    """
    try:
        cutoff = float(cutoff_nm)
    except (TypeError, ValueError) as exc:
        msg = f"cutoff_nm must be a real number in nanometres, got {cutoff_nm!r}"
        raise ValueError(msg) from exc
    if not (cutoff > 0.0) or cutoff != cutoff:  # NaN check via !=
        msg = f"cutoff_nm must be finite and > 0 nm, got {cutoff_nm!r}"
        raise ValueError(msg)
    if cutoff > _MAX_REASONABLE_CUTOFF_NM:
        msg = (
            f"cutoff_nm={cutoff} nm exceeds {_MAX_REASONABLE_CUTOFF_NM} nm; "
            f"biomolecular contacts are typically ≤ 1.5 nm — check units "
            f"(ångströms must be converted: 1 Å = 0.1 nm)"
        )
        raise ValueError(msg)
    return cutoff


def _validate_positions_nm(positions_nm: Tensor) -> Tensor:
    """Validate coordinate tensor shape and finiteness.

    Parameters
    ----------
    positions_nm
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(positions_nm, Tensor):
        msg = f"positions_nm must be a Tensor, got {type(positions_nm).__name__}"
        raise TypeError(msg)
    if positions_nm.ndim != 2 or positions_nm.shape[1] != 3:
        msg = (
            "positions_nm must have shape (num_atoms, 3) in nanometres, "
            f"got {tuple(positions_nm.shape)}"
        )
        raise ValueError(msg)
    if int(positions_nm.shape[0]) < 1:
        msg = "positions_nm must contain at least one atom"
        raise ValueError(msg)
    if not bool(torch.isfinite(positions_nm).all()):
        msg = "positions_nm must be finite (nanometres)"
        raise ValueError(msg)
    return positions_nm.to(dtype=torch.float64)


def _validate_granularity(granularity: str) -> ContactGranularity:
    """Normalize the contact granularity string.

    Parameters
    ----------
    granularity
        See signature.

    Returns
    -------
        See signature."""
    if granularity not in ("atom", "residue"):
        msg = f"granularity must be 'atom' or 'residue', got {granularity!r}"
        raise ValueError(msg)
    return granularity  # type: ignore[return-value]


def _validate_residue_ids(residue_ids: Tensor | None, *, num_atoms: int) -> Tensor:
    """Validate per-atom residue labels for residue-granularity contacts.

    Parameters
    ----------
    residue_ids
        See signature.
    num_atoms
        See signature.

    Returns
    -------
        See signature."""
    if residue_ids is None:
        msg = "residue_ids is required when granularity='residue'"
        raise ValueError(msg)
    if not isinstance(residue_ids, Tensor):
        msg = f"residue_ids must be a Tensor, got {type(residue_ids).__name__}"
        raise TypeError(msg)
    if residue_ids.ndim != 1 or int(residue_ids.shape[0]) != num_atoms:
        msg = (
            f"residue_ids must have shape ({num_atoms},), "
            f"got {tuple(residue_ids.shape)}"
        )
        raise ValueError(msg)
    if residue_ids.dtype.is_floating_point or residue_ids.dtype == torch.bool:
        msg = f"residue_ids must be an integer tensor, got dtype {residue_ids.dtype}"
        raise ValueError(msg)
    return residue_ids.to(dtype=torch.long)


def _pairs_to_edge_index(pairs: list[tuple[int, int]]) -> Tensor:
    """Convert sorted undirected ``(i, j)`` pairs with ``i < j`` to edge_index.

    Parameters
    ----------
    pairs
        See signature.

    Returns
    -------
        See signature."""
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    ordered = sorted(set(pairs))
    src = torch.tensor([i for i, _ in ordered], dtype=torch.long)
    tgt = torch.tensor([j for _, j in ordered], dtype=torch.long)
    return torch.stack((src, tgt), dim=0)


def _atom_contact_edges(positions_nm: Tensor, cutoff_nm: float) -> Tensor:
    """Atom-node contacts from pairwise Euclidean distances (nm).

    Parameters
    ----------
    positions_nm
        See signature.
    cutoff_nm
        See signature.

    Returns
    -------
        See signature."""
    num_atoms = int(positions_nm.shape[0])
    # (N, N, 3) differences; diagonal ignored.
    delta = positions_nm.unsqueeze(0) - positions_nm.unsqueeze(1)
    dist = torch.linalg.vector_norm(delta, dim=-1)
    pairs: list[tuple[int, int]] = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if float(dist[i, j]) <= cutoff_nm:
                pairs.append((i, j))
    return _pairs_to_edge_index(pairs)


def _residue_contact_edges(
    positions_nm: Tensor,
    residue_ids: Tensor,
    cutoff_nm: float,
) -> Tensor:
    """Residue-node contacts when any inter-residue atom pair is within cutoff.

    Parameters
    ----------
    positions_nm
        See signature.
    residue_ids
        See signature.
    cutoff_nm
        See signature.

    Returns
    -------
        See signature."""
    unique_ids = torch.unique(residue_ids, sorted=True)
    id_list = [int(r) for r in unique_ids.tolist()]
    # Contiguous node index per sorted unique residue id.
    id_to_node = {rid: idx for idx, rid in enumerate(id_list)}
    atom_nodes = torch.tensor(
        [id_to_node[int(r)] for r in residue_ids.tolist()],
        dtype=torch.long,
        device=positions_nm.device,
    )

    num_atoms = int(positions_nm.shape[0])
    delta = positions_nm.unsqueeze(0) - positions_nm.unsqueeze(1)
    dist = torch.linalg.vector_norm(delta, dim=-1)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            ni = int(atom_nodes[i])
            nj = int(atom_nodes[j])
            if ni == nj:
                continue
            if float(dist[i, j]) > cutoff_nm:
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            key = (a, b)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return _pairs_to_edge_index(pairs)
