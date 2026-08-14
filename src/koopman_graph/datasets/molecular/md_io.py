"""Lazy ``[md]`` boundary for mdtraj trajectory I/O.

Importing this module does **not** import mdtraj. Call
:func:`require_mdtraj` (or :func:`load_md_trajectory`) at use sites so core
``import koopman_graph`` stays free of the optional extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

_MDTRAJ_INSTALL_HINT = (
    "mdtraj is required for molecular trajectory I/O. "
    "Install with: pip install 'koopman-graph[md]'"
)


def require_mdtraj() -> ModuleType:
    """Import ``mdtraj`` or raise a guided ``ImportError``.

    Returns
    -------
    module
        The ``mdtraj`` package.
    """
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise ImportError(_MDTRAJ_INSTALL_HINT) from exc
    return mdtraj


@dataclass(frozen=True)
class MDTrajectory:
    """In-repo MD coordinates after an mdtraj load.

    Attributes
    ----------
    xyz : Tensor
        Coordinates in nanometres with shape ``(T, n_atoms, 3)``.
    n_atoms : int
        Atom count.
    """

    xyz: Tensor
    n_atoms: int


def load_md_trajectory(path: str | Path, **kwargs: Any) -> MDTrajectory:
    """Load an MD trajectory via mdtraj into tensors.

    Parameters
    ----------
    path : str or Path
        Trajectory file.
    **kwargs
        Forwarded to ``mdtraj.load``.

    Returns
    -------
    MDTrajectory
        Coordinates in nanometres.
    """
    mdtraj = require_mdtraj()
    traj = mdtraj.load(str(path), **kwargs)
    xyz = torch.from_numpy(traj.xyz.copy()).to(dtype=torch.float32)
    return MDTrajectory(xyz=xyz, n_atoms=int(xyz.shape[1]))


def alanine_dipeptide_card() -> dict[str, Any]:
    """Return the packaged alanine-dipeptide teaching card.

    CI uses this JSON metadata rather than downloading Folding@home-scale
    trajectories.

    Returns
    -------
    dict of str to object
        Packaged teaching-card fields.
    """
    package = resources.files("koopman_graph.datasets.molecular").joinpath("data")
    payload = package.joinpath("alanine_dipeptide_v1.json").read_text(encoding="utf-8")
    return json.loads(payload)
