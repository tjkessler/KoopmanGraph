"""Lazy ``[md]`` boundary for mdtraj trajectory I/O stubs.

Importing this module does **not** import mdtraj. Call
:func:`require_mdtraj` (or :func:`load_md_trajectory`) at use sites so core
``import koopman_graph`` stays free of the optional extra.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

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

    Raises
    ------
    ImportError
        If mdtraj is not installed (``pip install 'koopman-graph[md]'``).
    """
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise ImportError(_MDTRAJ_INSTALL_HINT) from exc
    return mdtraj


def load_md_trajectory(*_args: Any, **_kwargs: Any) -> Any:
    """Load an MD trajectory via mdtraj into an in-repo representation.

    Stub until molecular loaders ship; signature mirrors the future mdtraj-backed
    API (positional and keyword arguments forwarded to the loader).

    Parameters
    ----------
    *_args
        Forwarded trajectory path / format arguments (not yet implemented).
    **_kwargs
        Forwarded mdtraj loader keyword arguments (not yet implemented).

    Returns
    -------
    Any
        Planned in-repo trajectory container (not implemented).

    Raises
    ------
    ImportError
        If the ``[md]`` extra is not installed.
    NotImplementedError
        Always, until contact-graph construction and molecular loaders ship.
    """
    require_mdtraj()
    msg = (
        "load_md_trajectory is not implemented yet; "
        "install 'koopman-graph[md]' for trajectory I/O when loaders ship"
    )
    raise NotImplementedError(msg)
