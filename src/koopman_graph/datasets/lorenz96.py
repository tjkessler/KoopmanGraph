"""Lorenz-96 chaotic ring-graph benchmark for Koopman stress tests.

Maps the Lorenz-96 ODE onto a cyclic ring topology for message-passing
stress tests that complement Laplacian-diffusion benchmarks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import make_generator
from koopman_graph.datasets.topology import ring_edge_index

IN_CHANNELS_SCALAR = 1


def _validate_positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    """Raise ``ValueError`` when ``value`` is below ``minimum``.

    Parameters
    ----------
    name : str
        Parameter name for the error message.
    value : int
        Candidate integer value.
    minimum : int, optional
        Inclusive lower bound. Default is ``1``.

    Raises
    ------
    ValueError
        If ``value < minimum``.
    """
    if value < minimum:
        msg = f"{name} must be >= {minimum}, got {value}"
        raise ValueError(msg)


class Lorenz96GraphBenchmark:
    """Lorenz-96 chaotic system mapped onto a ring graph.

    Each node stores one Lorenz-96 state variable. Nearest-neighbor coupling in
    the ODE matches the cyclic ring topology used for message passing.

    Notes
    -----
    Integration uses classical RK4 with an optional burn-in transient.
    """

    IN_CHANNELS = IN_CHANNELS_SCALAR

    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int = 40,
        num_timesteps: int = 500,
        forcing: float = 8.0,
        dt: float = 0.01,
        burn_in: int = 200,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a Lorenz-96 ring snapshot sequence via RK4.

        Parameters
        ----------
        num_nodes : int, optional
            System dimension / ring size. Default is ``40``.
        num_timesteps : int, optional
            Number of stored snapshots after burn-in. Default is ``500``.
        forcing : float, optional
            Lorenz-96 forcing ``F``. Default is ``8.0`` (chaotic regime).
        dt : float, optional
            Integration step. Default is ``0.01``.
        burn_in : int, optional
            Discarded transient steps. Default is ``200``.
        seed : int, optional
            RNG seed for the initial state (``None`` = unseeded).
        dtype : torch.dtype, optional
            Feature dtype. Default is ``torch.float32``.

        Returns
        -------
        GraphSnapshotSequence
            Snapshots with shape ``(T, N, 1)`` on a ring graph.

        Raises
        ------
        ValueError
            If sizes or ``dt`` are invalid.
        """
        _validate_positive_int("num_nodes", num_nodes, minimum=4)
        _validate_positive_int("num_timesteps", num_timesteps)
        if burn_in < 0:
            msg = f"burn_in must be >= 0, got {burn_in}"
            raise ValueError(msg)
        if dt <= 0.0:
            msg = f"dt must be > 0, got {dt}"
            raise ValueError(msg)

        generator = make_generator(seed)
        if generator is None:
            x = forcing * torch.ones(num_nodes, dtype=torch.float64)
            x = x + 0.1 * torch.randn(num_nodes, dtype=torch.float64)
        else:
            x = forcing * torch.ones(num_nodes, dtype=torch.float64)
            x = x + 0.1 * torch.randn(
                num_nodes, dtype=torch.float64, generator=generator
            )

        def rhs(state: Tensor) -> Tensor:
            """Evaluate the Lorenz-96 vector field.

            Parameters
            ----------
            state : Tensor
                Current state with shape ``(num_nodes,)``.

            Returns
            -------
            Tensor
                Time derivative with the same shape as ``state``.
            """
            xp1 = torch.roll(state, -1)
            xm1 = torch.roll(state, 1)
            xm2 = torch.roll(state, 2)
            return (xp1 - xm2) * xm1 - state + forcing

        def rk4_step(state: Tensor) -> Tensor:
            """Advance one RK4 step of Lorenz-96.

            Parameters
            ----------
            state : Tensor
                Current state with shape ``(num_nodes,)``.

            Returns
            -------
            Tensor
                Updated state.
            """
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        for _ in range(burn_in):
            x = rk4_step(x)

        frames: list[Tensor] = []
        for _ in range(num_timesteps):
            frames.append(x.to(dtype=dtype).unsqueeze(-1))
            x = rk4_step(x)

        features = torch.stack(frames, dim=0)
        return GraphSnapshotSequence.from_arrays(
            features, ring_edge_index(num_nodes), dtype=dtype
        )
