"""Kuramoto–Sivashinsky PDE benchmark for Koopman stress tests.

Integrates the 1D KS equation with a spectral ETDRK4 scheme on a path or
ring discretization, complementing Laplacian-diffusion benchmarks.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import make_generator
from koopman_graph.datasets.topology import (
    path_edge_index,
    ring_edge_index,
)

KSTopologyName = Literal["path", "ring"]

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


class KuramotoSivashinskyBenchmark:
    """1D Kuramoto–Sivashinsky PDE on a path or ring discretization.

    Integrates the periodic KS equation with a spectral ETDRK4 scheme
    (Kassam & Trefethen), the classic chaotic stress test used in EDMD
    dictionary-learning literature.

    Notes
    -----
    The PDE integration is always periodic; ``topology`` only selects the
    message-passing graph overlay (path vs ring).
    """

    IN_CHANNELS = IN_CHANNELS_SCALAR

    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int = 64,
        num_timesteps: int = 400,
        domain_length: float = 22.0,
        dt: float = 0.25,
        topology: KSTopologyName = "ring",
        burn_in: int = 100,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a Kuramoto–Sivashinsky snapshot sequence.

        Parameters
        ----------
        num_nodes : int, optional
            Spatial grid size. Default is ``64``.
        num_timesteps : int, optional
            Number of stored snapshots after burn-in. Default is ``400``.
        domain_length : float, optional
            Periodic domain length ``L``. Default is ``22.0``.
        dt : float, optional
            Integration step. Default is ``0.25``.
        topology : {"path", "ring"}, optional
            Graph topology overlay. Integration is always periodic; ``path``
            only changes the message-passing graph. Default is ``"ring"``.
        burn_in : int, optional
            Discarded transient steps. Default is ``100``.
        seed : int, optional
            RNG seed for the initial field (``None`` = unseeded).
        dtype : torch.dtype, optional
            Feature dtype. Default is ``torch.float32``.

        Returns
        -------
        GraphSnapshotSequence
            Snapshots with shape ``(T, N, 1)``.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        _validate_positive_int("num_nodes", num_nodes, minimum=8)
        _validate_positive_int("num_timesteps", num_timesteps)
        if burn_in < 0:
            msg = f"burn_in must be >= 0, got {burn_in}"
            raise ValueError(msg)
        if dt <= 0.0 or domain_length <= 0.0:
            msg = (
                f"dt and domain_length must be > 0, got dt={dt}, "
                f"domain_length={domain_length}"
            )
            raise ValueError(msg)
        if topology not in ("path", "ring"):
            msg = f"Unsupported topology {topology!r}; expected 'path' or 'ring'"
            raise ValueError(msg)

        n = num_nodes
        generator = make_generator(seed)
        x = torch.linspace(0.0, domain_length, n + 1, dtype=torch.float64)[:-1]
        if generator is None:
            u = torch.cos(2.0 * torch.pi * x / domain_length) * (
                1.0 + 0.1 * torch.sin(x)
            )
        else:
            noise = 0.01 * torch.randn(n, dtype=torch.float64, generator=generator)
            u = torch.cos(2.0 * torch.pi * x / domain_length) + noise

        k = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(n, d=domain_length / n, dtype=torch.float64)
        )
        linear = k**2 - k**4
        e_dt = torch.exp(dt * linear)
        e_dt2 = torch.exp(dt * linear / 2.0)

        # Contour integrals for ETDRK4 coefficients (Kassam–Trefethen).
        m = 32
        r = torch.exp(
            1j * torch.pi * (torch.arange(1, m + 1, dtype=torch.float64) - 0.5) / m
        )
        lr = dt * linear.unsqueeze(1) + r.unsqueeze(0)
        q = dt * torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=1).real
        f1 = (
            dt
            * torch.mean(
                (-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3,
                dim=1,
            ).real
        )
        f2 = (
            dt
            * torch.mean(
                (2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr**3,
                dim=1,
            ).real
        )
        f3 = (
            dt
            * torch.mean(
                (-4.0 - 3.0 * lr - lr**2 + torch.exp(lr) * (4.0 - lr)) / lr**3,
                dim=1,
            ).real
        )

        def nonlinear(field: Tensor) -> Tensor:
            """Evaluate the KS nonlinear term in physical space.

            Parameters
            ----------
            field : Tensor
                Real spatial field with shape ``(num_nodes,)``.

            Returns
            -------
            Tensor
                Nonlinear contribution ``-0.5 * ∂_x (u^2)``.
            """
            return -0.5 * torch.fft.ifft(1j * k * torch.fft.fft(field**2)).real

        def etdrk4_step(field: Tensor) -> Tensor:
            """Advance one ETDRK4 step of the KS equation.

            Parameters
            ----------
            field : Tensor
                Real spatial field with shape ``(num_nodes,)``.

            Returns
            -------
            Tensor
                Updated real field.
            """
            u_hat = torch.fft.fft(field)
            nv = torch.fft.fft(nonlinear(field))
            a = e_dt2 * u_hat + q * nv
            na = torch.fft.fft(nonlinear(torch.fft.ifft(a).real))
            b = e_dt2 * u_hat + q * na
            nb = torch.fft.fft(nonlinear(torch.fft.ifft(b).real))
            c = e_dt2 * a + q * (2.0 * nb - nv)
            nc = torch.fft.fft(nonlinear(torch.fft.ifft(c).real))
            u_next = e_dt * u_hat + nv * f1 + 2.0 * (na + nb) * f2 + nc * f3
            return torch.fft.ifft(u_next).real

        for _ in range(burn_in):
            u = etdrk4_step(u)

        frames: list[Tensor] = []
        for _ in range(num_timesteps):
            frames.append(u.to(dtype=dtype).unsqueeze(-1))
            u = etdrk4_step(u)

        features = torch.stack(frames, dim=0)
        edges = (
            ring_edge_index(num_nodes)
            if topology == "ring"
            else path_edge_index(num_nodes)
        )
        return GraphSnapshotSequence.from_arrays(features, edges, dtype=dtype)
