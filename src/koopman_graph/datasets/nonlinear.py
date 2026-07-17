"""Nonlinear and chaotic graph benchmarks for Koopman stress tests.

These generators complement the Laplacian-diffusion benchmarks with dynamics that
are intrinsically nonlinear on the graph: networked SIR epidemics, Lorenz-96 on a
ring, Kuramoto–Sivashinsky on a path/ring discretization, and a small cached
cylinder-wake Hopf surrogate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import make_generator
from koopman_graph.datasets.topology import TopologyPayload

EpidemicTopologyName = Literal["ring", "small_world", "custom"]
KSTopologyName = Literal["path", "ring"]

DEFAULT_WAKE_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cylinder_wake"
DEFAULT_WAKE_NUM_NODES = 72
DEFAULT_WAKE_NUM_TIMESTEPS = 120
IN_CHANNELS_SCALAR = 1
IN_CHANNELS_SIR = 3


def _path_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional path-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes on the path.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, E)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes - 1):
        src.extend([node, node + 1])
        dst.extend([node + 1, node])
    return torch.tensor([src, dst], dtype=torch.long)


def _ring_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional ring-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes on the ring.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, E)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        src.extend([node, nxt])
        dst.extend([nxt, node])
    return torch.tensor([src, dst], dtype=torch.long)


def _watts_strogatz_edge_index(
    num_nodes: int,
    *,
    k: int = 4,
    rewire_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Build an undirected Watts–Strogatz small-world graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    k : int, optional
        Even ring degree (each node links to ``k // 2`` neighbors on each side).
        Default is ``4``.
    rewire_prob : float, optional
        Probability of rewiring each directed half-edge. Default is ``0.1``.
    generator : torch.Generator or None, optional
        Optional RNG for deterministic rewiring.

    Returns
    -------
    Tensor
        Bidirectional edge index with shape ``(2, E)``.

    Raises
    ------
    ValueError
        If ``k`` is invalid for ``num_nodes``.
    """
    if num_nodes < 3:
        msg = f"small_world topology requires num_nodes >= 3, got {num_nodes}"
        raise ValueError(msg)
    if k < 2 or k % 2 != 0:
        msg = f"small_world k must be an even integer >= 2, got {k}"
        raise ValueError(msg)
    if k >= num_nodes:
        msg = f"small_world k must be < num_nodes, got k={k}, num_nodes={num_nodes}"
        raise ValueError(msg)
    if not 0.0 <= rewire_prob <= 1.0:
        msg = f"rewire_prob must be in [0, 1], got {rewire_prob}"
        raise ValueError(msg)

    half = k // 2
    undirected: set[tuple[int, int]] = set()
    for node in range(num_nodes):
        for offset in range(1, half + 1):
            nbr = (node + offset) % num_nodes
            undirected.add((min(node, nbr), max(node, nbr)))

    rewired: set[tuple[int, int]] = set()
    for src, dst in undirected:
        do_rewire = (
            torch.rand(1, generator=generator).item() < rewire_prob
            if rewire_prob > 0.0
            else False
        )
        if not do_rewire:
            rewired.add((src, dst))
            continue
        candidates = [
            j
            for j in range(num_nodes)
            if j != src and (min(src, j), max(src, j)) not in rewired
        ]
        if not candidates:
            rewired.add((src, dst))
            continue
        idx = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        new_dst = candidates[idx]
        rewired.add((min(src, new_dst), max(src, new_dst)))

    src_list: list[int] = []
    dst_list: list[int] = []
    for u, v in sorted(rewired):
        src_list.extend([u, v])
        dst_list.extend([v, u])
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def _row_normalized_adjacency(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Return a dense row-stochastic adjacency matrix.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index with shape ``(2, E)``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    Tensor
        Dense row-normalized adjacency with shape ``(N, N)``.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    row, col = edge_index
    adj[row, col] = 1.0
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


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


class EpidemicNetworkBenchmark:
    """Networked SIR epidemic dynamics on ring / small-world / custom graphs.

    Node features are susceptible / infected / recovered fractions
    ``(S, I, R)`` that sum to one. Infection uses a discrete-time force-of-
    infection update matching ``examples/06_epidemic_ring.ipynb`` (patient-zero
    seed at node 0 plus a small neighbor seed at node 1).

    Optional **contact-reduction** controls ``u_t`` in ``[0, 1]`` scale the infection
    term as ``beta * (1 - u_t) * S * force``, a control-affine / multiplicative
    intervention suitable for bilinear Koopman demos.

    Notes
    -----
    Use :meth:`generate` to sample trajectories. Topology choices are ring,
    Watts–Strogatz small-world, or a caller-supplied edge index.
    """

    IN_CHANNELS = IN_CHANNELS_SIR

    @staticmethod
    def default_intervention_schedule(
        num_timesteps: int,
        *,
        onset_fraction: float = 1.0 / 3.0,
        max_reduction: float = 0.7,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Build a mid-horizon contact-reduction schedule.

        ``u_t = 0`` until ``onset_fraction`` of the horizon, then linearly ramps
        to ``max_reduction`` and holds. Shape is ``(num_timesteps, 1)``.

        Parameters
        ----------
        num_timesteps : int
            Trajectory length ``T``.
        onset_fraction : float, optional
            Fraction of the horizon that remains uncontrolled. Default is
            ``1/3``.
        max_reduction : float, optional
            Peak contact reduction in ``[0, 1]``. Default is ``0.7``.
        dtype : torch.dtype, optional
            Output dtype. Default is ``torch.float32``.

        Returns
        -------
        Tensor
            Intervention controls with shape ``(num_timesteps, 1)``.
        """
        _validate_positive_int("num_timesteps", num_timesteps)
        if not 0.0 <= onset_fraction <= 1.0:
            msg = f"onset_fraction must be in [0, 1], got {onset_fraction}"
            raise ValueError(msg)
        if not 0.0 <= max_reduction <= 1.0:
            msg = f"max_reduction must be in [0, 1], got {max_reduction}"
            raise ValueError(msg)
        onset = int(onset_fraction * num_timesteps)
        schedule = torch.zeros(num_timesteps, 1, dtype=dtype)
        if onset >= num_timesteps:
            return schedule
        ramp_len = max(num_timesteps - onset, 1)
        for t in range(onset, num_timesteps):
            progress = (t - onset + 1) / ramp_len
            schedule[t, 0] = max_reduction * min(progress, 1.0)
        return schedule

    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int = 36,
        num_timesteps: int = 60,
        topology: EpidemicTopologyName = "ring",
        beta: float = 0.45,
        gamma: float = 0.12,
        edge_index: Tensor | None = None,
        small_world_k: int = 4,
        rewire_prob: float = 0.1,
        patient_zero: float = 0.08,
        neighbor_seed: float = 0.03,
        expose_intervention_control: bool = False,
        intervention: Tensor | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a networked SIR snapshot sequence.

        Parameters
        ----------
        num_nodes : int, optional
            Number of contact-graph nodes. Default is ``36``.
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``60``.
        topology : {"ring", "small_world", "custom"}, optional
            Contact topology. ``custom`` requires ``edge_index``. Default is
            ``"ring"``.
        beta : float, optional
            Infection rate. Default is ``0.45``.
        gamma : float, optional
            Recovery rate. Default is ``0.12``.
        edge_index : Tensor, optional
            Custom ``(2, E)`` edge index when ``topology="custom"``.
        small_world_k : int, optional
            Watts–Strogatz degree (even). Default is ``4``.
        rewire_prob : float, optional
            Watts–Strogatz rewiring probability. Default is ``0.1``.
        patient_zero : float, optional
            Initial infected fraction at node 0. Default is ``0.08``.
        neighbor_seed : float, optional
            Initial infected fraction at node 1 (when ``num_nodes > 1``).
            Default is ``0.03``.
        expose_intervention_control : bool, optional
            When ``True``, apply contact reduction ``u_t`` in the infection
            term and attach ``control_inputs`` with shape ``(T, 1)``. Default
            is ``False``.
        intervention : Tensor, optional
            Contact-reduction schedule with shape ``(T,)`` or ``(T, 1)`` and
            values in ``[0, 1]``. Used when
            ``expose_intervention_control=True``; when omitted, uses
            :meth:`default_intervention_schedule`.
        seed : int, optional
            RNG seed (``None`` = unseeded). Tutorials should pass an explicit
            seed.
        dtype : torch.dtype, optional
            Feature dtype. Default is ``torch.float32``.

        Returns
        -------
        GraphSnapshotSequence
            Snapshots with ``in_channels == 3`` (``S, I, R``). When
            interventions are exposed, ``control_dim == 1``.

        Raises
        ------
        ValueError
            If parameters are invalid, ``custom`` lacks ``edge_index``, or
            intervention controls are malformed.
        """
        _validate_positive_int("num_nodes", num_nodes)
        _validate_positive_int("num_timesteps", num_timesteps)
        if beta < 0.0 or gamma < 0.0:
            msg = f"beta and gamma must be >= 0, got beta={beta}, gamma={gamma}"
            raise ValueError(msg)
        if not 0.0 <= patient_zero <= 1.0 or not 0.0 <= neighbor_seed <= 1.0:
            msg = (
                "patient_zero and neighbor_seed must be in [0, 1], "
                f"got patient_zero={patient_zero}, neighbor_seed={neighbor_seed}"
            )
            raise ValueError(msg)
        if patient_zero + neighbor_seed > 1.0 + 1e-6:
            msg = (
                "patient_zero + neighbor_seed must be <= 1, "
                f"got {patient_zero + neighbor_seed}"
            )
            raise ValueError(msg)
        if intervention is not None and not expose_intervention_control:
            msg = "intervention requires expose_intervention_control=True"
            raise ValueError(msg)

        controls: Tensor | None = None
        if expose_intervention_control:
            if intervention is None:
                controls = cls.default_intervention_schedule(
                    num_timesteps,
                    dtype=dtype,
                )
            else:
                controls = intervention.to(dtype=dtype)
                if controls.ndim == 1:
                    controls = controls.unsqueeze(-1)
                if controls.shape != (num_timesteps, 1):
                    msg = (
                        "intervention must have shape (num_timesteps,) or "
                        f"(num_timesteps, 1), got {tuple(controls.shape)}"
                    )
                    raise ValueError(msg)
                if bool((controls < 0.0).any() or (controls > 1.0).any()):
                    msg = "intervention values must lie in [0, 1]"
                    raise ValueError(msg)

        generator = make_generator(seed)
        if topology == "ring":
            edges = _ring_edge_index(num_nodes)
        elif topology == "small_world":
            edges = _watts_strogatz_edge_index(
                num_nodes,
                k=small_world_k,
                rewire_prob=rewire_prob,
                generator=generator,
            )
        elif topology == "custom":
            if edge_index is None:
                msg = "topology='custom' requires edge_index"
                raise ValueError(msg)
            edges = edge_index.to(dtype=torch.long)
            inferred = int(edges.max().item()) + 1 if edges.numel() else 0
            if inferred > num_nodes:
                msg = (
                    f"edge_index references node {inferred - 1} but "
                    f"num_nodes={num_nodes}"
                )
                raise ValueError(msg)
        else:
            msg = (
                f"Unsupported topology {topology!r}; expected "
                "'ring', 'small_world', or 'custom'"
            )
            raise ValueError(msg)

        adj = _row_normalized_adjacency(edges, num_nodes)
        s = torch.ones(num_nodes, dtype=dtype)
        i = torch.zeros(num_nodes, dtype=dtype)
        r = torch.zeros(num_nodes, dtype=dtype)
        i[0] = float(patient_zero)
        if num_nodes > 1:
            i[1] = float(neighbor_seed)
        s = (1.0 - i - r).clamp(min=0.0)

        snapshots: list[Tensor] = []
        for step in range(num_timesteps):
            state = torch.stack([s, i, r], dim=1)
            snapshots.append(state.clone())
            if step == num_timesteps - 1:
                break
            neighbor_i = adj @ i
            contact_scale = 1.0
            if controls is not None:
                contact_scale = float(1.0 - controls[step, 0].item())
            infection = beta * contact_scale * s * neighbor_i
            recovered = gamma * i
            i = (i + infection - recovered).clamp(min=0.0)
            r = (r + recovered).clamp(min=0.0)
            s = (1.0 - i - r).clamp(min=0.0)

        features = torch.stack(snapshots, dim=0)
        return GraphSnapshotSequence.from_arrays(
            features,
            edges,
            control_inputs=controls,
            dtype=dtype,
        )


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
            features, _ring_edge_index(num_nodes), dtype=dtype
        )


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
            _ring_edge_index(num_nodes)
            if topology == "ring"
            else _path_edge_index(num_nodes)
        )
        return GraphSnapshotSequence.from_arrays(features, edges, dtype=dtype)


def _default_wake_path(cache_dir: Path | None = None) -> Path:
    """Return the default on-disk path for the cylinder-wake cache.

    Parameters
    ----------
    cache_dir : Path or None, optional
        Optional cache directory override.

    Returns
    -------
    Path
        Path to ``wake.pt``.
    """
    root = cache_dir if cache_dir is not None else DEFAULT_WAKE_CACHE_DIR
    return root / "wake.pt"


def _cylinder_wake_mesh(
    *,
    num_nodes: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Build a coarse unstructured wake mesh and k-NN edge index.

    Points fill a rectangular channel excluding a unit-diameter cylinder at the
    origin. Edges are symmetric k-nearest-neighbor links.

    Parameters
    ----------
    num_nodes : int
        Number of mesh nodes to keep.
    seed : int
        RNG seed for point sampling.

    Returns
    -------
    tuple of Tensor
        ``(coords, edge_index)`` with shapes ``(N, 2)`` and ``(2, E)``.
    """
    generator = torch.Generator().manual_seed(seed)
    # Oversample then reject interior points until we have num_nodes.
    coords_list: list[Tensor] = []
    radius = 0.5
    while len(coords_list) < num_nodes:
        batch = torch.rand(num_nodes * 2, 2, generator=generator)
        xs = -1.5 + batch[:, 0] * 10.0
        ys = -2.5 + batch[:, 1] * 5.0
        pts = torch.stack([xs, ys], dim=1)
        keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2) >= radius**2
        for row in pts[keep]:
            coords_list.append(row)
            if len(coords_list) >= num_nodes:
                break
    coords = torch.stack(coords_list[:num_nodes], dim=0)

    # Symmetric 6-NN graph.
    dists = torch.cdist(coords, coords)
    dists.fill_diagonal_(float("inf"))
    knn = torch.topk(dists, k=min(6, num_nodes - 1), largest=False).indices
    undirected: set[tuple[int, int]] = set()
    for i in range(num_nodes):
        for j in knn[i].tolist():
            undirected.add((min(i, j), max(i, j)))
    src: list[int] = []
    dst: list[int] = []
    for u, v in sorted(undirected):
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return coords, edge_index


def build_cylinder_wake_payload(
    *,
    num_nodes: int = DEFAULT_WAKE_NUM_NODES,
    num_timesteps: int = DEFAULT_WAKE_NUM_TIMESTEPS,
    dt: float = 0.15,
    omega: float = 0.8,
    mu: float = 0.15,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Assemble a small Hopf/POD cylinder-wake teaching cache.

    The field is a mean wake plus a complex Stuart–Landau oscillator modulating
    two spatial modes that approximate a von Kármán street. This is a
    reproducible reduced-order surrogate for tutorial and unit-test use, not
    Navier–Stokes DNS.

    Parameters
    ----------
    num_nodes : int, optional
        Mesh size. Default is ``72``.
    num_timesteps : int, optional
        Number of stored snapshots. Default is ``120``.
    dt : float, optional
        Temporal spacing of the oscillator. Default is ``0.15``.
    omega : float, optional
        Oscillation frequency. Default is ``0.8``.
    mu : float, optional
        Stuart–Landau growth parameter. Default is ``0.15``.
    seed : int, optional
        Mesh / phase seed. Default is ``0``.
    dtype : torch.dtype, optional
        Feature dtype. Default is ``torch.float32``.

    Returns
    -------
    dict
        Serializable cache payload for ``wake.pt``.
    """
    coords, edge_index = _cylinder_wake_mesh(num_nodes=num_nodes, seed=seed)
    x = coords[:, 0]
    y = coords[:, 1]
    # Mean wake deficit + two oscillatory modes (streamwise fluctuation).
    mean = -0.35 * torch.exp(-((y / 1.2) ** 2)) * torch.sigmoid(x)
    mode_r = torch.exp(-((y / 1.4) ** 2)) * torch.sin(0.9 * x) * torch.sigmoid(x + 0.5)
    mode_i = torch.exp(-((y / 1.4) ** 2)) * torch.cos(0.9 * x) * torch.sigmoid(x + 0.5)

    # Complex Stuart–Landau: ż = (μ + iω)z − |z|² z
    z = complex(0.05, 0.02)
    frames: list[Tensor] = []
    for _ in range(num_timesteps):
        amp_r = z.real
        amp_i = z.imag
        field = mean + amp_r * mode_r + amp_i * mode_i
        frames.append(field.to(dtype=dtype).unsqueeze(-1))
        mag2 = abs(z) ** 2
        dz = (mu + 1j * omega) * z - mag2 * z
        z = z + dt * dz

    features = torch.stack(frames, dim=0)
    return {
        "features": features,
        "edge_index": edge_index,
        "coords": coords.to(dtype=dtype),
        "num_nodes": int(num_nodes),
        "num_timesteps": int(num_timesteps),
        "dt": float(dt),
        "omega": float(omega),
        "mu": float(mu),
        "seed": int(seed),
        "description": (
            "Hopf/Stuart-Landau cylinder-wake surrogate on a coarse wake mesh "
            "(teaching cache; not DNS)."
        ),
    }


def ensure_wake_cache(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Build the cylinder-wake cache if missing.

    Parameters
    ----------
    cache_dir : Path, optional
        Cache directory. Defaults to ``data/cylinder_wake``.
    force : bool, optional
        Rebuild even when the cache exists. Default is ``False``.

    Returns
    -------
    Path
        Path to ``wake.pt``.
    """
    path = _default_wake_path(cache_dir)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_cylinder_wake_payload()
    torch.save(payload, path)
    return path


def load_wake_cache(
    cache_dir: Path | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    """Load the cached cylinder-wake payload, building it if needed.

    Parameters
    ----------
    cache_dir : Path, optional
        Cache directory. Defaults to ``data/cylinder_wake``.
    dtype : torch.dtype, optional
        Floating dtype for returned tensors.

    Returns
    -------
    dict
        Cache payload with ``features``, ``edge_index``, and metadata.
    """
    path = ensure_wake_cache(cache_dir)
    payload = torch.load(path, weights_only=False)
    payload["edge_index"] = payload["edge_index"].to(dtype=torch.long)
    payload["features"] = payload["features"].to(dtype=dtype)
    if "coords" in payload:
        payload["coords"] = payload["coords"].to(dtype=dtype)
    return payload


class CylinderWakeBenchmark:
    """Cached cylinder-wake Hopf surrogate on an unstructured wake mesh.

    Public entry points mirror METR-LA: ``load_topology`` / ``load_sequence``.
    The default cache is a small on-disk teaching dataset generated by
    :func:`build_cylinder_wake_payload` (Stuart–Landau modulated spatial modes),
    not full CFD. Rebuild with :func:`ensure_wake_cache` ``force=True``.

    Notes
    -----
    This is a teaching surrogate for Hopf wake dynamics, not a CFD result.
    """

    NUM_NODES = DEFAULT_WAKE_NUM_NODES
    IN_CHANNELS = IN_CHANNELS_SCALAR

    @classmethod
    def load_topology(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> TopologyPayload:
        """Load cached wake-mesh topology.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing ``wake.pt``.
        dtype : torch.dtype, optional
            Floating dtype for optional coordinate metadata.

        Returns
        -------
        TopologyPayload
            Frozen topology with ``edge_index`` and ``num_nodes``.
        """
        payload = load_wake_cache(cache_dir, dtype=dtype)
        return TopologyPayload(
            edge_index=payload["edge_index"],  # type: ignore[arg-type]
            num_nodes=int(payload["num_nodes"]),  # type: ignore[arg-type]
        )

    @classmethod
    def load_sequence(
        cls,
        cache_dir: Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Load the cached wake snapshot sequence.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing ``wake.pt``.
        dtype : torch.dtype, optional
            Feature dtype.

        Returns
        -------
        GraphSnapshotSequence
            Time-ordered streamwise-fluctuation snapshots.
        """
        payload = load_wake_cache(cache_dir, dtype=dtype)
        features = payload["features"]
        assert isinstance(features, Tensor)
        edge_index = payload["edge_index"]
        assert isinstance(edge_index, Tensor)
        if features.ndim != 3 or features.shape[2] != IN_CHANNELS_SCALAR:
            msg = (
                f"Expected features shape (T, N, {IN_CHANNELS_SCALAR}), "
                f"got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        return GraphSnapshotSequence.from_arrays(features, edge_index, dtype=dtype)
