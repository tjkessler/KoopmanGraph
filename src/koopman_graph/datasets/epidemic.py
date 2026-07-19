"""Networked SIR epidemic benchmark for Koopman stress tests.

Complements Laplacian-diffusion benchmarks with intrinsically nonlinear
contact dynamics on ring, Watts–Strogatz small-world, or custom graphs.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import make_generator
from koopman_graph.datasets.topology import ring_edge_index

EpidemicTopologyName = Literal["ring", "small_world", "custom"]

IN_CHANNELS_SIR = 3


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
            edges = ring_edge_index(num_nodes)
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
