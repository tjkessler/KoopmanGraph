"""Auxiliary-network spectral parameterization for continuous generators.

Implements a Lusch-style parametric / locally linear spectrum: an MLP maps
latent state ``z`` to instantaneous eigenvalues that assemble a block-diagonal
rotation–scaling generator ``L(z)``. This is **not** a fixed global Koopman
matrix and does **not** certify global spectral-radius / Hurwitz bounds.

Also hosts state-dependent advance / inverse / reset helpers used by
:class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator` when
``parameterization="auxiliary_spectral"``. Van Loan factor construction lives
in :mod:`koopman_graph.operators.continuous_van_loan`; structural / dense
assembly in :mod:`koopman_graph.operators.continuous_parameterizations`;
fixed-generator advance / inverse execution in
:mod:`koopman_graph.operators.continuous_propagation`; continuous remains the
thin orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from koopman_graph.operators.continuous_van_loan import van_loan_factors
from koopman_graph.operators.contract import InitMode
from koopman_graph.operators.control import (
    ControlMode,
    broadcast_control_term,
    effective_bilinear_matrix,
)

DEFAULT_AUXILIARY_HIDDEN_DIMS: tuple[int, ...] = (64, 64)

# Frozen-generator Van Loan step: ``(z, delta_t, control, *, generator) -> z'``.
VanLoanAdvanceFn = Callable[..., Tensor]


def normalize_auxiliary_hidden_dims(
    hidden_dims: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate and normalize auxiliary MLP hidden widths.

    Parameters
    ----------
    hidden_dims : sequence of int or None
        Per-layer widths. ``None`` selects
        :data:`DEFAULT_AUXILIARY_HIDDEN_DIMS`.

    Returns
    -------
    tuple of int
        Non-empty tuple of positive hidden widths.

    Raises
    ------
    ValueError
        If empty or any width is not a positive integer.
    """
    dims = (
        DEFAULT_AUXILIARY_HIDDEN_DIMS
        if hidden_dims is None
        else tuple(int(width) for width in hidden_dims)
    )
    if not dims:
        msg = "auxiliary_hidden_dims must be non-empty"
        raise ValueError(msg)
    if any(width < 1 for width in dims):
        msg = f"auxiliary_hidden_dims must be positive integers, got {dims}"
        raise ValueError(msg)
    return dims


def spectral_output_dim(latent_dim: int) -> int:
    """Return the auxiliary network output size for ``latent_dim``.

    Each complex conjugate pair contributes ``(μ, ω)`` (two scalars). An odd
    leftover dimension contributes one real eigenvalue.

    Parameters
    ----------
    latent_dim : int
        Generator dimension.

    Returns
    -------
    int
        Output feature count ``2 * (latent_dim // 2) + (latent_dim % 2)``.
    """
    n_pairs = latent_dim // 2
    n_real = latent_dim % 2
    return 2 * n_pairs + n_real


def assemble_block_diagonal_generator(
    mu: Tensor,
    omega: Tensor,
    real_eig: Tensor | None,
) -> Tensor:
    """Assemble a real block-diagonal rotation–scaling generator.

    Complex pairs use blocks::

        [[μ, -ω],
         [ω,  μ]]

    with eigenvalues ``μ ± iω``. An optional trailing real eigenvalue fills an
    odd leftover dimension.

    Parameters
    ----------
    mu : Tensor
        Real parts with shape ``(..., n_pairs)``.
    omega : Tensor
        Imaginary / angular frequencies with shape ``(..., n_pairs)``.
    real_eig : Tensor or None
        Optional real eigenvalues with shape ``(..., 1)`` when ``latent_dim``
        is odd; ``None`` when even.

    Returns
    -------
    Tensor
        Generators with shape ``(..., latent_dim, latent_dim)``.

    Raises
    ------
    ValueError
        If pair / real shapes are inconsistent.
    """
    if mu.shape != omega.shape:
        msg = (
            f"mu and omega must share shape, got mu={tuple(mu.shape)} "
            f"omega={tuple(omega.shape)}"
        )
        raise ValueError(msg)
    n_pairs = mu.shape[-1]
    leading = mu.shape[:-1]
    latent_dim = 2 * n_pairs + (0 if real_eig is None else 1)
    flat_mu = mu.reshape(-1, n_pairs)
    flat_omega = omega.reshape(-1, n_pairs)
    batch = flat_mu.shape[0]
    generator = mu.new_zeros(batch, latent_dim, latent_dim)

    for pair in range(n_pairs):
        i0 = 2 * pair
        i1 = i0 + 1
        generator[:, i0, i0] = flat_mu[:, pair]
        generator[:, i0, i1] = -flat_omega[:, pair]
        generator[:, i1, i0] = flat_omega[:, pair]
        generator[:, i1, i1] = flat_mu[:, pair]

    if real_eig is not None:
        if real_eig.shape[-1] != 1:
            msg = (
                "real_eig trailing dimension must be 1, "
                f"got shape {tuple(real_eig.shape)}"
            )
            raise ValueError(msg)
        if real_eig.shape[:-1] != leading:
            msg = (
                "real_eig leading shape must match mu/omega, "
                f"got {tuple(real_eig.shape[:-1])} vs {tuple(leading)}"
            )
            raise ValueError(msg)
        generator[:, -1, -1] = real_eig.reshape(batch)

    return generator.reshape(*leading, latent_dim, latent_dim)


def split_auxiliary_spectrum(
    raw: Tensor,
    *,
    latent_dim: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Split auxiliary network outputs into ``(μ, ω, real_eig)``.

    Parameters
    ----------
    raw : Tensor
        Network outputs with shape ``(..., spectral_output_dim)``.
    latent_dim : int
        Target generator dimension.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor or None]
        Real parts, frequencies, and optional real eigenvalue tensor.

    Raises
    ------
    ValueError
        If the trailing dimension does not match ``latent_dim``.
    """
    expected = spectral_output_dim(latent_dim)
    if raw.shape[-1] != expected:
        msg = (
            f"Expected auxiliary output dim {expected} for latent_dim={latent_dim}, "
            f"got {raw.shape[-1]}"
        )
        raise ValueError(msg)
    n_pairs = latent_dim // 2
    mu = raw[..., :n_pairs]
    omega = raw[..., n_pairs : 2 * n_pairs]
    real_eig = None
    if latent_dim % 2 == 1:
        real_eig = raw[..., 2 * n_pairs : 2 * n_pairs + 1]
    return mu, omega, real_eig


class AuxiliarySpectralNetwork(nn.Module):
    """MLP mapping latent state to instantaneous spectral parameters.

    Parameters
    ----------
    latent_dim : int
        Latent / generator dimension.
    hidden_dims : sequence of int or None, optional
        Hidden layer widths. Default is ``(64, 64)``.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        """Build the MLP for instantaneous spectral parameters.

        Parameters
        ----------
        latent_dim : int
            Latent / generator dimension.
        hidden_dims : sequence of int or None, optional
            Hidden layer widths. Default is ``(64, 64)``.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        self.latent_dim = latent_dim
        self.hidden_dims = normalize_auxiliary_hidden_dims(hidden_dims)
        out_dim = spectral_output_dim(latent_dim)
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize layers toward a near-zero (mildly damped) spectrum.

        Returns
        -------
        None
        """
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Near-constant initial spectrum: zero final weights, mild biases.
        # Gradients can then differentiate ω(z) / μ(z) from the advance loss.
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.zero_()
            n_pairs = self.latent_dim // 2
            if n_pairs > 0:
                final.bias[:n_pairs].fill_(-0.05)
                # softplus(0) ≈ 0.693 → ω ≈ 0.694 after the +1e-3 floor
                final.bias[n_pairs : 2 * n_pairs].zero_()
            if self.latent_dim % 2 == 1:
                final.bias[-1].fill_(-0.05)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Map latent states to ``(μ, ω, real_eig)``.

        Frequencies ``ω`` are passed through ``softplus`` so instantaneous
        oscillatory rates stay positive (Lusch-style continuous spectra).

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor or None]
            Spectral parameters for :func:`assemble_block_diagonal_generator`.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        mu, omega_raw, real_eig = split_auxiliary_spectrum(
            self.net(z),
            latent_dim=self.latent_dim,
        )
        omega = torch.nn.functional.softplus(omega_raw) + 1e-3
        return mu, omega, real_eig

    def generator_at(self, z: Tensor) -> Tensor:
        """Return the state-dependent generator ``L(z)``.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            ``L(z)`` with shape ``(..., latent_dim, latent_dim)`` (or
            ``(latent_dim, latent_dim)`` when ``z`` is 1-D).
        """
        mu, omega, real_eig = self.forward(z)
        return assemble_block_diagonal_generator(mu, omega, real_eig)


def apply_batched_transition(z: Tensor, transition: Tensor) -> Tensor:
    """Apply ``z @ K.T`` for a possibly batched transition ``K``.

    Parameters
    ----------
    z : Tensor
        Latents ``(..., d)``.
    transition : Tensor
        ``K`` with shape ``(d, d)`` or ``(..., d, d)``.

    Returns
    -------
    Tensor
        Advanced latents with the same shape as ``z``.
    """
    if transition.ndim == 2:
        return z @ transition.T
    return torch.matmul(z.unsqueeze(-2), transition.transpose(-1, -2)).squeeze(-2)


def advance_uncontrolled(
    z: Tensor,
    generator: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """Uncontrolled advance with a frozen state-dependent generator ``L(z)``.

    Parameters
    ----------
    z : Tensor
        Latent states.
    generator : Tensor
        Frozen ``L(z)`` matching ``z``'s batch axes.
    delta_t : Tensor
        Integration interval.

    Returns
    -------
    Tensor
        Advanced latents.
    """
    transition = torch.linalg.matrix_exp(generator * delta_t)
    return apply_batched_transition(z, transition)


def advance_controlled(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_mode: ControlMode,
    latent_dim: int,
    coupling: Tensor | None,
    advance_van_loan: VanLoanAdvanceFn,
) -> Tensor:
    """Controlled advance with ``L(z)`` frozen at the step start (locally linear).

    Parameters
    ----------
    z : Tensor
        Latent states.
    delta_t : Tensor
        Integration interval.
    control : Tensor
        Control with shape ``(control_dim,)`` or ``(num_nodes, control_dim)``.
    generator : Tensor
        Frozen ``L(z)`` matching ``z``'s batch axes.
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    latent_dim : int
        Trailing latent dimension.
    coupling : Tensor or None
        Bilinear coupling stack when ``control_mode="bilinear"``.
    advance_van_loan : callable
        Frozen-generator Van Loan step
        ``(z, delta_t, control, *, generator) -> Tensor``.

    Returns
    -------
    Tensor
        Advanced latents.
    """
    if control_mode == "additive":
        if control.ndim == 1:
            if z.ndim == 1:
                return advance_van_loan(z, delta_t, control, generator=generator)
            flat_z = z.reshape(-1, latent_dim)
            flat_l = generator.reshape(-1, latent_dim, latent_dim)
            advanced = torch.empty_like(flat_z)
            for idx in range(flat_z.shape[0]):
                advanced[idx] = advance_van_loan(
                    flat_z[idx],
                    delta_t,
                    control,
                    generator=flat_l[idx],
                )
            return advanced.reshape_as(z)
        if control.ndim == 2:
            if z.ndim < 2 or z.shape[-2] != control.shape[0]:
                msg = (
                    "per-node additive control requires matching node axes, "
                    f"got z={tuple(z.shape)}, u={tuple(control.shape)}"
                )
                raise ValueError(msg)
            advanced = torch.empty_like(z)
            for node_idx in range(control.shape[0]):
                advanced[..., node_idx, :] = advance_van_loan(
                    z[..., node_idx, :],
                    delta_t,
                    control[node_idx],
                    generator=generator[..., node_idx, :, :],
                )
            return advanced
        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    if coupling is None:
        msg = "bilinear coupling tensor is required for bilinear auxiliary advance"
        raise ValueError(msg)

    if control.ndim == 1:
        if z.ndim == 1:
            l_eff = effective_bilinear_matrix(generator, control, coupling)
            return advance_van_loan(z, delta_t, control, generator=l_eff)
        flat_z = z.reshape(-1, latent_dim)
        flat_l = generator.reshape(-1, latent_dim, latent_dim)
        advanced = torch.empty_like(flat_z)
        for idx in range(flat_z.shape[0]):
            l_eff = effective_bilinear_matrix(flat_l[idx], control, coupling)
            advanced[idx] = advance_van_loan(
                flat_z[idx],
                delta_t,
                control,
                generator=l_eff,
            )
        return advanced.reshape_as(z)

    if control.ndim == 2:
        if z.ndim < 2 or z.shape[-2] != control.shape[0]:
            msg = (
                "per-node bilinear control requires z with a matching "
                f"node axis, got z={tuple(z.shape)}, u={tuple(control.shape)}"
            )
            raise ValueError(msg)
        advanced = torch.empty_like(z)
        for node_idx in range(control.shape[0]):
            l_eff = effective_bilinear_matrix(
                generator[..., node_idx, :, :],
                control[node_idx],
                coupling,
            )
            advanced[..., node_idx, :] = advance_van_loan(
                z[..., node_idx, :],
                delta_t,
                control[node_idx],
                generator=l_eff,
            )
        return advanced

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def _invert_van_loan_step(
    state: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    delta_t: Tensor,
    latent_dim: int,
) -> Tensor:
    """Invert one Van Loan step for a frozen generator.

    Parameters
    ----------
    state : Tensor
        Latents after the forward interval.
    control : Tensor
        Control applied during the interval.
    generator : Tensor
        Frozen generator ``L(z)`` or ``L_eff``.
    control_matrix : Tensor
        Continuous control matrix ``B``.
    delta_t : Tensor
        Integration interval.
    latent_dim : int
        Trailing latent dimension.

    Returns
    -------
    Tensor
        Approximate pre-step latents.
    """
    phi11, phi12 = van_loan_factors(generator, control_matrix, delta_t)
    if control.ndim == 1:
        offset = control @ phi12.T
        if state.ndim > 1:
            offset = broadcast_control_term(state, offset, latent_dim=latent_dim)
        adjusted = state - offset
    else:
        adjusted = state - control @ phi12.T
    try:
        inverse_phi = torch.linalg.inv(phi11)
    except RuntimeError:
        inverse_phi = torch.linalg.pinv(phi11)
    return adjusted @ inverse_phi.T


def inverse_advance_bilinear(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    coupling: Tensor,
    latent_dim: int,
) -> Tensor:
    """Inverse bilinear step under auxiliary ``L(z)`` linearization.

    Parameters
    ----------
    z : Tensor
        Latents after advancing over ``Δt``.
    delta_t : Tensor
        Integration interval.
    control : Tensor
        Control applied during the forward interval.
    generator : Tensor
        Frozen ``L(z)`` at the post-step state.
    control_matrix : Tensor
        Continuous control matrix ``B``.
    coupling : Tensor
        Bilinear coupling stack.
    latent_dim : int
        Trailing latent dimension.

    Returns
    -------
    Tensor
        Approximate recovered latents.
    """
    if control.ndim == 1:
        if z.ndim == 1:
            l_eff = effective_bilinear_matrix(generator, control, coupling)
            return _invert_van_loan_step(
                z,
                control,
                generator=l_eff,
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        flat_z = z.reshape(-1, latent_dim)
        flat_l = generator.reshape(-1, latent_dim, latent_dim)
        recovered = torch.empty_like(flat_z)
        for idx in range(flat_z.shape[0]):
            l_eff = effective_bilinear_matrix(flat_l[idx], control, coupling)
            recovered[idx] = _invert_van_loan_step(
                flat_z[idx],
                control,
                generator=l_eff,
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        return recovered.reshape_as(z)

    if control.ndim == 2:
        if z.ndim < 2 or z.shape[-2] != control.shape[0]:
            msg = (
                "per-node bilinear inverse requires matching node axes, "
                f"got z={tuple(z.shape)}, u={tuple(control.shape)}"
            )
            raise ValueError(msg)
        recovered = torch.empty_like(z)
        for node_idx in range(control.shape[0]):
            l_eff = effective_bilinear_matrix(
                generator[..., node_idx, :, :],
                control[node_idx],
                coupling,
            )
            recovered[..., node_idx, :] = _invert_van_loan_step(
                z[..., node_idx, :],
                control[node_idx],
                generator=l_eff,
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        return recovered

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def inverse_advance(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor | None,
    *,
    generator: Tensor,
    control_dim: int,
    control_mode: ControlMode,
    latent_dim: int,
    control_matrix: Tensor | None,
    coupling: Tensor | None,
) -> Tensor:
    """Approximate inverse step using ``L(z)`` at the post-step state.

    Exact inversion would require the pre-step state that parameterized
    ``L(z_t)``. This uses the local linearization at the available post-step
    ``z`` (suitable for soft backward-consistency losses).

    Parameters
    ----------
    z : Tensor
        Latents after advancing over ``Δt``.
    delta_t : Tensor
        Integration interval.
    control : Tensor or None
        Control applied during the forward interval.
    generator : Tensor
        Frozen ``L(z)`` at the post-step state.
    control_dim : int
        Operator control dimension (``0`` for uncontrolled).
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    latent_dim : int
        Trailing latent dimension.
    control_matrix : Tensor or None
        Continuous control matrix ``B`` when ``control_dim > 0``.
    coupling : Tensor or None
        Bilinear coupling stack when ``control_mode="bilinear"``.

    Returns
    -------
    Tensor
        Approximate pre-step latents.
    """
    if control_dim == 0:
        transition = torch.linalg.matrix_exp(generator * (-delta_t))
        return apply_batched_transition(z, transition)

    assert control is not None
    assert control_matrix is not None
    if control_mode == "bilinear":
        if coupling is None:
            msg = "bilinear coupling tensor is required for bilinear inverse"
            raise ValueError(msg)
        return inverse_advance_bilinear(
            z,
            delta_t,
            control,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )

    if control.ndim == 1:
        if z.ndim == 1:
            return _invert_van_loan_step(
                z,
                control,
                generator=generator,
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        flat_z = z.reshape(-1, latent_dim)
        flat_l = generator.reshape(-1, latent_dim, latent_dim)
        recovered = torch.empty_like(flat_z)
        for idx in range(flat_z.shape[0]):
            recovered[idx] = _invert_van_loan_step(
                flat_z[idx],
                control,
                generator=flat_l[idx],
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        return recovered.reshape_as(z)

    if control.ndim == 2:
        if z.ndim < 2 or z.shape[-2] != control.shape[0]:
            msg = (
                "per-node control inverse requires matching node axes, "
                f"got z={tuple(z.shape)}, u={tuple(control.shape)}"
            )
            raise ValueError(msg)
        recovered = torch.empty_like(z)
        for node_idx in range(control.shape[0]):
            recovered[..., node_idx, :] = _invert_van_loan_step(
                z[..., node_idx, :],
                control[node_idx],
                generator=generator[..., node_idx, :, :],
                control_matrix=control_matrix,
                delta_t=delta_t,
                latent_dim=latent_dim,
            )
        return recovered

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def reset_auxiliary_network(
    network: AuxiliarySpectralNetwork,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Reinitialize an auxiliary spectral network for continuous init modes.

    Parameters
    ----------
    network : AuxiliarySpectralNetwork
        Auxiliary MLP to reset.
    init_mode : InitMode
        Continuous operator init mode (``identity``, ``identity_noise``,
        ``xavier``).
    init_scale : float
        Noise scale for ``identity_noise``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``init_mode`` is unknown.
    """
    network.reset_parameters()
    if init_mode == "identity_noise":
        with torch.no_grad():
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    module.weight.add_(torch.randn_like(module.weight) * init_scale)
    elif init_mode == "xavier":
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    elif init_mode != "identity":
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)
