"""Global / local discrete Koopman operator for non-stationary series.

Implements a shared global operator ``K_g`` plus a per-step low-rank local
correction ``K_ℓ(z_{t-w:t})`` from a small MLP over a recent latent window
(inspired by Koopman Neural Forecaster / global–local splits; see Wang et al.
and related non-stationary Koopman forecasting work). Discrete-time only.

``matrix`` / ``spectral_radius`` / ``bound_metric`` report the stationary
backbone ``K_g``. The effective step operator is state-dependent and does
**not** provide a single global spectral certificate (same vocabulary
precedent as continuous ``parameterization="auxiliary_spectral"``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import advance_step
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

DEFAULT_LOCAL_WINDOW = 4
DEFAULT_LOCAL_RANK = 2
DEFAULT_LOCAL_HIDDEN_DIMS: tuple[int, ...] = (32,)


def normalize_local_hidden_dims(
    hidden_dims: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate and normalize local-correction MLP hidden widths.

    Parameters
    ----------
    hidden_dims : sequence of int or None
        Per-layer widths. ``None`` selects :data:`DEFAULT_LOCAL_HIDDEN_DIMS`.

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
        DEFAULT_LOCAL_HIDDEN_DIMS
        if hidden_dims is None
        else tuple(int(width) for width in hidden_dims)
    )
    if not dims:
        msg = "local_hidden_dims must be non-empty"
        raise ValueError(msg)
    if any(width < 1 for width in dims):
        msg = f"local_hidden_dims must be positive integers, got {dims}"
        raise ValueError(msg)
    return dims


def pad_latent_window(z: Tensor, window: int) -> Tensor:
    """Build a cold-start latent window by repeating ``z``.

    Parameters
    ----------
    z : Tensor
        Current latent state with shape ``(..., latent_dim)``.
    window : int
        Window length ``w``.

    Returns
    -------
    Tensor
        Stacked window with shape ``(window, ..., latent_dim)``.
    """
    return z.unsqueeze(0).expand(window, *z.shape).contiguous()


def stack_latent_window(
    history: Sequence[Tensor],
    *,
    window: int,
    current: Tensor,
) -> Tensor:
    """Stack a rolling history into a length-``window`` latent window.

    Pads on the left by repeating the earliest available state when history is
    shorter than ``window``. The current state is always the last frame.

    Parameters
    ----------
    history : sequence of Tensor
        Prior latent states (oldest first), each matching ``current`` shape.
    window : int
        Target window length.
    current : Tensor
        Latent state at the step being advanced.

    Returns
    -------
    Tensor
        Window with shape ``(window, ..., latent_dim)``.
    """
    frames = list(history) + [current]
    if len(frames) < window:
        pad = [frames[0]] * (window - len(frames))
        frames = pad + frames
    else:
        frames = frames[-window:]
    return torch.stack(frames, dim=0)


class GlobalLocalKoopmanOperator(nn.Module):
    """Discrete Koopman step with global backbone and low-rank local correction.

    Advances latents via::

        z_{t+1} = z_t (K_g + K_ℓ(z_{t-w:t}))^T + control

    where ``K_ℓ = U V^T`` and ``(U, V)`` come from a small MLP over a recent
    latent window. Control (when enabled) is owned by the global factor.

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Parameterization of the global backbone ``K_g``.
    local_window : int
        Latent history length ``w``.
    local_rank : int
        Rank ``r`` of the local correction.
    local_hidden_dims : tuple of int
        MLP hidden widths for the local network.
    max_spectral_radius : float
        Stability bound forwarded to ``K_g``.
    """

    _LOCAL_SCALE_MAX = 1.0

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 1.0,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        local_window: int = DEFAULT_LOCAL_WINDOW,
        local_rank: int = DEFAULT_LOCAL_RANK,
        local_hidden_dims: Sequence[int] | None = None,
    ) -> None:
        """Initialize global backbone and local-correction MLP.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``K_g``.
        init_scale : float, optional
            Noise scale for ``identity_noise``.
        parameterization : Parameterization, optional
            Parameterization for ``K_g`` (not the local MLP).
        max_spectral_radius : float, optional
            Spectral bound for soft/structural modes on ``K_g``.
        control_dim : int, optional
            Additive / bilinear control dimension. Default ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling on the global factor.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        local_window : int, optional
            History length ``w`` for ``K_ℓ``. Default ``4``.
        local_rank : int, optional
            Rank of ``U V^T``. Default ``2``.
        local_hidden_dims : sequence of int or None, optional
            Local MLP hidden widths. Default ``(32,)``.

        Raises
        ------
        ValueError
            If window/rank/hidden dims are invalid or construction args fail.
        """
        super().__init__()
        if local_window < 1:
            msg = f"local_window must be positive, got {local_window}"
            raise ValueError(msg)
        if local_rank < 1:
            msg = f"local_rank must be positive, got {local_rank}"
            raise ValueError(msg)
        if parameterization == "auxiliary_spectral":
            msg = (
                "parameterization='auxiliary_spectral' is continuous-only; "
                "GlobalLocalKoopmanOperator is discrete-only"
            )
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_spectral_radius = max_spectral_radius
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank
        self.local_window = int(local_window)
        self.local_rank = int(local_rank)
        self.local_hidden_dims = normalize_local_hidden_dims(local_hidden_dims)

        self._global = KoopmanOperator(
            latent_dim,
            init_mode=init_mode,
            init_scale=init_scale,
            parameterization=parameterization,
            max_spectral_radius=max_spectral_radius,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
        )
        self._local_net = self._build_local_mlp()
        # Soft gate keeps K_ℓ small at start without a vanishing bilinear Jacobian
        # (zeroing both U and V kills gradients through U Vᵀ). Cap at
        # ``_LOCAL_SCALE_MAX`` so corrections stay a bounded perturbation of K_g.
        self._local_logit = nn.Parameter(torch.tensor(-4.0))
        self.reset_local_parameters()

    def _build_local_mlp(self) -> nn.Sequential:
        """Construct the MLP mapping a flattened window to low-rank factors.

        Returns
        -------
        nn.Sequential
            See summary line."""
        in_dim = self.local_window * self.latent_dim
        out_dim = 2 * self.latent_dim * self.local_rank
        layers: list[nn.Module] = []
        width_in = in_dim
        for width in self.local_hidden_dims:
            layers.append(nn.Linear(width_in, width))
            layers.append(nn.Tanh())
            width_in = width
        layers.append(nn.Linear(width_in, out_dim))
        return nn.Sequential(*layers)

    def reset_local_parameters(self) -> None:
        """Initialize the local MLP toward a small (but trainable) correction.

        Returns
        -------
        None
            See summary line."""
        for module in self._local_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.05)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self._local_logit.fill_(-4.0)

    def reset_parameters(self) -> None:
        """Reinitialize ``K_g`` (and control) plus the local MLP.

        Returns
        -------
        None
            See summary line."""
        self._global.reset_parameters()
        if self.control_dim > 0:
            self._global.reset_control_parameters()
        self.reset_local_parameters()

    @property
    def matrix(self) -> Tensor:
        """Stationary global backbone ``K_g`` (not the state-dependent ``K_eff``).

        Returns
        -------
        Tensor
            See summary line."""
        return self._global.matrix

    @property
    def K(self) -> Tensor:
        """Alias of :attr:`matrix` (global backbone).

        Returns
        -------
        Tensor
            See summary line."""
        return self.matrix

    @property
    def B(self) -> Tensor | None:
        """Control matrix from the global factor, when controlled.

        Returns
        -------
        Tensor | None
            See summary line."""
        if self.control_dim <= 0:
            return None
        return self._global.B

    def bound_metric(self) -> Tensor:
        """Cheap monitoring bound for the global backbone ``K_g``.

        Returns
        -------
        Tensor
            See summary line."""
        return self._global.bound_metric()

    def spectral_radius(self) -> Tensor:
        """True spectral radius of the global backbone ``K_g``.

        Returns
        -------
        Tensor
            See summary line."""
        return self._global.spectral_radius()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Structural certificate for ``K_g`` when the parameterization provides one.

        Returns
        -------
        StabilityCertificate | None
            See summary line."""
        return self._global.stability_certificate()

    def spectrum(self, *, time_step: float = 1.0) -> KoopmanSpectrum:
        """Eigenanalysis of the global backbone ``K_g``.

        Parameters
        ----------

        time_step : float, optional
            Discrete time step forwarded to :func:`compute_spectrum`.

        Returns
        -------

        KoopmanSpectrum
            See summary line.

        Notes
        -----

        The effective per-step operator ``K_g + K_ℓ(·)`` is state-dependent;
        this spectrum is **not** a certificate for the local correction."""
        return compute_spectrum(self.matrix, time_step=time_step)

    def local_correction(self, latent_window: Tensor) -> Tensor:
        """Assemble the low-rank local correction ``K_ℓ`` from a latent window.

        Parameters
        ----------
        latent_window : Tensor
            History with shape ``(window, ..., latent_dim)`` matching
            :attr:`local_window`.

        Returns
        -------
        Tensor
            Correction with shape ``(..., latent_dim, latent_dim)`` broadcast
            to the batch dims of the window (excluding the leading time axis).

        Raises
        ------
        ValueError
            If the window length or trailing dimension is invalid.
        """
        if latent_window.ndim < 2:
            msg = (
                "latent_window must have shape (window, ..., latent_dim), "
                f"got {tuple(latent_window.shape)}"
            )
            raise ValueError(msg)
        if latent_window.shape[0] != self.local_window:
            msg = (
                f"latent_window length must be {self.local_window}, "
                f"got {latent_window.shape[0]}"
            )
            raise ValueError(msg)
        if latent_window.shape[-1] != self.latent_dim:
            msg = (
                f"latent_window trailing dim must be {self.latent_dim}, "
                f"got {latent_window.shape[-1]}"
            )
            raise ValueError(msg)

        # (w, *batch, d) -> (*batch, w, d) -> (*batch, w*d)
        batch_shape = latent_window.shape[1:-1]
        permuted = latent_window.movedim(0, -2)
        features = permuted.reshape(*batch_shape, self.local_window * self.latent_dim)
        raw = self._local_net(features)
        d = self.latent_dim
        r = self.local_rank
        u = raw[..., : d * r].reshape(*batch_shape, d, r)
        v = raw[..., d * r :].reshape(*batch_shape, d, r)
        scale = self._LOCAL_SCALE_MAX * torch.sigmoid(self._local_logit)
        return scale * (u @ v.transpose(-1, -2))

    def effective_matrix(self, latent_window: Tensor) -> Tensor:
        """Return ``K_g + K_ℓ(latent_window)`` for diagnostics.

        Parameters
        ----------

        latent_window : Tensor
            See the function signature / summary for ``latent_window``.

        Returns
        -------

        Tensor
            See summary line."""
        return self.matrix + self.local_correction(latent_window)

    def resolve_latent_window(
        self,
        z: Tensor,
        latent_window: Tensor | None,
    ) -> Tensor:
        """Validate or cold-start a latent window for ``z``.

        Parameters
        ----------
        z : Tensor
            Current latent state ``(..., d)``.
        latent_window : Tensor or None
            Optional history ``(w, ..., d)``. When ``None``, repeats ``z``.

        Returns
        -------
        Tensor
            Window with shape ``(w, ..., d)``.
        """
        if latent_window is None:
            return pad_latent_window(z, self.local_window)
        if latent_window.shape[1:] != z.shape:
            msg = (
                "latent_window batch/feature shape must match z; "
                f"got window {tuple(latent_window.shape)} vs z {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if latent_window.shape[0] != self.local_window:
            msg = (
                f"latent_window length must be {self.local_window}, "
                f"got {latent_window.shape[0]}"
            )
            raise ValueError(msg)
        return latent_window

    def forward(
        self,
        z: Tensor,
        control: Tensor | None = None,
        *,
        latent_window: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states one discrete step with local correction.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        control : Tensor or None, optional
            Exogenous control when :attr:`control_dim` is positive.
        latent_window : Tensor or None, optional
            Recent latent history ``(w, ..., d)``. Cold-starts by repeating
            ``z`` when omitted.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        window = self.resolve_latent_window(z, latent_window)
        k_ell = self.local_correction(window)
        coupling = (
            self._global.bilinear_matrices()
            if self.control_mode == "bilinear"
            else None
        )
        # Global backbone + control via the shared discrete helper, then add
        # the batched low-rank local correction ``z @ K_ℓ.T``.
        z_next = advance_step(
            z,
            control,
            matrix=self.matrix,
            control_matrix=getattr(self._global, "B", None)
            if self.control_dim > 0
            else None,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            latent_dim=self.latent_dim,
            coupling=coupling,
        )
        local_delta = (z.unsqueeze(-2) @ k_ell.transpose(-1, -2)).squeeze(-2)
        return z_next + local_delta

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        latent_window: Tensor | None = None,
    ) -> Tensor:
        """Contract advance; optional ``latent_window`` for the local MLP.

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.
        delta_t : float | Tensor | None
            See the function signature / summary for ``delta_t``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.
        latent_window : Tensor | None
            See the function signature / summary for ``latent_window``.

        Returns
        -------

        Tensor
            See summary line.

        Notes
        -----

        Topology kwargs are accepted for API symmetry and ignored. ``delta_t``
        is ignored (discrete-only)."""
        _ = delta_t, edge_index, edge_weight
        return self.forward(z, control=control, latent_window=latent_window)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        latent_window: Tensor | None = None,
    ) -> Tensor:
        """Approximate inverse using the global backbone ``K_g`` only.

        The true forward map is state-dependent; recovering the exact prior
        state would require the same ``K_ℓ`` used at the forward step. This
        method inverts ``K_g`` (plus control) as a documented approximation
        and rejects caller-supplied ``inverse_matrix`` values that would
        imply an exact state-dependent inverse.

        Parameters
        ----------

        z : Tensor
            Latent states at time ``t+1``.
        latent_window : Tensor or None, optional
            Ignored (approximate inverse does not use ``K_ℓ``).
        delta_t : float | Tensor | None
            See the function signature / summary for ``delta_t``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        inverse_matrix : Tensor | None
            See the function signature / summary for ``inverse_matrix``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            If ``inverse_matrix`` is provided."""
        _ = delta_t, edge_index, edge_weight, latent_window
        if inverse_matrix is not None:
            msg = (
                "inverse_matrix is not supported for GlobalLocalKoopmanOperator "
                "(state-dependent local correction); inverse_advance uses the "
                "global backbone K_g only"
            )
            raise ValueError(msg)
        return self._global.inverse_advance(z, control=control, inverse_matrix=None)
