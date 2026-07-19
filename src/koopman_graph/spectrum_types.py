"""Neutral value types and discrete spectrum assembly.

Power-user module: importable as ``koopman_graph.spectrum_types``, documented
in architecture docs, and **not** re-exported in package ``__all__``.
:class:`KoopmanSpectrum` and :func:`compute_spectrum` are re-exported from
:mod:`koopman_graph.analysis` and (for the type / discrete helper) the package
root public surface. Operators import discrete spectrum assembly from here so
they never depend on :mod:`koopman_graph.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class KoopmanSpectrum:
    """Eigendecomposition and time scales of a Koopman operator or generator.

    Eigenpairs are sorted by descending eigenvalue magnitude. Frequencies are
    reported in cycles per unit time; multiply by ``2 * pi`` for angular
    frequency.

    Semantics depend on how the spectrum was produced:

    - :func:`~koopman_graph.spectrum_types.compute_spectrum` (discrete ``K``;
      also re-exported as :func:`~koopman_graph.analysis.compute_spectrum`):
      ``growth_rates = log(|lambda|) / time_step`` and
      ``frequencies = angle(lambda) / (2 * pi * time_step)``, with
      ``time_step`` equal to the discrete sampling interval.
    - :func:`~koopman_graph.analysis.compute_generator_spectrum`
      (continuous ``L``): ``growth_rates = Re(mu)`` and
      ``frequencies = Im(mu) / (2 * pi)``, with ``time_step`` set to ``1.0``
      as a placeholder (native continuous-time units).

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(latent_dim,)``.
    eigenvectors : Tensor
        Complex right eigenvectors stored as columns, with shape
        ``(latent_dim, latent_dim)``.
    magnitudes : Tensor
        Eigenvalue magnitudes with shape ``(latent_dim,)``.
    growth_rates : Tensor
        Continuous-time growth rates (see semantics above).
    frequencies : Tensor
        Signed frequencies in cycles per unit time (see semantics above).
    time_step : float
        Discrete sampling interval for discrete spectra; ``1.0`` for
        generator spectra.
    """

    eigenvalues: Tensor
    eigenvectors: Tensor
    magnitudes: Tensor
    growth_rates: Tensor
    frequencies: Tensor
    time_step: float

    def mode_amplitudes(self, latent_states: Tensor) -> Tensor:
        """Project latent states onto the Koopman eigenvector basis.

        For a latent row vector ``z``, the returned amplitudes ``a`` satisfy
        ``z.T = eigenvectors @ a``. Any leading dimensions are preserved.

        Parameters
        ----------
        latent_states : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Complex mode amplitudes with the same shape as ``latent_states``.

        Raises
        ------
        ValueError
            If the trailing latent dimension does not match the spectrum.
        RuntimeError
            If the eigenvector matrix is singular.
        """
        latent_dim = self.eigenvectors.shape[0]
        if latent_states.ndim == 0 or latent_states.shape[-1] != latent_dim:
            msg = (
                f"Expected trailing dimension {latent_dim}, "
                f"got shape {tuple(latent_states.shape)}"
            )
            raise ValueError(msg)

        vectors = self.eigenvectors.to(device=latent_states.device)
        states = latent_states.to(dtype=vectors.dtype)
        flat_states = states.reshape(-1, latent_dim)
        amplitudes = torch.linalg.solve(vectors, flat_states.T).T
        return amplitudes.reshape(latent_states.shape)


def compute_spectrum(operator: Tensor, time_step: float) -> KoopmanSpectrum:
    """Compute the sorted spectrum and continuous-time mode characteristics.

    Neutral-leaf discrete spectrum assembly used by operators (for example
    :meth:`~koopman_graph.operators.graph.GraphKoopmanOperator.spectrum`) and
    re-exported from :mod:`koopman_graph.analysis` for the public API.

    Parameters
    ----------
    operator : Tensor
        Square discrete-time Koopman matrix with shape
        ``(latent_dim, latent_dim)``.
    time_step : float
        Positive physical duration represented by one operator step.

    Returns
    -------
    KoopmanSpectrum
        Eigenpairs sorted by descending magnitude, plus growth rates and
        frequencies converted using ``time_step``.

    Raises
    ------
    ValueError
        If ``operator`` is not a non-empty square matrix or ``time_step`` is
        not positive.
    TypeError
        If ``operator`` is not floating-point or complex.
    """
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        msg = f"operator must be a square matrix, got shape {tuple(operator.shape)}"
        raise ValueError(msg)
    if operator.shape[0] == 0:
        raise ValueError("operator must be non-empty")
    if time_step <= 0:
        msg = f"time_step must be positive, got {time_step}"
        raise ValueError(msg)
    if not (operator.is_floating_point() or operator.is_complex()):
        msg = f"operator must be floating-point or complex, got {operator.dtype}"
        raise TypeError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(operator)
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]

    growth_rates = torch.log(magnitudes) / time_step
    frequencies = torch.angle(eigenvalues) / (2 * torch.pi * time_step)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=float(time_step),
    )
