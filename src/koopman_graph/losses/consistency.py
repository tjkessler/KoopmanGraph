"""Forward, backward, and Lie consistency losses."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from koopman_graph.graph_utils import (
    KoopmanPropagator,
    inverse_propagate_latent,
    propagate_latent,
)
from koopman_graph.losses.reconstruction import masked_mse_loss


class ForwardConsistencyLoss(nn.Module):
    """Penalize deviation from linear latent evolution under the Koopman operator.

    For latent row encodings ``z_t`` and ``z_{t+1}``, the loss is the element-wise
    mean squared error between the propagated state ``z_t @ K.T`` (plus optional
    control) and ``z_{t+1}``:

    .. math::

        \\mathcal{L}_{\\mathrm{fc}}
        = \\mathrm{mean}\\big((z_t K^{\\top} - z_{t+1})^2\\big)

    This matches ``torch.nn.functional.mse_loss`` (average over all entries),
    not an unnormalized Frobenius or Euclidean squared norm.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operators.KoopmanOperator`.
    """

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
        koopman: KoopmanPropagator,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
        default_delta_t: float | Tensor = 1.0,
        mask: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute forward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : KoopmanOperator or ContinuousKoopmanOperator
            Learnable linear propagator applied to ``z_t``.
        control : Tensor or None, optional
            Control input driving the transition from ``t`` to ``t+1``.
        delta_t : float, Tensor, or None, optional
            Integration interval for continuous-time operators. Ignored for
            discrete operators.
        default_delta_t : float or Tensor, optional
            Fallback when ``delta_t is None`` in continuous mode. Pass the
            model ``time_step`` for model-backed training; bare calls default
            to ``1.0``.
        mask : Tensor or None, optional
            Optional boolean node mask with shape ``(num_nodes,)``. When set,
            MSE is averaged over observed nodes only.
        edge_index : Tensor or None, optional
            Topology for networked operators (ignored by per-node operators).
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Scalar mean-squared error between propagated ``z_t`` and ``z_t1``.
        """
        z_pred = propagate_latent(
            koopman,
            z_t,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if mask is None:
            return nn.functional.mse_loss(z_pred, z_t1)
        return masked_mse_loss(z_pred, z_t1, mask)


class BackwardConsistencyLoss(nn.Module):
    """Penalize deviation from inverse linear latent evolution under **K**.

    For latent row encodings ``z_t`` and ``z_{t+1}`` with forward dynamics
    ``z_{t+1} = z_t @ K.T``, the backward consistency term is the element-wise
    mean squared error between ``z_t`` and the inverse propagation of
    ``z_{t+1}`` (``z_{t+1} @ (K^{\\dagger}).T`` after removing control):

    .. math::

        \\mathcal{L}_{\\mathrm{bc}}
        = \\mathrm{mean}\\big((z_t - z_{t+1} (K^{\\dagger})^{\\top})^2\\big)

    where ``K^{\\dagger}`` denotes an inverse or pseudo-inverse of ``K``.
    As with the forward term, this is ``mse_loss`` (mean over entries), not
    ``\\|\\cdot\\|_F^2``.

    Trade-offs
    ----------
    **Benefits:** Enforces bidirectional linear consistency, which can improve
    Koopman operator identifiability and training stability when paired with
    the forward consistency term (see Lusch et al., 2018; Mezić, 2021).

    **Costs:** Dense unconstrained ``K`` requires a matrix inverse or
    pseudo-inverse. The ODO parameterization
    (:attr:`~koopman_graph.operators.KoopmanOperator.parameterization`
    ``"odo"``) provides a cheap exact factorized inverse of the diagonal
    factors (discrete ODO still lacks a structural ε-interior certificate).
    For dense ``K``, sequence-level training precomputes the inverse once per
    step rather than per snapshot pair.

    Notes
    -----
    This module is stateless. Call :meth:`forward` with consecutive latent
    encodings and a :class:`~koopman_graph.operators.KoopmanOperator`.
    """

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
        koopman: KoopmanPropagator,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        delta_t: float | Tensor | None = None,
        default_delta_t: float | Tensor = 1.0,
        mask: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute backward consistency loss between consecutive latent states.

        Parameters
        ----------
        z_t : Tensor
            Latent encoding at time ``t``, shape ``(..., latent_dim)``.
        z_t1 : Tensor
            Latent encoding at time ``t+1``, same shape as ``z_t``.
        koopman : :class:`~koopman_graph.operators.KoopmanOperator`
            Learnable linear propagator whose inverse step is applied to
            ``z_t1``.
        control : Tensor or None, optional
            Control input that drove the forward transition from ``t`` to
            ``t+1``.
        inverse_matrix : Tensor or None, optional
            Precomputed dense inverse matrix reused across pair evaluations.
        delta_t : float, Tensor, or None, optional
            Integration interval for continuous-time operators. Ignored for
            discrete operators.
        default_delta_t : float or Tensor, optional
            Fallback when ``delta_t is None`` in continuous mode. Pass the
            model ``time_step`` for model-backed training; bare calls default
            to ``1.0``.
        mask : Tensor or None, optional
            Optional boolean node mask with shape ``(num_nodes,)``. When set,
            MSE is averaged over observed nodes only.
        edge_index : Tensor or None, optional
            Topology for networked operators (ignored by per-node operators).
        edge_weight : Tensor or None, optional
            Optional edge weights for networked operators.

        Returns
        -------
        Tensor
            Scalar mean-squared error between ``z_t`` and the inverse
            propagation of ``z_t1``.
        """
        z_recovered = inverse_propagate_latent(
            koopman,
            z_t1,
            control=control,
            inverse_matrix=inverse_matrix,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if mask is None:
            return nn.functional.mse_loss(z_recovered, z_t)
        return masked_mse_loss(z_recovered, z_t, mask)


class LieConsistencyLoss(nn.Module):
    r"""Penalize disagreement between a known vector field and latent dynamics.

    For observables ``phi`` and a known autonomous continuous vector field
    ``f``, this implements the physics-informed Koopman-network residual

    .. math::

        \mathcal{L}_{\mathrm{Lie}}
        = \operatorname{mean}\left(
          \left(\nabla_x \phi(x) f(x) - L(\phi(x))\phi(x)\right)^2
          \right),

    where ``L(z)`` is either a fixed continuous generator or the instantaneous
    generator of an ``auxiliary_spectral`` operator. The Jacobian-vector
    product is evaluated by autograd without materializing the full Jacobian.

    Notes
    -----
    This is a continuous-time consistency penalty. It does not by itself
    impose Hamiltonian, symplectic, or other structure preservation.
    """

    def forward(
        self,
        state: Tensor,
        *,
        observable_fn: Callable[[Tensor], Tensor],
        dynamics_fn: Callable[[Tensor], Tensor],
        koopman: KoopmanPropagator,
    ) -> Tensor:
        """Compute the Lie-derivative residual.

        Parameters
        ----------
        state : Tensor
            Physical state at which to evaluate the residual.
        observable_fn : callable
            Differentiable map from ``state`` to latent observables.
        dynamics_fn : callable
            Known vector field mapping ``state`` to ``dx/dt``.
        koopman : KoopmanOperatorContract
            Continuous operator exposing ``generator_at(z)``.

        Returns
        -------
        Tensor
            Scalar mean-squared Lie residual.

        Raises
        ------
        ValueError
            If the vector field or generator has an incompatible shape.
        TypeError
            If the operator does not expose ``generator_at``.
        """
        vector_field = dynamics_fn(state)
        if vector_field.shape != state.shape:
            msg = (
                "dynamics_fn output must match state shape, got "
                f"{tuple(vector_field.shape)} and {tuple(state.shape)}"
            )
            raise ValueError(msg)
        if koopman.control_dim != 0:
            msg = (
                "LieConsistencyLoss currently supports uncontrolled continuous "
                "operators only"
            )
            raise ValueError(msg)

        latent, lie_derivative = torch.autograd.functional.jvp(
            observable_fn,
            state,
            vector_field,
            create_graph=torch.is_grad_enabled(),
        )
        generator_at = getattr(koopman, "generator_at", None)
        if not callable(generator_at):
            msg = (
                "LieConsistencyLoss requires a continuous operator with generator_at(z)"
            )
            raise TypeError(msg)
        generator = generator_at(latent)
        expected_matrix_shape = (latent.shape[-1], latent.shape[-1])
        if generator.shape[-2:] != expected_matrix_shape:
            msg = (
                "generator_at output must end with shape "
                f"{expected_matrix_shape}, got {tuple(generator.shape)}"
            )
            raise ValueError(msg)

        latent_derivative = torch.matmul(
            latent.unsqueeze(-2),
            generator.transpose(-1, -2),
        ).squeeze(-2)
        if latent_derivative.shape != lie_derivative.shape:
            msg = (
                "latent generator derivative must match observable JVP shape, got "
                f"{tuple(latent_derivative.shape)} and {tuple(lie_derivative.shape)}"
            )
            raise ValueError(msg)
        return nn.functional.mse_loss(lie_derivative, latent_derivative)
