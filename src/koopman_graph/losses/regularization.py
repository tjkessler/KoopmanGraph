"""Eigenvalue hinge and sparsity regularization losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.graph_utils import KoopmanPropagator
from koopman_graph.operators.graph import GraphKoopmanOperator
from koopman_graph.operators.heterogeneous import HeteroGraphKoopmanOperator
from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator
from koopman_graph.protocols import DynamicsMode

# Assembled networked dense/ODO eig-reg no-ops when ``N·d`` exceeds this
# (``O((N·d)^3)`` eigendecomposition). Prefer structural modes or modest
# ``N·d``; see dense-ceiling notes in ``limitations.rst``.
MAX_ASSEMBLED_EIGREG_SIZE = 4096


class EigenvalueRegularizationLoss(nn.Module):
    """Penalize Koopman eigenvalues outside the stable region.

    Implements a hinge-style eigenloss. Discrete operators penalize magnitudes
    outside the unit circle:

    .. math::

        \\mathcal{L}_{\\mathrm{eig}} =
        \\mathrm{mean}\\big(\\max(|\\lambda_i| - 1, 0)^2\\big)

    Continuous generators penalize positive real parts.

    Path selection:

    - ``"dense"`` and ``"odo"`` on ordinary / custom operators —
      ``eigvals`` on
      :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix` (true
      spectrum). Continuous ODO can be Hurwitz-unstable even when diagonal
      factors are negative, so the factor
      :meth:`~koopman_graph.operators.KoopmanOperatorContract.bound_metric`
      must not be used here.
    - ``"dense"`` and ``"odo"`` on
      :class:`~koopman_graph.operators.GraphKoopmanOperator`,
      :class:`~koopman_graph.operators.HypergraphKoopmanOperator`, or
      :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator` —
      ``eigvals`` on the topology-coupled ``effective_matrix`` (requires
      topology / ``num_nodes``). Never falls back to ``K_self`` alone.
      When ``N·d >`` :data:`MAX_ASSEMBLED_EIGREG_SIZE`, the assembled hinge
      **no-ops** (returns zero) rather than allocating a huge eigendecomposition.
    - ``"schur"`` / ``"lyapunov"`` — cheap
      :meth:`~koopman_graph.operators.KoopmanOperatorContract.bound_metric`
      (closed-form certified bound for ordinary operators; for networked
      operators, max of factor bounds is a **factor-level surrogate** and is
      **not** a whole-network / joint ``ρ(K_eff)`` certificate —
      Gershgorin / joint spectral radius territory).
    - ``"dissipative"`` — always zero (structurally Hurwitz / contractive
      factors; for networked operators this does **not** certify the
      assembled effective map).

    Trade-offs
    ----------
    **Benefits:** Encourages stability without hard-constraining dense
    operators. For continuous ODO, penalizes the assembled generator's true
    spectrum (DeepKoopFormer-style eigenloss literature). For networked
    dense/ODO modes, the hinge matches advance / spectrum topology semantics
    when ``N·d`` is modest.

    **Costs:** ``"dense"`` / ``"odo"`` require ``torch.linalg.eigvals`` each
    evaluation (``N·d`` for networked effective matrices). Prefer structural
    modes (``schur`` / ``lyapunov`` / ``dissipative``) when a cheap
    ``bound_metric`` path is enough, understanding the networked structural
    caveat above — factor constraints do **not** guarantee
    ``ρ(K_eff) < 1``.

    Notes
    -----
    Pass ``dynamics_mode`` matching the operator semantics (the training
    loop uses :attr:`~koopman_graph.model.GraphKoopmanModel.dynamics_mode`).
    Defaults to ``"discrete"`` for standalone call sites.
    """

    def forward(
        self,
        koopman: KoopmanPropagator,
        *,
        dynamics_mode: DynamicsMode = "discrete",
        edge_index: Tensor | None = None,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
    ) -> Tensor:
        """Compute the stability eigenvalue hinge penalty.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Operator whose eigenvalues (or bound metric) are penalized.
        dynamics_mode : {"discrete", "continuous"}, optional
            Selects the discrete unit-circle hinge vs continuous Hurwitz hinge.
            Default is ``"discrete"``.
        edge_index : Tensor or None, optional
            Topology for networked
            :class:`~koopman_graph.operators.GraphKoopmanOperator` dense/ODO
            modes. Required for those modes; ignored for ordinary operators and
            for graph structural parameterizations.
        num_nodes : int or None, optional
            Node count ``N`` for the effective ``N·d`` operator. Required with
            topology for graph/hypergraph/hetero dense/ODO.
        edge_weight : Tensor or None, optional
            Optional edge weights with the same semantics as latent advance.
        hyperedge_index : Tensor or None, optional
            Incidence for
            :class:`~koopman_graph.operators.HypergraphKoopmanOperator`
            dense/ODO modes.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        edge_indices : sequence of Tensor or None, optional
            Ordered relation banks for
            :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`
            dense/ODO modes.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights for hetero dense/ODO.

        Returns
        -------
        Tensor
            Scalar hinge penalty (zero when assembled ``N·d`` exceeds
            :data:`MAX_ASSEMBLED_EIGREG_SIZE`).

        Raises
        ------
        ValueError
            If ``dynamics_mode`` is invalid, or a networked dense/ODO operator
            is regularized without required topology arguments.
        """
        if dynamics_mode not in {"discrete", "continuous"}:
            msg = (
                "dynamics_mode must be 'discrete' or 'continuous', "
                f"got {dynamics_mode!r}"
            )
            raise ValueError(msg)

        device = next(koopman.parameters()).device
        if koopman.parameterization in {"dissipative", "auxiliary_spectral"}:
            return torch.zeros((), device=device)

        if koopman.parameterization in {"schur", "lyapunov"}:
            bound = koopman.bound_metric()
            if dynamics_mode == "continuous":
                violation = torch.relu(bound)
            else:
                violation = torch.relu(bound - 1.0)
            return violation**2

        if _assembled_eigreg_exceeds_ceiling(koopman, num_nodes=num_nodes):
            return torch.zeros((), device=device)

        matrix = self._spectrum_matrix(
            koopman,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            edge_indices=edge_indices,
            edge_weights=edge_weights,
        )
        eigenvalues = torch.linalg.eigvals(matrix)
        if dynamics_mode == "continuous":
            violation = torch.relu(eigenvalues.real)
        else:
            violation = torch.relu(eigenvalues.abs() - 1.0)
        return (violation**2).mean()

    @staticmethod
    def _spectrum_matrix(
        koopman: KoopmanPropagator,
        *,
        edge_index: Tensor | None,
        num_nodes: int | None,
        edge_weight: Tensor | None,
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
    ) -> Tensor:
        """Return the matrix whose spectrum is regularized for dense/ODO.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Dense or ODO operator.
        edge_index : Tensor or None
            Topology for graph operators.
        num_nodes : int or None
            Node count for networked operators.
        edge_weight : Tensor or None
            Optional edge weights for graph operators.
        hyperedge_index : Tensor or None, optional
            Incidence for hypergraph operators.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        edge_indices : sequence of Tensor or None, optional
            Relation banks for hetero operators.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation weights for hetero operators.

        Returns
        -------
        Tensor
            Per-node ``matrix`` or networked ``effective_matrix``.

        Raises
        ------
        ValueError
            If a networked operator is missing required topology arguments.
        """
        from koopman_graph.operators import ContinuousGraphKoopmanOperator

        if isinstance(koopman, HeteroGraphKoopmanOperator):
            if edge_indices is None or num_nodes is None:
                msg = (
                    "edge_indices and num_nodes are required for "
                    "EigenvalueRegularizationLoss on HeteroGraphKoopmanOperator "
                    "dense/odo modes (topology-coupled effective operator); "
                    "the per-node contract matrix K_self is not a substitute"
                )
                raise ValueError(msg)
            return koopman.effective_matrix(
                edge_indices,
                num_nodes,
                edge_weights=edge_weights,
            )
        if isinstance(koopman, HypergraphKoopmanOperator):
            if hyperedge_index is None or num_nodes is None:
                msg = (
                    "hyperedge_index and num_nodes are required for "
                    "EigenvalueRegularizationLoss on HypergraphKoopmanOperator "
                    "dense/odo modes (topology-coupled effective operator); "
                    "the per-node contract matrix K_self is not a substitute"
                )
                raise ValueError(msg)
            return koopman.effective_matrix(
                hyperedge_index,
                num_nodes,
                hyperedge_weight=hyperedge_weight,
            )
        if isinstance(koopman, ContinuousGraphKoopmanOperator):
            if edge_index is None or num_nodes is None:
                msg = (
                    "edge_index and num_nodes are required for "
                    "EigenvalueRegularizationLoss on "
                    "ContinuousGraphKoopmanOperator dense/odo modes "
                    "(topology-coupled effective generator); the per-node "
                    "contract matrix L_self is not a substitute"
                )
                raise ValueError(msg)
            return koopman.effective_generator(
                edge_index,
                num_nodes,
                edge_weight=edge_weight,
            )
        if not isinstance(koopman, GraphKoopmanOperator):
            return koopman.matrix
        if edge_index is None or num_nodes is None:
            msg = (
                "edge_index and num_nodes are required for "
                "EigenvalueRegularizationLoss on GraphKoopmanOperator "
                "dense/odo modes (topology-coupled effective operator); "
                "the per-node contract matrix K_self is not a substitute"
            )
            raise ValueError(msg)
        return koopman.effective_matrix(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )


def _assembled_eigreg_exceeds_ceiling(
    koopman: KoopmanPropagator,
    *,
    num_nodes: int | None,
) -> bool:
    """Return whether assembled networked eig-reg should no-op.

    Parameters
    ----------
    koopman : KoopmanOperatorContract
        Operator under regularization.
    num_nodes : int or None
        Node count for the assembled ``N·d`` map.

    Returns
    -------
    bool
        ``True`` when ``koopman`` is a networked dense/ODO operator and
        ``N·d`` exceeds :data:`MAX_ASSEMBLED_EIGREG_SIZE`.
    """
    from koopman_graph.operators import ContinuousGraphKoopmanOperator

    if num_nodes is None:
        return False
    if koopman.parameterization not in {"dense", "odo"}:
        return False
    if not isinstance(
        koopman,
        (
            GraphKoopmanOperator,
            HypergraphKoopmanOperator,
            HeteroGraphKoopmanOperator,
            ContinuousGraphKoopmanOperator,
        ),
    ):
        return False
    return num_nodes * koopman.latent_dim > MAX_ASSEMBLED_EIGREG_SIZE


class KoopmanSparsityLoss(nn.Module):
    """Penalize magnitude of Koopman operator entries (SINDy-style sparsity).

    Applies an element-wise :math:`L_1` penalty (default) or a smoothed
    :math:`L_p` penalty with :math:`p < 1`:

    .. math::

        \\mathcal{L}_{\\mathrm{sp}}
        = \\mathrm{mean}\\big((|K_{ij}|^2 + \\varepsilon)^{p/2}\\big)

    Target matrices
    ---------------
    * Ordinary / custom operators — public
      :attr:`~koopman_graph.operators.KoopmanOperatorContract.matrix`
      (assembled ``K`` or continuous generator ``L``).
    * :class:`~koopman_graph.operators.GraphKoopmanOperator` — public
      :attr:`~koopman_graph.operators.GraphKoopmanOperator.K_self` and
      :attr:`~koopman_graph.operators.GraphKoopmanOperator.K_nbr`
      **only**. This is **parameter** sparsity on the self/neighbor factors,
      not sparsity of the topology-bound
      :meth:`~koopman_graph.operators.GraphKoopmanOperator.effective_matrix`.

    Notes
    -----
    Sparse latent factors do **not** imply a sparse physical adjacency or an
    interpretable governing equation in node coordinates (contrast SINDy on
    identified observables; Brunton et al., 2016). This term is a soft training
    regularizer.
    """

    def __init__(self, *, p: float = 1.0, eps: float = 1e-8) -> None:
        """Configure the sparsity exponent and smoother.

        Parameters
        ----------
        p : float, optional
            Penalty exponent. ``1.0`` is pure mean absolute value; values in
            ``(0, 1)`` enable the smoothed :math:`L_p` form. Default is ``1.0``.
        eps : float, optional
            Smoother for :math:`p < 1` (ignored when ``p == 1``). Must be
            positive. Default is ``1e-8``.

        Raises
        ------
        ValueError
            If ``p`` is not in ``(0, 1]`` or ``eps`` is not positive.
        """
        super().__init__()
        if not 0.0 < p <= 1.0:
            msg = f"p must be in (0, 1], got {p}"
            raise ValueError(msg)
        if eps <= 0.0:
            msg = f"eps must be positive, got {eps}"
            raise ValueError(msg)
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, koopman: KoopmanPropagator) -> Tensor:
        """Compute the mean sparsity penalty over targeted operator entries.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Operator whose public matrix factors are penalized.

        Returns
        -------
        Tensor
            Scalar sparsity penalty.
        """
        entries = self._target_entries(koopman)
        if self.p == 1.0:
            return entries.abs().mean()
        return (entries.square() + self.eps).pow(self.p / 2.0).mean()

    @staticmethod
    def _target_entries(koopman: KoopmanPropagator) -> Tensor:
        """Return the flattened entries used as the sparsity target.

        Parameters
        ----------
        koopman : KoopmanOperatorContract
            Operator providing public ``matrix`` or graph ``K_self`` /
            ``K_nbr`` factors.

        Returns
        -------
        Tensor
            1-D view of targeted entries.
        """
        if isinstance(koopman, GraphKoopmanOperator):
            return torch.cat(
                (koopman.K_self.reshape(-1), koopman.K_nbr.reshape(-1)),
                dim=0,
            )
        if isinstance(koopman, HypergraphKoopmanOperator):
            return torch.cat(
                (koopman.K_self.reshape(-1), koopman.K_hedge.reshape(-1)),
                dim=0,
            )
        return koopman.matrix.reshape(-1)
