"""Joint latent-state and graph-factor observer (separable dictionary).

Composes :class:`~koopman_graph.adaptation.KoopmanObserver` with either
group-sparse :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}`
identification (graph operators) or
:class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` (dense
per-node :math:`K`). This is **not** Koopman-GKFA: it does not run
ADMM on :math:`A`, does not certify a three-term MSE bound, and does
not claim the Structural Homomorphism Lemma unless
``claim_homomorphism=True`` **and** the encoder is a separable
node-wise dictionary (Peng, Shen & Zhu, arXiv:2606.17797,
``Peng2026KoopmanGKFA``). Default GNN encoders mix neighbors and are
not separable.

References
----------
Peng, C., Shen, X. & Zhu, Y. (2026). Koopman lifting with certified
error bounds for joint inference in nonlinear networks. *arXiv*
2606.17797. (``Peng2026KoopmanGKFA``). Provisional preprint. Cited
for the separable-dictionary homomorphism precondition only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.adaptation.kalman import FilterResult
from koopman_graph.adaptation.observer import KoopmanObserver, ObservationModel
from koopman_graph.adaptation.rls import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.data import GraphSnapshotSequence, resolve_pair_delta_t
from koopman_graph.graph_utils import snapshot_edge_weight
from koopman_graph.identification.sparse_factors import (
    SparseFactorGroup,
    SparseFactorMethod,
    SparseFactorReport,
    identify_sparse_graph_factors,
)
from koopman_graph.nn.separable import is_separable_dictionary
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.protocols import ModeShapeModel

__all__ = [
    "JointObserverResult",
    "JointStateTopologyObserver",
]


@dataclass(frozen=True)
class JointObserverResult:
    """One window of joint filtering and operator write-back.

    Attributes
    ----------
    filter : FilterResult
        Kalman filtered latents ``(T, N, d)``.
    sparse_factors : SparseFactorReport or None
        Graph-factor identification when the live operator is
        :class:`~koopman_graph.operators.GraphKoopmanOperator`.
    rls_steps : tuple of AdaptationStepResult
        Per-interval RLS updates on dense per-node :math:`K` (empty on
        the graph path).
    """

    filter: FilterResult
    sparse_factors: SparseFactorReport | None
    rls_steps: tuple[AdaptationStepResult, ...]


def _require_separable_graph_operator(
    model: ModeShapeModel,
    *,
    claim_homomorphism: bool,
) -> None:
    """Refuse theorem-tagged use without a separable graph lift.

    Parameters
    ----------
    model : ModeShapeModel
        Model exposing ``encoder`` and ``koopman``.
    claim_homomorphism : bool
        When ``True``, require a separable encoder and a one-tap graph
        operator.

    Raises
    ------
    ValueError
        If the homomorphism flag is set without the precondition.
    """
    if not claim_homomorphism:
        return
    encoder = getattr(model, "encoder", None)
    if not is_separable_dictionary(encoder):
        msg = (
            "claim_homomorphism=True requires encoder_kind='separable' "
            "(node-wise dictionary). Default GNN encoders mix neighbors "
            "and the Structural Homomorphism Lemma does not apply "
            "(Peng2026KoopmanGKFA)."
        )
        raise ValueError(msg)
    if not isinstance(model.koopman, GraphKoopmanOperator):
        msg = (
            "claim_homomorphism=True requires GraphKoopmanOperator "
            "(block sparsity of coupling vs graph edges). Dense per-node "
            "K has no graph topology to identify."
        )
        raise ValueError(msg)


def _write_graph_factors(
    koopman: GraphKoopmanOperator,
    k_self: Tensor,
    k_nbr: Tensor,
) -> None:
    """Copy identified dense factors onto the live graph operator.

    Parameters
    ----------
    koopman : GraphKoopmanOperator
        Dense one-tap graph operator.
    k_self, k_nbr : Tensor
        Factors with shape ``(d, d)``.

    Raises
    ------
    ValueError
        If the operator is not densely parameterized.
    """
    if koopman.parameterization != "dense":
        msg = (
            "joint observer write-back requires parameterization='dense', "
            f"got {koopman.parameterization!r}"
        )
        raise ValueError(msg)
    target_self = koopman.K_self
    target_nbr = koopman.K_nbr
    with torch.no_grad():
        target_self.copy_(k_self.to(device=target_self.device, dtype=target_self.dtype))
        target_nbr.copy_(k_nbr.to(device=target_nbr.device, dtype=target_nbr.dtype))


class JointStateTopologyObserver:
    """Kalman observer plus group-sparse graph factors or dense RLS.

    Graph path
        :meth:`KoopmanObserver.filter` then
        :func:`~koopman_graph.identification.identify_sparse_graph_factors`
        on consecutive filtered latents, writing :math:`K_{\\mathrm{self}}`
        / :math:`K_{\\mathrm{nbr}}`. Does **not** infer a new edge set.

    Dense per-node path
        Filter then :class:`RecursiveKoopmanAdapter` on latent pairs.
        Homomorphism claims are refused (no graph :math:`A`).

    Parameters
    ----------
    model : ModeShapeModel
        Fitted or seeded model with encode / decode / Koopman step.
    claim_homomorphism : bool, optional
        Theorem tag. Default ``False``. When ``True``, raise unless the
        encoder is separable **and** the operator is a one-tap graph
        Koopman operator.
    process_noise, observation_noise, observation_model
        Forwarded to :class:`KoopmanObserver`.
    sparse_group, sparse_method, sparse_threshold
        Forwarded to :func:`identify_sparse_graph_factors` on the graph
        path. ``sparse_threshold`` is dimensionless.
    rls_forgetting_factor, rls_regularization
        Forwarded to :class:`RecursiveKoopmanAdapter` on the dense path.
        ``rls_forgetting_factor`` is dimensionless in ``(0, 1]``.
        ``rls_regularization`` is the initial covariance scale on the
        regressor covariance (same units as RLS ``P``).
    """

    def __init__(
        self,
        model: ModeShapeModel,
        *,
        claim_homomorphism: bool = False,
        process_noise: float = 1e-3,
        observation_noise: float = 1e-2,
        observation_model: ObservationModel = "latent_encode",
        sparse_group: SparseFactorGroup = "self_nbr",
        sparse_method: SparseFactorMethod = "group_lasso",
        sparse_threshold: float = 0.0,
        rls_forgetting_factor: float = 0.99,
        rls_regularization: float = 1e3,
    ) -> None:
        """Initialize observer, optional RLS adapter, and theorem flag.

        Parameters
        ----------
        model : ModeShapeModel
            Encode / decode / Koopman façade.
        claim_homomorphism : bool, optional
            Require separable graph lift. Default ``False``.
        process_noise : float, optional
            Observer process-noise scale. Default ``1e-3``.
        observation_noise : float, optional
            Observer observation-noise scale. Default ``1e-2``.
        observation_model : {"latent_encode", "decoder_jacobian"}, optional
            Observer measurement linearization. Default ``"latent_encode"``.
        sparse_group : {"none", "self_nbr", "orbit"}, optional
            Factor grouping. Default ``"self_nbr"``.
        sparse_method : {"stlsq", "group_lasso"}, optional
            Sparse identification method. Default ``"group_lasso"``.
        sparse_threshold : float, optional
            Dimensionless cutoff / group-lasso ``λ``. Default ``0.0``.
        rls_forgetting_factor : float, optional
            Dense-path RLS ``λ``. Default ``0.99``.
        rls_regularization : float, optional
            Dense-path initial ``P`` scale. Default ``1e3``.

        Raises
        ------
        ValueError
            If ``claim_homomorphism=True`` without a separable graph
            operator, or the graph operator is not a one-tap dense
            symmetric / random-walk factorization.
        TypeError
            If the Koopman module is unsupported.
        """
        _require_separable_graph_operator(
            model,
            claim_homomorphism=claim_homomorphism,
        )
        koopman = model.koopman
        if isinstance(koopman, GraphKoopmanOperator):
            if int(koopman.filter_degree) != 1:
                msg = (
                    "JointStateTopologyObserver graph path requires "
                    f"filter_degree=1, got {koopman.filter_degree}"
                )
                raise ValueError(msg)
            if koopman.adjacency == "dual_random_walk":
                msg = (
                    "JointStateTopologyObserver does not support "
                    "adjacency='dual_random_walk'"
                )
                raise ValueError(msg)
            if koopman.parameterization != "dense":
                msg = (
                    "JointStateTopologyObserver graph path requires "
                    "parameterization='dense'"
                )
                raise ValueError(msg)
            self._rls: RecursiveKoopmanAdapter | None = None
            self._rls_mode: str | None = None
        elif isinstance(koopman, ContinuousKoopmanOperator):
            self._rls = RecursiveKoopmanAdapter.from_operator(
                koopman,
                mode="continuous",
                forgetting_factor=rls_forgetting_factor,
                regularization=rls_regularization,
            )
            self._rls_mode = "continuous"
        elif isinstance(koopman, KoopmanOperator):
            self._rls = RecursiveKoopmanAdapter.from_operator(
                koopman,
                mode="discrete",
                forgetting_factor=rls_forgetting_factor,
                regularization=rls_regularization,
            )
            self._rls_mode = "discrete"
        else:
            msg = (
                "JointStateTopologyObserver supports GraphKoopmanOperator, "
                "KoopmanOperator, or ContinuousKoopmanOperator, got "
                f"{type(koopman).__name__}"
            )
            raise TypeError(msg)

        self.model = model
        self.claim_homomorphism = bool(claim_homomorphism)
        self.sparse_group: SparseFactorGroup = sparse_group
        self.sparse_method: SparseFactorMethod = sparse_method
        self.sparse_threshold = float(sparse_threshold)
        self.observer = KoopmanObserver(
            model,
            process_noise=process_noise,
            observation_noise=observation_noise,
            observation_model=observation_model,
        )

    def filter_and_adapt(self, sequence: GraphSnapshotSequence) -> JointObserverResult:
        """Filter latents, then update graph factors or dense :math:`K`.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Homogeneous trajectory with ``T >= 2``.

        Returns
        -------
        JointObserverResult
            Filtered latents and the operator write-back payload.

        Raises
        ------
        ValueError
            If ``T < 2``.
        """
        if len(sequence) < 2:
            msg = f"filter_and_adapt requires T >= 2 snapshots, got T={len(sequence)}"
            raise ValueError(msg)
        filtered = self.observer.filter(sequence)
        koopman = self.model.koopman
        if isinstance(koopman, GraphKoopmanOperator):
            last = sequence[-1]
            report = identify_sparse_graph_factors(
                filtered.latents,
                last.edge_index,
                group=self.sparse_group,
                method=self.sparse_method,
                threshold=self.sparse_threshold,
                edge_weight=snapshot_edge_weight(last),
                adjacency=(
                    "random_walk" if koopman.adjacency == "random_walk" else "symmetric"
                ),
            )
            _write_graph_factors(koopman, report.K_self, report.K_nbr)
            return JointObserverResult(
                filter=filtered,
                sparse_factors=report,
                rls_steps=(),
            )
        assert self._rls is not None
        steps: list[AdaptationStepResult] = []
        latents = filtered.latents
        for timestep in range(int(latents.shape[0]) - 1):
            if self._rls_mode == "continuous":
                steps.append(
                    self._rls.update(
                        latents[timestep],
                        latents[timestep + 1],
                        delta_t=resolve_pair_delta_t(
                            sequence,
                            timestep,
                            default_time_step=float(self.model.time_step),
                        ),
                    )
                )
            else:
                steps.append(self._rls.update(latents[timestep], latents[timestep + 1]))
        self._rls.apply_to(koopman)
        return JointObserverResult(
            filter=filtered,
            sparse_factors=None,
            rls_steps=tuple(steps),
        )
