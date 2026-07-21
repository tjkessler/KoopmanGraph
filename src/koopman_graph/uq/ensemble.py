"""Deep-ensemble predictive intervals for GraphKoopman models.

Composes independently seeded :class:`~koopman_graph.model.GraphKoopmanModel`
members without subclassing. Member forecasts call
:func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` directly
(same hold-last topology policy as ``predict``). Predictive uncertainty
follows the deep-ensemble practice of Lakshminarayanan et al. (NeurIPS
2017): empirical mean and quantiles across member forecasts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    hold_last_topology_at,
    pack_rollout_snapshots,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.validation import validate_controls
from koopman_graph.training import FitHistory
from koopman_graph.uq.common import (
    PredictionInterval,
    quantile_levels,
    snapshot_with_features,
)

MemberFactory = Callable[[], GraphKoopmanModel]

_MANIFEST_NAME = "ensemble_manifest.json"
_MEMBER_PATTERN = "member_{index:02d}.pt"


@runtime_checkable
class IntervalForecastModel(Protocol):
    """Optional façade for models that expose predictive intervals.

    Detect via ``isinstance(..., IntervalForecastModel)`` or
    ``hasattr(model, "predict_interval")``. This is **not** part of the
    required :class:`~koopman_graph.protocols.ForecastModel` contract.

    Notes
    -----
    Prefer this Protocol for static typing of optional UQ call sites.
    """

    def predict_interval(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        *args: Any,
        level: float = 0.9,
        **kwargs: Any,
    ) -> PredictionInterval:
        """Return mean and empirical predictive bounds.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot for the rollout.
        steps : int
            Number of forecast steps.
        *args, **kwargs
            Implementer-specific options forwarded to member predictors.
        level : float, optional
            Nominal central coverage in ``(0, 1)``. Default is ``0.9``.

        Returns
        -------
        PredictionInterval
            Mean forecast plus lower/upper empirical quantiles.
        """
        ...


def empirical_coverage(
    targets: Sequence[Data],
    interval: PredictionInterval,
) -> float:
    """Fraction of target entries that fall inside an empirical interval.

    Coverage is computed entrywise over all forecast steps, nodes, and
    features present in ``targets``. Topology fields on ``Data`` objects are
    ignored.

    Parameters
    ----------
    targets : sequence of Data
        Ground-truth snapshots aligned with ``interval.mean`` (same length and
        ``x`` shapes).
    interval : PredictionInterval
        Ensemble interval whose ``lower`` / ``upper`` bound the targets.

    Returns
    -------
    float
        Empirical coverage in ``[0, 1]``.

    Raises
    ------
    ValueError
        If lengths or feature shapes disagree.
    """
    n_steps = len(targets)
    if n_steps == 0:
        msg = "targets must contain at least one snapshot"
        raise ValueError(msg)
    if (
        len(interval.lower) != n_steps
        or len(interval.upper) != n_steps
        or len(interval.mean) != n_steps
    ):
        msg = (
            "targets and interval must share the same number of steps; "
            f"got targets={n_steps}, mean={len(interval.mean)}, "
            f"lower={len(interval.lower)}, upper={len(interval.upper)}"
        )
        raise ValueError(msg)

    hits = 0
    total = 0
    for target, lower, upper in zip(
        targets, interval.lower, interval.upper, strict=True
    ):
        if target.x is None or lower.x is None or upper.x is None:
            msg = "all snapshots must define node features ``x``"
            raise ValueError(msg)
        if target.x.shape != lower.x.shape or target.x.shape != upper.x.shape:
            msg = (
                "target and interval feature shapes must match; "
                f"got target={tuple(target.x.shape)}, "
                f"lower={tuple(lower.x.shape)}, upper={tuple(upper.x.shape)}"
            )
            raise ValueError(msg)
        inside = (target.x >= lower.x) & (target.x <= upper.x)
        hits += int(inside.sum().item())
        total += int(inside.numel())
    return hits / total


def _member_autoregressive_rollout(
    member: GraphKoopmanModel,
    initial_graph: Tensor | Data,
    steps: int,
    *,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    controls: Sequence[Tensor] | None,
    future_topologies: Sequence[Data] | None,
    history: Sequence[Data] | None,
) -> list[Data]:
    """Roll out one ensemble member via the shared latent rollout primitive.

    Mirrors :meth:`~koopman_graph.model.GraphKoopmanModel.predict` semantics
    (eval mode, no grad, hold-last topology, control validation) but calls
    :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` directly
    instead of going through ``member.predict``.

    Returns
    -------
    list[Data]
        Decoded forecast snapshots for the requested horizon.
    """
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)

    validate_controls(
        control_dim=member.control_dim,
        controls=controls,
        steps=steps,
    )

    was_training = member.training
    member.eval()
    try:
        with torch.no_grad():
            z, origin_edge, origin_weight = member.encode_rollout_origin(
                initial_graph,
                edge_index=edge_index,
                edge_weight=edge_weight,
                history=history,
            )
            control_at = None if controls is None else (lambda step: controls[step])
            rollout = autoregressive_latent_rollout(
                member.koopman,
                member.decoder,
                z,
                steps=steps,
                topology_at=hold_last_topology_at(
                    origin_edge,
                    origin_weight,
                    future_topologies,
                ),
                control_at=control_at,
                default_delta_t=member.time_step,
            )
            return pack_rollout_snapshots(rollout)
    finally:
        member.train(was_training)


def _stack_member_features(
    member_rollouts: Sequence[Sequence[Data]],
) -> list[Tensor]:
    """Stack member ``x`` tensors per step into ``(n_members, ...)`` tensors.

    Parameters
    ----------
    member_rollouts : sequence of sequence of Data
        One rollout sequence per ensemble member.

    Returns
    -------
    list of Tensor
        Per-step stacked features with leading member dimension.

    Raises
    ------
    ValueError
        If rollouts are empty, unequal in length, or missing ``x``.
    """
    if not member_rollouts:
        msg = "ensemble must contain at least one member rollout"
        raise ValueError(msg)
    n_steps = len(member_rollouts[0])
    if any(len(rollout) != n_steps for rollout in member_rollouts):
        msg = "all member rollouts must have the same number of steps"
        raise ValueError(msg)

    stacked: list[Tensor] = []
    for step in range(n_steps):
        features = [rollout[step].x for rollout in member_rollouts]
        if any(feat is None for feat in features):
            msg = "all predicted snapshots must define node features ``x``"
            raise ValueError(msg)
        stacked.append(torch.stack(features, dim=0))
    return stacked


class EnsembleGraphKoopmanModel:
    """Deep ensemble of independently seeded GraphKoopman models.

    Aggregates member autoregressive forecasts into an empirical mean and
    quantile interval. Each member rollout uses
    :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` with the
    same hold-last topology policy as
    :meth:`~koopman_graph.model.GraphKoopmanModel.predict`. Members remain
    ordinary :class:`~koopman_graph.model.GraphKoopmanModel` instances:
    spectrum, regularization, and ``koopman="graph"`` topology contracts are
    unchanged.

    This wrapper is a **power-user** type (``koopman_graph.uq``). It is not
    re-exported on the root façade.

    Parameters
    ----------
    members : sequence of GraphKoopmanModel
        Independently initialized (and typically independently trained)
        ensemble members. Must be non-empty.
    """

    def __init__(self, members: Sequence[GraphKoopmanModel]) -> None:
        """Store ensemble members without copying model parameters.

        Parameters
        ----------
        members : sequence of GraphKoopmanModel
            Non-empty sequence of independently seeded models.

        Raises
        ------
        ValueError
            If ``members`` is empty.
        """
        if len(members) == 0:
            msg = "EnsembleGraphKoopmanModel requires at least one member"
            raise ValueError(msg)
        self._members = list(members)

    @classmethod
    def from_factory(
        cls,
        factory: MemberFactory,
        n_members: int,
        *,
        seeds: Sequence[int] | None = None,
    ) -> EnsembleGraphKoopmanModel:
        """Build members by calling ``factory`` under distinct RNG seeds.

        Parameters
        ----------
        factory : callable
            Zero-argument callable returning a fresh
            :class:`~koopman_graph.model.GraphKoopmanModel`.
        n_members : int
            Ensemble size (must be >= 1).
        seeds : sequence of int or None, optional
            One seed per member. When omitted, uses ``0 .. n_members-1``.

        Returns
        -------
        EnsembleGraphKoopmanModel
            Ensemble whose members were constructed under the given seeds.
        """
        if n_members < 1:
            msg = f"n_members must be >= 1; got {n_members}"
            raise ValueError(msg)
        resolved = list(range(n_members) if seeds is None else seeds)
        if len(resolved) != n_members:
            msg = (
                "seeds must have length n_members; "
                f"got len(seeds)={len(resolved)}, n_members={n_members}"
            )
            raise ValueError(msg)

        members: list[GraphKoopmanModel] = []
        for seed in resolved:
            torch.manual_seed(int(seed))
            members.append(factory())
        return cls(members)

    @property
    def members(self) -> tuple[GraphKoopmanModel, ...]:
        """Ensemble members in construction order.

        Returns
        -------
        tuple of GraphKoopmanModel
            Immutable view of the stored members.
        """
        return tuple(self._members)

    @property
    def n_members(self) -> int:
        """Number of ensemble members.

        Returns
        -------
        int
            Ensemble size.
        """
        return len(self._members)

    def fit(
        self,
        data_sequence: Any,
        *,
        seeds: Sequence[int] | None = None,
        **fit_kwargs: Any,
    ) -> list[FitHistory]:
        """Fit each member independently (optionally under distinct seeds).

        Parameters
        ----------
        data_sequence
            Training input forwarded to each member's ``fit``.
        seeds : sequence of int or None, optional
            When provided, ``torch.manual_seed`` is set before each member
            ``fit``. Length must equal :attr:`n_members`.
        **fit_kwargs
            Forwarded unchanged to
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.

        Returns
        -------
        list of FitHistory
            Per-member training histories in member order.
        """
        if seeds is not None and len(seeds) != self.n_members:
            msg = (
                "seeds must have length n_members; "
                f"got len(seeds)={len(seeds)}, n_members={self.n_members}"
            )
            raise ValueError(msg)

        histories: list[FitHistory] = []
        for index, member in enumerate(self._members):
            if seeds is not None:
                torch.manual_seed(int(seeds[index]))
            histories.append(member.fit(data_sequence, **fit_kwargs))
        return histories

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Return the ensemble-mean autoregressive forecast.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot for the rollout.
        steps : int
            Number of future snapshots to predict.
        edge_index : Tensor or None, optional
            Edge index used when ``initial_graph`` is a feature tensor.
        edge_weight : Tensor or None, optional
            Edge weights used when ``initial_graph`` is a feature tensor.
        controls : sequence of Tensor or None, optional
            Future controls for each member rollout.
        future_topologies : sequence of Data or None, optional
            Known future topologies for hold-last rollout scheduling.
        history : sequence of Data or None, optional
            Delay-embedding history for each member encode.

        Returns
        -------
        list of Data
            Ensemble-mean predicted snapshots.
        """
        interval = self.predict_interval(
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
            level=0.9,
        )
        return interval.mean

    def predict_interval(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        *,
        level: float = 0.9,
    ) -> PredictionInterval:
        """Return mean and empirical quantile bounds across members.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Initial graph snapshot for the rollout.
        steps : int
            Number of future snapshots to predict.
        edge_index : Tensor or None, optional
            Edge index used when ``initial_graph`` is a feature tensor.
        edge_weight : Tensor or None, optional
            Edge weights used when ``initial_graph`` is a feature tensor.
        controls : sequence of Tensor or None, optional
            Future controls for each member rollout.
        future_topologies : sequence of Data or None, optional
            Known future topologies for hold-last rollout scheduling.
        history : sequence of Data or None, optional
            Delay-embedding history for each member encode.
        level : float, optional
            Nominal central coverage in ``(0, 1)``. Default ``0.9`` uses the
            5th and 95th empirical percentiles across members.

        Returns
        -------
        PredictionInterval
            Mean forecast plus lower/upper empirical quantiles.
        """
        lower_q, upper_q = quantile_levels(level)
        member_rollouts = [
            _member_autoregressive_rollout(
                member,
                initial_graph,
                steps,
                edge_index=edge_index,
                edge_weight=edge_weight,
                controls=controls,
                future_topologies=future_topologies,
                history=history,
            )
            for member in self._members
        ]
        stacked = _stack_member_features(member_rollouts)
        template = member_rollouts[0]

        mean_snaps: list[Data] = []
        lower_snaps: list[Data] = []
        upper_snaps: list[Data] = []
        for step, features in enumerate(stacked):
            mean_x = features.mean(dim=0)
            if features.shape[0] == 1:
                lower_x = mean_x.clone()
                upper_x = mean_x.clone()
            else:
                # torch.quantile expects float; keep device/dtype of members.
                q = torch.tensor(
                    [lower_q, upper_q],
                    device=features.device,
                    dtype=features.dtype,
                )
                bounds = torch.quantile(features.float(), q, dim=0).to(
                    dtype=features.dtype
                )
                lower_x = bounds[0]
                upper_x = bounds[1]
            mean_snaps.append(snapshot_with_features(template[step], mean_x))
            lower_snaps.append(snapshot_with_features(template[step], lower_x))
            upper_snaps.append(snapshot_with_features(template[step], upper_x))

        return PredictionInterval(
            mean=mean_snaps,
            lower=lower_snaps,
            upper=upper_snaps,
            level=level,
            n_members=self.n_members,
        )

    def save(self, directory: str | Path) -> None:
        """Persist each member via format-1 ``GraphKoopmanModel`` checkpoints.

        Writes ``member_XX.pt`` files plus ``ensemble_manifest.json`` under
        ``directory``. Does not introduce a new checkpoint schema version.

        Parameters
        ----------
        directory : str or Path
            Destination directory (created when missing).
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        member_files: list[str] = []
        for index, member in enumerate(self._members):
            filename = _MEMBER_PATTERN.format(index=index)
            member.save(root / filename)
            member_files.append(filename)
        manifest = {
            "kind": "EnsembleGraphKoopmanModel",
            "n_members": self.n_members,
            "members": member_files,
            # Members use the library format-1 baseline; this wrapper does not
            # bump FORMAT_VERSION.
            "member_format": "GraphKoopmanModel.save",
        }
        (root / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> EnsembleGraphKoopmanModel:
        """Load an ensemble saved by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Directory containing ``ensemble_manifest.json`` and member
            checkpoints.
        map_location : str, torch.device, or None, optional
            Forwarded to each member
            :meth:`~koopman_graph.model.GraphKoopmanModel.load`.

        Returns
        -------
        EnsembleGraphKoopmanModel
            Reconstructed ensemble.
        """
        root = Path(directory)
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.is_file():
            msg = f"ensemble manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        member_files = manifest.get("members")
        if not isinstance(member_files, list) or not member_files:
            msg = "ensemble manifest must list at least one member checkpoint"
            raise ValueError(msg)
        members = [
            GraphKoopmanModel.load(root / filename, map_location=map_location)
            for filename in member_files
        ]
        return cls(members)
