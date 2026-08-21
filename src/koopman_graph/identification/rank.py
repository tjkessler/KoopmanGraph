"""Package-owned latent-rank selection on frozen encodings.

For each candidate rank :math:`r`, encodings are projected onto the
leading :math:`r` right singular vectors of the train cloud and scored
by VAMP-2, a finite-dictionary ResDMD residual, or a
stability-penalized held-out one-step mean squared error (MSE). This
selects a numerical dictionary rank. It is **not** Ray Tune / AutoML
for encoder ``latent_dim``, and it does not train a model per
candidate.

``criterion="vamp2"`` uses the in-tree topology-blind
:func:`~koopman_graph.baselines.vamp2.vamp2_score` (Wu and Noé, 2020).
deeptime (``[msm]``) is an optional cross-check, not a runtime
requirement. ``criterion="resdmd_elbow"`` is not a certified
infinite-dimensional residual bound.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`. ``analysis.resdmd`` and ``baselines.vamp2``
are imported lazily inside the matching criterion branch so loading
this package does not import :mod:`koopman_graph.model`.

References
----------
Wu, H. & Noé, F. (2020). Variational approach for learning Markov
processes from time series data. *Journal of Nonlinear Science*,
30(1), 23–66. https://doi.org/10.1007/s00332-019-09567-y
(``Wu2020VAMP``)

Colbrook, M. J. and Townsend, A. (2023/2024). Rigorous data-driven
computation of spectral properties of Koopman operators for dynamical
systems. *Communications on Pure and Applied Mathematics*, 77(1),
221–283. https://doi.org/10.1002/cpa.22125
(``ColbrookTownsend2023ResDMD``)

Hoffmann, M., Scherer, M., Hempel, T., Mardt, A., de Silva, B.,
Husic, B. E., Klus, S., Wu, H., Kutz, J. N., Brunton, S. L. & Noé, F.
(2022). Deeptime: a Python library for machine learning dynamical
models from time series data. *Machine Learning: Science and
Technology*, 3(1), 015009. https://doi.org/10.1088/2632-2153/ac3de0
(``deeptime2021``)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from koopman_graph.identification.config import IdentificationConfig
from koopman_graph.identification.invariance import SINGULAR_VALUE_REL_CUTOFF
from koopman_graph.identification.protocol import LatentPairs
from koopman_graph.identification.solvers import identify_operator

__all__ = [
    "DEFAULT_RANK_RIDGE",
    "DEFAULT_STABILITY_PENALTY",
    "LatentRankCriterion",
    "LatentRankReport",
    "VAMP2_PLATEAU_RELATIVE",
    "select_latent_rank",
]

LatentRankCriterion = Literal["vamp2", "resdmd_elbow", "stability_penalized"]

DEFAULT_RANK_RIDGE = 1e-4
DEFAULT_STABILITY_PENALTY = 1.0
VAMP2_PLATEAU_RELATIVE = 1e-3
_MIN_HELDOUT_TIMES = 4
_CRITERIA = frozenset({"vamp2", "resdmd_elbow", "stability_penalized"})


@dataclass(frozen=True)
class LatentRankReport:
    """Outcome of :func:`select_latent_rank`.

    ``scores`` align with ``candidates`` (ranks that were scored).
    ``rejected_alternatives`` lists candidate ranks dropped above the
    numerical SVD rank. This is not a certificate that encoder
    ``latent_dim`` should equal ``selected_rank``.

    Attributes
    ----------
    selected_rank : int
        Winning candidate.
    criterion : {"vamp2", "resdmd_elbow", "stability_penalized"}
        Scoring rule.
    candidates : tuple of int
        Sorted ranks that were scored.
    scores : tuple of float
        Criterion values for ``candidates`` (VAMP-2 score, max ResDMD
        residual, or penalized held-out MSE).
    rejected_alternatives : tuple of str
        Candidate ranks discarded as above numerical rank.
    numerical_rank : int
        Truncated-SVD rank of the train cloud.
    n_samples : int
        Number of train lag pairs used to form the basis / fit.
    """

    selected_rank: int
    criterion: LatentRankCriterion
    candidates: tuple[int, ...]
    scores: tuple[float, ...]
    rejected_alternatives: tuple[str, ...]
    numerical_rank: int
    n_samples: int

    def __post_init__(self) -> None:
        """Validate ranks, scores, and counts.

        Raises
        ------
        ValueError
            If ranks, lengths, or scores are invalid.
        """
        if self.criterion not in _CRITERIA:
            msg = (
                f"criterion must be one of {sorted(_CRITERIA)}, got {self.criterion!r}"
            )
            raise ValueError(msg)
        if type(self.selected_rank) is not int or self.selected_rank < 1:
            msg = f"selected_rank must be an int >= 1, got {self.selected_rank!r}"
            raise ValueError(msg)
        if type(self.numerical_rank) is not int or self.numerical_rank < 1:
            msg = f"numerical_rank must be an int >= 1, got {self.numerical_rank!r}"
            raise ValueError(msg)
        if type(self.n_samples) is not int or self.n_samples < 1:
            msg = f"n_samples must be an int >= 1, got {self.n_samples!r}"
            raise ValueError(msg)
        if len(self.candidates) != len(self.scores) or len(self.candidates) < 1:
            msg = "candidates and scores must be non-empty and the same length"
            raise ValueError(msg)
        if self.selected_rank not in self.candidates:
            msg = (
                f"selected_rank {self.selected_rank} is not among scored "
                f"candidates {self.candidates}"
            )
            raise ValueError(msg)
        for rank in self.candidates:
            if type(rank) is not int or rank < 1:
                msg = f"candidates must be positive ints, got {rank!r}"
                raise ValueError(msg)
        for score in self.scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                msg = f"scores must be finite floats, got {type(score).__name__}"
                raise ValueError(msg)
            if not math.isfinite(float(score)):
                msg = f"scores must be finite, got {score!r}"
                raise ValueError(msg)
        if not isinstance(self.rejected_alternatives, tuple) or any(
            not isinstance(name, str) or name == ""
            for name in self.rejected_alternatives
        ):
            msg = "rejected_alternatives must be a tuple of non-empty strings"
            raise ValueError(msg)


def _finite_float(value: float) -> bool:
    """Return whether ``value`` is a finite Python float.

    Parameters
    ----------
    value : float
        Scalar to test.

    Returns
    -------
    bool
        ``True`` when ``value`` is finite.
    """
    return value == value and value not in (float("inf"), float("-inf"))


def _as_time_major(encodings: Tensor, *, name: str) -> Tensor:
    """Require time-major encodings with trailing width.

    Parameters
    ----------
    encodings : Tensor
        ``(T, d)`` or ``(T, N, d)``.
    name : str
        Field name for errors.

    Returns
    -------
    Tensor
        The input tensor if the layout is valid.

    Raises
    ------
    ValueError
        If the layout is not 2-D / 3-D or ``T < 2``.
    """
    if encodings.ndim not in {2, 3}:
        msg = (
            f"{name} must have shape (T, d) or (T, N, d), got {tuple(encodings.shape)}"
        )
        raise ValueError(msg)
    if not encodings.is_floating_point():
        msg = f"{name} must be floating-point, got {encodings.dtype}"
        raise ValueError(msg)
    if int(encodings.shape[0]) < 2:
        msg = (
            f"{name} requires T >= 2 consecutive encodings, got T={encodings.shape[0]}"
        )
        raise ValueError(msg)
    if int(encodings.shape[-1]) < 1:
        msg = f"{name} requires trailing width d >= 1, got d={encodings.shape[-1]}"
        raise ValueError(msg)
    return encodings


def _lag_pairs(encodings: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten consecutive snapshots to ``(M, d)`` lag pairs.

    Parameters
    ----------
    encodings : Tensor
        Time-major ``(T, d)`` or ``(T, N, d)``.

    Returns
    -------
    tuple of Tensor
        Source and target with shape ``((T-1) N, d)``.
    """
    width = encodings.shape[-1]
    return encodings[:-1].reshape(-1, width), encodings[1:].reshape(-1, width)


def _right_singular_basis(rows: Tensor) -> Tensor:
    """Leading right singular vectors of the encoding cloud.

    Parameters
    ----------
    rows : Tensor
        Stacked samples with shape ``(n_samples, d)``.

    Returns
    -------
    Tensor
        ``Vh`` with shape ``(rank, d)``.

    Raises
    ------
    ValueError
        If every singular value is below the relative cutoff.
    """
    _, singular_values, vh = torch.linalg.svd(rows, full_matrices=False)
    peak = float(singular_values[0].item()) if singular_values.numel() else 0.0
    if not math.isfinite(peak) or peak <= 0.0:
        msg = "encoding basis is degenerate (no positive singular value)"
        raise ValueError(msg)
    cutoff = SINGULAR_VALUE_REL_CUTOFF * peak
    rank = int((singular_values > cutoff).sum().item())
    if rank < 1:
        msg = (
            "encoding basis rank is 0 after relative SVD cutoff "
            f"{SINGULAR_VALUE_REL_CUTOFF:g}"
        )
        raise ValueError(msg)
    return vh[:rank]


def _project(encodings: Tensor, basis: Tensor) -> Tensor:
    """Project trailing features onto ``basis`` rows.

    Parameters
    ----------
    encodings : Tensor
        Time-major encodings with trailing width ``d``.
    basis : Tensor
        Right singular vectors with shape ``(r, d)``.

    Returns
    -------
    Tensor
        Encodings with trailing width ``r``.
    """
    return encodings @ basis.to(dtype=encodings.dtype, device=encodings.device).T


def _parse_candidates(candidates: Sequence[int]) -> tuple[int, ...]:
    """Return unique sorted positive ranks.

    Parameters
    ----------
    candidates : sequence of int
        Proposed ranks.

    Returns
    -------
    tuple of int
        Unique ranks in increasing order.

    Raises
    ------
    ValueError
        If ``candidates`` is empty or contains a non-positive int.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        msg = "candidates must be a sequence of positive ints"
        raise ValueError(msg)
    parsed: list[int] = []
    for rank in candidates:
        if type(rank) is not int or rank < 1:
            msg = f"candidates must be positive ints, got {rank!r}"
            raise ValueError(msg)
        parsed.append(rank)
    if not parsed:
        msg = "candidates must be a non-empty sequence of positive ints"
        raise ValueError(msg)
    return tuple(sorted(set(parsed)))


def _spectral_radius(matrix: Tensor) -> float:
    """Return :math:`\\rho(K)` as a finite float.

    Parameters
    ----------
    matrix : Tensor
        Square operator.

    Returns
    -------
    float
        Spectral radius.

    Raises
    ------
    ValueError
        If the radius is non-finite.
    """
    radius = float(torch.linalg.eigvals(matrix).abs().max().real.item())
    if not math.isfinite(radius):
        msg = f"spectral radius is non-finite, got {radius!r}"
        raise ValueError(msg)
    return radius


def _vamp2_score(left: Tensor, right: Tensor) -> float:
    """Lazy in-tree VAMP-2 score on lag pairs.

    Parameters
    ----------
    left, right : Tensor
        Flattened lag pair.

    Returns
    -------
    float
        Topology-blind VAMP-2 score.
    """
    from koopman_graph.baselines.vamp2 import vamp2_score

    return float(vamp2_score(left, right).item())


def _resdmd_max_residual(left: Tensor, right: Tensor) -> float:
    """Lazy max finite-dictionary ResDMD residual on a projected dictionary.

    Parameters
    ----------
    left, right : Tensor
        Flattened lag pair (the dictionary).

    Returns
    -------
    float
        Maximum Colbrook–Townsend residual.
    """
    from koopman_graph.analysis.resdmd import resdmd

    report = resdmd(left, right)
    return float(report.residuals.max().item())


def _stability_score(
    train_left: Tensor,
    train_right: Tensor,
    score_left: Tensor,
    score_right_ambient: Tensor,
    basis_r: Tensor,
    *,
    ridge: float,
    penalty: float,
) -> float:
    """Held-out ambient reconstruction MSE plus a spectral-radius excess penalty.

    Parameters
    ----------
    train_left, train_right : Tensor
        Projected pairs used to fit ``K``.
    score_left : Tensor
        Held-out projected source encodings.
    score_right_ambient : Tensor
        Held-out target encodings in the original trailing width.
    basis_r : Tensor
        Top-``r`` right singular vectors with shape ``(r, d)``.
    ridge : float
        Tikhonov weight.
    penalty : float
        Weight on ``max(0, ρ(K) - 1)``.

    Returns
    -------
    float
        Penalized held-out ambient MSE.
    """
    snapshot = identify_operator(
        LatentPairs(z_t=train_left, z_next=train_right),
        IdentificationConfig(solver="ridge", ridge=ridge),
    )
    matrix = snapshot.matrix
    if matrix is None:
        msg = "identify_operator must return a dense matrix"
        raise ValueError(msg)
    predicted = (score_left @ matrix.T) @ basis_r.to(
        dtype=score_left.dtype,
        device=score_left.device,
    )
    mse = float((predicted - score_right_ambient).square().mean().item())
    excess = max(0.0, _spectral_radius(matrix) - 1.0)
    return mse + penalty * excess


def _pick_rank(
    candidates: tuple[int, ...],
    scores: tuple[float, ...],
    *,
    criterion: LatentRankCriterion,
) -> int:
    """Apply the documented selection rule.

    Parameters
    ----------
    candidates, scores : tuple
        Aligned scored ranks.
    criterion : str
        Scoring rule.

    Returns
    -------
    int
        Selected rank.
    """
    if criterion == "vamp2":
        peak = max(scores)
        floor = (1.0 - VAMP2_PLATEAU_RELATIVE) * peak
        eligible = [
            rank
            for rank, score in zip(candidates, scores, strict=True)
            if score >= floor
        ]
        return min(eligible)
    best = scores[0]
    chosen = candidates[0]
    for rank, score in zip(candidates[1:], scores[1:], strict=True):
        if score < best:
            best = score
            chosen = rank
    return chosen


def select_latent_rank(
    z_pairs: Tensor,
    candidates: Sequence[int],
    *,
    criterion: LatentRankCriterion,
    validation_z: Tensor | None = None,
    ridge: float = DEFAULT_RANK_RIDGE,
    stability_penalty: float = DEFAULT_STABILITY_PENALTY,
) -> LatentRankReport:
    """Select a numerical rank of frozen encodings over a candidate grid.

    Distinct from :mod:`koopman_graph.tuning` Ray Tune scaffolds and from
    choosing encoder ``latent_dim`` by held-out forecast HPO. Existing
    VAMP-2 training weights and ResDMD helpers still ship.

    Parameters
    ----------
    z_pairs : Tensor
        Train encodings with shape ``(T, d)`` or ``(T, N, d)``, ``T >= 2``.
    candidates : sequence of int
        Positive ranks to score. Values above the numerical SVD rank are
        rejected rather than scored.
    criterion : {"vamp2", "resdmd_elbow", "stability_penalized"}
        ``"vamp2"`` maximizes in-tree VAMP-2 and then takes the smallest
        rank within relative gap :data:`VAMP2_PLATEAU_RELATIVE` of the
        peak. ``"resdmd_elbow"`` minimizes the max finite-dictionary
        ResDMD residual (ties → smallest rank). ``"stability_penalized"``
        minimizes held-out one-step MSE plus
        ``stability_penalty * max(0, ρ(K) - 1)`` (ties → smallest rank).
        The MSE is ambient reconstruction of the held-out encodings, not
        the error in the truncated coordinates.
    validation_z : Tensor or None, optional
        Optional hold-out encodings with the same trailing width (and
        node axis when 3-D). Default is ``None``. For
        ``stability_penalized`` without ``validation_z``, the last half
        of ``z_pairs`` is the hold-out (requires ``T >= 4``).
    ridge : float, optional
        Tikhonov weight for the ridge map used by
        ``stability_penalized``. Default ``1e-4``.
    stability_penalty : float, optional
        Weight on spectral-radius excess. Default ``1.0``.

    Returns
    -------
    LatentRankReport
        Selected rank, scores, and rejected candidates.

    Raises
    ------
    ValueError
        If layouts, ``criterion``, ``candidates``, or scalars are
        invalid, or no candidate is at most the numerical rank.
    """
    if criterion not in _CRITERIA:
        msg = f"criterion must be one of {sorted(_CRITERIA)}, got {criterion!r}"
        raise ValueError(msg)
    if not _finite_float(float(ridge)) or float(ridge) < 0.0:
        msg = f"ridge must be a finite non-negative float, got {ridge}"
        raise ValueError(msg)
    if not _finite_float(float(stability_penalty)) or float(stability_penalty) < 0.0:
        msg = (
            "stability_penalty must be a finite non-negative float, "
            f"got {stability_penalty}"
        )
        raise ValueError(msg)

    train = _as_time_major(z_pairs, name="z_pairs")
    ranks = _parse_candidates(candidates)
    score_encodings = train
    fit_encodings = train
    if validation_z is not None:
        held = _as_time_major(validation_z, name="validation_z")
        if held.shape[1:] != train.shape[1:]:
            msg = (
                "validation_z trailing layout must match z_pairs, "
                f"got {tuple(held.shape)} vs {tuple(train.shape)}"
            )
            raise ValueError(msg)
        score_encodings = held
    elif criterion == "stability_penalized":
        n_times = int(train.shape[0])
        if n_times < _MIN_HELDOUT_TIMES:
            msg = (
                "stability_penalized without validation_z requires "
                f"T >= {_MIN_HELDOUT_TIMES}, got T={n_times}"
            )
            raise ValueError(msg)
        split = n_times // 2
        fit_encodings = train[:split]
        score_encodings = train[split:]

    basis = _right_singular_basis(fit_encodings.reshape(-1, fit_encodings.shape[-1]))
    numerical_rank = int(basis.shape[0])
    scored_ranks = tuple(rank for rank in ranks if rank <= numerical_rank)
    rejected = tuple(str(rank) for rank in ranks if rank > numerical_rank)
    if not scored_ranks:
        msg = (
            "no candidate rank is at most the numerical encoding rank "
            f"{numerical_rank}; candidates={ranks}"
        )
        raise ValueError(msg)

    scores: list[float] = []
    train_left_full, train_right_full = _lag_pairs(_project(fit_encodings, basis))
    score_left_full, score_right_full = _lag_pairs(_project(score_encodings, basis))
    _, score_right_ambient = _lag_pairs(score_encodings)
    n_samples = int(train_left_full.shape[0])
    for rank in scored_ranks:
        train_left = train_left_full[:, :rank]
        train_right = train_right_full[:, :rank]
        score_left = score_left_full[:, :rank]
        score_right = score_right_full[:, :rank]
        if criterion == "vamp2":
            scores.append(_vamp2_score(score_left, score_right))
        elif criterion == "resdmd_elbow":
            scores.append(_resdmd_max_residual(score_left, score_right))
        else:
            scores.append(
                _stability_score(
                    train_left,
                    train_right,
                    score_left,
                    score_right_ambient,
                    basis[:rank],
                    ridge=float(ridge),
                    penalty=float(stability_penalty),
                )
            )

    score_tuple = tuple(scores)
    selected = _pick_rank(scored_ranks, score_tuple, criterion=criterion)
    return LatentRankReport(
        selected_rank=selected,
        criterion=criterion,
        candidates=scored_ranks,
        scores=score_tuple,
        rejected_alternatives=rejected,
        numerical_rank=numerical_rank,
        n_samples=n_samples,
    )
