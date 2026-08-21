"""Latent-rank selection versus dense SVD rank and Ray Tune HPO."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT

import koopman_graph
import koopman_graph.identification as identification
from koopman_graph.identification import LatentRankReport, select_latent_rank

_SOURCE = REPO_ROOT / "src" / "koopman_graph" / "identification" / "rank.py"
_TRUE_RANK = 3
_AMBIENT = 8
_CANDIDATES = (1, 2, 3, 4, 5, 6)


def _rank3_ambient_encodings(
    *,
    num_times: int,
    seed: int,
    process_noise: float = 0.25,
) -> torch.Tensor:
    """Build a rank-3 linear Gaussian trajectory in ambient 8.

    Latent dynamics are diagonal AR(1) plus optional isotropic process
    noise, then an orthonormal embedding into :math:`\\mathbb{R}^{8}`.

    Parameters
    ----------
    num_times : int
        Length ``T``.
    seed : int
        RNG seed for the embedding, initial state, and process noise.
    process_noise : float, optional
        Standard deviation of latent process noise. Default ``0.25``.
        Use ``0`` for a noiseless embedding (ResDMD elbow).

    Returns
    -------
    Tensor
        Encodings with shape ``(T, 8)``, float64.
    """
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    true_k = torch.diag(torch.tensor([0.95, 0.90, 0.85], dtype=dtype))
    embedding, _ = torch.linalg.qr(
        torch.randn(_AMBIENT, _TRUE_RANK, dtype=dtype, generator=generator)
    )
    state = torch.randn(_TRUE_RANK, dtype=dtype, generator=generator)
    frames = [state]
    for _ in range(num_times - 1):
        state = state @ true_k.T
        if process_noise > 0.0:
            state = state + process_noise * torch.randn(
                _TRUE_RANK,
                dtype=dtype,
                generator=generator,
            )
        frames.append(state)
    latent = torch.stack(frames, dim=0)
    return latent @ embedding.T


def _finite_float(value: float) -> bool:
    """Return whether ``value`` is a finite Python float.

    Parameters
    ----------
    value : float
        Scalar.

    Returns
    -------
    bool
        ``True`` when finite.
    """
    return value == value and value not in (float("inf"), float("-inf"))


def test_rank_helpers_exported_and_not_on_root_all() -> None:
    """Rank helpers are identification exports, not root symbols."""
    for name in ("LatentRankReport", "select_latent_rank"):
        assert name in identification.__all__
        assert name not in set(koopman_graph.__all__)
        assert not hasattr(koopman_graph, name)
    text = _SOURCE.read_text(encoding="utf-8")
    assert "Wu2020VAMP" in text
    assert "10.1007/s00332-019-09567-y" in text
    assert "deeptime2021" in text
    assert "10.1088/2632-2153/ac3de0" in text
    assert "ColbrookTownsend2023ResDMD" in text
    assert "Ray Tune" in text
    assert "latent_dim" in text
    assert "koopman_graph.tuning" in text


@pytest.mark.parametrize(
    "criterion",
    ("vamp2", "resdmd_elbow", "stability_penalized"),
)
def test_linear_gaussian_oracle_recovers_true_rank(criterion: str) -> None:
    """Seeded rank-3 linear Gaussian map in ambient 8.

    ``vamp2`` / ``stability_penalized`` use driven AR(1) noise so all
    three modes stay excited. ``resdmd_elbow`` uses the noiseless
    embedding so max finite-dictionary residuals drop until the true
    rank (process noise inflates extra-mode residuals).
    """
    noise = 0.0 if criterion == "resdmd_elbow" else 0.25
    encodings = _rank3_ambient_encodings(
        num_times=48,
        seed=0,
        process_noise=noise,
    )
    report = select_latent_rank(encodings, _CANDIDATES, criterion=criterion)
    assert report.selected_rank == _TRUE_RANK
    assert report.numerical_rank == _TRUE_RANK
    assert report.candidates == (1, 2, 3)
    assert report.rejected_alternatives == ("4", "5", "6")
    assert report.criterion == criterion
    expected_pairs = 23 if criterion == "stability_penalized" else 47
    assert report.n_samples == expected_pairs
    assert len(report.scores) == 3
    assert all(_finite_float(score) for score in report.scores)


def test_node_layout_and_validation_split_recover_rank() -> None:
    """``(T, N, d)`` encodings and an explicit validation split still pick 3.

    Same construction as the oracle; the selected rank is an integer.
    """
    train = _rank3_ambient_encodings(num_times=64, seed=1)
    nodes = train[:40].unsqueeze(1).expand(-1, 3, -1).contiguous()
    report = select_latent_rank(
        nodes,
        _CANDIDATES,
        criterion="stability_penalized",
    )
    assert report.selected_rank == _TRUE_RANK
    split = select_latent_rank(
        train[:40],
        (1, 2, 3, 4),
        criterion="vamp2",
        validation_z=train[40:],
    )
    assert split.selected_rank == _TRUE_RANK
    assert split.n_samples == 39


def test_rank_selection_distinct_from_ray_tune() -> None:
    """The selector is not a Tune search space or latent_dim AutoML helper."""
    from koopman_graph.tuning.search_spaces import example_lr_latent_dim_space

    assert select_latent_rank is not example_lr_latent_dim_space
    assert not hasattr(select_latent_rank, "param_space")
    assert LatentRankReport is not type(example_lr_latent_dim_space)


def test_deeptime_vamp2_ordering_when_installed() -> None:
    """Optional deeptime VAMP-2 agrees that rank 3 beats rank 2 on the oracle.

    deeptime's score includes the constant singular value (+1); the
    in-tree mean-free score is compared after that shift. Relative
    ``0.05`` / absolute ``0.05`` match
    ``test_deeptime_oracle_agrees_when_installed``.
    """
    deeptime = pytest.importorskip("deeptime")
    from deeptime.decomposition import VAMP

    from koopman_graph.baselines.vamp2 import vamp2_score

    encodings = _rank3_ambient_encodings(num_times=64, seed=4)
    _, _, vh = torch.linalg.svd(encodings, full_matrices=False)
    projected = encodings @ vh[:_TRUE_RANK].T
    left = projected[:-1, :_TRUE_RANK]
    right = projected[1:, :_TRUE_RANK]
    our_score = float(vamp2_score(left, right))
    model = VAMP(lagtime=1, dim=None).fit(projected[:, :_TRUE_RANK].numpy())
    dt_score = float(model.fetch_model().score(2))
    assert our_score == pytest.approx(dt_score - 1.0, rel=0.05, abs=0.05)
    two = float(vamp2_score(projected[:-1, :2], projected[1:, :2]))
    assert two < our_score
    del deeptime


def test_rank_guards() -> None:
    """Invalid layouts, criteria, and empty grids raise with the constraint."""
    encodings = _rank3_ambient_encodings(num_times=16, seed=3)
    with pytest.raises(ValueError, match="criterion must be one of"):
        select_latent_rank(encodings, (1, 2, 3), criterion="aic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive ints"):
        select_latent_rank(encodings, (0, 2), criterion="vamp2")
    with pytest.raises(ValueError, match="non-empty sequence"):
        select_latent_rank(encodings, (), criterion="vamp2")
    with pytest.raises(ValueError, match="T >= 2"):
        select_latent_rank(encodings[:1], (1, 2), criterion="vamp2")
    with pytest.raises(ValueError, match="T >= 4"):
        select_latent_rank(
            encodings[:3],
            (1, 2),
            criterion="stability_penalized",
        )
    with pytest.raises(ValueError, match="no candidate rank is at most"):
        select_latent_rank(encodings, (16, 17), criterion="vamp2")
    with pytest.raises(ValueError, match="trailing layout must match"):
        select_latent_rank(
            encodings,
            (1, 2, 3),
            criterion="vamp2",
            validation_z=encodings[:, :4],
        )
    with pytest.raises(ValueError, match="ridge must be a finite"):
        select_latent_rank(encodings, (1, 2, 3), criterion="vamp2", ridge=-1.0)
