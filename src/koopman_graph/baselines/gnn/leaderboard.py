"""Protocol-matched traffic leaderboard adapters (not teaching ports).

``LeaderboardProtocol`` may have empty ``deviations`` because it names a
LibCity / BasicTS-style split rather than a teaching simplification.
Teaching :class:`~koopman_graph.baselines.gnn.ForecasterProtocol` still
rejects empty deviations.

References
----------
Wang, J., Jiang, J., Jiang, W., Li, C. & Zhao, W. X. (2021). LibCity:
An open library for traffic prediction. In *Proceedings of the 29th
International Conference on Advances in Geographic Information Systems*
(pp. 145–148). https://doi.org/10.1145/3474717.3483923
(``LibCity2021``)

Shao, Z., Wang, F., Xu, Y., Wei, W., Yu, C., Zhang, Z., Yao, D.,
Sun, T., Jin, G., Cao, X., Cong, G., Jensen, C. S. & Cheng, X. (2025).
Exploring progress in multivariate time series forecasting:
comprehensive benchmarking and heterogeneity analysis. *IEEE
Transactions on Knowledge and Data Engineering*, 37(1), 291–305.
https://doi.org/10.1109/TKDE.2024.3484454
(``BasicTS2024``)

Named split / horizon / metric objects follow those protocol papers.
The adapters are **not** LibCity or BasicTS implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.baselines.gnn.protocol import ForecastMetric

_RATIO_SUM_ATOL = 1e-6


@dataclass(frozen=True)
class LeaderboardProtocol:
    """Named split / horizon / metric for protocol-matched evaluation.

    Parameters
    ----------
    name : str
        Dataset or adapter identifier.
    history_len : int
        Lookback length (LibCity/BasicTS traffic default is 12).
    horizon : int
        Forecast horizon (default 12).
    train_ratio, val_ratio, test_ratio : float
        Temporal split fractions summing to 1.
    metric : {"mae", "rmse", "mape"}
        Primary reported metric.
    deviations : tuple of str, optional
        Optional notes. Empty is allowed (unlike teaching ports).
    zscore : bool, optional
        Whether inputs are z-scored on the train split. Default True.
    """

    name: str
    history_len: int
    horizon: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    metric: ForecastMetric
    deviations: tuple[str, ...] = ()
    zscore: bool = True

    def __post_init__(self) -> None:
        """Validate ratios and positive lengths.

        Notes
        -----
        Empty ``deviations`` is allowed for this protocol type.
        """
        if int(self.history_len) < 1 or int(self.horizon) < 1:
            raise ValueError("history_len and horizon must be positive")
        ratio_sum = (
            float(self.train_ratio) + float(self.val_ratio) + float(self.test_ratio)
        )
        if abs(ratio_sum - 1.0) > _RATIO_SUM_ATOL:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1, got {ratio_sum}"
            )


def metr_la_leaderboard_protocol() -> LeaderboardProtocol:
    """Return the 12/12, 0.7/0.1/0.2 z-score METR-LA protocol.

    Returns
    -------
    LeaderboardProtocol
        Named LibCity/BasicTS-style split.
    """
    return LeaderboardProtocol(
        name="metr-la",
        history_len=12,
        horizon=12,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        metric="mae",
        zscore=True,
    )


def pems_leaderboard_protocol(name: str = "pems-bay") -> LeaderboardProtocol:
    """Return the same 12-step protocol for a PEMS-family dataset.

    Parameters
    ----------
    name : str, optional
        Dataset label.

    Returns
    -------
    LeaderboardProtocol
        Named 12-step protocol.
    """
    return LeaderboardProtocol(
        name=name,
        history_len=12,
        horizon=12,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        metric="mae",
        zscore=True,
    )


def multi_seed_summary(values: Sequence[float]) -> tuple[Tensor, Tensor]:
    """Return mean and standard deviation of scalar runs.

    Parameters
    ----------
    values : sequence of float
        Per-seed scores.

    Returns
    -------
    tuple of Tensor
        ``(mean, std)``.
    """
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() < 1:
        raise ValueError("values must be non-empty")
    std = tensor.std(unbiased=False) if tensor.numel() == 1 else tensor.std()
    return tensor.mean(), std
